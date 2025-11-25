#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
from PIL import Image
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.topk_utils import (
    bmode_normalize,
    compute_depth_lateral_ratio,
    frame_tensor_to_numpy,
    load_lazy_dataset,
    parse_topk_table,
)


def load_frame_times(mat_path: Path) -> np.ndarray:
    with h5py.File(mat_path, "r") as f:
        acq = f["Acq"]
        time_ds = f[acq["x"][2, 0]]
        times = time_ds[...].astype(np.float64).ravel()
    return times


def make_reference_frames(
    dataset,
    frame_indices: Sequence[int],
    aspect_ratio: float,
    resize_width: int,
    depth_stride: int,
    lateral_stride: int,
) -> List[np.ndarray]:
    refs: List[np.ndarray] = []
    for idx in frame_indices:
        tensor = dataset[idx]
        vec = frame_to_vector(tensor, aspect_ratio, resize_width, depth_stride, lateral_stride)
        refs.append(vec)
    return refs


def frame_to_vector(
    frame_tensor: torch.Tensor,
    aspect_ratio: float,
    resize_width: int,
    depth_stride: int,
    lateral_stride: int,
) -> np.ndarray:
    chw = frame_tensor_to_numpy(frame_tensor)
    chw = chw[
        :,
        :: max(1, depth_stride),
        :: max(1, lateral_stride),
    ]
    normalized = bmode_normalize(chw)
    gray = normalized.squeeze()
    img = Image.fromarray((gray * 255.0).astype(np.uint8), mode="L")
    effective_ratio = aspect_ratio * (max(1, depth_stride) / max(1, lateral_stride))
    resize_height = max(1, int(round(resize_width * effective_ratio)))
    if img.size != (resize_width, resize_height):
        img = img.resize((resize_width, resize_height), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    vec = arr.reshape(-1)
    vec -= vec.mean()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def compute_reference_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("No reference vectors provided.")
    mean_vec = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm
    return mean_vec


def cosine_similarities(
    dataset,
    aspect_ratio: float,
    resize_width: int,
    reference_vector: np.ndarray,
    depth_stride: int,
    lateral_stride: int,
    frame_indices: Sequence[int],
) -> np.ndarray:
    sims = []
    for idx in frame_indices:
        tensor = dataset[idx]
        vec = frame_to_vector(tensor, aspect_ratio, resize_width, depth_stride, lateral_stride)
        sims.append(float(np.dot(vec, reference_vector)))
    return np.array(sims, dtype=np.float32)


def find_peaks(
    series: np.ndarray,
    threshold: float,
    min_distance: int,
) -> List[int]:
    peak_indices: List[int] = []
    for i in range(1, len(series) - 1):
        if series[i] < threshold:
            continue
        if series[i] < series[i - 1] or series[i] < series[i + 1]:
            continue
        if peak_indices and (i - peak_indices[-1]) < min_distance:
            if series[i] > series[peak_indices[-1]]:
                peak_indices[-1] = i
            continue
        peak_indices.append(i)
    return peak_indices


def compute_autocorrelation(series: np.ndarray) -> np.ndarray:
    centered = series - np.mean(series)
    corr = np.correlate(centered, centered, mode="full")
    return corr[corr.size // 2 :]


def summarize_periods(
    peak_indices: Sequence[int],
    frame_times: np.ndarray,
) -> Dict[str, float]:
    if len(peak_indices) < 2:
        return {
            "period_frames_mean": float("nan"),
            "period_frames_std": float("nan"),
            "period_seconds_mean": float("nan"),
            "period_seconds_std": float("nan"),
        }
    diffs_frames = np.diff(peak_indices)
    diffs_seconds = np.diff(frame_times[peak_indices])
    return {
        "period_frames_mean": float(np.mean(diffs_frames)),
        "period_frames_std": float(np.std(diffs_frames)),
        "period_seconds_mean": float(np.mean(diffs_seconds)),
        "period_seconds_std": float(np.std(diffs_seconds)),
    }


def plot_series(
    frame_times: np.ndarray,
    similarities: np.ndarray,
    peak_positions: Sequence[int],
    peak_values: Sequence[float],
    autocorr: np.ndarray,
    frame_step: int,
    period_estimate: float,
    output_path: Path,
) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    ax0.plot(frame_times, similarities, label="Cosine similarity", color="#1f77b4")
    if peak_positions:
        peak_times = frame_times[np.array(peak_positions)]
        ax0.scatter(peak_times, peak_values, color="#d62728", label="Detected peaks")
    ax0.set_title("Similarity vs. time")
    ax0.set_ylabel("Cosine similarity")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    lags = np.arange(autocorr.size) * frame_step
    ax1.plot(lags, autocorr, color="#2ca02c")
    ax1.set_title("Autocorrelation of similarity series")
    ax1.set_xlabel("Lag (frames)")
    ax1.set_ylabel("Autocorrelation")
    if not math.isnan(period_estimate):
        ax1.axvline(period_estimate, color="#ff7f0e", linestyle="--", label=f"~{period_estimate:.1f} frames")
        ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze periodicity of high-similarity frames.")
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--topk-md", type=Path, help="Markdown table with top-k frames for reference selection.")
    parser.add_argument("--reference-frames", type=int, nargs="*", help="Explicit frame indices to use as references.")
    parser.add_argument("--image-key", type=str, default="Acq/Amp")
    parser.add_argument("--image-axes", type=int, nargs="+", default=(0, 2, 1))
    parser.add_argument("--resize-width", type=int, default=128)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--lateral-stride", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=4, help="Sample every Nth frame when computing similarities.")
    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile threshold for peak detection.")
    parser.add_argument("--min-distance", type=int, default=20, help="Minimum distance between peaks in frames.")
    parser.add_argument("--output-json", type=Path, help="Path to save periodicity summary as JSON.")
    parser.add_argument("--output-plot", type=Path, help="Path to save similarity & autocorr plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.image_axes) not in (3, 4):
        raise ValueError("image_axes must have length 3 or 4")

    dataset = load_lazy_dataset(args.mat_path, args.image_key, args.image_axes, normalize_255=False)
    aspect_ratio = compute_depth_lateral_ratio(args.mat_path)
    frame_times = load_frame_times(args.mat_path)

    reference_frames: List[int] = []
    if args.reference_frames:
        reference_frames = list(args.reference_frames)
    elif args.topk_md:
        rows = parse_topk_table(args.topk_md)
        positive_frames = [frame for _, frame, _, label in rows if label == 1]
        reference_frames = positive_frames or [frame for _, frame, _, _ in rows[:3]]
    if not reference_frames:
        raise ValueError("Unable to determine reference frames; provide --reference-frames or --topk-md.")

    reference_vectors = make_reference_frames(
        dataset,
        reference_frames,
        aspect_ratio,
        args.resize_width,
        args.depth_stride,
        args.lateral_stride,
    )
    reference_vector = compute_reference_vector(reference_vectors)

    frame_step = max(1, args.frame_step)
    sampled_indices = list(range(0, len(dataset), frame_step))

    similarities = cosine_similarities(
        dataset,
        aspect_ratio,
        args.resize_width,
        reference_vector,
        args.depth_stride,
        args.lateral_stride,
        sampled_indices,
    )
    frame_times_sampled = frame_times[sampled_indices]
    threshold = float(np.percentile(similarities, args.percentile))
    min_distance_samples = max(1, int(round(args.min_distance / frame_step)))
    peak_positions = find_peaks(similarities, threshold=threshold, min_distance=min_distance_samples)
    peak_indices = [sampled_indices[p] for p in peak_positions]
    autocorr = compute_autocorrelation(similarities)

    summary = {
        "reference_frames": reference_frames,
        "resize_width": args.resize_width,
        "depth_stride": args.depth_stride,
        "lateral_stride": args.lateral_stride,
        "frame_step": frame_step,
        "threshold_percentile": args.percentile,
        "threshold_value": threshold,
        "min_distance_frames": args.min_distance,
        "min_distance_samples": min_distance_samples,
        "sampled_frame_indices": sampled_indices,
        "peak_sample_positions": peak_positions,
        "peak_indices": peak_indices,
        "peak_times": [float(frame_times[i]) for i in peak_indices],
        "peak_similarities": [float(similarities[p]) for p in peak_positions],
    }
    summary.update(summarize_periods(peak_indices, frame_times))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    period_estimate = summary["period_frames_mean"]
    if args.output_plot:
        plot_series(
            frame_times=frame_times_sampled,
            similarities=similarities,
            peak_positions=peak_positions,
            peak_values=[similarities[p] for p in peak_positions],
            autocorr=autocorr,
            frame_step=frame_step,
            period_estimate=period_estimate,
            output_path=args.output_plot,
        )

    dataset.close()

    if math.isnan(summary["period_frames_mean"]):
        print("Less than two peaks detected; unable to estimate period.")
    else:
        print(
            f"Detected {len(peak_indices)} peaks. "
            f"Period ~{summary['period_frames_mean']:.1f} frames "
            f"(~{summary['period_seconds_mean']:.2f} s)."
        )


if __name__ == "__main__":
    main()
