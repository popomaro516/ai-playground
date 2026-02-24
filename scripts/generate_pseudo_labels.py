#!/usr/bin/env python3
"""Generate pseudo labels for ultrasound .mat sequences via lazy loading.

This script **is not executed automatically**. It is designed to be run
manually once you are ready to generate pseudo labels. It now supports
optional GPU acceleration (if `--device cuda` is specified and available).

Default paths and hyperparameters are defined below in this file.
Override any of them via CLI flags when necessary, for example:

    python scripts/generate_pseudo_labels.py \
        --mat_path data/custom.mat \
        --output_csv annotations/custom_labels.csv \
        --percentile 10 \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mat_ssl.datasets.lazy_mat import LazyMatImageDataset


# ---------------------------------------------------------------------------
# Default configuration (keep notebook/script self-contained)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_ANNOTATIONS_DIR = Path("annotations")
DEFAULT_DATASET_MAT = DEFAULT_DATA_DIR / "dataset.mat"
DEFAULT_OUTPUT_CSV = DEFAULT_ANNOTATIONS_DIR / "dataset_labels.csv"
DEFAULT_IMAGE_KEY = "Acq/Amp"
DEFAULT_IMAGE_AXES = (0, 2, 1)
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_PERCENTILE = 20.0
DEFAULT_MIN_STABLE_LENGTH = 3
DEFAULT_DTYPE = "float32"
DEFAULT_REFERENCE_PERCENTILE = 50.0
DEFAULT_REFERENCE_METRIC = "mse"
DEFAULT_REFERENCE_MODE = "bmode"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-label ultrasound frames")
    parser.add_argument(
        "--mat_path",
        type=str,
        default=str(DEFAULT_DATASET_MAT),
        help="Path to v7.3 .mat file",
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default=DEFAULT_IMAGE_KEY,
        help="Dataset key inside the .mat file",
    )
    parser.add_argument(
        "--image_axes",
        type=int,
        nargs="*",
        default=list(DEFAULT_IMAGE_AXES),
        help="Optional axis override (e.g. 0 2 1 for (frame, depth, lateral))",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Destination CSV for labels",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Window size for moving-average smoothing (set 1 to disable)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="Percentile threshold (lower values => more frames marked stable)",
    )
    parser.add_argument(
        "--min_stable_length",
        type=int,
        default=DEFAULT_MIN_STABLE_LENGTH,
        help="Minimum consecutive frames to keep as stable (shorter runs relabeled unstable)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        help="Conversion dtype for frames before score computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (e.g. 'cpu', 'cuda'). Stays on CPU if CUDA unavailable.",
    )
    parser.add_argument(
        "--reference_image",
        type=str,
        default="",
        help="Optional reference image path (e.g. docs/invivo_normalized.png).",
    )
    parser.add_argument(
        "--reference_percentile",
        type=float,
        default=DEFAULT_REFERENCE_PERCENTILE,
        help="Percentile threshold for reference similarity (lower => more similar).",
    )
    parser.add_argument(
        "--reference_metric",
        type=str,
        choices=("mse", "cosine"),
        default=DEFAULT_REFERENCE_METRIC,
        help="Similarity metric to reference image.",
    )
    parser.add_argument(
        "--reference_mode",
        type=str,
        choices=("bmode", "raw"),
        default=DEFAULT_REFERENCE_MODE,
        help="Preprocess frames before reference matching.",
    )
    return parser.parse_args()


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(arr, (window // 2,), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    if smoothed.shape[0] > arr.shape[0]:
        smoothed = smoothed[: arr.shape[0]]
    return smoothed.astype(np.float64)


def _bmode_tensor(frame: torch.Tensor) -> torch.Tensor:
    frame = torch.abs(frame)
    frame = frame / (frame.max() + 1e-12)
    frame = 20.0 * torch.log10(frame + 1e-12)
    frame = torch.clamp(frame, -60.0, 0.0)
    frame = (frame + 60.0) / 60.0
    return frame


def _load_reference_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # 1x1xH xW


def compute_motion_scores(dataset: LazyMatImageDataset, device: torch.device) -> np.ndarray:
    """Compute frame-to-frame mean absolute difference lazily."""
    scores: List[float] = []
    prev: Optional[torch.Tensor] = None
    for idx in range(len(dataset)):
        frame = dataset[idx]
        if isinstance(frame, (tuple, list)):
            frame = frame[0]
        frame = frame.to(device=device, dtype=torch.float32, non_blocking=True)
        if prev is None:
            scores.append(0.0)
        else:
            diff = torch.abs(frame - prev)
            scores.append(float(diff.mean().item()))
        prev = frame
    if prev is not None:
        prev = prev.detach().cpu()
    return np.asarray(scores, dtype=np.float64)


def compute_reference_scores(
    dataset: LazyMatImageDataset,
    device: torch.device,
    ref_image: torch.Tensor,
    mode: str,
    metric: str,
) -> np.ndarray:
    scores: List[float] = []
    ref = ref_image.to(device=device, dtype=torch.float32)
    ref_hw = (ref.shape[-2], ref.shape[-1])
    for idx in range(len(dataset)):
        frame = dataset[idx]
        if isinstance(frame, (tuple, list)):
            frame = frame[0]
        frame = frame.to(device=device, dtype=torch.float32, non_blocking=True)
        if frame.dim() == 2:
            frame = frame.unsqueeze(0)
        if frame.shape[0] > 1:
            frame = frame.mean(dim=0, keepdim=True)
        if mode == "bmode":
            frame = _bmode_tensor(frame)
        frame = frame.unsqueeze(0)  # 1x1xH xW
        frame = F.interpolate(frame, size=ref_hw, mode="bilinear", align_corners=False)
        if metric == "mse":
            score = (frame - ref).pow(2).mean().item()
        else:
            flat_frame = frame.view(1, -1)
            flat_ref = ref.view(1, -1)
            score = float(1.0 - F.cosine_similarity(flat_frame, flat_ref).item())
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def enforce_min_segment(labels: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1:
        return labels
    labels = labels.copy()
    run_start = None
    for idx, val in enumerate(labels):
        if val == 1 and run_start is None:
            run_start = idx
        elif val == 0 and run_start is not None:
            run_length = idx - run_start
            if run_length < min_len:
                labels[run_start:idx] = 0
            run_start = None
    if run_start is not None:
        run_length = len(labels) - run_start
        if run_length < min_len:
            labels[run_start:] = 0
    return labels


def save_csv(
    path: Path,
    scores: np.ndarray,
    smooth_scores: np.ndarray,
    labels: np.ndarray,
    ref_scores: Optional[np.ndarray] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame", "label", "score", "smooth_score"]
        if ref_scores is not None:
            header.append("ref_score")
        writer.writerow(header)
        if ref_scores is None:
            for idx, (lab, sc, sm) in enumerate(zip(labels, scores, smooth_scores)):
                writer.writerow([idx, int(lab), float(sc), float(sm)])
        else:
            for idx, (lab, sc, sm, rs) in enumerate(zip(labels, scores, smooth_scores, ref_scores)):
                writer.writerow([idx, int(lab), float(sc), float(sm), float(rs)])


def generate_labels(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    mat_path = Path(args.mat_path)
    output_csv = Path(args.output_csv)

    image_axes = tuple(args.image_axes) if args.image_axes else None

    dataset = LazyMatImageDataset(
        [str(mat_path)],
        image_key=args.image_key,
        transform=None,
        dtype=args.dtype,
        normalize_255=False,
        image_axes=image_axes,
    )
    scores = compute_motion_scores(dataset, device)
    smooth_scores = _moving_average(scores, args.smooth_window)
    threshold = np.percentile(smooth_scores, args.percentile)
    labels = (smooth_scores <= threshold).astype(np.int32)
    ref_scores: Optional[np.ndarray] = None
    if args.reference_image:
        ref_path = Path(args.reference_image)
        if not ref_path.exists():
            raise FileNotFoundError(f"reference_image not found: {ref_path}")
        ref_image = _load_reference_image(ref_path)
        ref_scores = compute_reference_scores(
            dataset,
            device,
            ref_image,
            mode=args.reference_mode,
            metric=args.reference_metric,
        )
        ref_threshold = np.percentile(ref_scores, args.reference_percentile)
        labels = labels & (ref_scores <= ref_threshold)
    labels = enforce_min_segment(labels, args.min_stable_length)
    save_csv(output_csv, scores, smooth_scores, labels, ref_scores)


def main() -> None:
    args = parse_args()
    generate_labels(args)


if __name__ == "__main__":
    main()
