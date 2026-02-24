#!/usr/bin/env python3
"""
Compute SSIM between an invivo image and a list of frame indices (Top-K).
Outputs a Markdown table.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mat_ssl.datasets.lazy_mat import LazyMatImageDataset
from scripts.measure_ssim_lazyload import (
    load_invivo,
    bmode_normalize,
    compute_ssim,
    resize_gray,
    normalize_pair,
    to_hw_float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SSIM for Top-K frames.")
    parser.add_argument("--mat_path", type=Path, default=Path("data/dataset.mat"))
    parser.add_argument("--image_key", type=str, default="Acq/Amp")
    parser.add_argument("--image_axes", type=int, nargs="*", default=[0, 2, 1])
    parser.add_argument("--invivo_path", type=Path, default=Path("docs/invivo_normalized.png"))
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="Frame indices in rank order.")
    parser.add_argument(
        "--resize",
        choices=["invivo", "frame", "none"],
        default="invivo",
        help="Resize target so shapes match (default: invivo).",
    )
    parser.set_defaults(bmode=True)
    parser.add_argument("--bmode", dest="bmode", action="store_true", help="Apply B-mode normalization to frame.")
    parser.add_argument("--no-bmode", dest="bmode", action="store_false", help="Disable B-mode normalization.")
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("docs/ssim_topk_experiment_b_new.md"),
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip out-of-range frames and mark SSIM/Load as N/A instead of failing.",
    )
    return parser.parse_args()


def compute_for_frame(
    dataset: LazyMatImageDataset,
    invivo: np.ndarray,
    idx: int,
    resize_mode: str,
    use_bmode: bool,
) -> Tuple[float, float]:
    t0 = time.perf_counter()
    frame = dataset[idx]
    t1 = time.perf_counter()

    if hasattr(frame, "numpy"):
        frame = frame.numpy()
    frame = to_hw_float(frame)
    if use_bmode:
        frame = bmode_normalize(frame[np.newaxis, ...])[0]

    if resize_mode == "invivo":
        frame = resize_gray(frame, (invivo.shape[1], invivo.shape[0]))
        invivo_use = invivo
    elif resize_mode == "frame":
        invivo_use = resize_gray(invivo, (frame.shape[1], frame.shape[0]))
    else:
        if frame.shape != invivo.shape:
            raise ValueError(f"Shapes differ with resize=none: frame={frame.shape} invivo={invivo.shape}")
        invivo_use = invivo

    frame, invivo_use = normalize_pair(frame, invivo_use)
    ssim_val = compute_ssim(invivo_use, frame)
    load_ms = (t1 - t0) * 1000.0
    return ssim_val, load_ms


def main() -> int:
    args = parse_args()
    if not args.mat_path.exists():
        raise FileNotFoundError(f"Missing mat file: {args.mat_path}")
    if not args.invivo_path.exists():
        raise FileNotFoundError(f"Missing invivo image: {args.invivo_path}")

    invivo = load_invivo(args.invivo_path)
    dataset = LazyMatImageDataset(
        [str(args.mat_path)],
        image_key=args.image_key,
        image_axes=tuple(args.image_axes),
        normalize_255=False,
        transform=None,
    )

    rows = []
    max_idx = len(dataset) - 1
    for rank, idx in enumerate(args.frames, 1):
        if not (0 <= idx < len(dataset)):
            if args.allow_missing:
                rows.append((rank, idx, None, None))
                continue
            raise IndexError(f"frame_index {idx} out of range (0..{max_idx})")
        ssim_val, load_ms = compute_for_frame(dataset, invivo, idx, args.resize, args.bmode)
        rows.append((rank, idx, ssim_val, load_ms))

    dataset.close()

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Top-K SSIM (Experiment B, new)\n")
    lines.append("## Settings\n")
    lines.append(f"- mat_path: `{args.mat_path}`\n")
    lines.append(f"- invivo_path: `{args.invivo_path}`\n")
    lines.append(f"- resize: `{args.resize}`\n")
    lines.append(f"- bmode: `{args.bmode}`\n")
    lines.append("\n## Results\n")
    lines.append("| Rank | Frame | SSIM | Load ms |\n")
    lines.append("| --- | --- | --- | --- |\n")
    for rank, idx, ssim_val, load_ms in rows:
        if ssim_val is None:
            lines.append(f"| {rank} | {idx} | N/A | N/A |\n")
        else:
            lines.append(f"| {rank} | {idx} | {ssim_val:.6f} | {load_ms:.2f} |\n")

    args.output_md.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
