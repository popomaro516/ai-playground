#!/usr/bin/env python3
"""Convert ultrasound MAT frames into a GIF animation (memory friendly).

This script loads frames lazily via LazyMatImageDataset, applies B-mode
log-compression, resizes them to a manageable resolution, and exports a GIF.

Usage example:
    python scripts/mat_to_video.py \
        --mat-path data/dataset.mat \
        --output result/20251030/dataset_preview.gif \
        --start 0 --end 200 --step 2 \
        --resize-width 256 --resize-height 256 \
        --fps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.topk_utils import (
    bmode_normalize,
    load_lazy_dataset,
)


def frame_indices(total: int, start: int, end: int | None, step: int) -> Sequence[int]:
    s = max(0, start)
    e = total if end is None else min(total, end)
    if s >= e:
        raise ValueError(f"Invalid range: start={start}, end={end}, dataset length={total}")
    if step <= 0:
        raise ValueError("step must be positive")
    return list(range(s, e, step))


def tensor_to_image(
    tensor: torch.Tensor,
    resize_width: int,
    resize_height: int,
    colormap: str,
) -> Image.Image:
    chw = tensor.detach().cpu().numpy()
    gray = bmode_normalize(chw).squeeze()
    img = Image.fromarray((gray * 255.0).astype(np.uint8), mode="L")
    if resize_width > 0 and resize_height > 0:
        img = img.resize((resize_width, resize_height), Image.BICUBIC)
    if colormap == "grayscale":
        return img.convert("P")
    # minimal false color via simple replication to RGB
    img_rgb = Image.merge("RGB", (img, img, img))
    return img_rgb


def save_gif(frames: List[Image.Image], output: Path, fps: float) -> None:
    if not frames:
        raise ValueError("No frames to save.")
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(fps, 0.1))))
    first, *tail = frames
    first.save(
        output,
        save_all=True,
        append_images=tail,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MAT sequence to GIF.")
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--image-key", type=str, default="Acq/Amp")
    parser.add_argument("--image-axes", type=int, nargs="+", default=(0, 2, 1))
    parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive).")
    parser.add_argument("--end", type=int, help="End frame index (exclusive).")
    parser.add_argument("--step", type=int, default=2, help="Frame sampling stride.")
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--resize-height", type=int, default=256)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--colormap", type=str, choices=("grayscale", "rgb"), default="grayscale")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_lazy_dataset(args.mat_path, args.image_key, args.image_axes, normalize_255=False)
    indices = frame_indices(len(dataset), args.start, args.end, args.step)

    frames: List[Image.Image] = []
    for idx in indices:
        tensor = dataset[idx]
        frames.append(
            tensor_to_image(tensor, args.resize_width, args.resize_height, args.colormap)
        )

    save_gif(frames, args.output, args.fps)
    dataset.close()
    print(f"Saved {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
