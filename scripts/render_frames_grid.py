#!/usr/bin/env python3
"""
Render selected frames from dataset.mat into a grid image.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mat_ssl.datasets.lazy_mat import LazyMatImageDataset
from scripts.measure_ssim_lazyload import bmode_normalize, to_hw_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render frames into a grid image.")
    parser.add_argument("--mat_path", type=Path, default=Path("data/dataset.mat"))
    parser.add_argument("--image_key", type=str, default="Acq/Amp")
    parser.add_argument("--image_axes", type=int, nargs="*", default=[0, 2, 1])
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="Frame indices in rank order.")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--tile_width", type=int, default=256)
    parser.add_argument("--tile_height", type=int, default=198)
    parser.add_argument("--output", type=Path, default=Path("image_ssim/frames_grid.png"))
    parser.set_defaults(bmode=True)
    parser.add_argument("--bmode", dest="bmode", action="store_true", help="Apply B-mode normalization to frame.")
    parser.add_argument("--no-bmode", dest="bmode", action="store_false", help="Disable B-mode normalization.")
    return parser.parse_args()


def frame_to_tile(frame: np.ndarray, size: Tuple[int, int], caption: str) -> Image.Image:
    img = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8), mode="L")
    img = img.resize(size, Image.BICUBIC).convert("RGB")
    info_h = 22
    tile = Image.new("RGB", (size[0], size[1] + info_h), color=(20, 20, 20))
    tile.paste(img, (0, 0))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    draw.text((6, size[1] + 4), caption, font=font, fill=(230, 230, 230))
    return tile


def arrange_grid(tiles: List[Image.Image], cols: int, padding: int) -> Image.Image:
    if not tiles:
        raise ValueError("No tiles to arrange")
    tile_w, tile_h = tiles[0].size
    rows = math.ceil(len(tiles) / cols)
    canvas_w = cols * tile_w + (cols + 1) * padding
    canvas_h = rows * tile_h + (rows + 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(12, 12, 12))
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = padding + c * (tile_w + padding)
        y = padding + r * (tile_h + padding)
        canvas.paste(tile, (x, y))
    return canvas


def main() -> int:
    args = parse_args()
    if not args.mat_path.exists():
        raise FileNotFoundError(f"Missing mat file: {args.mat_path}")

    dataset = LazyMatImageDataset(
        [str(args.mat_path)],
        image_key=args.image_key,
        image_axes=tuple(args.image_axes),
        normalize_255=False,
        transform=None,
    )

    tiles: List[Image.Image] = []
    for rank, idx in enumerate(args.frames, 1):
        if not (0 <= idx < len(dataset)):
            raise IndexError(f"frame_index {idx} out of range (0..{len(dataset)-1})")
        frame = dataset[idx]
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        frame = to_hw_float(frame)
        if args.bmode:
            frame = bmode_normalize(frame[np.newaxis, ...])[0]
        caption = f"#{rank} f={idx}"
        tile = frame_to_tile(frame, (args.tile_width, args.tile_height), caption)
        tiles.append(tile)

    dataset.close()

    grid = arrange_grid(tiles, cols=args.cols, padding=args.padding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
