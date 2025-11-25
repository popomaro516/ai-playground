#!/usr/bin/env python
import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.topk_utils import (
    bmode_normalize,
    compute_depth_lateral_ratio,
    load_lazy_dataset,
    parse_topk_table,
)


def frame_to_tile(
    frame_tensor: torch.Tensor,
    target_width: int,
    aspect_ratio: float,
    caption: str,
    caption_color: Tuple[int, int, int],
) -> Image.Image:
    chw = frame_tensor.detach().cpu().numpy()
    normalized = bmode_normalize(chw)
    gray = (normalized.squeeze() * 255.0).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    new_height = max(1, int(round(target_width * aspect_ratio)))
    img = img.resize((target_width, new_height), Image.BICUBIC)
    img = img.convert("RGB")

    info_height = 22
    tile = Image.new("RGB", (target_width, new_height + info_height), color=(20, 20, 20))
    tile.paste(img, (0, 0))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    draw.text((6, new_height + 4), caption, font=font, fill=caption_color)
    return tile


def arrange_grid(tiles: List[Image.Image], cols: int, padding: int, bg_color=(12, 12, 12)) -> Image.Image:
    if not tiles:
        raise ValueError("No tiles to arrange")
    tile_w, tile_h = tiles[0].size
    rows = math.ceil(len(tiles) / cols)
    canvas_w = cols * tile_w + (cols + 1) * padding
    canvas_h = rows * tile_h + (rows + 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = padding + c * (tile_w + padding)
        y = padding + r * (tile_h + padding)
        canvas.paste(tile, (x, y))
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Render Top-K frames into a grid image.")
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--topk-md", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-key", type=str, default="Acq/Amp")
    parser.add_argument("--image-axes", type=int, nargs="+", default=(0, 2, 1))
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--padding", type=int, default=16)
    args = parser.parse_args()

    if len(args.image_axes) not in (3, 4):
        raise ValueError("image_axes must have length 3 or 4")

    rows = parse_topk_table(args.topk_md)

    dataset = load_lazy_dataset(args.mat_path, args.image_key, args.image_axes, normalize_255=False)

    aspect_ratio = compute_depth_lateral_ratio(args.mat_path)

    tiles: List[Image.Image] = []
    for rank, frame_idx, sim, label in rows:
        tensor = dataset[frame_idx]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(np.asarray(tensor))
        caption = f"Rank {rank} | Frame {frame_idx} | Sim {sim:.3f} | Label {label}"
        caption_color = (0, 200, 120) if label == 1 else (220, 80, 60)
        tile = frame_to_tile(tensor, args.width, aspect_ratio, caption, caption_color)
        tiles.append(tile)

    grid = arrange_grid(tiles, cols=args.cols, padding=args.padding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    dataset.close()


if __name__ == "__main__":
    main()
