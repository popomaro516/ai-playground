#!/usr/bin/env python3
"""
Export a SimSiam-style frame pair (base + augmented) and the next frame (base).

This script reads a single frame lazily from a .mat (HDF5) dataset and saves:
  - frame_{i}_base.png: B-mode normalized base image
  - frame_{i}_aug.png: SimSiam augmentation result (denormalized for viewing)
  - frame_{i+1}_base.png: B-mode normalized base image for next frame
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_image_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    if key in f:
        return f[key]
    # search shallow for datasets ending with key
    for k in f.keys():
        if k.endswith(key):
            return f[k]
        if isinstance(f[k], h5py.Group):
            # handle common nested path like Acq/Amp
            if key in f[k]:
                return f[k][key]
            for subk in f[k].keys():
                full = f"{k}/{subk}"
                if full.endswith(key):
                    return f[full]
    raise KeyError(f"image_key '{key}' not found in file")


def load_coords(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    xref = f["Acq"]["x"]
    depth_ref = xref[0, 0]
    lateral_ref = xref[1, 0]
    depth = np.array(f[depth_ref]).reshape(-1)
    lateral = np.array(f[lateral_ref]).reshape(-1)
    return depth, lateral


def interpret_image_shape(
    shape: Tuple[int, ...], override: Optional[Tuple[int, ...]] = None
) -> Tuple[int, int, int, int, Tuple[int, ...]]:
    if override is not None:
        axes = tuple(override)
        if len(axes) not in (3, 4):
            raise ValueError("image_axes override must have length 3 or 4")
        if len(axes) != len(shape):
            raise ValueError("image_axes override must match dataset rank")
        dims = [shape[a] for a in axes]
        if len(axes) == 4:
            n, c, h, w = dims
            return n, c, h, w, axes
        n, h, w = dims
        return n, 1, h, w, axes

    rank = len(shape)
    if rank == 4:
        candidates = [
            (0, 3, 1, 2),
            (0, 1, 2, 3),
            (3, 2, 0, 1),
            (3, 0, 1, 2),
        ]
        for axes in candidates:
            n = shape[axes[0]]
            c = shape[axes[1]]
            h = shape[axes[2]]
            w = shape[axes[3]]
            if all(v > 0 for v in (n, c, h, w)):
                return n, c, h, w, axes
        raise ValueError(f"Unable to infer N,C,H,W from shape {shape}")
    if rank == 3:
        candidates = [
            (0, 1, 2),
            (0, 2, 1),
            (2, 0, 1),
            (2, 1, 0),
            (1, 0, 2),
            (1, 2, 0),
        ]
        for axes in candidates:
            n = shape[axes[0]]
            h = shape[axes[1]]
            w = shape[axes[2]]
            if all(v > 0 for v in (n, h, w)):
                return n, 1, h, w, axes
        raise ValueError(f"Unable to infer N,H,W from shape {shape}")
    raise ValueError(f"Unsupported image rank {rank}; expected 3D/4D")


def ensure_chw(arr: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
    n_axis = axes[0]
    remaining = []
    for a in axes[1:]:
        remaining.append(a if a < n_axis else a - 1)

    if len(remaining) == 3:
        c_pos, h_pos, w_pos = remaining
        return np.moveaxis(arr, (c_pos, h_pos, w_pos), (0, 1, 2))
    if len(remaining) == 2:
        h_pos, w_pos = remaining
        hw = np.moveaxis(arr, (h_pos, w_pos), (0, 1))
        return np.expand_dims(hw, axis=0)
    raise ValueError("Unexpected axes configuration for image array")


def bmode_normalize(chw: np.ndarray) -> np.ndarray:
    x = np.abs(chw.astype(np.float32))
    m = float(x.max())
    x = x / (m + 1e-12)
    x = 20.0 * np.log10(x + 1e-12)
    x = np.clip((x + 60.0) / 60.0, 0.0, 1.0)
    return x


def to_pil_3ch_from_chw01(chw01: np.ndarray) -> Image.Image:
    if chw01.shape[0] == 1:
        chw01 = np.repeat(chw01, 3, axis=0)
    hwc = np.transpose(chw01, (1, 2, 0))
    hwc = (hwc * 255.0).astype(np.uint8)
    return Image.fromarray(hwc)


def resize_bmode_for_view(img01: np.ndarray, depth: np.ndarray, lateral: np.ndarray) -> np.ndarray:
    depth_range = float(depth.max() - depth.min())
    lateral_range = float(lateral.max() - lateral.min())
    if lateral_range > 0:
        ratio = depth_range / lateral_range
    else:
        ratio = img01.shape[0] / max(1, img01.shape[1])

    width = img01.shape[1]
    new_h = max(1, int(round(width * ratio)))
    pil_img = Image.fromarray(img01.astype(np.float32), mode="F")
    pil_img = pil_img.resize((width, new_h), resample=Image.BILINEAR)
    out = np.array(pil_img)
    return np.clip(out, 0.0, 1.0)


def build_simsiam_aug() -> T.Compose:
    return T.Compose([
        T.Lambda(lambda x: to_pil_3ch_from_chw01(bmode_normalize(x))),
        T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_mild_aug() -> T.Compose:
    """Doc-friendly augmentation: keep identity recognizable."""
    return T.Compose([
        T.Lambda(lambda x: to_pil_3ch_from_chw01(bmode_normalize(x))),
        T.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(0.08, 0.08, 0.08, 0.02),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_dark_center(img01: np.ndarray) -> Tuple[float, float]:
    h, w = img01.shape
    stride = max(1, min(h, w) // 256)
    small = img01[::stride, ::stride]
    thresh = np.percentile(small, 2.0)
    ys, xs = np.where(small <= thresh)
    if ys.size == 0:
        y, x = np.unravel_index(np.argmin(small), small.shape)
    else:
        y = float(np.mean(ys))
        x = float(np.mean(xs))
    return y * stride, x * stride


def crop_box(center_yx: Tuple[float, float], crop_size: int, h: int, w: int) -> Tuple[int, int, int, int]:
    size = min(crop_size, h, w)
    half = size / 2.0
    cy, cx = center_yx
    top = int(round(cy - half))
    left = int(round(cx - half))
    top = max(0, min(top, h - size))
    left = max(0, min(left, w - size))
    return left, top, left + size, top + size


def denorm_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().float()
    if t.ndim == 2:
        t = t.unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    t = t * std + mean
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export base + SimSiam-augmented frame views.")
    parser.add_argument("--mat_path", type=Path, default=Path("data/dataset.mat"))
    parser.add_argument("--image_key", type=str, default="Acq/Amp")
    parser.add_argument("--image_axes", type=int, nargs="*", default=[0, 2, 1])
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/simsiam_views"))
    parser.add_argument(
        "--aug_mode",
        type=str,
        choices=("simsiam", "mild", "crop", "none"),
        default="crop",
        help="Augmentation mode for visualization.",
    )
    parser.add_argument("--crop_size", type=int, default=224, help="Base crop size for aug_mode=crop.")
    parser.add_argument("--crop_scale", type=float, default=0.7, help="Scale factor for crop size.")
    parser.add_argument("--crop_jitter", type=int, default=0, help="Center jitter in pixels for crop.")
    parser.add_argument(
        "--crop_anchor",
        type=str,
        choices=("left_bottom", "dark", "center"),
        default="left_bottom",
        help="Anchor for crop center (default: left_bottom).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_frame(
    dset: h5py.Dataset, idx: int, axes: Tuple[int, ...]
) -> np.ndarray:
    slicer = [slice(None)] * dset.ndim
    slicer[axes[0]] = idx
    arr = np.asarray(dset[tuple(slicer)])
    return ensure_chw(arr, axes)


def main() -> None:
    args = parse_args()
    if not args.mat_path.exists():
        raise FileNotFoundError(f"Missing mat file: {args.mat_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.mat_path, "r") as f:
        dset = resolve_image_dataset(f, args.image_key)
        n, _, _, _, axes = interpret_image_shape(dset.shape, override=tuple(args.image_axes))
        depth, lateral = load_coords(f)

        if args.frame_index < 0 or args.frame_index >= n:
            raise IndexError(f"frame_index {args.frame_index} out of range (0..{n-1})")
        if args.frame_index + 1 >= n:
            raise IndexError(f"frame_index+1 {args.frame_index+1} out of range (0..{n-1})")

        frame0 = load_frame(dset, args.frame_index, axes)
        frame1 = load_frame(dset, args.frame_index + 1, axes)

    # Base images (B-mode, resized with physical aspect ratio)
    base0_chw = bmode_normalize(frame0)
    base1_chw = bmode_normalize(frame1)
    base0_hw = base0_chw[0] if base0_chw.shape[0] == 1 else base0_chw.mean(axis=0)
    base1_hw = base1_chw[0] if base1_chw.shape[0] == 1 else base1_chw.mean(axis=0)
    base0_hw = resize_bmode_for_view(base0_hw, depth, lateral)
    base1_hw = resize_bmode_for_view(base1_hw, depth, lateral)
    base0 = to_pil_3ch_from_chw01(base0_hw[None, ...])
    base1 = to_pil_3ch_from_chw01(base1_hw[None, ...])

    # Augmented view from frame0 (for visualization)
    if args.aug_mode == "simsiam":
        aug = build_simsiam_aug()
        aug0 = aug(frame0)
        aug0_img = denorm_tensor_to_pil(aug0)
    elif args.aug_mode == "mild":
        aug = build_mild_aug()
        aug0 = aug(frame0)
        aug0_img = denorm_tensor_to_pil(aug0)
    elif args.aug_mode == "crop":
        if args.crop_anchor == "dark":
            center = find_dark_center(base0_hw)
        elif args.crop_anchor == "center":
            center = (base0_hw.shape[0] * 0.5, base0_hw.shape[1] * 0.5)
        else:
            center = (base0_hw.shape[0] * 0.8, base0_hw.shape[1] * 0.2)
        rng = np.random.default_rng(args.seed)
        jitter_y = int(rng.integers(-args.crop_jitter, args.crop_jitter + 1))
        jitter_x = int(rng.integers(-args.crop_jitter, args.crop_jitter + 1))
        center = (center[0] + jitter_y, center[1] + jitter_x)
        crop_size = max(64, int(round(args.crop_size * args.crop_scale)))
        box = crop_box(center, crop_size, base0.height, base0.width)
        aug0_img = base0.crop(box).resize((224, 224), Image.BICUBIC)
    else:
        aug0_img = base0.copy()

    base0_path = out_dir / f"frame_{args.frame_index:05d}_base.png"
    aug0_path = out_dir / f"frame_{args.frame_index:05d}_aug.png"
    base1_path = out_dir / f"frame_{args.frame_index + 1:05d}_base.png"

    base0.save(base0_path)
    aug0_img.save(aug0_path)
    base1.save(base1_path)

    print("Saved:")
    print(f"  {base0_path}")
    print(f"  {aug0_path}")
    print(f"  {base1_path}")


if __name__ == "__main__":
    main()
