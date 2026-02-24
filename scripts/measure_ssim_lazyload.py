#!/usr/bin/env python3
"""
Measure latency for lazily loading a single frame from dataset.mat and computing SSIM
against an invivo image.

Example:
  python scripts/measure_ssim_lazyload.py \\
    --mat_path data/dataset.mat \\
    --invivo_path docs/invivo_normalized.png \\
    --frame_index 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mat_ssl.datasets.lazy_mat import LazyMatImageDataset


def bmode_normalize(chw: np.ndarray) -> np.ndarray:
    x = np.abs(chw.astype(np.float32))
    m = float(x.max())
    x = x / (m + 1e-12)
    x = 20.0 * np.log10(x + 1e-12)
    x = np.clip((x + 60.0) / 60.0, 0.0, 1.0)
    return x


def gaussian_kernel_1d(win_size: int, sigma: float) -> np.ndarray:
    ax = np.arange(win_size) - win_size // 2
    kernel = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def gaussian_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.size // 2
    img_p = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=img_p)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=tmp)
    return out.astype(np.float32, copy=False)


def ssim_global(img1: np.ndarray, img2: np.ndarray, data_range: float) -> float:
    mu1 = float(img1.mean())
    mu2 = float(img2.mean())
    var1 = float(((img1 - mu1) ** 2).mean())
    var2 = float(((img2 - mu2) ** 2).mean())
    cov12 = float(((img1 - mu1) * (img2 - mu2)).mean())
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2.0 * mu1 * mu2 + c1) * (2.0 * cov12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
    return float(num / den) if den > 0 else 0.0


def compute_ssim(img1: np.ndarray, img2: np.ndarray, win_size: int = 11, sigma: float = 1.5) -> float:
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM requires same shape, got {img1.shape} vs {img2.shape}")

    h, w = img1.shape
    if min(h, w) < win_size:
        win_size = min(h, w)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        data_range = float(max(img1.max(), img2.max()) - min(img1.min(), img2.min()))
        data_range = data_range if data_range > 0 else 1.0
        return ssim_global(img1, img2, data_range)

    kernel = gaussian_kernel_1d(win_size, sigma)
    mu1 = gaussian_filter(img1, kernel)
    mu2 = gaussian_filter(img2, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 * img1, kernel) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, kernel) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, kernel) - mu1_mu2

    data_range = float(max(img1.max(), img2.max()) - min(img1.min(), img2.min()))
    data_range = data_range if data_range > 0 else 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


def load_invivo(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def to_hw_float(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.mean(axis=0)
    return arr.astype(np.float32, copy=False)


def resize_gray(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    pil = pil.resize(size, resample=Image.BICUBIC)
    return np.asarray(pil).astype(np.float32) / 255.0


def normalize_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    minv = float(min(a.min(), b.min()))
    maxv = float(max(a.max(), b.max()))
    if maxv - minv < 1e-12:
        return np.zeros_like(a), np.zeros_like(b)
    a = (a - minv) / (maxv - minv)
    b = (b - minv) / (maxv - minv)
    return a, b


def save_gray_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure SSIM latency with lazy-loaded frame.")
    parser.add_argument("--mat_path", type=Path, default=Path("data/dataset.mat"))
    parser.add_argument("--image_key", type=str, default="Acq/Amp")
    parser.add_argument("--image_axes", type=int, nargs="*", default=[0, 2, 1])
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--invivo_path", type=Path, default=Path("docs/invivo_normalized.png"))
    parser.add_argument(
        "--resize",
        choices=["invivo", "frame", "none"],
        default="invivo",
        help="Resize target so shapes match (default: invivo).",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeated measurements.")
    parser.set_defaults(bmode=True)
    parser.add_argument("--bmode", dest="bmode", action="store_true", help="Apply B-mode normalization to frame.")
    parser.add_argument("--no-bmode", dest="bmode", action="store_false", help="Disable B-mode normalization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mat_path.exists():
        raise FileNotFoundError(f"Missing mat file: {args.mat_path}")
    if not args.invivo_path.exists():
        raise FileNotFoundError(f"Missing invivo image: {args.invivo_path}")

    t0 = time.perf_counter()
    invivo = load_invivo(args.invivo_path)
    invivo_load_ms = (time.perf_counter() - t0) * 1000.0

    init_start = time.perf_counter()
    dataset = LazyMatImageDataset(
        [str(args.mat_path)],
        image_key=args.image_key,
        image_axes=tuple(args.image_axes),
        normalize_255=False,
        transform=None,
    )
    init_ms = (time.perf_counter() - init_start) * 1000.0

    if not (0 <= args.frame_index < len(dataset)):
        raise IndexError(f"frame_index {args.frame_index} out of range (0..{len(dataset)-1})")

    load_times = []
    prep_times = []
    ssim_times = []
    total_times = []
    ssim_values = []

    for run_idx in range(max(1, args.repeat)):
        t_load0 = time.perf_counter()
        frame = dataset[args.frame_index]
        t_load1 = time.perf_counter()

        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        frame = to_hw_float(frame)
        if args.bmode:
            frame = bmode_normalize(frame[np.newaxis, ...])[0]

        if args.resize == "invivo":
            frame = resize_gray(frame, (invivo.shape[1], invivo.shape[0]))
            invivo_use = invivo
        elif args.resize == "frame":
            invivo_use = resize_gray(invivo, (frame.shape[1], frame.shape[0]))
        else:
            if frame.shape != invivo.shape:
                raise ValueError(
                    f"Shapes differ with resize=none: frame={frame.shape} invivo={invivo.shape}"
                )
            invivo_use = invivo

        frame, invivo_use = normalize_pair(frame, invivo_use)
        if run_idx == 0:
            out_dir = Path("image_ssim")
            save_gray_image(out_dir / "invivo_used.png", invivo_use)
            save_gray_image(out_dir / "frame_used.png", frame)
        t_prep = time.perf_counter()

        ssim_val = compute_ssim(invivo_use, frame)
        t_done = time.perf_counter()

        load_times.append((t_load1 - t_load0) * 1000.0)
        prep_times.append((t_prep - t_load1) * 1000.0)
        ssim_times.append((t_done - t_prep) * 1000.0)
        total_times.append((t_done - t_load0) * 1000.0)
        ssim_values.append(ssim_val)

    dataset.close()

    print(f"invivo_load_ms={invivo_load_ms:.2f}")
    print(f"dataset_init_ms={init_ms:.2f}")
    for i, (lt, pt, st, tt, sv) in enumerate(zip(load_times, prep_times, ssim_times, total_times, ssim_values), 1):
        print(
            f"run={i} load_ms={lt:.2f} prep_ms={pt:.2f} ssim_ms={st:.2f} total_ms={tt:.2f} ssim={sv:.6f}"
        )

    if len(total_times) > 1:
        print(
            "avg_ms "
            f"load={np.mean(load_times):.2f} "
            f"prep={np.mean(prep_times):.2f} "
            f"ssim={np.mean(ssim_times):.2f} "
            f"total={np.mean(total_times):.2f}"
        )


if __name__ == "__main__":
    main()
