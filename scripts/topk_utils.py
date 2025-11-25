#!/usr/bin/env python
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mat_ssl.datasets.lazy_mat import LazyMatImageDataset

_ROW_PATTERN = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(\d+)\s*\|"
)


def parse_topk_table(md_path: Path) -> List[Tuple[int, int, float, int]]:
    rows: List[Tuple[int, int, float, int]] = []
    with md_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = _ROW_PATTERN.match(line)
            if m:
                rank = int(m.group(1))
                frame = int(m.group(2))
                sim = float(m.group(3))
                label = int(m.group(4))
                rows.append((rank, frame, sim, label))
    if not rows:
        raise ValueError(f"No table rows parsed from {md_path}")
    rows.sort(key=lambda x: x[0])
    return rows


def load_lazy_dataset(
    mat_path: Path,
    image_key: str,
    image_axes: Sequence[int],
    normalize_255: bool = False,
) -> LazyMatImageDataset:
    return LazyMatImageDataset(
        [str(mat_path)],
        image_key=image_key,
        image_axes=tuple(image_axes),
        normalize_255=normalize_255,
    )


def bmode_normalize(chw: np.ndarray) -> np.ndarray:
    x = np.abs(chw.astype(np.float32))
    x /= float(x.max() + 1e-12)
    x = 20.0 * np.log10(x + 1e-12)
    x = np.clip((x + 60.0) / 60.0, 0.0, 1.0)
    return x


def compute_depth_lateral_ratio(mat_path: Path) -> float:
    with h5py.File(mat_path, "r") as f:
        acq = f["Acq"]
        depth = f[acq["x"][0, 0]][...].astype(np.float64).ravel()
        lateral = f[acq["x"][1, 0]][...].astype(np.float64).ravel()
    depth_range = float(depth.max() - depth.min())
    lateral_range = float(lateral.max() - lateral.min())
    if lateral_range <= 0.0:
        raise ValueError("Invalid lateral range for aspect ratio")
    return depth_range / lateral_range


def frame_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
