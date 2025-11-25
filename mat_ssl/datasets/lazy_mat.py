import glob
import os
from typing import List, Optional, Tuple, Sequence, Dict, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LazyMatImageDataset(Dataset):
    """
    Lazily iterates images stored across multiple MATLAB v7.3 (HDF5) .mat files.

    Each .mat file is expected to contain an HDF5 dataset for images, e.g. /images.
    Supported per-file shapes:
      - [N, H, W, C] (common)
      - [N, C, H, W]
      - [H, W, C, N]
      - [C, H, W, N]

    Optionally, a label dataset can be provided (for supervised fine-tuning), same N.

    This dataset opens files lazily and reads a single sample per __getitem__ without
    loading all data into memory. Files are kept open on demand and closed on __del__.
    """

    def __init__(
        self,
        mat_files: Sequence[str],
        image_key: str = "images",
        label_key: Optional[str] = None,
        transform=None,
        dtype: str = "float32",
        normalize_255: bool = True,
        image_axes: Optional[Tuple[int, ...]] = None,
    ):
        self.mat_files = list(mat_files)
        if len(self.mat_files) == 1 and any(ch in self.mat_files[0] for ch in "*?[]"):
            # Support passing a glob as a single string
            self.mat_files = sorted(glob.glob(self.mat_files[0]))

        if not self.mat_files:
            raise ValueError("No .mat files found for dataset")

        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        self.dtype = dtype
        self.normalize_255 = normalize_255
        self._axes_override = image_axes

        # Internal index mapping: global_idx -> (file_idx, local_idx)
        self._index: List[Tuple[int, int]] = []
        # File handles and metadata
        self._files: List[Optional[h5py.File]] = [None] * len(self.mat_files)
        self._per_file_counts: List[int] = []
        self._image_axes: List[Tuple[int, int, int, int]] = []  # mapping to N,C,H,W

        for fi, path in enumerate(self.mat_files):
            f = h5py.File(path, "r")
            try:
                if self.image_key not in f:
                    # Try common nested paths
                    if f.get("/", None) is not None and isinstance(f["/"], h5py.Group):
                        # search shallow
                        found = None
                        for k in f.keys():
                            if k.endswith(self.image_key):
                                found = k
                                break
                        if found is None:
                            raise KeyError(f"image_key '{self.image_key}' not found in {path}")
                        img_ds = f[found]
                    else:
                        raise KeyError(f"image_key '{self.image_key}' not found in {path}")
                else:
                    img_ds = f[self.image_key]

                shape = tuple(img_ds.shape)
                n, c, h, w, axes = _interpret_image_shape(shape, override=self._axes_override)

                # If labels requested, validate existence and length
                if self.label_key is not None:
                    if self.label_key not in f:
                        raise KeyError(f"label_key '{self.label_key}' not found in {path}")
                    lbl_ds = f[self.label_key]
                    if lbl_ds.shape[0] != n and lbl_ds.shape[-1] != n:
                        raise ValueError(
                            f"Labels size mismatch in {path}: labels shape {lbl_ds.shape}, N={n}"
                        )

                self._per_file_counts.append(n)
                base = len(self._index)
                for li in range(n):
                    self._index.append((fi, li))
                self._image_axes.append(axes)
                self._files[fi] = f  # keep open for speed
            except Exception:
                f.close()
                raise

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        file_idx, local_idx = self._index[idx]
        f = self._files[file_idx]
        if f is None:
            f = h5py.File(self.mat_files[file_idx], "r")
            self._files[file_idx] = f

        img_ds = f[self.image_key] if self.image_key in f else f[next(k for k in f.keys() if k.endswith(self.image_key))]
        axes = self._image_axes[file_idx]
        # Read a single item lazily via slicing on N dimension
        # Get view with N at axis 0
        slicer = [slice(None)] * img_ds.ndim
        slicer[axes[0]] = local_idx
        arr = np.array(img_ds[tuple(slicer)])  # loads only one sample

        # Move to C,H,W with helper (handles axis drop of N)
        arr = _ensure_chw(arr, axes)

        # Type and scale
        if arr.dtype.kind in ("u", "i"):
            arr = arr.astype(self.dtype)
            if self.normalize_255:
                arr = arr / 255.0
        else:
            arr = arr.astype(self.dtype)

        if self.transform is not None:
            img = self.transform(arr)
        else:
            img = torch.from_numpy(arr)

        if self.label_key is not None:
            lbl_ds = f[self.label_key]
            label = _read_label(lbl_ds, local_idx)
            return img, label
        else:
            return img

    def close(self):
        for i, f in enumerate(self._files):
            if f is not None:
                f.close()
                self._files[i] = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _interpret_image_shape(
    shape: Tuple[int, ...], override: Optional[Tuple[int, ...]] = None
) -> Tuple[int, int, int, int, Tuple[int, ...]]:
    """Infer N, C, H, W and axes order given a shape.

    Returns (N, C, H, W, axes) where axes maps original axes to the logical order
    (N, C, H, W). For 2D grayscale (N, H, W) inputs the channel dimension is
    synthesized and axes is length 3 (mapping to N, H, W).
    """
    if override is not None:
        axes = tuple(override)
        rank = len(shape)
        if len(axes) not in (3, 4):
            raise ValueError("image_axes override must have length 3 or 4")
        if len(axes) != rank:
            raise ValueError(f"image_axes override length {len(axes)} does not match rank {rank}")
        dims = [shape[a] for a in axes]
        if len(axes) == 4:
            n, c, h, w = dims
            return n, c, h, w, axes
        else:
            n, h, w = dims
            return n, 1, h, w, axes

    rank = len(shape)
    if rank == 4:
        candidates = [
            (0, 3, 1, 2),  # N,H,W,C -> N,C,H,W
            (0, 1, 2, 3),  # N,C,H,W -> N,C,H,W
            (3, 2, 0, 1),  # H,W,C,N -> N,C,H,W
            (3, 0, 1, 2),  # C,H,W,N -> N,C,H,W
        ]
        for axes in candidates:
            n = shape[axes[0]]
            c = shape[axes[1]]
            h = shape[axes[2]]
            w = shape[axes[3]]
            if all(x > 0 for x in (n, c, h, w)):
                return n, c, h, w, axes
        raise ValueError(f"Unable to infer N,C,H,W from shape {shape}")
    elif rank == 3:
        candidates = [
            (0, 1, 2),  # N,H,W
            (0, 2, 1),  # N,W,H
            (2, 0, 1),  # W,N,H
            (2, 1, 0),  # W,H,N
            (1, 0, 2),  # H,N,W
            (1, 2, 0),  # H,W,N
        ]
        for axes in candidates:
            n = shape[axes[0]]
            h = shape[axes[1]]
            w = shape[axes[2]]
            if all(x > 0 for x in (n, h, w)):
                return n, 1, h, w, axes
        raise ValueError(f"Unable to infer N,H,W from shape {shape}")
    else:
        raise ValueError(f"Unsupported image rank {rank}; expected 3D/4D, got {shape}")


def _ensure_chw(arr: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
    """Move axes to CHW given original axes mapping for N,(C),H,W."""
    n_axis = axes[0]
    remaining_axes = []
    for a in axes[1:]:
        if a < n_axis:
            remaining_axes.append(a)
        else:
            remaining_axes.append(a - 1)

    if len(remaining_axes) == 3:
        c_pos, h_pos, w_pos = remaining_axes
        return np.moveaxis(arr, (c_pos, h_pos, w_pos), (0, 1, 2))
    elif len(remaining_axes) == 2:
        h_pos, w_pos = remaining_axes
        arr_hw = np.moveaxis(arr, (h_pos, w_pos), (0, 1))
        return np.expand_dims(arr_hw, axis=0)
    else:
        raise ValueError("Unexpected axes configuration for image array")


def _read_label(lbl_ds: h5py.Dataset, idx: int) -> int:
    arr = np.array(lbl_ds[idx])
    if arr.ndim == 0:
        val = int(arr)
    else:
        val = int(arr.reshape(-1)[0])
    return val


class TwoCropsTransform:
    """Return two transformed views of the same input (for SimSiam)."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x: np.ndarray):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k
