from __future__ import annotations

from typing import Tuple

import h5py


def inspect_mat_hdf5(path: str):
    """Print a summary of top-level datasets and shapes for a v7.3 .mat file."""
    with h5py.File(path, "r") as f:
        for k, v in f.items():
            try:
                shape = getattr(v, "shape", None)
            except Exception:
                shape = None
            print(k, type(v), shape)


def is_v73_hdf5(path: str) -> bool:
    try:
        with h5py.File(path, "r") as _:
            return True
    except Exception:
        return False

