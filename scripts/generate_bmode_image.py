#!/usr/bin/env python
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def load_coords(h5file):
    xref = h5file["Acq"]["x"]
    depth_ref = xref[0, 0]
    lateral_ref = xref[1, 0]
    depth = np.array(h5file[depth_ref]).reshape(-1)
    lateral = np.array(h5file[lateral_ref]).reshape(-1)
    return depth, lateral


def bmode(frame: np.ndarray) -> np.ndarray:
    frame = np.abs(frame)
    frame = frame / (frame.max() + 1e-12)
    frame = 20.0 * np.log10(frame + 1e-12)
    frame = np.clip(frame, -60.0, 0.0)
    frame = (frame + 60.0) / 60.0
    return frame


def main():
    parser = argparse.ArgumentParser(description="Generate a B-mode image from one frame.")
    parser.add_argument("--input", required=True, help="Path to .mat (HDF5) dataset")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to export")
    parser.add_argument("--flip-ud", action="store_true", help="Flip vertically (optional)")
    parser.add_argument("--use-key", default="Acq/Amp", help="Dataset key (default: Acq/Amp)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(in_path, "r") as f:
        if args.use_key not in f:
            raise KeyError(f"Dataset key not found: {args.use_key}")
        dset = f[args.use_key]
        frame = np.array(dset[args.frame])

        depth, lateral = load_coords(f)
        depth_len = depth.shape[0]
        lateral_len = lateral.shape[0]

        # Orient to (depth, lateral)
        if frame.shape == (lateral_len, depth_len):
            frame = frame.T
        elif frame.shape == (depth_len, lateral_len):
            pass
        elif frame.shape[::-1] == (depth_len, lateral_len):
            frame = frame.T

        img = bmode(frame)

        if args.flip_ud:
            img = np.flipud(img)

        depth_range = float(depth.max() - depth.min())
        lateral_range = float(lateral.max() - lateral.min())
        if lateral_range > 0:
            ratio = depth_range / lateral_range
        else:
            ratio = img.shape[0] / max(1, img.shape[1])

        width = img.shape[1]
        new_h = max(1, int(round(width * ratio)))

        # Resize using bilinear in float space
        pil_img = Image.fromarray(img.astype(np.float32), mode="F")
        pil_img = pil_img.resize((width, new_h), resample=Image.BILINEAR)

        img_resized = np.array(pil_img)
        img_resized = np.clip(img_resized, 0.0, 1.0)
        img_u8 = (img_resized * 255.0).astype(np.uint8)

        out = Image.fromarray(img_u8, mode="L")
        out.save(out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
