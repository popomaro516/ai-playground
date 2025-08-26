import argparse
import os
import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output .mat (HDF5) path")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--w", type=int, default=32)
    parser.add_argument("--c", type=int, default=3)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--image_key", type=str, default="images")
    parser.add_argument("--label_key", type=str, default="labels")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Create tiny uint8 images (0..255) and simple labels
    imgs = np.random.randint(0, 256, size=(args.n, args.h, args.w, args.c), dtype=np.uint8)
    labels = np.arange(args.n, dtype=np.int64) % max(1, args.classes)

    with h5py.File(args.out, "w") as f:
        f.create_dataset(args.image_key, data=imgs, compression="gzip")
        f.create_dataset(args.label_key, data=labels, compression="gzip")
    print(f"Wrote dummy HDF5 .mat to {args.out} with {args.n} samples")


if __name__ == "__main__":
    main()

