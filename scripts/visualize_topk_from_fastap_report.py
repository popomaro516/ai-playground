#!/usr/bin/env python3
"""
Visualize Top-k retrieval examples from a FastAP report markdown.

This is intentionally lightweight and self-contained so it can run in notebooks/colab
or locally without depending on the training code.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.topk_utils import load_lazy_dataset


def parse_retrieval_table(report_text: str, section_title: str) -> list[dict]:
    # Expect the exact table header used in the generated reports.
    pat = (
        re.escape(section_title)
        + r"\s*\n\| Rank \| Frame \| Score \| Label \|\n\|[^\n]*\n((?:\|[^\n]*\n)+)"
    )
    m = re.search(pat, report_text)
    if not m:
        return []
    rows_block = m.group(1)
    rows: list[dict] = []
    for line in rows_block.strip().splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 4:
            continue
        try:
            rank = int(cols[0])
            frame = int(cols[1])
            score = float(cols[2])
        except Exception:
            continue
        # Label in report may be -1 when labels are unavailable; we overwrite with CSV labels if present.
        try:
            label = int(cols[3])
        except Exception:
            label = -1
        rows.append({"rank": rank, "frame": frame, "score": score, "label": label})
    return rows


def bmode_normalize(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = np.squeeze(frame)
    a = np.abs(frame.astype(np.float64))
    amax = float(a.max()) if a.size else 1.0
    amax = max(amax, 1e-12)
    x = a / amax
    I = 20.0 * np.log10(np.maximum(x, 1e-12))
    I = np.clip(I, -60.0, 0.0)
    return (I + 60.0) / 60.0


def load_aspect_ratio_from_mat(f: h5py.File) -> float:
    # Acq/x holds refs: depth coords, lateral coords, time.
    xrefs = f["Acq/x"][...]
    z = np.array(f[xrefs[0, 0]]).reshape(-1)
    x = np.array(f[xrefs[1, 0]]).reshape(-1)
    dz = (float(z.max()) - float(z.min())) / max(1, (len(z) - 1))
    dx = (float(x.max()) - float(x.min())) / max(1, (len(x) - 1))
    if dx <= 0:
        return 1.0
    return dz / dx


def frame_from_lazy_dataset(dataset, idx: int) -> np.ndarray:
    tensor = dataset[idx]
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def to_pil_gray01(img01: np.ndarray) -> Image.Image:
    arr = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def aspect_correct(pil_img: Image.Image, r: float) -> Image.Image:
    # Depth axis is anisotropic; scale height by dz/dx.
    w, h = pil_img.size
    new_h = max(1, int(round(h * float(r))))
    return pil_img.resize((w, new_h), Image.Resampling.BICUBIC)


def crop_external_query_image(pil: Image.Image, white_thresh: int = 245) -> Image.Image:
    """Crop plot decorations (axes/ticks/colorbar/whitespace) from exported query images."""
    try:
        gray = np.asarray(pil.convert("L"))
        mask = gray < int(white_thresh)
        if not mask.any():
            return pil

        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        best_area = 0
        best_bbox = None  # (miny, minx, maxy_excl, maxx_excl)

        def _neighbors(y: int, x: int):
            if y > 0:
                yield y - 1, x
            if y + 1 < h:
                yield y + 1, x
            if x > 0:
                yield y, x - 1
            if x + 1 < w:
                yield y, x + 1

        for y in range(h):
            xs = np.where(mask[y] & (~visited[y]))[0]
            for x in xs:
                stack = [(int(y), int(x))]
                visited[y, x] = True
                area = 0
                miny = 10**9
                minx = 10**9
                maxy = -1
                maxx = -1
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    if cy < miny:
                        miny = cy
                    if cx < minx:
                        minx = cx
                    if cy > maxy:
                        maxy = cy
                    if cx > maxx:
                        maxx = cx
                    for ny, nx in _neighbors(cy, cx):
                        if mask[ny, nx] and (not visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                if area < 1000:
                    continue
                if area > best_area:
                    best_area = area
                    best_bbox = (miny, minx, maxy + 1, maxx + 1)

        if best_bbox is None:
            return pil

        miny, minx, maxy, maxx = best_bbox
        bw = max(1, maxx - minx)
        bh = max(1, maxy - miny)
        # Inset a bit to remove tick labels that touch the plot border.
        pad_l = int(round(bw * 0.03))
        pad_r = int(round(bw * 0.03))
        pad_t = int(round(bh * 0.03))
        pad_b = int(round(bh * 0.08))
        minx = max(0, minx + pad_l)
        maxx = min(w, maxx - pad_r)
        miny = max(0, miny + pad_t)
        maxy = min(h, maxy - pad_b)
        if maxx <= minx or maxy <= miny:
            return pil
        return pil.crop((minx, miny, maxx, maxy))
    except Exception:
        return pil


def make_tile(img: Image.Image, title: str, out_size: tuple[int, int]) -> Image.Image:
    img = img.convert("L")
    img.thumbnail(out_size, Image.Resampling.BICUBIC)
    canvas = Image.new("L", out_size, color=0)
    ox = (out_size[0] - img.size[0]) // 2
    oy = (out_size[1] - img.size[1]) // 2
    canvas.paste(img, (ox, oy))

    canvas_rgb = canvas.convert("RGB")
    draw = ImageDraw.Draw(canvas_rgb)
    font = ImageFont.load_default()
    bar_h = 22
    draw.rectangle([0, out_size[1] - bar_h, out_size[0], out_size[1]], fill=(0, 0, 0))
    draw.text((6, out_size[1] - bar_h + 4), title[:60], fill=(255, 255, 255), font=font)
    return canvas_rgb


def make_grid(tiles: list[Image.Image], cols: int, pad: int = 10, bg=(245, 245, 245)) -> Image.Image:
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out_w = cols * w + (cols + 1) * pad
    out_h = rows * h + (rows + 1) * pad
    out = Image.new("RGB", (out_w, out_h), color=bg)
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        out.paste(tile, (x, y))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset.mat"))
    ap.add_argument("--labels", type=Path, default=Path("annotations/dataset_labels.csv"))
    ap.add_argument("--external", type=Path, default=Path("data/invivo_alt.jpg"))
    ap.add_argument("--query-index", type=int, default=0)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--tile-w", type=int, default=360)
    ap.add_argument("--tile-h", type=int, default=320)
    ap.add_argument("--image-key", type=str, default="Acq/Amp")
    ap.add_argument("--image-axes", type=int, nargs="+", default=(0, 2, 1))
    args = ap.parse_args()

    report_text = args.report.read_text(encoding="utf-8", errors="ignore")
    internal = parse_retrieval_table(report_text, f"### Internal query (index={args.query_index})")
    external = parse_retrieval_table(report_text, "### External query (/content/data/invivo_normalized.png)")

    labels_map: dict[int, int] = {}
    if args.labels.exists():
        df = pd.read_csv(args.labels).set_index("frame").sort_index()
        if "label" in df.columns:
            labels_map = df["label"].astype(int).to_dict()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare cropped external query image for visualization.
    query_cropped_path = None
    if args.external.exists():
        q = Image.open(args.external).convert("RGB")
        q = crop_external_query_image(q)
        query_cropped_path = args.out_dir / "external_query_cropped.png"
        q.save(query_cropped_path)

    def _label(idx: int) -> str:
        if idx in labels_map:
            return str(labels_map[idx])
        return "?"

    if len(args.image_axes) not in (3, 4):
        raise ValueError("image_axes must have length 3 or 4")

    dataset = load_lazy_dataset(args.dataset, args.image_key, args.image_axes, normalize_255=False)
    with h5py.File(args.dataset, "r") as f:
        aspect_ratio = load_aspect_ratio_from_mat(f)

    # Internal grid: query frame + top-10.
    internal_tiles: list[Image.Image] = []
    if internal:
        q_raw = frame_from_lazy_dataset(dataset, args.query_index)
        q_img = aspect_correct(to_pil_gray01(bmode_normalize(q_raw)), aspect_ratio)
        internal_tiles.append(
            make_tile(q_img, f"query frame={args.query_index} y={_label(args.query_index)}", (args.tile_w, args.tile_h))
        )
        for row in internal:
            idx = int(row["frame"])
            raw = frame_from_lazy_dataset(dataset, idx)
            img = aspect_correct(to_pil_gray01(bmode_normalize(raw)), aspect_ratio)
            internal_tiles.append(
                make_tile(
                    img,
                    f"r{row['rank']} idx={idx} s={row['score']:.4f} y={_label(idx)}",
                    (args.tile_w, args.tile_h),
                )
            )
        grid = make_grid(internal_tiles, cols=args.cols)
        grid.save(args.out_dir / "topk_internal_query0.png")

    # External grid: cropped query + top-10.
    external_tiles: list[Image.Image] = []
    if query_cropped_path is not None:
        q = Image.open(query_cropped_path).convert("L")
        external_tiles.append(make_tile(q, f"query {query_cropped_path.name}", (args.tile_w, args.tile_h)))
    if external:
        for row in external:
            idx = int(row["frame"])
            raw = frame_from_lazy_dataset(dataset, idx)
            img = aspect_correct(to_pil_gray01(bmode_normalize(raw)), aspect_ratio)
            external_tiles.append(
                make_tile(
                    img,
                    f"r{row['rank']} idx={idx} s={row['score']:.4f} y={_label(idx)}",
                    (args.tile_w, args.tile_h),
                )
            )
        grid = make_grid(external_tiles, cols=args.cols)
        grid.save(args.out_dir / "topk_external_query.png")

    dataset.close()

    # Write a small summary for humans.
    def count_pos(rows: list[dict]) -> tuple[int, int]:
        if not labels_map:
            return (0, 0)
        ys = [labels_map.get(int(r["frame"])) for r in rows if int(r["frame"]) in labels_map]
        return (sum(1 for y in ys if y == 1), len(ys))

    internal_pos, internal_n = count_pos(internal)
    external_pos, external_n = count_pos(external)
    md = []
    md.append("# Top-k Visualization Notes")
    md.append("")
    md.append(f"- report: `{args.report}`")
    md.append(f"- dataset: `{args.dataset}`")
    md.append(f"- labels: `{args.labels}`" if args.labels.exists() else "- labels: (missing)")
    md.append(f"- external_query: `{args.external}`" if args.external.exists() else "- external_query: (missing)")
    md.append("")
    if internal:
        md.append(f"- internal_top10_pos: {internal_pos}/{internal_n} (from labels csv)")
    if external:
        md.append(f"- external_top10_pos: {external_pos}/{external_n} (from labels csv)")
    md.append("")
    md.append("- files:")
    if internal:
        md.append("  - `topk_internal_query0.png`")
    if external:
        md.append("  - `topk_external_query.png`")
    if query_cropped_path is not None:
        md.append("  - `external_query_cropped.png`")
    (args.out_dir / "topk_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
