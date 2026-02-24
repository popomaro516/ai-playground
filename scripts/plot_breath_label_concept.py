#!/usr/bin/env python3
"""Create a conceptual breathing waveform with exhale segments labeled as 1.

This is a schematic (not data-derived) figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot conceptual inhale/exhale with label=1 on exhale.")
    parser.add_argument("--output", type=Path, default=Path("outputs/breath_label_concept.png"))
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--points", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t = np.linspace(0.0, 2.0 * np.pi * args.cycles, args.points)

    # Base breathing waveform (smooth with slight asymmetry)
    wave = np.sin(t) + 0.2 * np.sin(2.0 * t + 0.6)

    # Exhale segments defined as only around troughs (narrow band)
    exhale = wave < (wave.min() * 0.92)
    # Canvas
    width, height = 1400, 520
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Layout
    left, right, top = 80, 40, 40
    plot_h = 280
    gap = 30
    bar_h = 80
    plot_top = top
    plot_bottom = plot_top + plot_h
    bar_top = plot_bottom + gap
    bar_bottom = bar_top + bar_h
    plot_w = width - left - right

    # Titles
    draw.text((left, 10), "Concept: label=1 during exhale (trough side)", fill=(0, 0, 0), font=font)
    draw.text((left, plot_top - 18), "Breathing signal", fill=(0, 0, 0), font=font)
    draw.text((left, bar_top - 18), "Labels", fill=(0, 0, 0), font=font)
    draw.text((width - 90, bar_bottom + 8), "Time", fill=(0, 0, 0), font=font)

    # Axes boxes
    draw.rectangle([left, plot_top, left + plot_w, plot_bottom], outline=(0, 0, 0), width=2)
    draw.rectangle([left, bar_top, left + plot_w, bar_bottom], outline=(0, 0, 0), width=2)

    # Map waveform to plot coords
    xmin, xmax = float(t.min()), float(t.max())
    ymin, ymax = float(wave.min()), float(wave.max())
    if ymax == ymin:
        ymax = ymin + 1.0

    def xpx(x: float) -> int:
        return int(round(left + (x - xmin) / (xmax - xmin) * plot_w))

    def ypx(y: float) -> int:
        return int(round(plot_top + (1.0 - (y - ymin) / (ymax - ymin)) * plot_h))

    # Zero line
    y0 = ypx(0.0)
    draw.line([(left, y0), (left + plot_w, y0)], fill=(150, 150, 150), width=1)

    # Waveform
    pts = [(xpx(float(x)), ypx(float(y))) for x, y in zip(t, wave)]
    if len(pts) >= 2:
        draw.line(pts, fill=(40, 40, 40), width=2)

    # Label bar segments
    step = max(1, len(t) // 220)
    for i in range(0, len(t) - step, step):
        x0, x1 = xpx(float(t[i])), xpx(float(t[i + step]))
        color = (58, 166, 85) if exhale[i] else (217, 217, 217)
        draw.rectangle([x0, bar_top + 1, x1, bar_bottom - 1], fill=color, outline=None)

    # Legend
    legend_x = width - 360
    legend_y = bar_bottom + 6
    draw.rectangle([legend_x, legend_y, legend_x + 16, legend_y + 16], fill=(58, 166, 85))
    draw.text((legend_x + 22, legend_y), "label=1 (exhale)", fill=(0, 0, 0), font=font)
    legend_y += 20
    draw.rectangle([legend_x, legend_y, legend_x + 16, legend_y + 16], fill=(217, 217, 217))
    draw.text((legend_x + 22, legend_y), "label=0 (inhale)", fill=(0, 0, 0), font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
