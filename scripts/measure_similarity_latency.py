#!/usr/bin/env python3
"""
Measure per-image latency from query ingestion to similarity ranking.

This script expects that you have already trained SimSiam on the ultrasound
dataset (e.g., via `notebook/simsiam_fastap_ultrasound.ipynb`) and exported:

- Checkpoint: checkpoints/simsiam_latest.pth  (dict with key 'state_dict')
- Frame embeddings: checkpoints/simsiam_embeddings.npy  (L2-normalized)

Given one or more query images, the script reports the time spent on:
  (a) loading & preprocessing the image
  (b) pushing it through the SimSiam backbone + projector
  (c) computing cosine similarities against the stored frame embeddings

Usage example:
    python scripts/measure_similarity_latency.py \\
        --checkpoint checkpoints/simsiam_latest.pth \\
        --embeddings checkpoints/simsiam_embeddings.npy \\
        --query data/invivo.jpg \\
        --topk 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------------------------------------------------------
# Model definitions (mirrors notebook implementation)

class Projector(nn.Module):
    def __init__(self, in_dim: int = 512, hid_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, in_dim: int = 2048, hid_dim: int = 512, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimSiam(nn.Module):
    """ResNet-18 backbone SimSiam (matching the notebook)."""

    def __init__(self, proj_dim: int = 2048):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # -> [B, 512, 1, 1]
        self.projector = Projector(in_dim=512, hid_dim=proj_dim, out_dim=proj_dim)
        self.predictor = Predictor(in_dim=proj_dim, hid_dim=512, out_dim=proj_dim)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        return torch.flatten(y, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.forward_backbone(x1)
        z1 = self.projector(h1)
        p1 = self.predictor(z1)

        h2 = self.forward_backbone(x2)
        z2 = self.projector(h2)
        p2 = self.predictor(z2)

        return p1, z1, p2, z2


# ---------------------------------------------------------------------------
# Utilities

def build_query_transform() -> transforms.Compose:
    """Preprocessing pipeline for external (breath-hold) images."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path: Path, device: torch.device) -> SimSiam:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    model = SimSiam()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """Ensure embeddings are float32 and L2-normalized (safety check)."""
    emb = emb.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return emb / norms


def iter_query_images(paths: Iterable[Path]) -> List[Path]:
    results: List[Path] = []
    for path in paths:
        if path.is_dir():
            results.extend(sorted(p for p in path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}))
        elif path.is_file():
            results.append(path)
        else:
            raise FileNotFoundError(f"Query path not found: {path}")
    if not results:
        raise ValueError("No query images found.")
    return results


def embed_single_image(model: SimSiam, img: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        xb = img.unsqueeze(0).to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        h = model.forward_backbone(xb)
        z = model.projector(h)
        z = F.normalize(z, dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return z.cpu().numpy()[0]


def measure_latency(
    model: SimSiam,
    embeddings: np.ndarray,
    query_paths: Iterable[Path],
    topk: int,
    device: torch.device,
) -> None:
    transform = build_query_transform()
    frame_count = embeddings.shape[0]
    topk = max(1, min(topk, frame_count))

    for path in query_paths:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        img = Image.open(path).convert("RGB")
        tensor = transform(img)
        query_emb = embed_single_image(model, tensor, device)
        sims = embeddings @ query_emb
        top_idx = np.argsort(-sims)[:topk]

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"[{path.name}] latency={elapsed*1000.0:.2f} ms (top-{topk} frames: {top_idx.tolist()})")


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure similarity latency per query image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SimSiam checkpoint (.pth) with state_dict")
    parser.add_argument("--embeddings", type=Path, required=True, help="Numpy file containing frame embeddings")
    parser.add_argument("--query", type=Path, nargs="+", required=True, help="Query image(s) or directory(ies)")
    parser.add_argument("--topk", type=int, default=10, help="Number of nearest frames to compute (default: 10)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")

    model = load_model(args.checkpoint, device=device)
    embeddings = normalize_embeddings(np.load(args.embeddings))

    query_paths = iter_query_images(args.query)
    print(f"Loaded {len(query_paths)} query image(s); dataset embeddings shape={embeddings.shape}; device={device}")

    measure_latency(model, embeddings, query_paths, topk=args.topk, device=device)


if __name__ == "__main__":
    main()

