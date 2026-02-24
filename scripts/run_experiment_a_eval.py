#!/usr/bin/env python3
"""Reproduce Experiment A (2025-11-05) evaluation on local dataset.

Pipeline:
- Load ultrasound frames from data/dataset.mat (Acq/Amp, axes 0 2 1)
- Train SimSiam (ResNet-18) with B-mode preprocessing
- Extract embeddings and compute mAP with pseudo labels (annotations/dataset_labels.csv)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as tvm

from mat_ssl.datasets.lazy_mat import LazyMatImageDataset, TwoCropsTransform


@dataclass
class SimSiamConfig:
    seed: int = 42
    image_key: str = "Acq/Amp"
    image_axes: Tuple[int, ...] = (0, 2, 1)
    train_batch_size: int = 128
    eval_batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    cosine_t_max: int = 100
    num_workers: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- B-mode preprocessing ---

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
    hwc255 = np.transpose(chw01, (1, 2, 0))
    hwc255 = (hwc255 * 255.0).astype(np.uint8)
    return Image.fromarray(hwc255)


def build_transforms():
    train_tf = T.Compose([
        T.Lambda(lambda x: to_pil_3ch_from_chw01(bmode_normalize(x))),
        T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = T.Compose([
        T.Lambda(lambda x: to_pil_3ch_from_chw01(bmode_normalize(x))),
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


# --- SimSiam (ResNet-18) ---

class Projector(nn.Module):
    def __init__(self, in_dim=512, hid_dim=2048, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, in_dim=2048, hid_dim=512, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimSiam(nn.Module):
    def __init__(self, proj_dim=2048):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = Projector(in_dim=512, hid_dim=proj_dim, out_dim=proj_dim)
        self.predictor = Predictor(in_dim=proj_dim, hid_dim=512, out_dim=proj_dim)

    def forward_backbone(self, x):
        h = self.backbone(x)
        return torch.flatten(h, 1)

    def forward(self, x1, x2):
        h1 = self.forward_backbone(x1)
        z1 = self.projector(h1)
        p1 = self.predictor(z1)
        h2 = self.forward_backbone(x2)
        z2 = self.projector(h2)
        p2 = self.predictor(z2)
        return p1, z1, p2, z2


def negative_cosine(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


# --- Evaluation ---

def average_precision_for_query(sim_vec: np.ndarray, rel: np.ndarray) -> float:
    order = np.argsort(-sim_vec)
    rel_sorted = rel[order]
    n_rel = int(rel_sorted.sum())
    if n_rel == 0:
        return 0.0
    cumsum = np.cumsum(rel_sorted)
    idx = np.arange(1, len(rel_sorted) + 1)
    prec_at_k = (cumsum / idx) * rel_sorted
    return float(prec_at_k.sum() / n_rel)


def compute_map(embs: np.ndarray, labels: np.ndarray):
    sim_all = embs @ embs.T
    n = sim_all.shape[0]
    ap_list = []
    for i in range(n):
        sim_i = sim_all[i].copy()
        rel_i = (labels == labels[i]).astype(np.int32)
        sim_i[i] = -np.inf
        rel_i[i] = 0
        ap = average_precision_for_query(sim_i, rel_i)
        ap_list.append(ap)
    return float(np.mean(ap_list)), ap_list


def extract_embeddings(model: SimSiam, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    embs = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            h = model.forward_backbone(xb)
            z = model.projector(h)
            z = F.normalize(z, dim=1)
            embs.append(z.cpu().numpy())
    return np.concatenate(embs, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment A evaluation")
    parser.add_argument("--mat_path", type=str, default="data/dataset.mat")
    parser.add_argument("--label_csv", type=str, default="annotations/dataset_labels.csv")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_key", type=str, default="Acq/Amp")
    parser.add_argument("--image_axes", type=int, nargs="*", default=[0, 2, 1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimSiamConfig(
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        image_key=args.image_key,
        image_axes=tuple(args.image_axes),
    )

    mat_path = Path(args.mat_path)
    label_csv = Path(args.label_csv)
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing dataset: {mat_path}")
    if not label_csv.exists():
        raise FileNotFoundError(f"Missing labels: {label_csv}. Run scripts/generate_pseudo_labels.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output_root)
    run_dir = out_root / f"experiment_a_{run_id}"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms()

    train_dataset = LazyMatImageDataset(
        [str(mat_path)],
        image_key=cfg.image_key,
        image_axes=cfg.image_axes,
        normalize_255=False,
        transform=TwoCropsTransform(train_tf),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = SimSiam().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.cosine_t_max)
    scaler = torch.amp.GradScaler(device=device, enabled=(device == "cuda"))

    total_batches = len(train_loader)
    if total_batches == 0:
        raise RuntimeError("No training batches. Check dataset and batch size.")
    log_interval = max(1, total_batches // 5)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, (x1, x2) in enumerate(train_loader, start=1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                p1, z1, p2, z2 = model(x1, x2)
                loss = 0.5 * negative_cosine(p1, z2) + 0.5 * negative_cosine(p2, z1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                avg_so_far = epoch_loss / batch_idx
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1}/{cfg.epochs} batch {batch_idx}/{total_batches} "
                    f"loss={loss.item():.4f} avg_loss={avg_so_far:.4f} lr={lr_now:.6f}",
                    flush=True,
                )
        scheduler.step()
        dt = time.time() - t0
        avg_loss = epoch_loss / max(1, total_batches)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "time_sec": dt})
        ckpt_path = checkpoints_dir / "simsiam_latest.pth"
        torch.save({"epoch": epoch + 1, "state_dict": model.state_dict()}, ckpt_path)
        print(f"Epoch {epoch+1} done loss={avg_loss:.4f} time={dt:.1f}s checkpoint={ckpt_path}")

    with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Embeddings
    eval_dataset = LazyMatImageDataset(
        [str(mat_path)],
        image_key=cfg.image_key,
        image_axes=cfg.image_axes,
        normalize_255=False,
        transform=eval_tf,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    embeddings = extract_embeddings(model, eval_loader, device)
    embeddings_path = checkpoints_dir / "simsiam_embeddings.npy"
    np.save(embeddings_path, embeddings)

    labels_df = pd.read_csv(label_csv)
    labels = labels_df.set_index("frame").loc[range(len(eval_dataset)), "label"].to_numpy().astype(int)

    mAP, ap_list = compute_map(embeddings, labels)
    metrics = {
        "mAP": mAP,
        "num_frames": len(eval_dataset),
        "num_positive_labels": int(labels.sum()),
        "timestamp": time.time(),
        "run_id": run_id,
        "config": asdict(cfg),
    }
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"mAP (pseudo labels): {mAP:.4f} -> {metrics_path}")


if __name__ == "__main__":
    main()
