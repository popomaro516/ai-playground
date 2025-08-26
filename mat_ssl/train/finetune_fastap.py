import argparse
import glob
import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from pytorch_metric_learning.losses import FastAPLoss

from mat_ssl.datasets.lazy_mat import LazyMatImageDataset
from mat_ssl.models.simsiam import ResNet50Backbone, EmbeddingHead
from mat_ssl.utils.logging_utils import setup_logging


class ToTensorFromNumpy(nn.Module):
    def forward(self, x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(x)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t.float()


def build_transform(image_size: int = 224):
    return transforms.Compose([
        ToTensorFromNumpy(),
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def save_checkpoint(state: dict, out_dir: str, name: str):
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    torch.save(state, os.path.join(out_dir, "checkpoints", name))


def train_one_epoch(backbone, head, loader, criterion, optimizer, device, fp16=False, limit_batches: int = 0):
    backbone.eval()  # keep backbone frozen by default
    head.train()
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    running = 0.0
    pbar = tqdm(loader, desc="Finetune")
    for bi, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=fp16):
            with torch.no_grad():
                feats = backbone(x)
            emb = head(feats)
            loss = criterion(emb, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
        pbar.set_postfix(loss=f"{running / (pbar.n + 1e-9):.4f}")
        if (bi + 1) % 10 == 0 or limit_batches:
            logging.info(f"step={bi+1} loss={loss.item():.4f}")
        if limit_batches and (bi + 1) >= limit_batches:
            break
    return running / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_glob", type=str, required=True)
    parser.add_argument("--image_key", type=str, default="images")
    parser.add_argument("--label_key", type=str, default="labels")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to SimSiam checkpoint (last.pth)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="runs/fastap_r50")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Fine-tune backbone too")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory (default mat_ssl/logs)")
    parser.add_argument("--limit_batches", type=int, default=0, help="For dry-run: cap batches per epoch")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU threads for low-spec machines")
    args = parser.parse_args()

    log_path = setup_logging("fastap_finetune", base_log_dir=args.log_dir)
    logging.info("FastAP fine-tuning started")
    logging.info(f"Args: {vars(args)}")

    try:
        torch.set_num_threads(max(1, int(args.threads)))
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    files = sorted(glob.glob(args.mat_glob))
    transform = build_transform(args.image_size)
    dataset = LazyMatImageDataset(files, image_key=args.image_key, label_key=args.label_key, transform=transform)
    logging.info(f"Dataset files={len(files)} samples={len(dataset)} image_key={args.image_key} label_key={args.label_key}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)

    # Build backbone + head
    backbone = ResNet50Backbone(pretrained_backbone=False)
    head = EmbeddingHead(in_dim=2048, emb_dim=args.emb_dim)

    # Load pretrained SimSiam weights (only backbone by mapping keys)
    ckpt = torch.load(args.pretrained, map_location="cpu")
    state = ckpt.get("model", ckpt)
    # Extract backbone weights
    backbone_prefix = "backbone."
    bb_state = {k[len(backbone_prefix):]: v for k, v in state.items() if k.startswith(backbone_prefix)}
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    logging.info(f"Loaded backbone weights from {args.pretrained}; missing={missing} unexpected={unexpected}")
    # Predictor/projector are ignored here

    if args.unfreeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = True
    else:
        for p in backbone.parameters():
            p.requires_grad = False

    backbone.to(device)
    head.to(device)

    params = list(head.parameters()) + (list(backbone.parameters()) if args.unfreeze_backbone else [])
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = FastAPLoss(num_bins=10)

    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs} start")
        loss = train_one_epoch(backbone, head, loader, criterion, optimizer, device, fp16=args.fp16, limit_batches=args.limit_batches)
        save_checkpoint({
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, args.output_dir, name="last.pth")
        logging.info(f"Epoch {epoch+1} done: loss={loss:.4f}; checkpoint saved")

    logging.info(f"Fine-tuning finished. Logs: {log_path}")


if __name__ == "__main__":
    main()
