import argparse
import glob
import math
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

from mat_ssl.datasets.lazy_mat import LazyMatImageDataset, TwoCropsTransform
from mat_ssl.models.simsiam import SimSiam, simsiam_loss
from mat_ssl.utils.logging_utils import setup_logging


def build_transform(image_size: int = 224):
    # SimSiam-style strong augmentations
    return transforms.Compose([
        ToTensorFromNumpy(),
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ToTensorFromNumpy(nn.Module):
    def forward(self, x: np.ndarray) -> torch.Tensor:
        # x: C,H,W in [0,1]
        t = torch.from_numpy(x)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t.float()


def save_checkpoint(state: dict, out_dir: str, name: str):
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    torch.save(state, os.path.join(out_dir, "checkpoints", name))


def train_one_epoch(model, loader, optimizer, device, epoch, epochs, fp16=False, limit_batches: int = 0):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    running = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for bi, (x1, x2) in enumerate(pbar):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=fp16):
            p1, p2, z1, z2 = model(x1, x2)
            loss = simsiam_loss(p1, z2) * 0.5 + simsiam_loss(p2, z1) * 0.5
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
        pbar.set_postfix(loss=f"{running / (pbar.n + 1e-9):.4f}")
        if (bi + 1) % 10 == 0 or limit_batches:
            logging.info(f"epoch={epoch+1} step={bi+1} loss={loss.item():.4f}")
        if limit_batches and (bi + 1) >= limit_batches:
            break
    return running / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_glob", type=str, required=True, help="Glob for .mat files")
    parser.add_argument("--image_key", type=str, default="images")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="runs/simsiam_r50")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--limit_batches", type=int, default=0, help="For dry-run: cap batches per epoch")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU threads for low-spec machines")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory (default mat_ssl/logs)")
    args = parser.parse_args()

    log_path = setup_logging("simsiam", base_log_dir=args.log_dir)
    logging.info("SimSiam training started")
    logging.info(f"Args: {vars(args)}")

    try:
        torch.set_num_threads(max(1, int(args.threads)))
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    files = sorted(glob.glob(args.mat_glob))
    transform = TwoCropsTransform(build_transform(args.image_size))
    dataset = LazyMatImageDataset(files, image_key=args.image_key, transform=transform)
    logging.info(f"Dataset files={len(files)} samples={len(dataset)} image_key={args.image_key}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)

    model = SimSiam(pretrained_backbone=args.pretrained_backbone)
    model.to(device)
    logging.info("Model constructed: SimSiam(ResNet50)")

    # LARS or SGD; for simplicity, SGD with cosine schedule
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs} start")
        loss = train_one_epoch(model, loader, optimizer, device, epoch, args.epochs, fp16=args.fp16, limit_batches=args.limit_batches)
        scheduler.step()
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, args.output_dir, name="last.pth")
        logging.info(f"Epoch {epoch+1} done: loss={loss:.4f}; checkpoint saved")

    logging.info(f"Training finished. Logs: {log_path}")


if __name__ == "__main__":
    main()
