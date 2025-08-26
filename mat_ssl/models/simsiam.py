from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers = []
    d_prev = in_dim
    for i in range(num_layers - 1):
        layers += [nn.Linear(d_prev, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        d_prev = hidden_dim
    layers += [nn.Linear(d_prev, out_dim, bias=True)]
    return nn.Sequential(*layers)


class SimSiam(nn.Module):
    """SimSiam with ResNet50 backbone.

    - Encoder: ResNet50 up to avgpool, outputs 2048-d
    - Projector: 3-layer MLP to 2048-d (default)
    - Predictor: 2-layer MLP to 2048-d
    """

    def __init__(
        self,
        proj_dim: int = 2048,
        pred_dim: int = 512,
        proj_hidden: int = 2048,
        pretrained_backbone: bool = False,
    ):
        super().__init__()

        # Backbone
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None)
        # Remove fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: [B, 2048, 1, 1]
        feat_dim = 2048

        # Projector (3-layer as in paper)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),  # no affine on last BN per paper
        )

        # Predictor (2-layer)
        self.predictor = _build_mlp(proj_dim, pred_dim, proj_dim, num_layers=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 3, H, W]
        z1 = self._encode(x1)
        z2 = self._encode(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)  # [B, 2048, 1, 1]
        y = torch.flatten(y, 1)
        z = self.projector(y)
        return z


def simsiam_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity between predictor p and stop-grad z.
    p and z are L2 normalized before dot product. Loss reduced over batch.
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return - (p * z).sum(dim=1).mean()


class EmbeddingHead(nn.Module):
    """Small MLP to produce embeddings for metric learning fine-tuning."""

    def __init__(self, in_dim: int = 2048, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet50Backbone(nn.Module):
    """ResNet50 feature extractor returning pooled 2048-d vectors."""

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        return torch.flatten(y, 1)

