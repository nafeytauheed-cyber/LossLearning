"""Normalized feature encoders and explicit final-linear-head parameterization."""

import torch
from torch import nn
from torch.nn import functional as F


class PotentialFieldHead(nn.Module):
    def __init__(self, dimension: int, classes: int, prototypes: int = 16):
        super().__init__()
        self.centers = nn.Parameter(F.normalize(torch.randn(prototypes, dimension), dim=1))
        self.raw_scale = nn.Parameter(torch.zeros(prototypes))
        self.raw_power = nn.Parameter(torch.zeros(prototypes))
        self.linear = nn.Linear(dimension + prototypes + 1, classes, bias=False)

    @property
    def scales(self):
        return F.softplus(self.raw_scale) + 0.05

    @property
    def powers(self):
        return F.softplus(self.raw_power) + 0.05

    def features(self, h):
        distance = (h[:, None] - self.centers[None]).square().sum(-1)
        phi = torch.exp(-self.powers * torch.log1p(distance / self.scales.square()))
        return torch.cat((h, phi, torch.ones_like(h[:, :1])), 1)

    def forward(self, h):
        return self.linear(self.features(h))


class LinearHead(nn.Module):
    def __init__(self, dimension: int, classes: int):
        super().__init__()
        self.linear = nn.Linear(dimension + 1, classes, bias=False)

    def features(self, h):
        return torch.cat((h, torch.ones_like(h[:, :1])), 1)

    def forward(self, h):
        return self.linear(self.features(h))


class LossLearningNet(nn.Module):
    def __init__(
        self,
        input_shape,
        n_classes: int,
        encoder="mlp",
        embedding_dim=32,
        hidden_dim=64,
        n_prototypes=16,
        head="linear",
        freeze_encoder=False,
    ):
        super().__init__()
        if encoder == "mlp" and len(input_shape) == 1:
            self.encoder = nn.Sequential(
                nn.Linear(input_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
        elif encoder == "cnn" and len(input_shape) == 3:
            self.encoder = nn.Sequential(
                nn.Conv2d(input_shape[0], 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(512, embedding_dim),
            )
        else:
            raise ValueError("mlp expects (N,D), cnn expects (N,C,H,W)")
        if head not in ("linear", "potential"):
            raise ValueError("unknown head")
        self.head = (
            PotentialFieldHead(embedding_dim, n_classes, n_prototypes)
            if head == "potential"
            else LinearHead(embedding_dim, n_classes)
        )
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def features(self, x):
        h = F.normalize(self.encoder(x), dim=1, eps=1e-8)
        return h, self.head.features(h)

    def forward(self, x):
        return self.head.linear(self.features(x)[1])
