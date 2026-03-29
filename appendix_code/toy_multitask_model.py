from __future__ import annotations

import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TinyMultiTaskNet(nn.Module):
    """A tiny shared-encoder model with classification and reconstruction heads."""

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        width: int = 256,
        depth: int = 4,
        num_classes: int = 100,
    ) -> None:
        super().__init__()
        channels, height, width_px = image_shape
        self.image_shape = image_shape
        self.flat_dim = channels * height * width_px
        self.input_proj = nn.Linear(self.flat_dim, width)
        self.blocks = nn.ModuleList([MLPBlock(width) for _ in range(depth)])
        self.classifier = nn.Linear(width, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, self.flat_dim),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = images.flatten(1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(images)
        logits = self.classifier(features)
        recon = self.decoder(features).view(images.shape[0], *self.image_shape)
        return {
            "features": features,
            "logits": logits,
            "recon": recon,
        }
