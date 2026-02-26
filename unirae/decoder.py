from typing import Optional, Tuple

import torch
import torch.nn as nn


def maybe_token_dropout(z: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return z
    if not z.requires_grad:
        return z
    keep = (torch.rand(z.shape[:2], device=z.device) > drop_prob).float().unsqueeze(-1)
    return z * keep


def apply_rae_latent_noise(z: torch.Tensor, noise_tau: float, training: bool) -> torch.Tensor:
    """RAE 风格噪声：每个样本独立采样 sigma ~ U(0, tau)。"""
    if (not training) or noise_tau <= 0:
        return z
    sigma = noise_tau * torch.rand(
        (z.shape[0],) + (1,) * (z.ndim - 1),
        device=z.device,
        dtype=z.dtype,
    )
    return z + sigma * torch.randn_like(z)


class ReconDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mode: str,
        feature_target_dim: Optional[int] = None,
        pixel_size: int = 64,
        hidden_dim: int = 1024,
        token_dropout: float = 0.0,
        noise_tau: float = 0.8,
    ):
        super().__init__()
        self.mode = mode
        self.pixel_size = pixel_size
        self.token_dropout = token_dropout
        # RAE-style: 训练阶段对 decoder 输入特征注入高斯噪声，增强鲁棒性。
        self.noise_tau = float(noise_tau)

        if mode == "feature_recon":
            out_dim = feature_target_dim if feature_target_dim is not None else in_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        elif mode == "pixel_recon":
            out_dim = 3 * pixel_size * pixel_size
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise ValueError(f"Unknown decoder mode: {mode}")

    def forward(self, z_dino: torch.Tensor) -> torch.Tensor:
        z = maybe_token_dropout(z_dino, self.token_dropout)
        z = apply_rae_latent_noise(z, noise_tau=self.noise_tau, training=self.training)
        if self.mode == "feature_recon":
            return self.net(z)

        pooled = z.mean(dim=1)
        px = self.net(pooled)
        px = px.view(px.shape[0], 3, self.pixel_size, self.pixel_size)
        return px


def downsample_target(images: torch.Tensor, size: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
