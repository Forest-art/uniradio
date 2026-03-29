from __future__ import annotations

import tempfile
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .grad_conflict import apply_cagrad, apply_gma_laga, apply_ma_laga_objective


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_norm_tensors(device: torch.device, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


class LightweightPatchDecoder(nn.Module):
    """MAE-style lightweight Transformer decoder on patch tokens."""

    def __init__(
        self,
        in_dim: int,
        num_patches: int,
        patch_dim: int,
        decoder_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.input_proj = nn.Linear(in_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        hidden_dim = int(decoder_dim * float(mlp_ratio))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=decoder_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=drop_rate,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_dim)

    def _resize_pos_embed(self, n_tokens: int) -> torch.Tensor:
        if n_tokens == self.pos_embed.shape[1]:
            return self.pos_embed
        x = self.pos_embed.transpose(1, 2)
        x = F.interpolate(x, size=n_tokens, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(patch_tokens)
        x = x + self._resize_pos_embed(x.shape[1]).to(device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.pred(x)


class DSGAModel(nn.Module):
    """DINO encoder + linear classification head + patch reconstruction decoder."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 224,
        encoder_name: str = "vit_small_patch14_reg4_dinov2",
        encoder_pretrained: bool = True,
        decoder_dim: int = 384,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        decoder_mlp_ratio: float = 4.0,
        decoder_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=bool(encoder_pretrained),
            num_classes=0,
            global_pool="",
            img_size=self.image_size,
        )

        self.embed_dim = int(self.encoder.embed_dim)
        patch_size = self.encoder.patch_embed.patch_size
        self.patch_size = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)
        self.num_patches = int(self.encoder.patch_embed.num_patches)
        self.patch_dim = 3 * self.patch_size * self.patch_size

        self.cls_head = nn.Linear(self.embed_dim, int(num_classes))
        nn.init.trunc_normal_(self.cls_head.weight, std=0.02)
        if self.cls_head.bias is not None:
            nn.init.zeros_(self.cls_head.bias)

        self.decoder = LightweightPatchDecoder(
            in_dim=self.embed_dim,
            num_patches=self.num_patches,
            patch_dim=self.patch_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            drop_rate=decoder_drop_rate,
        )

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def encode_tokens(self, images_norm: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(images_norm)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[-1]
        if isinstance(tokens, dict):
            for k in ("x_prenorm", "x", "last_hidden_state"):
                if k in tokens:
                    tokens = tokens[k]
                    break
        if tokens.ndim == 2:
            x = self.encoder.patch_embed(images_norm)
            x = self.encoder._pos_embed(x)
            x = self.encoder.patch_drop(x)
            x = self.encoder.norm_pre(x)
            for blk in self.encoder.blocks:
                x = blk(x)
            x = self.encoder.norm(x)
            tokens = x
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected [B,T,C] tokens, got {tuple(tokens.shape)}")
        return tokens

    def encode_cls(self, images_norm: torch.Tensor) -> torch.Tensor:
        return self.encode_tokens(images_norm)[:, 0]

    def patchify(self, images_01: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images_01.shape
        p = self.patch_size
        if (h % p) != 0 or (w % p) != 0:
            raise ValueError(f"image shape {(h, w)} is not divisible by patch size {p}")
        gh, gw = h // p, w // p
        return images_01.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, p * p * c)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, d = patches.shape
        p = self.patch_size
        expected = 3 * p * p
        if d != expected:
            raise ValueError(f"patch dim mismatch: got={d}, expected={expected}")
        gh = gw = int(n**0.5)
        if gh * gw != n:
            raise ValueError(f"number of patches {n} is not a perfect square")
        return patches.reshape(b, gh, gw, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, gh * p, gw * p)

    def forward(self, images_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.encode_tokens(images_norm)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, -self.num_patches :]
        logits = self.cls_head(cls_token)
        pred_patches = self.decoder(patch_tokens)
        return {
            "tokens": tokens,
            "cls_token": cls_token,
            "logits": logits,
            "pred_patches": pred_patches,
        }


def _to_image_01(
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    input_is_normalized: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if input_is_normalized:
        images_norm = images
        images_01 = (images * std + mean).clamp_(0.0, 1.0)
    else:
        images_01 = images.clamp_(0.0, 1.0)
        images_norm = (images_01 - mean) / std
    return images_norm, images_01


def _build_recon_objective(
    pred_patches: torch.Tensor,
    gt_patches: torch.Tensor,
    recon_loss: Literal["mse", "rmse"] = "rmse",
    rmse_eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(pred_patches, gt_patches)
    if recon_loss == "mse":
        return mse, mse
    if recon_loss == "rmse":
        return torch.sqrt(mse + float(rmse_eps)), mse
    raise ValueError(f"Unsupported recon_loss={recon_loss}")


def _materialize_grads(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[torch.nn.Parameter],
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        out.append(torch.zeros_like(p, memory_format=torch.preserve_format) if g is None else g)
    return out


def _grad_cosine_from_losses(
    loss_u: torch.Tensor,
    loss_g: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> float:
    if len(params) == 0:
        return 0.0
    gu = torch.autograd.grad(loss_u, params, retain_graph=True, allow_unused=True)
    gg = torch.autograd.grad(loss_g, params, retain_graph=True, allow_unused=True)
    gu = _materialize_grads(gu, params)
    gg = _materialize_grads(gg, params)
    v_u = torch.cat([g.reshape(-1) for g in gu], dim=0)
    v_g = torch.cat([g.reshape(-1) for g in gg], dim=0)
    denom = (v_u.norm() * v_g.norm()).clamp_min(1e-12)
    return float((torch.dot(v_u, v_g) / denom).item())


def _split_params(model: DSGAModel) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    aux_params: List[torch.nn.Parameter] = []
    aux_params.extend([p for p in model.cls_head.parameters() if p.requires_grad])
    aux_params.extend([p for p in model.decoder.parameters() if p.requires_grad])
    return encoder_params, aux_params


def train_step_joint_naive(
    model: DSGAModel,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    lambda_cls: float = 1.0,
    lambda_recon: float = 1.0,
    recon_loss: Literal["mse", "rmse"] = "rmse",
    recon_rmse_eps: float = 1e-8,
    input_is_normalized: bool = True,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Phase-2: Naive joint fine-tuning (Encoder+Head+Decoder all trainable)."""
    model.train()
    images, labels = batch
    device = next(model.parameters()).device
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    images_norm, images_01 = _to_image_01(images, mean=mean, std=std, input_is_normalized=input_is_normalized)
    gt_patches = model.patchify(images_01)

    out = model(images_norm)
    logits = out["logits"]
    pred_patches = out["pred_patches"]

    loss_cls = F.cross_entropy(logits, labels)
    loss_recon, loss_recon_mse = _build_recon_objective(
        pred_patches,
        gt_patches,
        recon_loss=recon_loss,
        rmse_eps=recon_rmse_eps,
    )
    loss_total = float(lambda_cls) * loss_cls + float(lambda_recon) * loss_recon

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    grad_cos = _grad_cosine_from_losses(loss_cls, loss_recon, encoder_params)

    optimizer.zero_grad(set_to_none=True)
    loss_total.backward()
    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()

    with torch.no_grad():
        acc = float((logits.argmax(dim=-1) == labels).float().mean().item())
        rmse = float(torch.sqrt(loss_recon_mse.detach().clamp_min(0.0)).item())

    return {
        "loss_cls": float(loss_cls.detach().item()),
        "loss_recon": float(loss_recon.detach().item()),
        "loss_recon_mse": float(loss_recon_mse.detach().item()),
        "loss_total": float(loss_total.detach().item()),
        "acc": acc,
        "rmse": rmse,
        "grad_cosine": float(grad_cos),
    }


def train_step_gradient_decompose(
    model: DSGAModel,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    method: Literal["cagrad", "ma_laga_global", "gma_laga"] = "cagrad",
    lambda_cls: float = 1.0,
    lambda_recon: float = 1.0,
    recon_loss: Literal["mse", "rmse"] = "rmse",
    recon_rmse_eps: float = 1e-8,
    input_is_normalized: bool = True,
    cagrad_beta: float = 0.5,
    ma_align_gamma: float = 0.5,
    ma_norm_restore: bool = True,
    ma_mode: str = "full",
    gma_max_scale: float = 100.0,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Phase-3: Gradient decomposition update on encoder grads."""
    model.train()
    images, labels = batch
    device = next(model.parameters()).device
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    images_norm, images_01 = _to_image_01(images, mean=mean, std=std, input_is_normalized=input_is_normalized)
    gt_patches = model.patchify(images_01)

    out = model(images_norm)
    logits = out["logits"]
    pred_patches = out["pred_patches"]

    loss_cls = F.cross_entropy(logits, labels)
    loss_recon, loss_recon_mse = _build_recon_objective(
        pred_patches,
        gt_patches,
        recon_loss=recon_loss,
        rmse_eps=recon_rmse_eps,
    )

    encoder_params, aux_params = _split_params(model)

    optimizer.zero_grad(set_to_none=True)
    if method == "cagrad":
        grad_cos = apply_cagrad(
            loss_txt=loss_cls,
            loss_rec=loss_recon,
            shared_params=encoder_params,
            aux_params=aux_params,
            lambda_txt=float(lambda_cls),
            lambda_rec=float(lambda_recon),
            beta=float(cagrad_beta),
        )
    elif method == "ma_laga_global":
        grad_cos = apply_ma_laga_objective(
            loss_txt=loss_cls,
            loss_rec=loss_recon,
            shared_params=encoder_params,
            aux_params=aux_params,
            lambda_txt=float(lambda_cls),
            lambda_rec=float(lambda_recon),
            group_to_indices=None,  # global
            align_gamma=float(ma_align_gamma),
            norm_restore=bool(ma_norm_restore),
            mode=str(ma_mode),
            eps=1e-8,
        )
    elif method == "gma_laga":
        grad_cos = apply_gma_laga(
            loss_txt=loss_cls,
            loss_rec=loss_recon,
            shared_params=encoder_params,
            aux_params=aux_params,
            group_to_indices=None,  # global
            lambda_txt=float(lambda_cls),
            lambda_rec=float(lambda_recon),
            max_scale=float(gma_max_scale),
            eps=1e-8,
        )
    else:
        raise ValueError(f"Unsupported method={method}. Use cagrad|ma_laga_global|gma_laga")

    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()

    with torch.no_grad():
        loss_total = float(lambda_cls) * loss_cls + float(lambda_recon) * loss_recon
        acc = float((logits.argmax(dim=-1) == labels).float().mean().item())
        rmse = float(torch.sqrt(loss_recon_mse.detach().clamp_min(0.0)).item())

    return {
        "loss_cls": float(loss_cls.detach().item()),
        "loss_recon": float(loss_recon.detach().item()),
        "loss_recon_mse": float(loss_recon_mse.detach().item()),
        "loss_total": float(loss_total.detach().item()),
        "acc": acc,
        "rmse": rmse,
        "grad_cosine": float(grad_cos),
    }


@torch.no_grad()
def _extract_cls_features(encoder: nn.Module, images_norm: torch.Tensor) -> torch.Tensor:
    tokens = encoder.forward_features(images_norm)
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[-1]
    if isinstance(tokens, dict):
        for k in ("x_prenorm", "x", "last_hidden_state"):
            if k in tokens:
                tokens = tokens[k]
                break
    if torch.is_tensor(tokens) and tokens.ndim == 2:
        return tokens.float()
    if torch.is_tensor(tokens) and tokens.ndim == 3:
        return tokens[:, 0].float()
    raise RuntimeError(f"Unsupported encoder output type for linear probing: {type(tokens)}")


def evaluate_linear_probing(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_classes: int,
    device: torch.device,
    max_steps: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> Dict[str, float]:
    """Phase-4 protocol: freeze encoder, train a fresh linear head, report Top-1."""
    prev_mode = encoder.training
    prev_req = [p.requires_grad for p in encoder.parameters()]

    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    first_images, _ = next(iter(train_loader))
    first_images = first_images.to(device, non_blocking=True)
    feat_dim = int(_extract_cls_features(encoder, first_images).shape[-1])

    head = nn.Linear(feat_dim, int(num_classes)).to(device)
    optimizer = AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_iter = cycle(train_loader)
    head.train()
    for _ in range(int(max_steps)):
        images, labels = next(train_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            feat = _extract_cls_features(encoder, images)
        logits = head(feat)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    head.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feat = _extract_cls_features(encoder, images)
            logits = head(feat)
            pred = logits.argmax(dim=-1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())

    # restore state
    for p, req in zip(encoder.parameters(), prev_req):
        p.requires_grad = req
    encoder.train(prev_mode)

    top1 = float(correct / max(total, 1))
    return {
        "linear_probe_top1": top1,
        "linear_probe_top1_percent": 100.0 * top1,
        "num_val_samples": int(total),
    }


@torch.no_grad()
def evaluate_reconstruction_rmse_rfid(
    model: DSGAModel,
    val_loader: DataLoader,
    *,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    input_is_normalized: bool = True,
    max_batches: int = 0,
    rfid_num_samples: int = 1024,
    rfid_batch_size: int = 64,
    rfid_tmp_dir: str = "/tmp",
) -> Dict[str, float]:
    """Phase-4 reconstruction protocol: direct forward rMSE + sampled rFID."""
    model.eval()
    sum_mse = 0.0
    n_samples = 0

    tmp_root = tempfile.TemporaryDirectory(prefix="dsga_rfid_", dir=(rfid_tmp_dir or None))
    root = Path(tmp_root.name)
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for bi, (images, _) in enumerate(val_loader):
        if max_batches > 0 and bi >= int(max_batches):
            break
        images = images.to(device, non_blocking=True)
        images_norm, images_01 = _to_image_01(images, mean=mean, std=std, input_is_normalized=input_is_normalized)

        out = model(images_norm)
        pred_img = model.unpatchify(out["pred_patches"]).clamp(0.0, 1.0)
        if pred_img.shape[-2:] != images_01.shape[-2:]:
            pred_img = F.interpolate(pred_img, size=images_01.shape[-2:], mode="bilinear", align_corners=False)

        mse_per = ((pred_img - images_01) ** 2).flatten(1).mean(dim=1)
        sum_mse += float(mse_per.sum().item())
        n_samples += int(images.shape[0])

        for i in range(int(images.shape[0])):
            if saved >= int(rfid_num_samples):
                break
            save_image(images_01[i].cpu(), str(real_dir / f"{saved:07d}.png"))
            save_image(pred_img[i].cpu(), str(fake_dir / f"{saved:07d}.png"))
            saved += 1
        if saved >= int(rfid_num_samples):
            break

    val_mse = float(sum_mse / max(n_samples, 1))
    val_rmse = float(val_mse**0.5)

    rfid = float("nan")
    rfid_error = ""
    if saved > 1:
        try:
            from torch_fidelity import calculate_metrics

            ret = calculate_metrics(
                input1=str(real_dir),
                input2=str(fake_dir),
                cuda=(device.type == "cuda"),
                isc=False,
                fid=True,
                kid=False,
                prc=False,
                verbose=False,
                batch_size=int(rfid_batch_size),
            )
            rfid = float(ret.get("frechet_inception_distance", float("nan")))
        except Exception as e:  # noqa: BLE001
            rfid_error = str(e)

    tmp_root.cleanup()
    return {
        "val_rmse": val_rmse,
        "val_mse": val_mse,
        "val_rfid": rfid,
        "val_num_samples": int(n_samples),
        "rfid_num_samples": int(saved),
        "rfid_error": rfid_error,
    }


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DSGAModel",
    "imagenet_norm_tensors",
    "train_step_joint_naive",
    "train_step_gradient_decompose",
    "evaluate_linear_probing",
    "evaluate_reconstruction_rmse_rfid",
]
