from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # 复用仓库现有 CAGrad 合并实现（私有函数，但公式与当前实验保持一致）。
    from unirae.grad_conflict import _cagrad_like_merge as _repo_cagrad_like_merge
except Exception:  # pragma: no cover
    _repo_cagrad_like_merge = None


@dataclass
class UniRAELossOutput:
    loss_u: torch.Tensor
    loss_g: torch.Tensor
    total: torch.Tensor


class LoRAQKV(nn.Module):
    """对 ViT Attention 的 qkv 线性层做 LoRA 增量注入。

    说明：
    - 原始 base 线性层参数被冻结。
    - 仅 LoRA 参数 lora_A / lora_B 可训练。
    - 适配 [B, T, C] 与 [N, C] 两种输入形状。
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRAQKV expects nn.Linear, got {type(base)}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        # LoRA: W + BA, 其中 A:[r,in], B:[out,r]
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        # 冻结原始 qkv 权重
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # delta = x @ A^T @ B^T
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta

    def lora_parameters(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


class LoRA_ViT_Block(nn.Module):
    """包装单个 ViT Block，仅替换其 Attention 的 qkv 为 LoRAQKV。"""

    def __init__(self, vit_block: nn.Module, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if not hasattr(vit_block, "attn") or not hasattr(vit_block.attn, "qkv"):
            raise ValueError("LoRA_ViT_Block expects a ViT block with attn.qkv.")
        if not isinstance(vit_block.attn.qkv, nn.Linear):
            raise TypeError(f"attn.qkv must be nn.Linear, got {type(vit_block.attn.qkv)}")

        self.block = vit_block
        self.block.attn.qkv = LoRAQKV(self.block.attn.qkv, rank=rank, alpha=alpha)

        # 先冻结整块，再显式开启 LoRA。
        for p in self.block.parameters():
            p.requires_grad = False
        for p in self.block.attn.qkv.lora_parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def lora_parameters(self) -> List[nn.Parameter]:
        qkv = self.block.attn.qkv
        if isinstance(qkv, LoRAQKV):
            return qkv.lora_parameters()
        return []


class TransformerDecoder(nn.Module):
    """RAE/MAE 风格的轻量 Transformer Decoder。

    数据流：
    patch tokens (encoder) -> 输入投影 -> +pos_embed -> 多层 Transformer Block
    -> LayerNorm -> Linear 预测 patch 像素向量。
    """

    def __init__(
        self,
        in_dim: int,
        num_patches: int,
        patch_dim: int,
        decoder_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        noise_tau: float = 0.8,
    ):
        super().__init__()
        if num_patches <= 0:
            raise ValueError(f"num_patches must be > 0, got {num_patches}")

        self.num_patches = int(num_patches)
        self.decoder_dim = int(decoder_dim)
        # RAE-style decoder input noise. 仅在 model.train() 时生效。
        self.noise_tau = float(noise_tau)

        self.input_proj = nn.Linear(in_dim, self.decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        hidden = int(self.decoder_dim * float(mlp_ratio))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.decoder_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden,
                    dropout=drop_rate,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.decoder_dim)
        self.pred = nn.Linear(self.decoder_dim, patch_dim)

    def _resize_pos_embed(self, n_tokens: int) -> torch.Tensor:
        if n_tokens == self.pos_embed.shape[1]:
            return self.pos_embed
        # 序列长度变化时做 1D 线性插值，保证 decoder 可跑通。
        pos = self.pos_embed.transpose(1, 2)  # [1, C, N]
        pos = F.interpolate(pos, size=n_tokens, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def _apply_rae_noise(self, x: torch.Tensor) -> torch.Tensor:
        # RAE 训练习惯：每个样本单独采样 noise sigma ~ U(0, tau)，再乘高斯噪声。
        if (not self.training) or self.noise_tau <= 0:
            return x
        sigma = self.noise_tau * torch.rand(
            (x.shape[0],) + (1,) * (x.ndim - 1),
            device=x.device,
            dtype=x.dtype,
        )
        return x + sigma * torch.randn_like(x)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens = self._apply_rae_noise(patch_tokens)
        x = self.input_proj(patch_tokens)
        x = x + self._resize_pos_embed(x.shape[1]).to(dtype=x.dtype, device=x.device)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.pred(x)


class UniRAEDinoLoRA(nn.Module):
    """UniRAE: 冻结 DINO + 共享 LoRA + 理解头 + 生成解码器。"""

    _DINO_NAME_MAP = {
        "vit_small": "vit_small_patch14_dinov2",
        "vit_base": "vit_base_patch14_dinov2",
        "vit_small_patch14_dinov2": "vit_small_patch14_dinov2",
        "vit_base_patch14_dinov2": "vit_base_patch14_dinov2",
    }

    def __init__(
        self,
        num_classes: int,
        dino_variant: str = "vit_small",
        image_size: int = 224,
        pretrained: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_last_n_blocks: int = 4,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        decoder_mlp_ratio: float = 4.0,
        decoder_drop_rate: float = 0.0,
        decoder_noise_tau: float = 0.8,
        use_grad_checkpointing: bool = False,
    ):
        super().__init__()
        import timm

        model_name = self._DINO_NAME_MAP.get(dino_variant.lower())
        if model_name is None:
            raise ValueError(
                f"Unsupported dino_variant={dino_variant}. "
                "Use vit_small|vit_base|vit_small_patch14_dinov2|vit_base_patch14_dinov2."
            )

        # global_pool='' 可拿到完整 token 序列 [CLS + patches]。
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=image_size,
        )
        if use_grad_checkpointing and hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(True)

        # 1) 冻结原始 DINO 参数
        for p in self.encoder.parameters():
            p.requires_grad = False

        if not hasattr(self.encoder, "blocks"):
            raise RuntimeError("DINO encoder has no .blocks; cannot inject LoRA to last N blocks.")
        n_blocks = len(self.encoder.blocks)
        if n_blocks <= 0:
            raise RuntimeError("DINO encoder has empty blocks.")

        n_lora = min(max(0, int(lora_last_n_blocks)), n_blocks)
        start = n_blocks - n_lora

        self._lora_wrapped_blocks = nn.ModuleList()
        self._shared_lora_params: List[nn.Parameter] = []

        # 2) 在最后 N 个 block 的 qkv 注入 LoRA（共享冲突区）
        if n_lora > 0:
            for i in range(start, n_blocks):
                wrapped = LoRA_ViT_Block(self.encoder.blocks[i], rank=lora_rank, alpha=lora_alpha)
                self.encoder.blocks[i] = wrapped
                self._lora_wrapped_blocks.append(wrapped)
                self._shared_lora_params.extend(wrapped.lora_parameters())

        self.embed_dim = int(self.encoder.embed_dim)
        p = self.encoder.patch_embed.patch_size
        self.patch_size = int(p[0] if isinstance(p, tuple) else p)
        self.num_patches = int(self.encoder.patch_embed.num_patches)
        self.patch_dim = 3 * self.patch_size * self.patch_size

        # 3) 理解分支：[CLS] -> 分类
        self.understanding_head = nn.Linear(self.embed_dim, num_classes)
        nn.init.trunc_normal_(self.understanding_head.weight, std=0.02)
        if self.understanding_head.bias is not None:
            nn.init.zeros_(self.understanding_head.bias)

        # 4) 生成分支：patch tokens -> 轻量 Transformer Decoder -> patch 像素
        self.decoder = TransformerDecoder(
            in_dim=self.embed_dim,
            num_patches=self.num_patches,
            patch_dim=self.patch_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            drop_rate=decoder_drop_rate,
            noise_tau=decoder_noise_tau,
        )

    def shared_lora_parameters(self) -> List[nn.Parameter]:
        # 去重并保持顺序，避免重复写 grad。
        uniq: List[nn.Parameter] = []
        seen = set()
        for p in self._shared_lora_params:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                uniq.append(p)
        return uniq

    def understanding_parameters(self) -> List[nn.Parameter]:
        return list(self.understanding_head.parameters())

    def generation_parameters(self) -> List[nn.Parameter]:
        return list(self.decoder.parameters())

    def trainable_parameters(self) -> List[nn.Parameter]:
        return self.shared_lora_parameters() + self.understanding_parameters() + self.generation_parameters()

    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        # timm DINO 在 global_pool='' 时 forward_features 返回 [B, 1+N, C]。
        tokens = self.encoder.forward_features(images)
        if tokens.ndim == 2:
            # 兼容少数实现返回池化向量时，改走手工 token 路径。
            x = self.encoder.patch_embed(images)
            x = self.encoder._pos_embed(x)
            x = self.encoder.patch_drop(x)
            x = self.encoder.norm_pre(x)
            for blk in self.encoder.blocks:
                x = blk(x)
            x = self.encoder.norm(x)
            tokens = x
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected token tensor [B, T, C], got {tuple(tokens.shape)}")
        return tokens

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """把原图切成 patch 向量，作为重建监督目标。"""
        b, c, h, w = images.shape
        p = self.patch_size
        if (h % p) != 0 or (w % p) != 0:
            raise ValueError(f"Input size {(h, w)} is not divisible by patch size {p}.")
        gh, gw = h // p, w // p
        # [B, C, H, W] -> [B, gh*gw, p*p*C]
        x = images.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, p * p * c)
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """把 patch 向量还原回图像，用于重建评估可视化。"""
        b, n, d = patches.shape
        p = self.patch_size
        expected = p * p * 3
        if d != expected:
            raise ValueError(f"Patch dim mismatch: got {d}, expected {expected}")
        gh = gw = int(n ** 0.5)
        if gh * gw != n:
            raise ValueError(f"Number of patches {n} is not a perfect square.")
        x = patches.reshape(b, gh, gw, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, gh * p, gw * p)
        return x

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.encode_tokens(images)
        cls_token = tokens[:, 0]       # [B, C]
        patch_tokens = tokens[:, 1:]   # [B, N, C]

        logits = self.understanding_head(cls_token)
        pred_patches = self.decoder(patch_tokens)

        return {
            "tokens": tokens,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "logits": logits,
            "pred_patches": pred_patches,
        }

    def compute_losses(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        lambda_u: float = 1.0,
        lambda_g: float = 1.0,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[UniRAELossOutput, Dict[str, torch.Tensor]]:
        if outputs is None:
            outputs = self(images)

        loss_u = F.cross_entropy(outputs["logits"], targets)
        gt_patches = self.patchify(images)
        pred_patches = outputs["pred_patches"]
        if pred_patches.shape != gt_patches.shape:
            raise RuntimeError(
                f"Decoder output shape {tuple(pred_patches.shape)} does not match patch target {tuple(gt_patches.shape)}"
            )
        loss_g = F.mse_loss(pred_patches, gt_patches)
        total = float(lambda_u) * loss_u + float(lambda_g) * loss_g
        return UniRAELossOutput(loss_u=loss_u, loss_g=loss_g, total=total), outputs


def _clone_or_zero_grads(params: Sequence[nn.Parameter]) -> List[torch.Tensor]:
    grads: List[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p, memory_format=torch.preserve_format))
        else:
            grads.append(p.grad.detach().clone())
    return grads


def _assign_grads(params: Sequence[nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g


def _flatten(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.zeros(1)
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def _mgda_two_task_merge(g_u: Sequence[torch.Tensor], g_g: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    v_u = _flatten(g_u)
    v_g = _flatten(g_g)
    diff = v_u - v_g
    denom = torch.dot(diff, diff)
    if float(denom.item()) <= 1e-12:
        alpha = 0.5
    else:
        alpha = float(torch.dot(v_g, v_g - v_u).item() / denom.item())
        alpha = max(0.0, min(1.0, alpha))
    return [alpha * a + (1.0 - alpha) * b for a, b in zip(g_u, g_g)]


def _cagrad_like_merge(g_u: Sequence[torch.Tensor], g_g: Sequence[torch.Tensor], beta: float) -> List[torch.Tensor]:
    beta = max(0.0, min(1.0, float(beta)))
    g_avg = [0.5 * (a + b) for a, b in zip(g_u, g_g)]
    g_mgda = _mgda_two_task_merge(g_u, g_g)
    return [(1.0 - beta) * a + beta * b for a, b in zip(g_avg, g_mgda)]


def apply_gradient_strategy(
    g_u: Sequence[torch.Tensor],
    g_g: Sequence[torch.Tensor],
    strategy: str = "cagrad",
    cagrad_beta: float = 0.35,
) -> List[torch.Tensor]:
    """共享 LoRA 区域的梯度路由插槽。

    - naive: g_total = g_u + g_g
    - cagrad: 调用仓库 CAGrad merge（若可用），否则使用等价本地实现
    """
    st = str(strategy).lower()
    if st == "naive":
        return [a + b for a, b in zip(g_u, g_g)]
    if st == "cagrad":
        if _repo_cagrad_like_merge is not None:
            return _repo_cagrad_like_merge(g_u, g_g, beta=cagrad_beta)
        return _cagrad_like_merge(g_u, g_g, beta=cagrad_beta)
    raise ValueError(f"Unsupported strategy={strategy}. Use naive|cagrad.")


def train_step(
    model: UniRAEDinoLoRA,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    strategy: str = "cagrad",
    cagrad_beta: float = 0.35,
    lambda_u: float = 1.0,
    lambda_g: float = 1.0,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """标准训练 step：两次 backward + 共享区梯度合成。

    关键规则：
    1) 共享区（LoRA）: 用 LU 与 LG 各自产生梯度，再做策略融合。
    2) 理解头（task-specific）: 只使用 LU 梯度。
    3) 生成解码器（task-specific）: 只使用 LG 梯度。
    """
    images, labels = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)

    losses, _ = model.compute_losses(images, labels, lambda_u=lambda_u, lambda_g=lambda_g)
    lu = float(lambda_u) * losses.loss_u
    lg = float(lambda_g) * losses.loss_g

    shared_params = model.shared_lora_parameters()
    u_head_params = model.understanding_parameters()

    # ===== 1) L_U backward，提取共享区 g_u =====
    lu.backward(retain_graph=True)
    g_u = _clone_or_zero_grads(shared_params)
    # 保存理解分支专有梯度（后续会清零）
    u_head_grads = _clone_or_zero_grads(u_head_params)

    # ===== 2) 清空梯度 =====
    optimizer.zero_grad(set_to_none=True)

    # ===== 3) L_G backward，提取共享区 g_g =====
    lg.backward()
    g_g = _clone_or_zero_grads(shared_params)
    # 此时 decoder 参数上保留的是 LG 梯度（符合需求）

    # ===== 4) 共享 LoRA 梯度路由 =====
    g_total = apply_gradient_strategy(g_u, g_g, strategy=strategy, cagrad_beta=cagrad_beta)
    _assign_grads(shared_params, g_total)
    # 理解头只用 LU 梯度，覆盖回去
    _assign_grads(u_head_params, u_head_grads)

    if max_grad_norm is not None and max_grad_norm > 0:
        trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if len(trainable) > 0:
            nn.utils.clip_grad_norm_(trainable, max_grad_norm)

    optimizer.step()

    with torch.no_grad():
        vu = _flatten(g_u)
        vg = _flatten(g_g)
        denom = (vu.norm() * vg.norm()).clamp_min(1e-12)
        cos = float((torch.dot(vu, vg) / denom).item())

    return {
        "loss_total": float((lu + lg).item()),
        "loss_u": float(losses.loss_u.item()),
        "loss_g": float(losses.loss_g.item()),
        "cos_lora_u_g": cos,
    }
