import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .grad_conflict import (
    apply_cagrad,
    apply_conflict_aware,
    apply_egd,
    apply_gma_laga,
    apply_la_cagrad,
    apply_naive,
)
try:
    from .losses import FeatureVarianceLoss
except Exception:  # noqa: BLE001
    class FeatureVarianceLoss(nn.Module):
        """Fallback VICReg-style variance regularization.

        Some clusters may run an older `unirae.losses` that does not expose
        FeatureVarianceLoss yet; keep the training entry self-contained.
        """

        def __init__(self, gamma: float = 1.0, eps: float = 1e-4):
            super().__init__()
            self.gamma = float(gamma)
            self.eps = float(eps)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            if features.ndim == 4:
                x = F.adaptive_avg_pool2d(features, output_size=(1, 1)).flatten(1)
            elif features.ndim == 2:
                x = features
            else:
                raise ValueError(
                    f"FeatureVarianceLoss expects [B, D] or [B, C, H, W], got shape={tuple(features.shape)}"
                )

            bsz, dim = x.shape
            if dim <= 0:
                raise ValueError("FeatureVarianceLoss got empty feature dimension.")

            x_centered = x - x.mean(dim=0, keepdim=True)
            if bsz > 1:
                var = (x_centered * x_centered).sum(dim=0) / float(bsz - 1)
            else:
                var = x.new_zeros((dim,))

            std = torch.sqrt(var + self.eps)
            return torch.relu(self.gamma - std).mean()
from .train_imagenet100_dynamics import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ViTBridgeModel,
    _autograd_grads_by_name,
    _global_grad_norm,
    _group_encoder_params,
    _init_distributed,
    _reduce_mean_scalar,
    _safe_cosine,
    _neg_ratio,
    build_encoder,
    build_imagenet100_dataloaders,
    evaluate_recon_and_understanding,
    save_layerwise_conflict_csv,
)
from .utils import append_jsonl, ensure_dir, save_json, seed_everything


def _materialize_grads(
    loss: torch.Tensor,
    params: Sequence[nn.Parameter],
    retain_graph: bool = False,
) -> List[torch.Tensor]:
    if len(params) == 0:
        return []
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    out: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p, memory_format=torch.preserve_format))
        else:
            out.append(g.detach().clone())
    return out


def _set_grads(params: Sequence[nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g


def _all_trainable_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _all_reduce_grads_if_needed(
    params: Sequence[nn.Parameter],
    is_distributed: bool,
    world_size: int,
) -> None:
    if not is_distributed:
        return
    ws = float(max(1, int(world_size)))
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(ws)


def _flatten(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(grads) == 0:
        return torch.zeros(1)
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def _is_deep_param(name: str) -> bool:
    # LACAR 深层定义：Blocks.11 与最终 norm。
    return name.startswith("blocks.11") or name.startswith("norm")


def _build_param_groups(base_model: ViTBridgeModel):
    enc_named = [(n, p) for n, p in base_model.encoder.named_parameters() if p.requires_grad]
    cls_named = [(f"cls_head.{n}", p) for n, p in base_model.cls_head.named_parameters() if p.requires_grad]
    dec_named = [(f"decoder.{n}", p) for n, p in base_model.decoder.named_parameters() if p.requires_grad]

    enc_params = [p for _, p in enc_named]
    cls_params = [p for _, p in cls_named]
    dec_params = [p for _, p in dec_named]

    shared_params = enc_params
    aux_params = cls_params + dec_params
    return enc_named, cls_named, dec_named, shared_params, aux_params


def _build_encoder_group_indices(enc_named: List[Tuple[str, nn.Parameter]]) -> Dict[str, List[int]]:
    """Build layer-wise groups for encoder parameters."""
    groups: Dict[str, List[int]] = {}
    for idx, (name, _) in enumerate(enc_named):
        if name.startswith("patch_embed"):
            gname = "patch_embed"
        elif (
            name.startswith("cls_token")
            or name.startswith("pos_embed")
            or name.startswith("register_tokens")
        ):
            gname = "embeddings"
        elif name.startswith("blocks."):
            parts = name.split(".")
            block_id = parts[1] if len(parts) > 1 else "0"
            gname = f"blocks.{block_id}"
        elif name.startswith("norm"):
            gname = "norm"
        else:
            gname = "other"
        groups.setdefault(gname, []).append(idx)
    return groups


def _build_encoder_group_indices_coarse(enc_named: List[Tuple[str, nn.Parameter]]) -> Dict[str, List[int]]:
    """Build coarse layer-wise groups to reduce per-block routing noise."""
    groups: Dict[str, List[int]] = {}
    for idx, (name, _) in enumerate(enc_named):
        if name.startswith("patch_embed") or name.startswith("cls_token") or name.startswith("pos_embed"):
            gname = "stem"
        elif name.startswith("register_tokens"):
            gname = "stem"
        elif name.startswith("blocks."):
            parts = name.split(".")
            try:
                block_id = int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                block_id = 0
            if block_id <= 3:
                gname = "blocks.0_3"
            elif block_id <= 7:
                gname = "blocks.4_7"
            else:
                gname = "blocks.8_11"
        elif name.startswith("norm"):
            gname = "norm"
        else:
            gname = "other"
        groups.setdefault(gname, []).append(idx)
    return groups


def _encoder_group_depth_ratio(group_name: str) -> float:
    """Approximate semantic depth for encoder groups.

    Early stem groups are allowed to absorb more reconstruction signal, while
    deeper blocks are more strongly anchored to semantics.
    """
    name = str(group_name)
    if name in {"patch_embed", "embeddings", "stem"}:
        return 0.0
    if name == "norm":
        return 1.0
    if name.startswith("blocks."):
        suffix = name.split(".", 1)[1]
        if "_" in suffix:
            try:
                hi = int(suffix.split("_")[-1])
            except ValueError:
                return 0.5
            return max(0.0, min(1.0, float(hi + 1) / 12.0))
        try:
            block_id = int(suffix)
        except ValueError:
            return 0.5
        return max(0.0, min(1.0, float(block_id + 1) / 12.0))
    if name == "global":
        return 0.5
    return 0.5


def _build_recon_objective(
    pred_patches: torch.Tensor,
    gt_patches: torch.Tensor,
    loss_type: str,
    rmse_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (reconstruction objective used for backward, raw mse for metrics)."""
    mse = F.mse_loss(pred_patches, gt_patches)
    if loss_type == "mse":
        return mse, mse
    if loss_type == "rmse":
        return torch.sqrt(mse + float(rmse_eps)), mse
    raise ValueError(f"Unsupported recon loss_type={loss_type}")


def _collect_probe_stats(
    *,
    base_model: ViTBridgeModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    images_01: torch.Tensor,
    run_dir: Path,
    global_step: int,
    encoder_groups: List[Tuple[str, int, List[Tuple[str, nn.Parameter]]]],
    enc_named: List[Tuple[str, nn.Parameter]],
    recon_loss_type: str,
    recon_rmse_eps: float,
) -> Dict[str, float]:
    probe_outputs = base_model(images)
    probe_logits = probe_outputs["logits"]
    probe_pred_patches = probe_outputs["pred_patches"]
    probe_gt_patches = base_model.patchify(images_01)

    probe_loss_u = F.cross_entropy(probe_logits, labels)
    probe_loss_g, _ = _build_recon_objective(
        probe_pred_patches,
        probe_gt_patches,
        loss_type=str(recon_loss_type),
        rmse_eps=float(recon_rmse_eps),
    )

    dict_gu = _autograd_grads_by_name(probe_loss_u, enc_named, retain_graph=True)
    dict_gg = _autograd_grads_by_name(probe_loss_g, enc_named, retain_graph=False)

    layer_stats = save_layerwise_conflict_csv(
        step=global_step,
        dict_gu=dict_gu,
        dict_gg=dict_gg,
        encoder_groups=encoder_groups,
        out_dir=run_dir,
    )

    # 全局冲突统计
    g_all_u = _flatten([dict_gu[n] for n, _ in enc_named])
    g_all_g = _flatten([dict_gg[n] for n, _ in enc_named])
    layer_stats["global_cosine"] = float(_safe_cosine(g_all_u, g_all_g))
    layer_stats["global_neg_ratio"] = float(_neg_ratio(g_all_u, g_all_g))
    layer_stats["global_gu_norm"] = float(torch.norm(g_all_u).item())
    layer_stats["global_gg_norm"] = float(torch.norm(g_all_g).item())

    # 深层冲突统计（blocks.11 + norm）
    deep_names = [n for n, _ in enc_named if _is_deep_param(n)]
    if len(deep_names) > 0:
        g_deep_u = _flatten([dict_gu[n] for n in deep_names])
        g_deep_g = _flatten([dict_gg[n] for n in deep_names])
        layer_stats["deep_cosine"] = float(_safe_cosine(g_deep_u, g_deep_g))
        layer_stats["deep_neg_ratio"] = float(_neg_ratio(g_deep_u, g_deep_g))
        layer_stats["deep_gu_norm"] = float(torch.norm(g_deep_u).item())
        layer_stats["deep_gg_norm"] = float(torch.norm(g_deep_g).item())
    else:
        layer_stats["deep_cosine"] = 0.0
        layer_stats["deep_neg_ratio"] = 0.0
        layer_stats["deep_gu_norm"] = 0.0
        layer_stats["deep_gg_norm"] = 0.0

    return layer_stats


def _run_lacar_step(
    *,
    loss_u: torch.Tensor,
    loss_g: torch.Tensor,
    loss_var: torch.Tensor,
    enc_named: List[Tuple[str, nn.Parameter]],
    cls_params: List[nn.Parameter],
    dec_params: List[nn.Parameter],
    lambda_u: float,
    lambda_g: float,
    lambda_var: float,
) -> float:
    enc_params = [p for _, p in enc_named]

    gu_list = _materialize_grads(loss_u, enc_params, retain_graph=True)
    gg_list = _materialize_grads(loss_g, enc_params, retain_graph=True)
    gv_list = _materialize_grads(loss_var, enc_params, retain_graph=True)

    merged_enc: List[torch.Tensor] = []
    eps = 1e-12
    for (name, _), gu, gg, gv in zip(enc_named, gu_list, gg_list, gv_list):
        if _is_deep_param(name):
            # 深层非对称零空间投影：保留理解梯度，去除生成梯度中沿理解方向的分量。
            gu_flat = gu.reshape(-1)
            gg_flat = gg.reshape(-1)
            denom = torch.dot(gu_flat, gu_flat).clamp_min(eps)
            proj_coeff = torch.dot(gg_flat, gu_flat) / denom
            gg_proj = gg - proj_coeff * gu
            g_merge = float(lambda_u) * gu + float(lambda_g) * gg_proj
        else:
            # 浅层协同路由：直接相加。
            g_merge = float(lambda_u) * gu + float(lambda_g) * gg

        # 全层附加 FVR 梯度。
        g_merge = g_merge + float(lambda_var) * gv
        merged_enc.append(g_merge)

    cls_grads = _materialize_grads(loss_u, cls_params, retain_graph=True)
    dec_grads = _materialize_grads(loss_g, dec_params, retain_graph=False)

    _set_grads(enc_params, merged_enc)
    _set_grads(cls_params, [float(lambda_u) * g for g in cls_grads])
    _set_grads(dec_params, [float(lambda_g) * g for g in dec_grads])

    return float(_safe_cosine(_flatten(gu_list), _flatten(gg_list)))


def _run_laga_step(
    *,
    loss_u: torch.Tensor,
    loss_g: torch.Tensor,
    enc_named: List[Tuple[str, nn.Parameter]],
    cls_params: List[nn.Parameter],
    dec_params: List[nn.Parameter],
    lambda_u: float,
    lambda_g: float,
    eps: float = 1e-12,
) -> float:
    """Layer-wise Asymmetric Gradient Arbitration (LAGA).

    For each encoder block k (0..11):
      c_k = cos(g_u,k, g_g,k)
      if c_k >= 0:
          g*_k = g_u,k + g_g,k
      else:
          g_g,k <- g_g,k - proj_{g_u,k}(g_g,k)
          g*_k = g_u,k + g_g,k_proj

    Implementation note:
      - Arbitration is block-wise (concatenate all params inside a block),
        then unflatten back to each tensor.
      - Non-block encoder params (e.g., embeddings/patch_embed/norm) use
        weighted direct sum for stability.
    """
    enc_params = [p for _, p in enc_named]

    gu_list = _materialize_grads(loss_u, enc_params, retain_graph=True)
    gg_list = _materialize_grads(loss_g, enc_params, retain_graph=True)
    cls_grads = _materialize_grads(loss_u, cls_params, retain_graph=True)
    dec_grads = _materialize_grads(loss_g, dec_params, retain_graph=False)

    # Apply loss weights before arbitration so fairness protocol is preserved.
    wg_u = [float(lambda_u) * g for g in gu_list]
    wg_g = [float(lambda_g) * g for g in gg_list]

    merged_enc = [u + g for u, g in zip(wg_u, wg_g)]

    block_to_indices: Dict[int, List[int]] = {k: [] for k in range(12)}
    for idx, (name, _) in enumerate(enc_named):
        m = re.match(r"^blocks\.(\d+)\.", name)
        if m is None:
            continue
        k = int(m.group(1))
        if 0 <= k <= 11:
            block_to_indices[k].append(idx)

    eps = float(max(1e-12, eps))

    for k in range(12):
        idxs = block_to_indices[k]
        if len(idxs) == 0:
            continue

        u_block = [wg_u[i] for i in idxs]
        g_block = [wg_g[i] for i in idxs]
        v_u = _flatten(u_block)
        v_g = _flatten(g_block)

        dot = torch.dot(v_u, v_g)
        nu = torch.linalg.norm(v_u)
        ng = torch.linalg.norm(v_g)
        cos = dot / (nu * ng + eps)

        if float(cos.item()) >= 0.0:
            v_m = v_u + v_g
        else:
            proj_coeff = dot / torch.dot(v_u, v_u).clamp_min(eps)
            v_g_proj = v_g - proj_coeff * v_u
            v_m = v_u + v_g_proj

        off = 0
        for i in idxs:
            ref = wg_u[i]
            n = int(ref.numel())
            merged_enc[i] = v_m[off : off + n].view_as(ref)
            off += n

    _set_grads(enc_params, merged_enc)
    _set_grads(cls_params, [float(lambda_u) * g for g in cls_grads])
    _set_grads(dec_params, [float(lambda_g) * g for g in dec_grads])

    return float(_safe_cosine(_flatten(gu_list), _flatten(gg_list)))


def _run_ma_laga_step(
    *,
    loss_u: torch.Tensor,
    loss_g: torch.Tensor,
    enc_named: List[Tuple[str, nn.Parameter]],
    cls_params: List[nn.Parameter],
    dec_params: List[nn.Parameter],
    lambda_u: float,
    lambda_g: float,
    align_gamma: float = 1.0,
    align_max_scale: float = 0.0,
    norm_restore: bool = False,
    mode: str = "full",
    group_to_indices: Optional[Dict[str, List[int]]] = None,
    conflict_tau: float = 0.0,
    eps: float = 1e-8,
    encoder_rec_gate_strength: float = 0.0,
    encoder_rec_gate_min: float = 1.0,
) -> float:
    """MA-LAGA family on ViT blocks.

    mode:
      - full: MA-LAGA (align magnitude then conflict projection)
      - capped_full: cap alignment scale then apply LAGA projection on conflicts
      - direction_only: pure LAGA (projection only)
      - magnitude_only: pure MA (alignment only, no projection)
      - nr_laga: projection + restore generation's own norm on conflicts
    """
    enc_params = [p for _, p in enc_named]

    gu_list = _materialize_grads(loss_u, enc_params, retain_graph=True)
    gg_list = _materialize_grads(loss_g, enc_params, retain_graph=True)
    cls_grads = _materialize_grads(loss_u, cls_params, retain_graph=True)
    dec_grads = _materialize_grads(loss_g, dec_params, retain_graph=False)

    # Keep fairness protocol: apply task weights before group arbitration.
    wg_u = [float(lambda_u) * g for g in gu_list]
    wg_g = [float(lambda_g) * g for g in gg_list]

    merged_enc = [u + g for u, g in zip(wg_u, wg_g)]

    if group_to_indices is None:
        group_to_indices = _build_encoder_group_indices(enc_named)

    valid_groups: List[Tuple[str, List[int]]] = []
    for group_name, raw_indices in group_to_indices.items():
        idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(enc_named)]
        if len(idxs) == 0:
            continue
        # Keep deterministic order and remove duplicates.
        idxs = list(dict.fromkeys(idxs))
        valid_groups.append((str(group_name), idxs))
    if len(valid_groups) == 0:
        valid_groups = [("global", list(range(len(enc_named))))]

    eps = float(max(1e-8, eps))
    align_gamma = float(max(0.0, align_gamma))
    align_max_scale = float(align_max_scale)
    conflict_tau = float(max(0.0, conflict_tau))
    encoder_rec_gate_strength = float(max(0.0, encoder_rec_gate_strength))
    encoder_rec_gate_min = float(min(1.0, max(0.0, encoder_rec_gate_min)))
    mode = str(mode).lower()
    if mode not in {"full", "capped_full", "direction_only", "magnitude_only", "nr_laga"}:
        raise ValueError(
            f"Unsupported ma_laga_mode={mode}. "
            "Use one of: full|capped_full|direction_only|magnitude_only|nr_laga."
        )
    if mode == "capped_full" and align_max_scale <= 0.0:
        raise ValueError(
            f"ma_laga_mode=capped_full requires ma_laga_max_scale>0, got {align_max_scale}"
        )

    for group_name, idxs in valid_groups:
        u_block = [wg_u[i] for i in idxs]
        g_block = [wg_g[i] for i in idxs]
        v_u = _flatten(u_block)
        rec_gate = 1.0
        if encoder_rec_gate_strength > 0.0:
            depth_ratio = _encoder_group_depth_ratio(group_name)
            rec_gate = max(encoder_rec_gate_min, 1.0 - encoder_rec_gate_strength * depth_ratio)
        v_g = rec_gate * _flatten(g_block)

        dot = torch.dot(v_u, v_g)
        nu = torch.linalg.norm(v_u)
        ng = torch.linalg.norm(v_g)
        cos = dot / (nu * ng + eps)
        is_conflict = float(cos.item()) < -conflict_tau

        if mode == "direction_only":
            if not is_conflict:
                v_m = v_u + v_g
            else:
                proj_coeff = dot / torch.dot(v_u, v_u).clamp_min(eps)
                v_g_perp = v_g - proj_coeff * v_u
                v_m = v_u + v_g_perp
        elif mode == "nr_laga":
            if not is_conflict:
                v_m = v_u + v_g
            else:
                proj_coeff = dot / torch.dot(v_u, v_u).clamp_min(eps)
                v_g_perp = v_g - proj_coeff * v_u
                ng_perp = torch.linalg.norm(v_g_perp)
                v_g_perp_restored = v_g_perp * (ng / (ng_perp + eps))
                v_m = v_u + v_g_perp_restored
        else:
            if mode == "capped_full":
                scale_factor = torch.clamp(nu / (ng + eps), max=align_max_scale).detach()
            else:
                scale_factor = torch.pow((nu / (ng + eps)).clamp_min(eps), align_gamma).detach()
            v_g_aligned = scale_factor * v_g

            if mode == "magnitude_only":
                v_m = v_u + v_g_aligned
            else:
                if not is_conflict:
                    v_m = v_u + v_g_aligned
                else:
                    proj_coeff = torch.dot(v_g_aligned, v_u) / torch.dot(v_u, v_u).clamp_min(eps)
                    v_g_perp = v_g_aligned - proj_coeff * v_u
                    if bool(norm_restore):
                        ng_aligned = torch.linalg.norm(v_g_aligned)
                        ng_perp = torch.linalg.norm(v_g_perp)
                        v_g_perp = v_g_perp * (ng_aligned / (ng_perp + eps))
                    v_m = v_u + v_g_perp

        off = 0
        for i in idxs:
            ref = wg_u[i]
            n = int(ref.numel())
            merged_enc[i] = v_m[off : off + n].view_as(ref)
            off += n

    _set_grads(enc_params, merged_enc)
    _set_grads(cls_params, [float(lambda_u) * g for g in cls_grads])
    _set_grads(dec_params, [float(lambda_g) * g for g in dec_grads])

    return float(_safe_cosine(_flatten(gu_list), _flatten(gg_list)))


def _run_laga_gbvc_step(
    *,
    loss_u: torch.Tensor,
    loss_g: torch.Tensor,
    loss_gbvc: torch.Tensor,
    enc_named: List[Tuple[str, nn.Parameter]],
    cls_params: List[nn.Parameter],
    dec_params: List[nn.Parameter],
    lambda_u: float,
    lambda_g: float,
    lambda_gbvc: float,
    eps: float = 1e-12,
) -> float:
    """Vanilla LAGA + GBVC regularization on encoder features."""
    enc_params = [p for _, p in enc_named]

    gu_list = _materialize_grads(loss_u, enc_params, retain_graph=True)
    gg_list = _materialize_grads(loss_g, enc_params, retain_graph=True)
    ggvc_list = _materialize_grads(loss_gbvc, enc_params, retain_graph=True)
    cls_grads = _materialize_grads(loss_u, cls_params, retain_graph=True)
    dec_grads = _materialize_grads(loss_g, dec_params, retain_graph=False)

    wg_u = [float(lambda_u) * g for g in gu_list]
    wg_g = [float(lambda_g) * g for g in gg_list]

    merged_enc = [u + g for u, g in zip(wg_u, wg_g)]

    block_to_indices: Dict[int, List[int]] = {k: [] for k in range(12)}
    for idx, (name, _) in enumerate(enc_named):
        m = re.match(r"^blocks\.(\d+)\.", name)
        if m is None:
            continue
        k = int(m.group(1))
        if 0 <= k <= 11:
            block_to_indices[k].append(idx)

    eps = float(max(1e-12, eps))

    for k in range(12):
        idxs = block_to_indices[k]
        if len(idxs) == 0:
            continue

        u_block = [wg_u[i] for i in idxs]
        g_block = [wg_g[i] for i in idxs]
        v_u = _flatten(u_block)
        v_g = _flatten(g_block)

        dot = torch.dot(v_u, v_g)
        nu = torch.linalg.norm(v_u)
        ng = torch.linalg.norm(v_g)
        cos = dot / (nu * ng + eps)

        if float(cos.item()) >= 0.0:
            v_m = v_u + v_g
        else:
            proj_coeff = dot / torch.dot(v_u, v_u).clamp_min(eps)
            v_g_proj = v_g - proj_coeff * v_u
            v_m = v_u + v_g_proj

        off = 0
        for i in idxs:
            ref = wg_u[i]
            n = int(ref.numel())
            merged_enc[i] = v_m[off : off + n].view_as(ref)
            off += n

    if float(lambda_gbvc) != 0.0:
        merged_enc = [g + float(lambda_gbvc) * gv for g, gv in zip(merged_enc, ggvc_list)]

    _set_grads(enc_params, merged_enc)
    _set_grads(cls_params, [float(lambda_u) * g for g in cls_grads])
    _set_grads(dec_params, [float(lambda_g) * g for g in dec_grads])

    return float(_safe_cosine(_flatten(gu_list), _flatten(gg_list)))


def main() -> None:
    parser = argparse.ArgumentParser("IN100 Multi-Method Benchmark with DINO-Small")

    parser.add_argument(
        "--method",
        type=str,
        default="joint",
        choices=[
            "und_only",
            "gen_only",
            "joint",
            "pcgrad",
            "cagrad",
            "la_cagrad",
            "gma_laga",
            "egd",
            "lacar",
            "laga",
            "ma_laga",
            "enc_anchor_dsga",
            "laga_gbvc",
        ],
        help="Experiment branch selector.",
    )
    parser.add_argument("--encoder_init", type=str, default="dinov2", choices=["scratch", "dinov2"])
    parser.add_argument("--encoder_ckpt", type=str, default="")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Local dataset root. Supports ImageFolder (train/val) or HF load_from_disk directory.",
    )
    parser.add_argument("--hf_dataset_id", type=str, default="clane9/imagenet-100")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--label_key", type=str, default="label")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10000)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_g", type=float, default=1.0)
    parser.add_argument(
        "--recon_loss_type",
        type=str,
        default="mse",
        choices=["mse", "rmse"],
        help="Reconstruction objective used for backward (mse or rmse).",
    )
    parser.add_argument(
        "--recon_rmse_eps",
        type=float,
        default=1e-12,
        help="Numerical epsilon in sqrt(mse + eps) when recon_loss_type=rmse.",
    )
    parser.add_argument(
        "--auto_align_lambda_g",
        action="store_true",
        default=True,
        help="At step 1, set lambda_g = detach(Lu/Lg) for 1:1 weighted initialization.",
    )
    parser.add_argument("--disable_auto_align_lambda_g", action="store_true")

    parser.add_argument("--cagrad_beta", type=float, default=0.35)
    parser.add_argument(
        "--gma_laga_max_scale",
        type=float,
        default=100.0,
        help="Global generation scale cap for GMA-LAGA.",
    )
    parser.add_argument(
        "--ma_laga_mode",
        type=str,
        default="full",
        choices=["full", "capped_full", "direction_only", "magnitude_only", "nr_laga"],
        help="MA-LAGA ablation mode.",
    )
    parser.add_argument(
        "--ma_laga_align_gamma",
        type=float,
        default=1.0,
        help="Magnitude alignment strength. 1.0 means strict norm matching.",
    )
    parser.add_argument(
        "--ma_laga_max_scale",
        type=float,
        default=3.0,
        help="Cap for alignment scale in ma_laga_mode=capped_full.",
    )
    parser.add_argument(
        "--ma_laga_norm_restore",
        action="store_true",
        default=False,
        help="If enabled, restore aligned generation norm after conflict projection.",
    )
    parser.add_argument(
        "--ma_laga_grouping",
        type=str,
        default="layerwise",
        choices=["layerwise", "layerwise_coarse", "global"],
        help="MA-LAGA grouping scope: fine layerwise, coarse layerwise, or a single global group.",
    )
    parser.add_argument(
        "--enc_anchor_rec_gate_strength",
        type=float,
        default=0.75,
        help="Depth-aware suppression strength for reconstruction gradients on encoder groups.",
    )
    parser.add_argument(
        "--enc_anchor_rec_gate_min",
        type=float,
        default=0.20,
        help="Minimum reconstruction gate kept on the deepest encoder groups.",
    )
    parser.add_argument(
        "--ma_laga_conflict_tau",
        type=float,
        default=0.0,
        help="Conflict margin: project only when cosine < -tau.",
    )
    parser.add_argument(
        "--ma_laga_conflict_tau_end",
        type=float,
        default=-1.0,
        help="If >=0, linearly anneal tau from ma_laga_conflict_tau to this value across training.",
    )
    parser.add_argument("--lambda_var", type=float, default=0.20)
    parser.add_argument("--var_gamma", type=float, default=1.0)
    parser.add_argument("--var_eps", type=float, default=1e-4)
    parser.add_argument("--lambda_gbvc", type=float, default=0.1)
    parser.add_argument("--gbvc_nu", type=float, default=1.0)

    parser.add_argument("--decoder_dim", type=int, default=384)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=6)
    parser.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--decoder_drop_rate", type=float, default=0.0)

    parser.add_argument("--probe_every", type=int, default=500)
    parser.add_argument("--probe_until", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_max_batches", type=int, default=50)
    parser.add_argument("--eval_rfid_every", type=int, default=0)
    parser.add_argument("--final_eval_rfid", action="store_true", default=True)
    parser.add_argument("--no_final_eval_rfid", action="store_true")
    parser.add_argument("--eval_rfid_num_samples", type=int, default=2048)
    parser.add_argument("--eval_rfid_batch_size", type=int, default=64)
    parser.add_argument("--eval_rfid_tmp_dir", type=str, default="/tmp")

    parser.add_argument("--output_root", type=str, default="results/in100_method_bench")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args()

    if args.disable_auto_align_lambda_g:
        args.auto_align_lambda_g = False
    if args.no_final_eval_rfid:
        args.final_eval_rfid = False
    if float(args.gma_laga_max_scale) <= 0.0:
        raise ValueError(f"gma_laga_max_scale must be > 0, got {args.gma_laga_max_scale}")
    if float(args.recon_rmse_eps) <= 0.0:
        raise ValueError(f"recon_rmse_eps must be > 0, got {args.recon_rmse_eps}")
    if float(args.ma_laga_align_gamma) < 0.0:
        raise ValueError(f"ma_laga_align_gamma must be >= 0, got {args.ma_laga_align_gamma}")
    if float(args.enc_anchor_rec_gate_strength) < 0.0:
        raise ValueError(
            f"enc_anchor_rec_gate_strength must be >= 0, got {args.enc_anchor_rec_gate_strength}"
        )
    if not (0.0 <= float(args.enc_anchor_rec_gate_min) <= 1.0):
        raise ValueError(
            f"enc_anchor_rec_gate_min must be in [0, 1], got {args.enc_anchor_rec_gate_min}"
        )
    if float(args.ma_laga_conflict_tau) < 0.0:
        raise ValueError(f"ma_laga_conflict_tau must be >= 0, got {args.ma_laga_conflict_tau}")
    if float(args.ma_laga_conflict_tau_end) < 0.0 and float(args.ma_laga_conflict_tau_end) != -1.0:
        raise ValueError(
            f"ma_laga_conflict_tau_end must be -1 (disabled) or >= 0, got {args.ma_laga_conflict_tau_end}"
        )
    if str(args.ma_laga_mode) == "capped_full" and float(args.ma_laga_max_scale) <= 0.0:
        raise ValueError(f"ma_laga_max_scale must be > 0 for capped_full, got {args.ma_laga_max_scale}")

    is_distributed, rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

    seed_everything(int(args.seed) + int(rank))

    if is_distributed and args.device == "cpu" and torch.cuda.is_available():
        raise RuntimeError("Distributed launch with CUDA available does not support --device=cpu.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda is set but CUDA is not available.")

    if is_distributed and torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_loader, val_loader, data_meta = build_imagenet100_dataloaders(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        dataset_path=(str(args.dataset_path).strip() if str(args.dataset_path).strip() else None),
        hf_dataset_id=str(args.hf_dataset_id),
        cache_dir=(str(args.cache_dir) if args.cache_dir else None),
        image_key=str(args.image_key),
        label_key=str(args.label_key),
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    encoder = build_encoder(args.encoder_init, image_size=224, encoder_ckpt=str(args.encoder_ckpt))
    model = ViTBridgeModel(
        encoder=encoder,
        num_classes=int(data_meta["num_classes"]),
        image_size=224,
        decoder_dim=int(args.decoder_dim),
        decoder_depth=int(args.decoder_depth),
        decoder_heads=int(args.decoder_heads),
        decoder_mlp_ratio=float(args.decoder_mlp_ratio),
        decoder_drop_rate=float(args.decoder_drop_rate),
    ).to(device)

    # 使用 DDP 封装做多卡一致前向；梯度由本脚本手动写入并 all_reduce。
    if is_distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        else:
            model = DDP(model, broadcast_buffers=False)

    base_model = model.module if isinstance(model, DDP) else model
    all_trainable = _all_trainable_params(base_model)

    optimizer = torch.optim.AdamW(
        all_trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )

    warmup_steps = max(1, int(args.warmup_steps))

    def _lr_lambda(step_idx: int) -> float:
        # 线性预热到基准 LR，预热后常数保持。
        s = step_idx + 1
        if s <= warmup_steps:
            return float(s) / float(warmup_steps)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    fvr_loss = FeatureVarianceLoss(gamma=float(args.var_gamma), eps=float(args.var_eps)).to(device)

    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = (
            f"in100_{args.method}_{args.encoder_init}_"
            f"bs{int(args.batch_size)}_s{int(args.seed)}"
        )
    run_dir = Path(args.output_root) / run_name
    ensure_dir(str(run_dir))

    if is_main_process:
        save_json(
            {
                "args": vars(args),
                "data_meta": data_meta,
                "device": str(device),
                "rank": int(rank),
                "world_size": int(world_size),
                "effective_global_batch": int(args.batch_size) * int(world_size),
                "num_trainable_params": int(sum(p.numel() for p in all_trainable)),
            },
            str(run_dir / "run_setup.json"),
        )

    metrics_path = run_dir / "train_metrics.jsonl"
    eval_metrics_path = run_dir / "eval_metrics.jsonl"

    encoder_groups = _group_encoder_params(base_model.encoder)
    enc_named, cls_named, dec_named, shared_params, aux_params = _build_param_groups(base_model)
    encoder_group_indices = _build_encoder_group_indices(enc_named)
    grouping_mode = str(args.ma_laga_grouping).lower()
    if grouping_mode == "global":
        ma_laga_group_indices = {"global": list(range(len(enc_named)))}
    elif grouping_mode == "layerwise_coarse":
        ma_laga_group_indices = _build_encoder_group_indices_coarse(enc_named)
    else:
        ma_laga_group_indices = dict(encoder_group_indices)
    cls_params = [p for _, p in cls_named]
    dec_params = [p for _, p in dec_named]

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    runtime_lambda_g = float(args.lambda_g)
    tau_start = float(args.ma_laga_conflict_tau)
    tau_end = float(args.ma_laga_conflict_tau_end) if float(args.ma_laga_conflict_tau_end) >= 0.0 else tau_start
    aligned_once = False

    global_step = 0
    train_epoch = 0
    train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    if train_sampler is not None:
        train_sampler.set_epoch(train_epoch)
    train_iter = iter(train_loader)

    pbar = (
        tqdm(total=int(args.max_steps), desc=f"in100-{args.method}", dynamic_ncols=True)
        if is_main_process
        else None
    )

    while global_step < int(args.max_steps):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(train_epoch)
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        global_step += 1
        if int(args.max_steps) > 1:
            tau_now = tau_start + (tau_end - tau_start) * (float(global_step - 1) / float(int(args.max_steps) - 1))
        else:
            tau_now = tau_end

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images_01 = (images * std + mean).clamp_(0.0, 1.0)

        outputs = model(images)
        logits = outputs["logits"]
        pred_patches = outputs["pred_patches"]
        gt_patches = base_model.patchify(images_01)

        loss_u = F.cross_entropy(logits, labels)
        loss_g, loss_g_mse = _build_recon_objective(
            pred_patches,
            gt_patches,
            loss_type=str(args.recon_loss_type),
            rmse_eps=float(args.recon_rmse_eps),
        )

        # Step-0 自动损失对齐：lambda_g = detach(Lu/Lg)
        if (
            (not aligned_once)
            and args.auto_align_lambda_g
            and (
                args.method
                in {
                    "joint",
                    "pcgrad",
                    "cagrad",
                    "la_cagrad",
                    "gma_laga",
                    "egd",
                    "lacar",
                    "laga",
                    "ma_laga",
                    "enc_anchor_dsga",
                    "laga_gbvc",
                }
            )
        ):
            lu_mean = _reduce_mean_scalar(float(loss_u.detach().item()), device, is_distributed, world_size)
            lg_mean = _reduce_mean_scalar(float(loss_g.detach().item()), device, is_distributed, world_size)
            runtime_lambda_g = float(lu_mean / max(lg_mean, 1e-12))
            aligned_once = True
            if is_main_process:
                print(
                    f"[align] step=1 raw_lu={lu_mean:.6f} raw_lg={lg_mean:.6f} "
                    f"=> lambda_g={runtime_lambda_g:.6f}"
                )

        loss_var = torch.tensor(0.0, device=device)
        loss_gbvc = torch.tensor(0.0, device=device)
        if args.method == "lacar":
            # 用 encoder patch token 均值做方差约束，抑制表征坍缩。
            patch_tokens = outputs["tokens"][:, -base_model.num_patches :]
            feat_for_var = patch_tokens.mean(dim=1)
            loss_var = fvr_loss(feat_for_var)
        elif args.method == "laga_gbvc":
            z = outputs["tokens"]
            variance_z = torch.var(z, dim=0).mean()
            loss_gbvc = torch.relu(torch.tensor(float(args.gbvc_nu), device=device) - variance_z)

        should_probe = (
            global_step <= int(args.probe_until)
            and (global_step % max(1, int(args.probe_every)) == 0)
        )

        probe_stats: Optional[Dict[str, float]] = None
        if should_probe and is_main_process:
            probe_stats = _collect_probe_stats(
                base_model=base_model,
                images=images,
                labels=labels,
                images_01=images_01,
                run_dir=run_dir,
                global_step=global_step,
                encoder_groups=encoder_groups,
                enc_named=enc_named,
                recon_loss_type=str(args.recon_loss_type),
                recon_rmse_eps=float(args.recon_rmse_eps),
            )

        optimizer.zero_grad(set_to_none=True)

        train_grad_cos = 0.0
        if args.method == "und_only":
            # encoder + cls_head 仅由 CE 监督。
            enc_params = [p for _, p in enc_named]
            enc_grads = _materialize_grads(loss_u, enc_params, retain_graph=True)
            cls_grads = _materialize_grads(loss_u, cls_params, retain_graph=False)
            _set_grads(enc_params, [float(args.lambda_u) * g for g in enc_grads])
            _set_grads(cls_params, [float(args.lambda_u) * g for g in cls_grads])

        elif args.method == "gen_only":
            # encoder + decoder 仅由重建监督。
            enc_params = [p for _, p in enc_named]
            enc_grads = _materialize_grads(loss_g, enc_params, retain_graph=True)
            dec_grads = _materialize_grads(loss_g, dec_params, retain_graph=False)
            _set_grads(enc_params, enc_grads)
            _set_grads(dec_params, dec_grads)

        elif args.method == "joint":
            train_grad_cos = apply_naive(
                loss_txt=loss_u,
                loss_rec=loss_g,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(runtime_lambda_g),
            )

        elif args.method == "pcgrad":
            train_grad_cos = apply_conflict_aware(
                loss_txt=loss_u,
                loss_rec=loss_g,
                lora_params=shared_params,
                aux_params=aux_params,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(runtime_lambda_g),
            )

        elif args.method == "cagrad":
            train_grad_cos = apply_cagrad(
                loss_txt=loss_u,
                loss_rec=loss_g,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(runtime_lambda_g),
                beta=float(args.cagrad_beta),
            )

        elif args.method == "la_cagrad":
            train_grad_cos = apply_la_cagrad(
                loss_txt=loss_u,
                loss_rec=loss_g,
                shared_params=shared_params,
                aux_params=aux_params,
                group_to_indices=encoder_group_indices,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(runtime_lambda_g),
                beta=float(args.cagrad_beta),
            )

        elif args.method == "gma_laga":
            train_grad_cos = apply_gma_laga(
                loss_txt=loss_u,
                loss_rec=loss_g,
                shared_params=shared_params,
                aux_params=aux_params,
                group_to_indices=encoder_group_indices,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(runtime_lambda_g),
                max_scale=float(args.gma_laga_max_scale),
                eps=1e-8,
            )

        elif args.method == "egd":
            train_grad_cos = apply_egd(
                loss_txt=loss_u,
                loss_rec=loss_g,
                shared_params=shared_params,
                aux_params=aux_params,
                group_to_indices=encoder_group_indices,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(runtime_lambda_g),
                eps=1e-8,
            )

        elif args.method == "lacar":
            train_grad_cos = _run_lacar_step(
                loss_u=loss_u,
                loss_g=loss_g,
                loss_var=loss_var,
                enc_named=enc_named,
                cls_params=cls_params,
                dec_params=dec_params,
                lambda_u=float(args.lambda_u),
                lambda_g=float(runtime_lambda_g),
                lambda_var=float(args.lambda_var),
            )

        elif args.method == "laga":
            train_grad_cos = _run_laga_step(
                loss_u=loss_u,
                loss_g=loss_g,
                enc_named=enc_named,
                cls_params=cls_params,
                dec_params=dec_params,
                lambda_u=float(args.lambda_u),
                lambda_g=float(runtime_lambda_g),
            )

        elif args.method == "ma_laga":
            train_grad_cos = _run_ma_laga_step(
                loss_u=loss_u,
                loss_g=loss_g,
                enc_named=enc_named,
                cls_params=cls_params,
                dec_params=dec_params,
                lambda_u=float(args.lambda_u),
                lambda_g=float(runtime_lambda_g),
                align_gamma=float(args.ma_laga_align_gamma),
                align_max_scale=float(args.ma_laga_max_scale),
                norm_restore=bool(args.ma_laga_norm_restore),
                mode=str(args.ma_laga_mode),
                group_to_indices=ma_laga_group_indices,
                conflict_tau=float(tau_now),
                eps=1e-8,
                encoder_rec_gate_strength=0.0,
                encoder_rec_gate_min=1.0,
            )

        elif args.method == "enc_anchor_dsga":
            train_grad_cos = _run_ma_laga_step(
                loss_u=loss_u,
                loss_g=loss_g,
                enc_named=enc_named,
                cls_params=cls_params,
                dec_params=dec_params,
                lambda_u=float(args.lambda_u),
                lambda_g=float(runtime_lambda_g),
                align_gamma=float(args.ma_laga_align_gamma),
                align_max_scale=float(args.ma_laga_max_scale),
                norm_restore=bool(args.ma_laga_norm_restore),
                mode=str(args.ma_laga_mode),
                group_to_indices=ma_laga_group_indices,
                conflict_tau=float(tau_now),
                eps=1e-8,
                encoder_rec_gate_strength=float(args.enc_anchor_rec_gate_strength),
                encoder_rec_gate_min=float(args.enc_anchor_rec_gate_min),
            )

        elif args.method == "laga_gbvc":
            train_grad_cos = _run_laga_gbvc_step(
                loss_u=loss_u,
                loss_g=loss_g,
                loss_gbvc=loss_gbvc,
                enc_named=enc_named,
                cls_params=cls_params,
                dec_params=dec_params,
                lambda_u=float(args.lambda_u),
                lambda_g=float(runtime_lambda_g),
                lambda_gbvc=float(args.lambda_gbvc),
            )

        else:
            raise ValueError(f"Unsupported method={args.method}")

        _all_reduce_grads_if_needed(all_trainable, is_distributed, world_size)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            rmse = torch.sqrt(loss_g_mse.detach().clamp_min(0.0))
            loss_u_weighted = float(args.lambda_u) * loss_u
            if args.method == "gen_only":
                loss_g_weighted = loss_g
            else:
                loss_g_weighted = float(runtime_lambda_g) * loss_g
            if args.method == "lacar":
                total_loss = loss_u_weighted + loss_g_weighted + float(args.lambda_var) * loss_var
            elif args.method == "laga_gbvc":
                total_loss = loss_u_weighted + loss_g_weighted + float(args.lambda_gbvc) * loss_gbvc
            elif args.method == "und_only":
                total_loss = loss_u_weighted
            elif args.method == "gen_only":
                total_loss = loss_g
            else:
                total_loss = loss_u_weighted + loss_g_weighted

        loss_u_log = _reduce_mean_scalar(float(loss_u.detach().item()), device, is_distributed, world_size)
        loss_g_log = _reduce_mean_scalar(float(loss_g.detach().item()), device, is_distributed, world_size)
        loss_g_mse_log = _reduce_mean_scalar(float(loss_g_mse.detach().item()), device, is_distributed, world_size)
        loss_var_log = _reduce_mean_scalar(float(loss_var.detach().item()), device, is_distributed, world_size)
        loss_gbvc_log = _reduce_mean_scalar(float(loss_gbvc.detach().item()), device, is_distributed, world_size)
        loss_total_log = _reduce_mean_scalar(float(total_loss.detach().item()), device, is_distributed, world_size)
        acc_log = _reduce_mean_scalar(float(acc.item()), device, is_distributed, world_size)
        rmse_log = _reduce_mean_scalar(float(rmse.item()), device, is_distributed, world_size)
        lr_now = float(optimizer.param_groups[0]["lr"])

        loss_u_weighted_log = float(args.lambda_u) * loss_u_log
        if args.method == "gen_only":
            loss_g_weighted_log = loss_g_log
        else:
            loss_g_weighted_log = float(runtime_lambda_g) * loss_g_log

        row = {
            "step": int(global_step),
            "method": str(args.method),
            "ma_laga_grouping": str(args.ma_laga_grouping),
            "ma_laga_conflict_tau": float(args.ma_laga_conflict_tau),
            "ma_laga_conflict_tau_now": float(tau_now),
            "ma_laga_conflict_tau_end": float(args.ma_laga_conflict_tau_end),
            "loss_u": float(loss_u_log),
            "loss_g": float(loss_g_log),
            "loss_g_mse": float(loss_g_mse_log),
            "loss_var": float(loss_var_log),
            "loss_gbvc": float(loss_gbvc_log),
            "loss_u_weighted": float(loss_u_weighted_log),
            "loss_g_weighted": float(loss_g_weighted_log),
            "loss_total": float(loss_total_log),
            "recon_loss_type": str(args.recon_loss_type),
            "lambda_u": float(args.lambda_u),
            "lambda_g": float(runtime_lambda_g),
            "lambda_var": float(args.lambda_var if args.method == "lacar" else 0.0),
            "lambda_gbvc": float(args.lambda_gbvc if args.method == "laga_gbvc" else 0.0),
            "train_grad_cosine": float(train_grad_cos),
            "acc": float(acc_log),
            "rmse": float(rmse_log),
            "lr": float(lr_now),
            "loss_ratio_u_over_g": float(loss_u_log / max(loss_g_log, 1e-12)),
            "loss_ratio_weighted_u_over_g": float(loss_u_weighted_log / max(loss_g_weighted_log, 1e-12)),
        }

        if is_main_process and probe_stats is not None:
            row["probe_mean_cosine"] = float(probe_stats["mean_cosine"])
            row["probe_mean_neg_ratio"] = float(probe_stats["mean_neg_ratio"])
            row["probe_global_cosine"] = float(probe_stats["global_cosine"])
            row["probe_global_neg_ratio"] = float(probe_stats["global_neg_ratio"])
            row["probe_deep_cosine"] = float(probe_stats["deep_cosine"])
            row["probe_deep_neg_ratio"] = float(probe_stats["deep_neg_ratio"])
            row["probe_csv"] = str(probe_stats["csv_path"])
            print(
                f"[probe] step={global_step} "
                f"global_cos={probe_stats['global_cosine']:.4f} global_neg={probe_stats['global_neg_ratio']:.4f} "
                f"deep_cos={probe_stats['deep_cosine']:.4f} deep_neg={probe_stats['deep_neg_ratio']:.4f} "
                f"csv={probe_stats['csv_path']}"
            )

        if is_main_process:
            append_jsonl(str(metrics_path), row)
            if pbar is not None:
                pbar.update(1)
                if (global_step == 1) or (global_step % int(args.log_every) == 0) or (global_step == int(args.max_steps)):
                    pbar.set_postfix(
                        lu=f"{row['loss_u']:.4f}",
                        lg=f"{row['loss_g']:.4f}",
                        luw=f"{row['loss_u_weighted']:.4f}",
                        lgw=f"{row['loss_g_weighted']:.4f}",
                        acc=f"{row['acc']:.3f}",
                        rmse=f"{row['rmse']:.4f}",
                        lr=f"{row['lr']:.2e}",
                    )

        if int(args.eval_every) > 0 and (global_step % int(args.eval_every) == 0):
            do_rfid = int(args.eval_rfid_every) > 0 and (global_step % int(args.eval_rfid_every) == 0)
            eval_ret = evaluate_recon_and_understanding(
                model=base_model,
                val_loader=val_loader,
                device=device,
                mean=mean,
                std=std,
                max_batches=int(args.eval_max_batches),
                compute_rfid=bool(do_rfid),
                rfid_num_samples=int(args.eval_rfid_num_samples),
                rfid_batch_size=int(args.eval_rfid_batch_size),
                rfid_tmp_dir=str(args.eval_rfid_tmp_dir),
            )
            base_model.train()
            if is_main_process:
                eval_row = {"step": int(global_step), **eval_ret}
                append_jsonl(str(eval_metrics_path), eval_row)
                print(
                    f"[eval] step={global_step} acc={eval_ret['val_top1_acc']:.4f} "
                    f"rMSE={eval_ret['val_rmse']:.6f} rFID={eval_ret['val_rfid']:.4f}"
                )

    # 训练结束后保存 checkpoint。
    if is_main_process:
        ckpt = {
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": int(global_step),
            "runtime_lambda_g": float(runtime_lambda_g),
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / "latest.pt")

    # 训练结束后强制评测一次 rFID。
    if bool(args.final_eval_rfid):
        final_ret = evaluate_recon_and_understanding(
            model=base_model,
            val_loader=val_loader,
            device=device,
            mean=mean,
            std=std,
            max_batches=int(args.eval_max_batches),
            compute_rfid=True,
            rfid_num_samples=int(args.eval_rfid_num_samples),
            rfid_batch_size=int(args.eval_rfid_batch_size),
            rfid_tmp_dir=str(args.eval_rfid_tmp_dir),
        )
        base_model.train()
        if is_main_process:
            final_row = {"step": int(global_step), "final_eval": True, **final_ret}
            append_jsonl(str(eval_metrics_path), final_row)
            save_json(final_row, str(run_dir / "final_eval.json"))
            print(
                f"[final_eval] step={global_step} acc={final_ret['val_top1_acc']:.4f} "
                f"rMSE={final_ret['val_rmse']:.6f} rFID={final_ret['val_rfid']:.4f}"
            )

    if pbar is not None:
        pbar.close()

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
