from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from .utils import ensure_dir, save_json


def _get_attr_by_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _iter_modules_by_paths(root, paths: Sequence[str]) -> List[torch.nn.Module]:
    out: List[torch.nn.Module] = []
    seen = set()
    for path in paths:
        mod = _get_attr_by_path(root, path)
        if isinstance(mod, torch.nn.Module):
            mid = id(mod)
            if mid not in seen:
                seen.add(mid)
                out.append(mod)
    return out


def _resolve_vit_blocks(encoder: torch.nn.Module):
    # Common for timm ViT and HF DINO wrappers.
    for path in ["blocks", "encoder.layer", "encoder.encoder.layer"]:
        blocks = _get_attr_by_path(encoder, path)
        if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
            return blocks, path
    raise ValueError(
        "Cannot locate transformer blocks on encoder. "
        "Tried: blocks | encoder.layer | encoder.encoder.layer"
    )


def freeze_encoder_except_last_n_blocks(encoder: torch.nn.Module, n: int = 4) -> Dict[str, object]:
    """Freeze shallow encoder, only keep last-n blocks + final norm trainable.

    Behavior:
    - Freeze all encoder params first.
    - Keep `patch_embed/patch_embeddings`, `cls_token`, `pos_embed/position_embeddings` frozen.
    - Unfreeze last `n` transformer blocks.
    - Unfreeze final norm modules (`norm/fc_norm/layernorm` variants).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    for p in encoder.parameters():
        p.requires_grad = False

    blocks, block_path = _resolve_vit_blocks(encoder)
    total_blocks = int(len(blocks))
    n = min(int(n), total_blocks)
    start = total_blocks - n

    for bi in range(start, total_blocks):
        for p in blocks[bi].parameters():
            p.requires_grad = True

    norm_modules = _iter_modules_by_paths(
        encoder,
        [
            "norm",
            "fc_norm",
            "layernorm",
            "encoder.layernorm",
            "encoder.encoder.layernorm",
        ],
    )
    for mod in norm_modules:
        for p in mod.parameters():
            p.requires_grad = True

    # Enforce tokens/embeddings stay frozen.
    token_name_hits = []
    for name, p in encoder.named_parameters():
        if any(
            k in name
            for k in (
                "patch_embed",
                "patch_embeddings",
                "cls_token",
                "pos_embed",
                "position_embeddings",
                "register_tokens",
            )
        ):
            if p.requires_grad:
                token_name_hits.append(name)
            p.requires_grad = False

    trainable_names = [name for name, p in encoder.named_parameters() if p.requires_grad]
    info = {
        "block_path": block_path,
        "total_blocks": total_blocks,
        "trainable_last_n_blocks": n,
        "trainable_block_range": [start, total_blocks - 1] if n > 0 else [],
        "num_trainable_params": int(sum(p.numel() for p in encoder.parameters() if p.requires_grad)),
        "num_frozen_params": int(sum(p.numel() for p in encoder.parameters() if not p.requires_grad)),
        "forced_frozen_token_params": token_name_hits,
    }
    return info


def setup_deep_conflict_bottleneck(encoder: torch.nn.Module, num_trainable_blocks: int = 4) -> Dict[str, object]:
    """Alias kept for training-loop readability."""
    return freeze_encoder_except_last_n_blocks(encoder=encoder, n=num_trainable_blocks)


def should_log_conflict(
    step: int,
    early_until: int = 1000,
    early_every: int = 50,
    late_every: int = 500,
) -> bool:
    if step <= 0:
        return False
    if step <= int(early_until):
        return int(step) % max(1, int(early_every)) == 0
    return int(step) % max(1, int(late_every)) == 0


def _to_flat_or_zeros(g: Optional[torch.Tensor], p: torch.nn.Parameter) -> torch.Tensor:
    if g is None:
        return torch.zeros_like(p, memory_format=torch.preserve_format).reshape(-1)
    return g.detach().reshape(-1)


def _safe_cos(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    nu = torch.norm(u)
    nv = torch.norm(v)
    if float(nu.item()) < eps or float(nv.item()) < eps:
        return 0.0
    return float(torch.dot(u, v).item() / (nu.item() * nv.item() + eps))


def _group_vit_param_name(name: str) -> Tuple[str, int]:
    # timm ViT: blocks.{i}.*
    m = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        return f"block{idx:02d}", idx + 1

    # HF DINO style: encoder.layer.{i}.* or encoder.encoder.layer.{i}.*
    m = re.search(r"(?:^|\.)encoder\.layer\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        return f"block{idx:02d}", idx + 1
    m = re.search(r"(?:^|\.)encoder\.encoder\.layer\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        return f"block{idx:02d}", idx + 1

    if (
        "patch_embed" in name
        or "patch_embeddings" in name
        or "embeddings.patch_embeddings" in name
    ):
        return "patch_embed", 0

    if any(k in name for k in ("cls_token", "pos_embed", "position_embeddings", "register_tokens", "embeddings")):
        return "embeddings", 0

    if any(k in name for k in ("fc_norm", ".norm", ".layernorm", "layernorm")):
        return "norm", 999

    return "other", 1000


def collect_layerwise_grad_conflict(
    *,
    loss_u: torch.Tensor,
    loss_g: torch.Tensor,
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    step: int,
    out_dir: str,
    tag: str = "conflict",
    retain_graph: bool = True,
) -> Dict[str, object]:
    """Collect and save step-wise layerwise gradient conflict statistics.

    Important:
    - Uses `torch.autograd.grad`, so it does NOT write to `.grad`.
    - If called before normal `backward()`, keep `retain_graph=True`.
    """
    params = [p for _, p in named_params if p.requires_grad]
    names = [n for n, p in named_params if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable params provided for conflict analysis.")

    gu = torch.autograd.grad(loss_u, params, retain_graph=retain_graph, allow_unused=True)
    gg = torch.autograd.grad(loss_g, params, retain_graph=retain_graph, allow_unused=True)

    by_group_u: Dict[str, List[torch.Tensor]] = {}
    by_group_g: Dict[str, List[torch.Tensor]] = {}
    group_depth: Dict[str, int] = {}
    for name, p, g_u, g_g in zip(names, params, gu, gg):
        group, depth = _group_vit_param_name(name)
        if group not in by_group_u:
            by_group_u[group] = []
            by_group_g[group] = []
            group_depth[group] = depth
        else:
            group_depth[group] = min(group_depth[group], depth)
        by_group_u[group].append(_to_flat_or_zeros(g_u, p))
        by_group_g[group].append(_to_flat_or_zeros(g_g, p))

    rows = []
    for group, depth in sorted(group_depth.items(), key=lambda x: (x[1], x[0])):
        g_u_flat = torch.cat(by_group_u[group], dim=0)
        g_g_flat = torch.cat(by_group_g[group], dim=0)
        rows.append(
            {
                "step": int(step),
                "layer": group,
                "depth": int(depth),
                "cos": _safe_cos(g_u_flat, g_g_flat),
                "gu_norm": float(torch.norm(g_u_flat).item()),
                "gg_norm": float(torch.norm(g_g_flat).item()),
                "lu": float(loss_u.detach().item()),
                "lg": float(loss_g.detach().item()),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows collected in conflict analysis.")

    ensure_dir(out_dir)
    out = Path(out_dir)
    step_csv = out / f"{tag}_layerwise_grad_summary_step_{int(step)}.csv"
    step_json = out / f"{tag}_conflict_stats_step_{int(step)}.json"
    df.to_csv(step_csv, index=False)

    stats = {
        "step": int(step),
        "num_layers": int(df.shape[0]),
        "global_cos_mean": float(df["cos"].mean()),
        "global_neg_ratio": float((df["cos"] < 0).mean()),
        "shallow_mean_cos": float(df[df["depth"] < df["depth"].median()]["cos"].mean()),
        "deep_mean_cos": float(df[df["depth"] >= df["depth"].median()]["cos"].mean()),
        "shallow_neg_ratio": float((df[df["depth"] < df["depth"].median()]["cos"] < 0).mean()),
        "deep_neg_ratio": float((df[df["depth"] >= df["depth"].median()]["cos"] < 0).mean()),
        "summary_csv": str(step_csv),
    }
    save_json(stats, str(step_json))

    # Rolling logs for later time-series plotting.
    roll_csv = out / f"{tag}_layerwise_grad_summary.csv"
    if roll_csv.exists():
        prev = pd.read_csv(roll_csv)
        pd.concat([prev, df], axis=0, ignore_index=True).to_csv(roll_csv, index=False)
    else:
        df.to_csv(roll_csv, index=False)

    return {
        "step_csv": str(step_csv),
        "step_json": str(step_json),
        "rolling_csv": str(roll_csv),
        "stats": stats,
    }
