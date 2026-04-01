"""Gradient merge utilities.

Baseline CIFAR-10 training currently uses:
- apply_naive
- apply_conflict_aware (PCGrad-style)
- apply_cagrad
- apply_saop
- apply_laga_objective
- apply_ma_laga_objective (DSGA-M + DSGA-D)
- compute_grad_cosine

Other layer-wise helpers below are kept for legacy run compatibility.
"""

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch


def _safe_zeros_like_param(p: torch.nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(p, memory_format=torch.preserve_format)


def _materialize_grads(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[torch.nn.Parameter],
) -> List[torch.Tensor]:
    out = []
    for g, p in zip(grads, params):
        out.append(_safe_zeros_like_param(p) if g is None else g)
    return out


def _flatten(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.zeros(1)
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def _unflatten_like(flat: torch.Tensor, refs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    off = 0
    for r in refs:
        n = int(r.numel())
        out.append(flat[off : off + n].view_as(r))
        off += n
    if off != int(flat.numel()):
        raise ValueError(f"Unflatten mismatch: used={off}, total={int(flat.numel())}.")
    return out


def _weighted(grads: Sequence[torch.Tensor], weight: float) -> List[torch.Tensor]:
    return [weight * g for g in grads]


def _add(g1: Sequence[torch.Tensor], g2: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [a + b for a, b in zip(g1, g2)]


def _group_l2_norm(grads: Sequence[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    if len(grads) == 0:
        return torch.tensor(0.0)
    acc = torch.zeros((), device=grads[0].device, dtype=grads[0].dtype)
    for g in grads:
        acc = acc + torch.sum(g * g)
    return torch.sqrt(acc.clamp_min(eps))


def _safe_cosine_flat(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> float:
    n1 = float(torch.linalg.norm(g1).item())
    n2 = float(torch.linalg.norm(g2).item())
    if n1 <= eps or n2 <= eps:
        return 0.0
    return float(torch.dot(g1, g2).item() / (n1 * n2 + eps))


def _encoder_group_depth_ratio(group_name: str) -> float:
    """Approximate encoder depth for group-wise reconstruction gating."""
    name = str(group_name).strip().lower()
    if name in {"patch_embed", "embeddings", "stem"}:
        return 0.0
    if name in {"norm", "fc_norm", "head"}:
        return 1.0
    if name.startswith("layer"):
        suffix = name[len("layer") :]
        if "." in suffix:
            suffix = suffix.split(".", 1)[0]
        try:
            stage = int(suffix)
        except ValueError:
            stage = -1
        if stage >= 1:
            return max(0.0, min(1.0, float(stage) / 4.0))
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


def _apply_encoder_rec_gate(
    grads_g: Sequence[torch.Tensor],
    layers: Optional[Dict[str, Sequence[int]]],
    encoder_rec_gate_strength: float,
    encoder_rec_gate_min: float,
) -> List[torch.Tensor]:
    strength = float(max(0.0, encoder_rec_gate_strength))
    if strength <= 0.0:
        return list(grads_g)

    gate_min = float(min(1.0, max(0.0, encoder_rec_gate_min)))
    groups = layers if layers else {"global": list(range(len(grads_g)))}
    gated = list(grads_g)
    for group_name, raw_indices in groups.items():
        idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(grads_g)]
        if len(idxs) == 0:
            continue
        depth_ratio = _encoder_group_depth_ratio(group_name)
        rec_gate = max(gate_min, 1.0 - strength * depth_ratio)
        for idx in idxs:
            gated[idx] = rec_gate * gated[idx]
    return gated


def _canonicalize_preserve_target(preserve_target: str) -> str:
    key = str(preserve_target).strip().lower()
    aliases = {
        "u": "understanding",
        "txt": "understanding",
        "text": "understanding",
        "understanding": "understanding",
        "g": "generation",
        "gen": "generation",
        "rec": "generation",
        "recon": "generation",
        "reconstruction": "generation",
        "generation": "generation",
        "sym": "symmetric",
        "symmetric": "symmetric",
        "neutral": "symmetric",
        "none": "symmetric",
        "neither": "symmetric",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported preserve_target={preserve_target}. "
            "Use understanding|generation|symmetric."
        )
    return aliases[key]


def _symmetric_balance_scales(
    norm_u: torch.Tensor,
    norm_g: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma = float(max(0.0, gamma))
    if gamma == 0.0:
        one = torch.ones((), device=norm_u.device, dtype=norm_u.dtype)
        return one, one
    half_gamma = 0.5 * gamma
    scale_u = torch.pow((norm_g / (norm_u + eps)).clamp_min(eps), half_gamma).detach()
    scale_g = torch.pow((norm_u / (norm_g + eps)).clamp_min(eps), half_gamma).detach()
    return scale_u, scale_g


def _symmetric_project_pair(
    grads_u: Sequence[torch.Tensor],
    grads_g: Sequence[torch.Tensor],
    eps: float = 1e-12,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    flat_u = _flatten(grads_u)
    flat_g = _flatten(grads_g)
    dot_ug = torch.dot(flat_u, flat_g)
    norm_uu = torch.dot(flat_u, flat_u).clamp_min(eps)
    norm_gg = torch.dot(flat_g, flat_g).clamp_min(eps)
    proj_u = (dot_ug / norm_gg).detach()
    proj_g = (dot_ug / norm_uu).detach()
    u_proj = [gu - proj_u * gg for gu, gg in zip(grads_u, grads_g)]
    g_proj = [gg - proj_g * gu for gu, gg in zip(grads_u, grads_g)]
    return u_proj, g_proj


def _blend_dsga_with_global_anchor(
    grads_u: Sequence[torch.Tensor],
    grads_g: Sequence[torch.Tensor],
    merged_global: Sequence[torch.Tensor],
    merged_local: Sequence[torch.Tensor],
    groups: Dict[str, Sequence[int]],
    adaptive_blend_strength: float,
    adaptive_blend_power: float,
    eps: float = 1e-8,
) -> List[torch.Tensor]:
    """Blend a global DSGA anchor with selective layerwise deviations.

    The layerwise branch only gets weight when a group is more conflicting than
    the global shared-parameter gradient geometry. This keeps the global method
    as a stable fallback while still letting truly heterogeneous layers deviate.
    """
    merged = [g.clone() for g in merged_global]
    if len(groups) <= 1:
        return merged

    adaptive_blend_strength = float(max(0.0, adaptive_blend_strength))
    adaptive_blend_power = float(max(1e-6, adaptive_blend_power))
    if adaptive_blend_strength <= 0.0:
        return merged

    gu_all = _flatten(grads_u)
    gg_all = _flatten(grads_g)
    global_cos = _safe_cosine_flat(gu_all, gg_all, eps=eps)
    total_energy = float(torch.linalg.norm(gu_all).item() + torch.linalg.norm(gg_all).item())

    covered: Set[int] = set()
    for raw_indices in groups.values():
        idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(grads_u)]
        if len(idxs) == 0:
            continue
        covered.update(idxs)

        gu_group = [grads_u[i] for i in idxs]
        gg_group = [grads_g[i] for i in idxs]
        gu_flat = _flatten(gu_group)
        gg_flat = _flatten(gg_group)
        local_cos = _safe_cosine_flat(gu_flat, gg_flat, eps=eps)

        local_energy = float(torch.linalg.norm(gu_flat).item() + torch.linalg.norm(gg_flat).item())
        energy_share = local_energy / max(total_energy, eps)
        conflict_gain = max(0.0, 0.5 * (global_cos - local_cos))
        blend_weight = adaptive_blend_strength * conflict_gain * (energy_share ** adaptive_blend_power)
        blend_weight = float(min(max(blend_weight, 0.0), 1.0))

        for idx in idxs:
            merged[idx] = (1.0 - blend_weight) * merged_global[idx] + blend_weight * merged_local[idx]

    for idx in range(len(grads_u)):
        if idx not in covered:
            merged[idx] = merged_global[idx]

    return merged


def _normalize_two_task_grads(
    g1: Sequence[torch.Tensor],
    g2: Sequence[torch.Tensor],
    mode: str = "none",
    indices: Optional[Sequence[int]] = None,
    conflict_only: bool = False,
    eps: float = 1e-12,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    mode = str(mode).lower()
    if mode in {"none", "off", "false", "0", ""}:
        return list(g1), list(g2)

    if indices is None:
        sel = list(range(len(g1)))
    else:
        sel = sorted({int(i) for i in indices if 0 <= int(i) < len(g1)})

    if len(sel) == 0:
        return list(g1), list(g2)

    g1_sel = [g1[i] for i in sel]
    g2_sel = [g2[i] for i in sel]

    if conflict_only:
        dot = torch.zeros((), device=g1_sel[0].device, dtype=g1_sel[0].dtype)
        for a, b in zip(g1_sel, g2_sel):
            dot = dot + torch.sum(a * b)
        if float(dot.item()) >= 0.0:
            return list(g1), list(g2)

    n1 = _group_l2_norm(g1_sel, eps=eps)
    n2 = _group_l2_norm(g2_sel, eps=eps)

    if mode in {"mean", "avg", "balance"}:
        target = 0.5 * (n1 + n2)
    elif mode in {"geom", "geometric", "geom_mean"}:
        target = torch.sqrt((n1 * n2).clamp_min(eps))
    elif mode in {"unit", "l2"}:
        target = torch.ones((), device=n1.device, dtype=n1.dtype)
    else:
        raise ValueError(
            f"Unsupported grad_norm_mode={mode}. "
            "Use one of: none|mean|geom|unit."
        )

    s1 = (target / n1.clamp_min(eps)).detach()
    s2 = (target / n2.clamp_min(eps)).detach()

    out1 = list(g1)
    out2 = list(g2)
    for i in sel:
        out1[i] = s1 * out1[i]
        out2[i] = s2 * out2[i]
    return out1, out2


def grad_cosine(g1: Sequence[torch.Tensor], g2: Sequence[torch.Tensor]) -> float:
    v1 = _flatten(g1)
    v2 = _flatten(g2)
    denom = (v1.norm() * v2.norm()).clamp_min(1e-12)
    return float((v1 @ v2 / denom).item())


def project_conflicting(g1: Sequence[torch.Tensor], g2: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    v1 = _flatten(g1)
    v2 = _flatten(g2)

    dot12 = torch.dot(v1, v2)
    n1 = torch.dot(v1, v1).clamp_min(1e-12)
    n2 = torch.dot(v2, v2).clamp_min(1e-12)

    if dot12 >= 0:
        return list(g1), list(g2)

    # Symmetric PCGrad-style projection.
    ratio12 = (dot12 / n2).detach()
    ratio21 = (dot12 / n1).detach()

    g1_proj = [a - ratio12 * b for a, b in zip(g1, g2)]
    g2_proj = [b - ratio21 * a for a, b in zip(g1, g2)]
    return g1_proj, g2_proj


def _mgda_two_task_merge(g1: Sequence[torch.Tensor], g2: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    # Minimize || alpha * g1 + (1-alpha) * g2 ||^2 over alpha in [0, 1].
    v1 = _flatten(g1)
    v2 = _flatten(g2)
    diff = v1 - v2
    denom = torch.dot(diff, diff)

    if float(denom.item()) <= 1e-12:
        alpha = 0.5
    else:
        alpha = float(torch.dot(v2, v2 - v1).item() / denom.item())
        alpha = max(0.0, min(1.0, alpha))

    return [alpha * a + (1.0 - alpha) * b for a, b in zip(g1, g2)]


def _cagrad_like_merge(g1: Sequence[torch.Tensor], g2: Sequence[torch.Tensor], beta: float) -> List[torch.Tensor]:
    # Interpolate between average gradient and MGDA direction.
    # beta in [0, 1]: 0 -> average, 1 -> MGDA.
    beta = max(0.0, min(1.0, float(beta)))
    g_avg = [0.5 * (a + b) for a, b in zip(g1, g2)]
    g_mgda = _mgda_two_task_merge(g1, g2)
    return [(1.0 - beta) * a + beta * b for a, b in zip(g_avg, g_mgda)]


def apply_saop_layerwise(
    g_u: torch.Tensor,
    g_g: torch.Tensor,
    is_deep_layer: bool,
    eps: float = 1e-8,
    log_norm_ratio: bool = False,
) -> torch.Tensor:
    """Strict Asymmetric Orthogonal Projection (SAOP) on flattened vectors.

    If no conflict (cos >= 0) or not target deep layer, return direct sum.
    If conflict (cos < 0) on deep layer:
      1) scale generation gradient to understanding norm,
      2) remove component along understanding direction (Gram-Schmidt),
      3) compose with full understanding gradient.
    """
    if g_u.ndim != 1 or g_g.ndim != 1:
        raise ValueError("apply_saop_layerwise expects flattened 1D tensors.")
    if g_u.numel() != g_g.numel():
        raise ValueError(
            f"Shape mismatch in SAOP: g_u={tuple(g_u.shape)} g_g={tuple(g_g.shape)}."
        )
    if not is_deep_layer:
        return g_u + g_g

    eps = float(max(1e-12, eps))
    dot_ug = torch.dot(g_u, g_g)
    norm_u = torch.linalg.norm(g_u)
    norm_g = torch.linalg.norm(g_g)
    cos = dot_ug / (norm_u * norm_g + eps)

    if float(cos.item()) >= 0.0:
        return g_u + g_g

    # Step A: align generation norm to understanding norm.
    scale = norm_u / (norm_g + eps)
    g_g_scaled = g_g * scale

    # Step B: remove projection on understanding direction.
    proj_coeff = torch.dot(g_g_scaled, g_u) / (torch.dot(g_u, g_u) + eps)
    g_g_orthogonal = g_g_scaled - proj_coeff * g_u

    if log_norm_ratio:
        before = float(norm_g.item())
        after = float(torch.linalg.norm(g_g_orthogonal).item())
        ratio = after / (before + eps)
        print(
            f"[saop] conflict cos={float(cos.item()):.4f}, "
            f"|g_g|={before:.6f}, |g_g_orth|={after:.6f}, ratio={ratio:.4f}"
        )

    # Step C: understanding gradient is preserved, generation contributes orthogonal component.
    return g_u + g_g_orthogonal


def apply_laga(
    grads_u: Sequence[torch.Tensor],
    grads_g: Sequence[torch.Tensor],
    layers: Optional[Dict[str, Sequence[int]]],
    preserve_target: str = "understanding",
    eps: float = 1e-8,
    conflict_threshold: float = 0.0,
    restore_ratio: float = 0.0,
    alpha_mode: str = "fixed",
    alpha_power: float = 1.0,
    alpha_min: float = 1.0,
    alpha_max: float = 1.0,
) -> List[torch.Tensor]:
    """Layer-wise Asymmetric Gradient Arbitration (LAGA).

    Args:
        grads_u: Understanding gradients (same layout as params).
        grads_g: Generation gradients (same layout as params).
        layers: Group name -> parameter index list. If empty/None, treat all params as one group.
        eps: Numerical stability epsilon.

    Returns:
        Merged gradients with the same structure as input gradients.
    """
    if len(grads_u) != len(grads_g):
        raise ValueError(
            f"LAGA gradient length mismatch: len(grads_u)={len(grads_u)} len(grads_g)={len(grads_g)}"
        )
    if len(grads_u) == 0:
        return []
    preserve_target = _canonicalize_preserve_target(preserve_target)
    if preserve_target == "generation":
        return apply_laga(
            grads_u=grads_g,
            grads_g=grads_u,
            layers=layers,
            preserve_target="understanding",
            eps=eps,
            conflict_threshold=conflict_threshold,
            restore_ratio=restore_ratio,
            alpha_mode=alpha_mode,
            alpha_power=alpha_power,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
    if preserve_target == "symmetric":
        eps = float(max(1e-12, eps))
        conflict_threshold = float(conflict_threshold)
        merged = [torch.zeros_like(g) for g in grads_u]
        if not layers:
            groups = {"all": list(range(len(grads_u)))}
        else:
            groups = layers

        covered: Set[int] = set()
        for _, raw_indices in groups.items():
            idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(grads_u)]
            if len(idxs) == 0:
                continue
            covered.update(idxs)

            gu_group = [grads_u[i] for i in idxs]
            gg_group = [grads_g[i] for i in idxs]
            gu_flat = _flatten(gu_group)
            gg_flat = _flatten(gg_group)
            dot_ug = torch.dot(gu_flat, gg_flat)
            norm_u = torch.linalg.norm(gu_flat)
            norm_g = torch.linalg.norm(gg_flat)
            cos = dot_ug / (norm_u * norm_g + eps)
            use_projection = (float(cos.item()) < conflict_threshold) and (float(dot_ug.item()) < 0.0)
            if use_projection:
                gu_group, gg_group = _symmetric_project_pair(gu_group, gg_group, eps=eps)
            merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
            for idx, g in zip(idxs, merged_group):
                merged[idx] = g

        for idx in range(len(grads_u)):
            if idx not in covered:
                merged[idx] = grads_u[idx] + grads_g[idx]
        return merged

    eps = float(max(1e-12, eps))
    conflict_threshold = float(conflict_threshold)
    restore_ratio = float(max(0.0, restore_ratio))
    alpha_mode = str(alpha_mode).lower()
    alpha_power = float(max(alpha_power, 1e-6))
    alpha_min = float(max(0.0, alpha_min))
    alpha_max = float(max(alpha_min, alpha_max))
    merged = [torch.zeros_like(g) for g in grads_u]

    if not layers:
        groups: Dict[str, Sequence[int]] = {"all": list(range(len(grads_u)))}
    else:
        groups = layers

    covered: Set[int] = set()
    for _, raw_indices in groups.items():
        idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(grads_u)]
        if len(idxs) == 0:
            continue
        covered.update(idxs)

        gu_group = [grads_u[i] for i in idxs]
        gg_group = [grads_g[i] for i in idxs]
        gu_flat = _flatten(gu_group)
        gg_flat = _flatten(gg_group)

        dot_ug = torch.dot(gu_flat, gg_flat)
        norm_u = torch.linalg.norm(gu_flat)
        norm_g = torch.linalg.norm(gg_flat)
        cos = dot_ug / (norm_u * norm_g + eps)

        # Only perform asymmetric projection under strong enough conflict.
        use_projection = (float(cos.item()) < conflict_threshold) and (float(dot_ug.item()) < 0.0)
        if not use_projection:
            merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
        else:
            proj_coeff = dot_ug / (torch.dot(gu_flat, gu_flat) + eps)
            if alpha_mode in {"fixed", "full"}:
                alpha = torch.tensor(alpha_max, device=gu_flat.device, dtype=gu_flat.dtype)
            elif alpha_mode in {"ratio", "adaptive_ratio"}:
                ratio = norm_g / (norm_u + eps)
                alpha = torch.pow(ratio.clamp_min(eps), alpha_power)
                alpha = alpha.clamp(min=alpha_min, max=alpha_max).detach()
            else:
                raise ValueError(
                    f"Unsupported LAGA alpha_mode={alpha_mode}. "
                    "Use one of: fixed|ratio."
                )

            gg_proj_group = [gg - alpha * proj_coeff * gu for gu, gg in zip(gu_group, gg_group)]
            if restore_ratio > 0.0:
                gg_proj_flat = _flatten(gg_proj_group)
                norm_proj = torch.linalg.norm(gg_proj_flat)
                restore_scale = norm_g / (norm_proj + eps)
                restore_mix = (1.0 - restore_ratio) + restore_ratio * restore_scale
                gg_proj_group = [restore_mix * g for g in gg_proj_group]
            merged_group = [gu + gg_proj for gu, gg_proj in zip(gu_group, gg_proj_group)]

        for idx, g in zip(idxs, merged_group):
            merged[idx] = g

    # Safety fallback for params not covered by groups.
    for idx in range(len(grads_u)):
        if idx not in covered:
            merged[idx] = grads_u[idx] + grads_g[idx]

    return merged


def apply_ma_laga(
    grads_u: Sequence[torch.Tensor],
    grads_g: Sequence[torch.Tensor],
    layers: Optional[Dict[str, Sequence[int]]],
    preserve_target: str = "understanding",
    align_gamma: float = 0.5,
    align_max_scale: float = 0.0,
    norm_restore: bool = True,
    mode: str = "full",
    conflict_threshold: float = 0.0,
    conflict_only: bool = False,
    eps: float = 1e-8,
    magnitude_scope: str = "layerwise",
    adaptive_layerwise_blend: bool = False,
    adaptive_blend_strength: float = 1.0,
    adaptive_blend_power: float = 0.5,
    encoder_rec_gate_strength: float = 0.0,
    encoder_rec_gate_min: float = 1.0,
) -> List[torch.Tensor]:
    """DSGA-M + DSGA-D merge core (legacy name: MA-LAGA).

    Supported modes:
      - full: DSGA-M + DSGA-D (align then conflict projection).
      - capped_full: cap alignment scale then apply LAGA projection on conflicts.
      - direction_only: pure LAGA without magnitude alignment.
      - magnitude_only: pure MA without projection (always sum aligned gradients).
      - nr_laga: project generation gradient on conflicts, then restore to its own norm.

    Args:
        magnitude_scope:
            - "layerwise": DSGA-M scale is computed per direction-routing group.
            - "global": DSGA-M scale is computed once from all shared params, while
              DSGA-D direction routing still follows ``layers`` grouping.
    """
    if len(grads_u) != len(grads_g):
        raise ValueError(
            f"DSGA-M/DSGA-D gradient length mismatch: len(grads_u)={len(grads_u)} len(grads_g)={len(grads_g)}"
        )
    if len(grads_u) == 0:
        return []
    preserve_target = _canonicalize_preserve_target(preserve_target)
    if preserve_target == "generation":
        return apply_ma_laga(
            grads_u=grads_g,
            grads_g=grads_u,
            layers=layers,
            preserve_target="understanding",
            align_gamma=align_gamma,
            align_max_scale=align_max_scale,
            norm_restore=norm_restore,
            mode=mode,
            conflict_threshold=conflict_threshold,
            conflict_only=conflict_only,
            eps=eps,
            magnitude_scope=magnitude_scope,
            adaptive_layerwise_blend=adaptive_layerwise_blend,
            adaptive_blend_strength=adaptive_blend_strength,
            adaptive_blend_power=adaptive_blend_power,
            encoder_rec_gate_strength=0.0,
            encoder_rec_gate_min=1.0,
        )

    eps = float(max(1e-12, eps))
    align_gamma = float(max(0.0, align_gamma))
    align_max_scale = float(align_max_scale)
    norm_restore = bool(norm_restore)
    conflict_threshold = float(conflict_threshold)
    if not (-1.0 <= conflict_threshold <= 1.0):
        raise ValueError(
            f"DSGA-D conflict_threshold must be in [-1, 1], got {conflict_threshold}."
        )
    conflict_only = bool(conflict_only)
    mode = str(mode).lower()
    if mode not in {"full", "capped_full", "direction_only", "magnitude_only", "nr_laga"}:
        raise ValueError(
            f"Unsupported DSGA-D mode={mode}. "
            "Use one of: full|capped_full|direction_only|magnitude_only|nr_laga."
        )
    if mode == "capped_full" and align_max_scale <= 0.0:
        raise ValueError(
            f"DSGA-M capped_full requires align_max_scale>0, got {align_max_scale}."
        )
    magnitude_scope = str(magnitude_scope).lower()
    if magnitude_scope not in {"layerwise", "global"}:
        raise ValueError(
            f"Unsupported DSGA-M magnitude_scope={magnitude_scope}. "
            "Use one of: layerwise|global."
        )
    merged = [torch.zeros_like(g) for g in grads_u]

    if not layers:
        groups: Dict[str, Sequence[int]] = {"all": list(range(len(grads_u)))}
    else:
        groups = layers
    grads_g_effective = _apply_encoder_rec_gate(
        grads_g=grads_g,
        layers=groups,
        encoder_rec_gate_strength=encoder_rec_gate_strength,
        encoder_rec_gate_min=encoder_rec_gate_min,
    )

    adaptive_layerwise_blend = bool(adaptive_layerwise_blend)
    adaptive_blend_strength = float(max(0.0, adaptive_blend_strength))
    adaptive_blend_power = float(max(1e-6, adaptive_blend_power))
    if adaptive_layerwise_blend and len(groups) > 1:
        merged_global = apply_ma_laga(
            grads_u=grads_u,
            grads_g=grads_g_effective,
            layers={"all": list(range(len(grads_u)))},
            preserve_target=preserve_target,
            align_gamma=align_gamma,
            align_max_scale=align_max_scale,
            norm_restore=norm_restore,
            mode=mode,
            conflict_threshold=conflict_threshold,
            conflict_only=conflict_only,
            eps=eps,
            magnitude_scope="global",
            adaptive_layerwise_blend=False,
            adaptive_blend_strength=adaptive_blend_strength,
            adaptive_blend_power=adaptive_blend_power,
            encoder_rec_gate_strength=0.0,
            encoder_rec_gate_min=1.0,
        )
        merged_local = apply_ma_laga(
            grads_u=grads_u,
            grads_g=grads_g_effective,
            layers=groups,
            preserve_target=preserve_target,
            align_gamma=align_gamma,
            align_max_scale=align_max_scale,
            norm_restore=norm_restore,
            mode=mode,
            conflict_threshold=conflict_threshold,
            conflict_only=conflict_only,
            eps=eps,
            magnitude_scope=magnitude_scope,
            adaptive_layerwise_blend=False,
            adaptive_blend_strength=adaptive_blend_strength,
            adaptive_blend_power=adaptive_blend_power,
            encoder_rec_gate_strength=0.0,
            encoder_rec_gate_min=1.0,
        )
        return _blend_dsga_with_global_anchor(
            grads_u=grads_u,
            grads_g=grads_g_effective,
            merged_global=merged_global,
            merged_local=merged_local,
            groups=groups,
            adaptive_blend_strength=adaptive_blend_strength,
            adaptive_blend_power=adaptive_blend_power,
            eps=eps,
        )

    global_scale_factor: Optional[torch.Tensor] = None
    global_scale_u: Optional[torch.Tensor] = None
    global_scale_g: Optional[torch.Tensor] = None
    if magnitude_scope == "global" and mode not in {"direction_only", "nr_laga"}:
        gu_all = _flatten(grads_u)
        gg_all = _flatten(grads_g_effective)
        m_u_all = torch.linalg.norm(gu_all)
        m_g_all = torch.linalg.norm(gg_all)
        if preserve_target == "symmetric":
            global_scale_u, global_scale_g = _symmetric_balance_scales(m_u_all, m_g_all, align_gamma, eps=eps)
        elif mode == "capped_full":
            global_scale_factor = torch.clamp(m_u_all / (m_g_all + eps), max=align_max_scale).detach()
        else:
            global_scale_factor = torch.pow((m_u_all / (m_g_all + eps)).clamp_min(eps), align_gamma).detach()

    covered: Set[int] = set()
    for _, raw_indices in groups.items():
        idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(grads_u)]
        if len(idxs) == 0:
            continue
        covered.update(idxs)

        gu_group = [grads_u[i] for i in idxs]
        gg_group = [grads_g_effective[i] for i in idxs]
        gu_flat = _flatten(gu_group)
        gg_flat = _flatten(gg_group)

        m_u = torch.linalg.norm(gu_flat)
        m_g = torch.linalg.norm(gg_flat)
        dot_ug = torch.dot(gu_flat, gg_flat)
        cos = dot_ug / (m_u * m_g + eps)
        cos_v = float(cos.item())
        is_conflict = cos_v < conflict_threshold

        if preserve_target == "symmetric" and mode == "direction_only":
            if not is_conflict:
                merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
            else:
                gu_proj_group, gg_proj_group = _symmetric_project_pair(gu_group, gg_group, eps=eps)
                merged_group = [gu + gg for gu, gg in zip(gu_proj_group, gg_proj_group)]
        elif preserve_target == "symmetric" and mode == "nr_laga":
            if not is_conflict:
                merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
            else:
                gu_proj_group, gg_proj_group = _symmetric_project_pair(gu_group, gg_group, eps=eps)
                norm_u_proj = torch.linalg.norm(_flatten(gu_proj_group))
                norm_g_proj = torch.linalg.norm(_flatten(gg_proj_group))
                restore_u = m_u / (norm_u_proj + eps)
                restore_g = m_g / (norm_g_proj + eps)
                gu_proj_group = [restore_u * gu for gu in gu_proj_group]
                gg_proj_group = [restore_g * gg for gg in gg_proj_group]
                merged_group = [gu + gg for gu, gg in zip(gu_proj_group, gg_proj_group)]
        elif mode == "direction_only":
            # Pure LAGA: no magnitude alignment.
            if not is_conflict:
                merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
            else:
                proj_coeff = dot_ug / (torch.dot(gu_flat, gu_flat) + eps)
                gg_perp_group = [gg - proj_coeff * gu for gu, gg in zip(gu_group, gg_group)]
                merged_group = [gu + gg_perp for gu, gg_perp in zip(gu_group, gg_perp_group)]
        elif mode == "nr_laga":
            # Norm-Restored LAGA: no alignment to understanding norm.
            if not is_conflict:
                merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
            else:
                proj_coeff = dot_ug / (torch.dot(gu_flat, gu_flat) + eps)
                gg_perp_group = [gg - proj_coeff * gu for gu, gg in zip(gu_group, gg_group)]
                norm_perp = torch.linalg.norm(_flatten(gg_perp_group))
                restore_scale = m_g / (norm_perp + eps)
                gg_perp_group = [restore_scale * gg_perp for gg_perp in gg_perp_group]
                merged_group = [gu + gg_perp for gu, gg_perp in zip(gu_group, gg_perp_group)]
        else:
            # Magnitude alignment stage.
            if preserve_target == "symmetric":
                if magnitude_scope == "global":
                    if global_scale_u is None or global_scale_g is None:
                        raise RuntimeError("global symmetric DSGA-M scales are unexpectedly None.")
                    scale_u = global_scale_u
                    scale_g = global_scale_g
                else:
                    scale_u, scale_g = _symmetric_balance_scales(m_u, m_g, align_gamma, eps=eps)
                gu_aligned_group = [scale_u * gu for gu in gu_group]
                gg_aligned_group = [scale_g * gg for gg in gg_group]
                if mode == "magnitude_only":
                    if conflict_only and (not is_conflict):
                        merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
                    else:
                        merged_group = [gu + gg for gu, gg in zip(gu_aligned_group, gg_aligned_group)]
                else:
                    if not is_conflict:
                        if conflict_only:
                            merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
                        else:
                            merged_group = [gu + gg for gu, gg in zip(gu_aligned_group, gg_aligned_group)]
                    else:
                        gu_proj_group, gg_proj_group = _symmetric_project_pair(gu_aligned_group, gg_aligned_group, eps=eps)
                        if norm_restore:
                            norm_u_aligned = torch.linalg.norm(_flatten(gu_aligned_group))
                            norm_g_aligned = torch.linalg.norm(_flatten(gg_aligned_group))
                            norm_u_proj = torch.linalg.norm(_flatten(gu_proj_group))
                            norm_g_proj = torch.linalg.norm(_flatten(gg_proj_group))
                            restore_u = norm_u_aligned / (norm_u_proj + eps)
                            restore_g = norm_g_aligned / (norm_g_proj + eps)
                            gu_proj_group = [restore_u * gu for gu in gu_proj_group]
                            gg_proj_group = [restore_g * gg for gg in gg_proj_group]
                        merged_group = [gu + gg for gu, gg in zip(gu_proj_group, gg_proj_group)]
            else:
                if magnitude_scope == "global":
                    if global_scale_factor is None:
                        raise RuntimeError("global_scale_factor is unexpectedly None for DSGA-M global mode.")
                    scale_factor = global_scale_factor
                elif mode == "capped_full":
                    scale_factor = torch.clamp(m_u / (m_g + eps), max=align_max_scale).detach()
                else:
                    scale_factor = torch.pow((m_u / (m_g + eps)).clamp_min(eps), align_gamma).detach()
                gg_aligned_group = [scale_factor * gg for gg in gg_group]

                if mode == "magnitude_only":
                    # Pure MA: never project, always add aligned generation gradient.
                    if conflict_only and (not is_conflict):
                        merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
                    else:
                        merged_group = [gu + gg_aligned for gu, gg_aligned in zip(gu_group, gg_aligned_group)]
                else:
                    # Full DSGA-M + DSGA-D.
                    if not is_conflict:
                        if conflict_only:
                            # Option-1: only decompose conflict groups; non-conflict groups use plain sum.
                            merged_group = [gu + gg for gu, gg in zip(gu_group, gg_group)]
                        else:
                            merged_group = [gu + gg_aligned for gu, gg_aligned in zip(gu_group, gg_aligned_group)]
                    else:
                        gg_aligned_flat = _flatten(gg_aligned_group)
                        proj_coeff = torch.dot(gg_aligned_flat, gu_flat) / (torch.dot(gu_flat, gu_flat) + eps)
                        gg_perp_group = [gg_aligned - proj_coeff * gu for gu, gg_aligned in zip(gu_group, gg_aligned_group)]
                        if norm_restore:
                            norm_aligned = torch.linalg.norm(gg_aligned_flat)
                            norm_perp = torch.linalg.norm(_flatten(gg_perp_group))
                            restore_scale = norm_aligned / (norm_perp + eps)
                            gg_perp_group = [restore_scale * gg_perp for gg_perp in gg_perp_group]
                        merged_group = [gu + gg_perp for gu, gg_perp in zip(gu_group, gg_perp_group)]

        for idx, g in zip(idxs, merged_group):
            merged[idx] = g

    for idx in range(len(grads_u)):
        if idx not in covered:
            merged[idx] = grads_u[idx] + grads_g_effective[idx]

    return merged


def set_grads(params: Sequence[torch.nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g


def _apply_multi_objective(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
    merge_fn: Callable[[List[torch.Tensor], List[torch.Tensor], float], List[torch.Tensor]],
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    total = lambda_txt * loss_txt + lambda_rec * loss_rec
    if extra_loss is not None:
        total = total + extra_loss

    if len(shared_params) == 0:
        if len(aux_params) > 0:
            aux_grads = torch.autograd.grad(total, aux_params, allow_unused=True)
            aux_grads = _materialize_grads(aux_grads, aux_params)
            set_grads(aux_params, aux_grads)
        return 0.0

    raw_txt = torch.autograd.grad(loss_txt, shared_params, retain_graph=True, allow_unused=True)
    raw_rec = torch.autograd.grad(loss_rec, shared_params, retain_graph=True, allow_unused=True)

    g_txt = _materialize_grads(raw_txt, shared_params)
    g_rec = _materialize_grads(raw_rec, shared_params)
    cos = grad_cosine(g_txt, g_rec)

    wg_txt = _weighted(g_txt, lambda_txt)
    wg_rec = _weighted(g_rec, lambda_rec)
    wg_txt, wg_rec = _normalize_two_task_grads(
        wg_txt,
        wg_rec,
        mode=grad_norm_mode,
        indices=grad_norm_indices,
        conflict_only=grad_norm_conflict_only,
    )
    merged_shared = merge_fn(wg_txt, wg_rec, cos)
    if extra_loss is not None:
        g_extra = torch.autograd.grad(extra_loss, shared_params, retain_graph=True, allow_unused=True)
        g_extra = _materialize_grads(g_extra, shared_params)
        merged_shared = _add(merged_shared, g_extra)

    if len(aux_params) > 0:
        aux_grads = torch.autograd.grad(total, aux_params, allow_unused=True)
        aux_grads = _materialize_grads(aux_grads, aux_params)
        set_grads(aux_params, aux_grads)

    set_grads(shared_params, merged_shared)
    return cos


def apply_conflict_aware(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    lora_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    # PCGrad-style symmetric projection on conflicts.
    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], cos: float) -> List[torch.Tensor]:
        if cos < 0:
            wg_txt, wg_rec = project_conflicting(wg_txt, wg_rec)
        return _add(wg_txt, wg_rec)

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=lora_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def apply_naive(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], _: float) -> List[torch.Tensor]:
        return _add(wg_txt, wg_rec)

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def apply_mgda_ub(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], _: float) -> List[torch.Tensor]:
        return _mgda_two_task_merge(wg_txt, wg_rec)

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def apply_cagrad(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
    beta: float = 0.5,
    conflict_only: bool = False,
    conflict_threshold: float = 0.0,
    nonconflict_merge: str = "cagrad",
    adaptive_beta: bool = False,
    adaptive_group_to_indices: Optional[Dict[str, Sequence[int]]] = None,
    adaptive_nonconflict_merge: str = "sum",
    adaptive_conflict_threshold: float = 0.0,
    adaptive_strength: float = 1.0,
    adaptive_power: float = 1.0,
    adaptive_beta_cap: float = 1.0,
    adaptive_online_beta: bool = False,
    adaptive_online_lr: float = 0.1,
    adaptive_state: Optional[Dict[str, float]] = None,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    nonconflict_merge = str(nonconflict_merge).lower()
    if nonconflict_merge not in {"cagrad", "sum", "avg", "average"}:
        raise ValueError(
            f"Unsupported nonconflict_merge={nonconflict_merge}. "
            "Use one of: cagrad|sum|avg."
        )

    adaptive_nonconflict_merge = str(adaptive_nonconflict_merge).lower()
    if adaptive_nonconflict_merge not in {"cagrad", "sum", "avg", "average"}:
        raise ValueError(
            f"Unsupported adaptive_nonconflict_merge={adaptive_nonconflict_merge}. "
            "Use one of: cagrad|sum|avg."
        )
    adaptive_power = max(1e-6, float(adaptive_power))
    adaptive_strength = max(0.0, float(adaptive_strength))
    adaptive_beta_cap = max(0.0, min(1.0, float(adaptive_beta_cap)))
    adaptive_conflict_threshold = float(adaptive_conflict_threshold)
    adaptive_online_lr = max(0.0, min(1.0, float(adaptive_online_lr)))

    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], cos: float) -> List[torch.Tensor]:
        if adaptive_beta and adaptive_group_to_indices:
            merged = [torch.zeros_like(g) for g in wg_txt]
            covered: Set[int] = set()

            for group_name, raw_indices in adaptive_group_to_indices.items():
                idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(wg_txt)]
                if len(idxs) == 0:
                    continue
                covered.update(idxs)

                g_txt_k = [wg_txt[i] for i in idxs]
                g_rec_k = [wg_rec[i] for i in idxs]
                cos_k = grad_cosine(g_txt_k, g_rec_k)

                if cos_k < adaptive_conflict_threshold:
                    # Conflict severity in [0, 1]:
                    # severity=0 at threshold, severity=1 at cos=-1.
                    denom = max(1e-6, 1.0 + adaptive_conflict_threshold)
                    severity = (adaptive_conflict_threshold - cos_k) / denom
                    severity = max(0.0, min(1.0, severity))
                    severity = severity**adaptive_power
                    beta_obs = float(beta) * (1.0 + adaptive_strength * severity)
                    beta_obs = max(0.0, min(adaptive_beta_cap, beta_obs))
                    if adaptive_online_beta and adaptive_state is not None:
                        beta_prev = float(adaptive_state.get(group_name, float(beta)))
                        beta_k = (1.0 - adaptive_online_lr) * beta_prev + adaptive_online_lr * beta_obs
                        beta_k = max(0.0, min(adaptive_beta_cap, beta_k))
                        adaptive_state[group_name] = beta_k
                    else:
                        beta_k = beta_obs
                    merged_k = _cagrad_like_merge(g_txt_k, g_rec_k, beta=beta_k)
                else:
                    if adaptive_online_beta and adaptive_state is not None:
                        beta_prev = float(adaptive_state.get(group_name, float(beta)))
                        beta_obs = max(0.0, min(adaptive_beta_cap, float(beta)))
                        beta_k = (1.0 - adaptive_online_lr) * beta_prev + adaptive_online_lr * beta_obs
                        beta_k = max(0.0, min(adaptive_beta_cap, beta_k))
                        adaptive_state[group_name] = beta_k
                    else:
                        beta_k = float(beta)
                    if adaptive_nonconflict_merge in {"sum"}:
                        merged_k = _add(g_txt_k, g_rec_k)
                    elif adaptive_nonconflict_merge in {"avg", "average"}:
                        merged_k = [0.5 * (a + b) for a, b in zip(g_txt_k, g_rec_k)]
                    else:
                        merged_k = _cagrad_like_merge(g_txt_k, g_rec_k, beta=beta_k)

                for idx, g in zip(idxs, merged_k):
                    merged[idx] = g

            # Fallback for uncovered params.
            for idx in range(len(wg_txt)):
                if idx in covered:
                    continue
                merged[idx] = _cagrad_like_merge([wg_txt[idx]], [wg_rec[idx]], beta=beta)[0]
            return merged

        if conflict_only and cos >= float(conflict_threshold):
            if nonconflict_merge in {"sum"}:
                return _add(wg_txt, wg_rec)
            if nonconflict_merge in {"avg", "average"}:
                return [0.5 * (a + b) for a, b in zip(wg_txt, wg_rec)]
        return _cagrad_like_merge(wg_txt, wg_rec, beta=beta)

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def apply_la_cagrad(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    group_to_indices: Optional[Dict[str, Sequence[int]]],
    lambda_txt: float,
    lambda_rec: float,
    beta: float = 0.5,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    """Layer-wise CAGrad (LA-CAGrad).

    Apply CAGrad merge independently on each parameter group, then write merged
    gradients back to shared params. Auxiliary params use weighted total loss
    gradients, matching global CAGrad fairness protocol.
    """
    beta = max(0.0, min(1.0, float(beta)))

    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], _: float) -> List[torch.Tensor]:
        if not group_to_indices:
            return _cagrad_like_merge(wg_txt, wg_rec, beta=beta)

        merged = [torch.zeros_like(g) for g in wg_txt]
        covered: Set[int] = set()

        for _, raw_indices in group_to_indices.items():
            idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(wg_txt)]
            if len(idxs) == 0:
                continue
            covered.update(idxs)
            g_txt_k = [wg_txt[i] for i in idxs]
            g_rec_k = [wg_rec[i] for i in idxs]
            merged_k = _cagrad_like_merge(g_txt_k, g_rec_k, beta=beta)
            for idx, g in zip(idxs, merged_k):
                merged[idx] = g

        # Safety fallback for params not covered by any group.
        for idx in range(len(wg_txt)):
            if idx not in covered:
                merged[idx] = _cagrad_like_merge([wg_txt[idx]], [wg_rec[idx]], beta=beta)[0]
        return merged

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def apply_gma_laga(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    group_to_indices: Optional[Dict[str, Sequence[int]]],
    lambda_txt: float,
    lambda_rec: float,
    max_scale: float = 100.0,
    eps: float = 1e-8,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    """Global Magnitude-Aligned LAGA (GMA-LAGA).

    1) Compute one global scale factor S from all encoder grads.
    2) Apply layer-wise asymmetric projection routing with scaled generation grads.
    """
    eps = float(max(1e-12, eps))
    max_scale = float(max_scale)

    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], _: float) -> List[torch.Tensor]:
        if len(wg_txt) == 0:
            return []

        # Global magnitude alignment scale.
        norm_u = _group_l2_norm(wg_txt, eps=eps)
        norm_g = _group_l2_norm(wg_rec, eps=eps)
        scale = (norm_u / (norm_g + eps)).detach()
        if max_scale > 0.0:
            scale = torch.clamp(scale, max=max_scale)

        if not group_to_indices:
            v_u = _flatten(wg_txt)
            v_g = _flatten(wg_rec)
            v_g_scaled = scale * v_g
            cos = float((torch.dot(v_u, v_g) / (torch.linalg.norm(v_u) * torch.linalg.norm(v_g) + eps)).item())
            if cos >= 0.0:
                merged_flat = v_u + v_g_scaled
            else:
                proj_coeff = torch.dot(v_g_scaled, v_u) / torch.dot(v_u, v_u).clamp_min(eps)
                v_g_perp = v_g_scaled - proj_coeff * v_u
                merged_flat = v_u + v_g_perp
            return _unflatten_like(merged_flat, wg_txt)

        merged = [torch.zeros_like(g) for g in wg_txt]
        covered: Set[int] = set()
        for _, raw_indices in group_to_indices.items():
            idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(wg_txt)]
            if len(idxs) == 0:
                continue
            covered.update(idxs)

            g_u_k = [wg_txt[i] for i in idxs]
            g_g_k = [wg_rec[i] for i in idxs]
            v_u_k = _flatten(g_u_k)
            v_g_k = _flatten(g_g_k)
            v_g_k_scaled = scale * v_g_k

            # Cosine with original generation gradient (scale-invariant by design).
            cos = float(
                (
                    torch.dot(v_u_k, v_g_k)
                    / (torch.linalg.norm(v_u_k) * torch.linalg.norm(v_g_k) + eps)
                ).item()
            )
            if cos >= 0.0:
                v_m_k = v_u_k + v_g_k_scaled
            else:
                proj_coeff = torch.dot(v_g_k_scaled, v_u_k) / torch.dot(v_u_k, v_u_k).clamp_min(eps)
                v_g_perp = v_g_k_scaled - proj_coeff * v_u_k
                v_m_k = v_u_k + v_g_perp

            off = 0
            for i in idxs:
                n = int(wg_txt[i].numel())
                merged[i] = v_m_k[off : off + n].view_as(wg_txt[i])
                off += n

        for idx in range(len(wg_txt)):
            if idx not in covered:
                merged[idx] = wg_txt[idx] + scale * wg_rec[idx]
        return merged

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def apply_egd(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    group_to_indices: Optional[Dict[str, Sequence[int]]],
    lambda_txt: float,
    lambda_rec: float,
    eps: float = 1e-8,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    """Element-wise Gradient Disentanglement (EGD).

    For each group k:
      1) normalize g_u,k / ||g_u,k|| and g_g,k / ||g_g,k||
      2) route per element:
         - same sign: g_u + g_g
         - opposite sign: pick winner by larger normalized magnitude
    """
    eps = float(max(1e-12, eps))

    def _merge_group(g_u_k: List[torch.Tensor], g_g_k: List[torch.Tensor]) -> List[torch.Tensor]:
        v_u = _flatten(g_u_k)
        v_g = _flatten(g_g_k)

        n_u = torch.linalg.norm(v_u)
        n_g = torch.linalg.norm(v_g)
        v_u_n = v_u / (n_u + eps)
        v_g_n = v_g / (n_g + eps)

        mask_syn = (v_u * v_g) >= 0
        mask_conf = ~mask_syn
        mask_u_wins = v_u_n.abs() >= v_g_n.abs()

        v_m = torch.zeros_like(v_u)
        v_m[mask_syn] = v_u[mask_syn] + v_g[mask_syn]

        mask_conf_u = mask_conf & mask_u_wins
        mask_conf_g = mask_conf & (~mask_u_wins)
        v_m[mask_conf_u] = v_u[mask_conf_u]
        v_m[mask_conf_g] = v_g[mask_conf_g]
        return _unflatten_like(v_m, g_u_k)

    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], _: float) -> List[torch.Tensor]:
        if len(wg_txt) == 0:
            return []
        if not group_to_indices:
            return _merge_group(wg_txt, wg_rec)

        merged = [torch.zeros_like(g) for g in wg_txt]
        covered: Set[int] = set()
        for _, raw_indices in group_to_indices.items():
            idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(wg_txt)]
            if len(idxs) == 0:
                continue
            covered.update(idxs)
            g_u_k = [wg_txt[i] for i in idxs]
            g_g_k = [wg_rec[i] for i in idxs]
            g_m_k = _merge_group(g_u_k, g_g_k)
            for idx, g in zip(idxs, g_m_k):
                merged[idx] = g

        for idx in range(len(wg_txt)):
            if idx not in covered:
                merged[idx] = wg_txt[idx] + wg_rec[idx]
        return merged

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def apply_saop(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
    deep_group_to_indices: Optional[Dict[str, Sequence[int]]] = None,
    eps: float = 1e-8,
    log_norm_ratio: bool = False,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    eps = float(max(1e-12, eps))

    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], _: float) -> List[torch.Tensor]:
        if not deep_group_to_indices:
            v_u = _flatten(wg_txt)
            v_g = _flatten(wg_rec)
            merged_flat = apply_saop_layerwise(
                g_u=v_u,
                g_g=v_g,
                is_deep_layer=True,
                eps=eps,
                log_norm_ratio=log_norm_ratio,
            )
            return _unflatten_like(merged_flat, wg_txt)

        merged = [torch.zeros_like(g) for g in wg_txt]
        covered: Set[int] = set()

        for _, raw_indices in deep_group_to_indices.items():
            idxs = [int(i) for i in raw_indices if 0 <= int(i) < len(wg_txt)]
            if len(idxs) == 0:
                continue
            covered.update(idxs)
            gu_group = [wg_txt[i] for i in idxs]
            gg_group = [wg_rec[i] for i in idxs]
            merged_flat = apply_saop_layerwise(
                g_u=_flatten(gu_group),
                g_g=_flatten(gg_group),
                is_deep_layer=True,
                eps=eps,
                log_norm_ratio=log_norm_ratio,
            )
            merged_group = _unflatten_like(merged_flat, gu_group)
            for idx, g in zip(idxs, merged_group):
                merged[idx] = g

        # Non-selected (typically shallow) params: direct sum.
        for idx in range(len(wg_txt)):
            if idx in covered:
                continue
            merged[idx] = wg_txt[idx] + wg_rec[idx]
        return merged

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def apply_laga_objective(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
    group_to_indices: Optional[Dict[str, Sequence[int]]] = None,
    preserve_target: str = "understanding",
    eps: float = 1e-8,
    conflict_threshold: float = 0.0,
    restore_ratio: float = 0.0,
    alpha_mode: str = "fixed",
    alpha_power: float = 1.0,
    alpha_min: float = 1.0,
    alpha_max: float = 1.0,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    eps = float(max(1e-12, eps))

    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], _: float) -> List[torch.Tensor]:
        return apply_laga(
            grads_u=wg_txt,
            grads_g=wg_rec,
            layers=group_to_indices,
            preserve_target=preserve_target,
            eps=eps,
            conflict_threshold=conflict_threshold,
            restore_ratio=restore_ratio,
            alpha_mode=alpha_mode,
            alpha_power=alpha_power,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def apply_ma_laga_objective(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
    group_to_indices: Optional[Dict[str, Sequence[int]]] = None,
    preserve_target: str = "understanding",
    align_gamma: float = 0.5,
    norm_restore: bool = True,
    mode: str = "full",
    conflict_threshold: float = 0.0,
    conflict_only: bool = False,
    eps: float = 1e-8,
    magnitude_scope: str = "layerwise",
    adaptive_layerwise_blend: bool = False,
    adaptive_blend_strength: float = 1.0,
    adaptive_blend_power: float = 0.5,
    encoder_rec_gate_strength: float = 0.0,
    encoder_rec_gate_min: float = 1.0,
    grad_norm_mode: str = "none",
    grad_norm_indices: Optional[Sequence[int]] = None,
    grad_norm_conflict_only: bool = False,
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    eps = float(max(1e-12, eps))

    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], _: float) -> List[torch.Tensor]:
        return apply_ma_laga(
            grads_u=wg_txt,
            grads_g=wg_rec,
            layers=group_to_indices,
            preserve_target=preserve_target,
            align_gamma=align_gamma,
            norm_restore=norm_restore,
            mode=mode,
            conflict_threshold=conflict_threshold,
            conflict_only=conflict_only,
            eps=eps,
            magnitude_scope=magnitude_scope,
            adaptive_layerwise_blend=adaptive_layerwise_blend,
            adaptive_blend_strength=adaptive_blend_strength,
            adaptive_blend_power=adaptive_blend_power,
            encoder_rec_gate_strength=encoder_rec_gate_strength,
            encoder_rec_gate_min=encoder_rec_gate_min,
        )

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
        grad_norm_mode=grad_norm_mode,
        grad_norm_indices=grad_norm_indices,
        grad_norm_conflict_only=grad_norm_conflict_only,
        extra_loss=extra_loss,
    )


def compute_grad_cosine(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> float:
    if len(params) == 0:
        return 0.0
    g1 = torch.autograd.grad(loss_txt, params, retain_graph=True, allow_unused=True)
    g2 = torch.autograd.grad(loss_rec, params, retain_graph=True, allow_unused=True)
    g1 = _materialize_grads(g1, params)
    g2 = _materialize_grads(g2, params)
    return grad_cosine(g1, g2)


def apply_layerwise_pcgrad(
    loss_u: torch.Tensor,
    loss_g: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    group_to_indices: Dict[str, List[int]],
    conflict_threshold: float = 0.0,
    eps: float = 1e-12,
) -> Dict[str, object]:
    if len(params) == 0:
        return {
            "global_cos": 0.0,
            "layer_cos_mean": 0.0,
            "layer_cos_neg_ratio": 0.0,
            "layer_cos": {},
        }

    raw_u = torch.autograd.grad(loss_u, params, retain_graph=True, allow_unused=True)
    raw_g = torch.autograd.grad(loss_g, params, retain_graph=True, allow_unused=True)
    g_u = _materialize_grads(raw_u, params)
    g_g = _materialize_grads(raw_g, params)

    merged = [_safe_zeros_like_param(p) for p in params]
    layer_cos: Dict[str, float] = {}

    for group_name, indices in group_to_indices.items():
        if len(indices) == 0:
            layer_cos[group_name] = 0.0
            continue

        dot = torch.zeros((), device=g_u[indices[0]].device, dtype=g_u[indices[0]].dtype)
        nu = torch.zeros_like(dot)
        ng = torch.zeros_like(dot)
        for idx in indices:
            gu = g_u[idx]
            gg = g_g[idx]
            dot = dot + torch.sum(gu * gg)
            nu = nu + torch.sum(gu * gu)
            ng = ng + torch.sum(gg * gg)

        denom = torch.sqrt(nu.clamp_min(eps) * ng.clamp_min(eps))
        cos = float((dot / denom).item()) if float(denom.item()) > 0 else 0.0
        layer_cos[group_name] = cos

        use_projection = (cos < conflict_threshold) and (float(dot.item()) < 0.0)
        if use_projection:
            ratio_u = (dot / ng.clamp_min(eps)).detach()
            ratio_g = (dot / nu.clamp_min(eps)).detach()
            for idx in indices:
                gu_orig = g_u[idx]
                gg_orig = g_g[idx]
                gu_proj = gu_orig - ratio_u * gg_orig
                gg_proj = gg_orig - ratio_g * gu_orig
                merged[idx] = gu_proj + gg_proj
        else:
            for idx in indices:
                merged[idx] = g_u[idx] + g_g[idx]

    # Safety fallback for params not covered by any group.
    covered = set()
    for idxs in group_to_indices.values():
        covered.update(idxs)
    for idx in range(len(params)):
        if idx not in covered:
            merged[idx] = g_u[idx] + g_g[idx]

    set_grads(params, merged)

    layer_values = list(layer_cos.values())
    if len(layer_values) == 0:
        layer_cos_mean = 0.0
        layer_cos_neg_ratio = 0.0
    else:
        layer_cos_mean = float(sum(layer_values) / len(layer_values))
        layer_cos_neg_ratio = float(sum(1 for c in layer_values if c < conflict_threshold) / len(layer_values))

    return {
        "global_cos": grad_cosine(g_u, g_g),
        "layer_cos_mean": layer_cos_mean,
        "layer_cos_neg_ratio": layer_cos_neg_ratio,
        "layer_cos": layer_cos,
    }


def apply_layerwise_cagrad(
    loss_u: torch.Tensor,
    loss_g: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    group_to_indices: Dict[str, List[int]],
    beta: float = 0.5,
    conflict_threshold: float = 0.0,
    variant: str = "v2_gate",
    gate_tau: float = 0.0,
    gate_scale: float = 8.0,
    prefer_understanding_on_conflict: bool = True,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    gate_conflict_only: bool = True,
    nonconflict_merge: str = "cagrad",
    conflict_groups: Optional[Sequence[str]] = None,
    layer_grad_normalize: bool = False,
    layer_grad_normalize_target: str = "understanding",
    layer_grad_normalize_strength: float = 1.0,
    layer_grad_normalize_conflict_only: bool = False,
    asym_proj_residual: float = 0.0,
    eps: float = 1e-12,
) -> Dict[str, object]:
    if len(params) == 0:
        return {
            "global_cos": 0.0,
            "layer_cos_mean": 0.0,
            "layer_cos_neg_ratio": 0.0,
            "layer_cos": {},
            "layer_alpha_mean": 0.5,
            "layer_alpha": {},
        }

    raw_u = torch.autograd.grad(loss_u, params, retain_graph=True, allow_unused=True)
    raw_g = torch.autograd.grad(loss_g, params, retain_graph=True, allow_unused=True)
    g_u = _materialize_grads(raw_u, params)
    g_g = _materialize_grads(raw_g, params)

    merged = [_safe_zeros_like_param(p) for p in params]
    layer_cos: Dict[str, float] = {}
    layer_alpha: Dict[str, float] = {}

    variant = str(variant).lower()
    alpha_min = float(max(0.0, min(1.0, alpha_min)))
    alpha_max = float(max(alpha_min, min(1.0, alpha_max)))
    gate_scale = float(gate_scale)
    gate_tau = float(gate_tau)
    nonconflict_merge = str(nonconflict_merge).lower()
    layer_grad_normalize_target = str(layer_grad_normalize_target).lower()
    layer_grad_normalize_strength = float(max(0.0, min(1.0, layer_grad_normalize_strength)))
    asym_proj_residual = float(max(0.0, asym_proj_residual))
    conflict_groups_set: Set[str] = set()
    if conflict_groups is not None:
        conflict_groups_set = {str(g).strip() for g in conflict_groups if str(g).strip()}

    for group_name, indices in group_to_indices.items():
        if len(indices) == 0:
            layer_cos[group_name] = 0.0
            layer_alpha[group_name] = 0.5
            continue

        gu_group = [g_u[idx] for idx in indices]
        gg_group = [g_g[idx] for idx in indices]
        cos = grad_cosine(gu_group, gg_group)
        layer_cos[group_name] = cos

        gu_merge = gu_group
        gg_merge = gg_group
        do_layer_norm = layer_grad_normalize
        if do_layer_norm and layer_grad_normalize_conflict_only and not (cos < conflict_threshold):
            do_layer_norm = False

        if do_layer_norm:
            norm_u = _group_l2_norm(gu_group, eps=eps)
            norm_g = _group_l2_norm(gg_group, eps=eps)
            if layer_grad_normalize_target in {"understanding", "u", "txt"}:
                scale = (norm_u / norm_g.clamp_min(eps)).detach()
                gg_scaled = [scale * gg for gg in gg_group]
                s = layer_grad_normalize_strength
                gg_merge = [(1.0 - s) * gg + s * ggs for gg, ggs in zip(gg_group, gg_scaled)]
            elif layer_grad_normalize_target in {"generation", "g", "rec"}:
                scale = (norm_g / norm_u.clamp_min(eps)).detach()
                gu_scaled = [scale * gu for gu in gu_group]
                s = layer_grad_normalize_strength
                gu_merge = [(1.0 - s) * gu + s * gus for gu, gus in zip(gu_group, gu_scaled)]
            else:
                raise ValueError(
                    "Unsupported layer_grad_normalize_target="
                    f"{layer_grad_normalize_target}. Use understanding|generation."
                )

        if variant in {"asym_proj", "asymmetric_proj", "layer_asym_proj"}:
            # Asymmetric Layer-PCGrad:
            # - if non-conflict, keep full signal from both tasks: g = gu + gg
            # - if conflict, remove from gg the component that hurts understanding.
            if len(conflict_groups_set) > 0 and group_name not in conflict_groups_set:
                layer_alpha[group_name] = 0.5
                merged_group = _add(gu_merge, gg_merge)
                for idx, gm in zip(indices, merged_group):
                    merged[idx] = gm
                continue

            dot = torch.zeros((), device=gu_merge[0].device, dtype=gu_merge[0].dtype)
            nu = torch.zeros_like(dot)
            for gu, gg in zip(gu_merge, gg_merge):
                dot = dot + torch.sum(gg * gu)
                nu = nu + torch.sum(gu * gu)

            is_conflict = cos < conflict_threshold
            if is_conflict and float(dot.item()) < 0.0:
                ratio = (dot / nu.clamp_min(eps)).detach()
                gg_proj_group = [gg - ratio * gu for gu, gg in zip(gu_merge, gg_merge)]
                # Re-inject a controllable fraction of the removed generation component:
                # residual=0.0 -> pure projection; residual=1.0 -> original gg;
                # residual>1.0 -> over-inject generation component.
                if asym_proj_residual > 0.0:
                    gg_merge_group = [
                        gg_proj + asym_proj_residual * (gg - gg_proj)
                        for gg, gg_proj in zip(gg_merge, gg_proj_group)
                    ]
                else:
                    gg_merge_group = gg_proj_group
                merged_group = [gu + gg_mix for gu, gg_mix in zip(gu_merge, gg_merge_group)]
                layer_alpha[group_name] = 1.0
            else:
                layer_alpha[group_name] = 0.5
                merged_group = _add(gu_merge, gg_merge)
        elif variant in {"v2", "v2_gate", "gate"}:
            if gate_conflict_only and cos >= conflict_threshold:
                layer_alpha[group_name] = 0.5
                if nonconflict_merge == "sum":
                    merged_group = _add(gu_merge, gg_merge)
                else:
                    merged_group = _cagrad_like_merge(gu_merge, gg_merge, beta=beta)
                for idx, gm in zip(indices, merged_group):
                    merged[idx] = gm
                continue
            if prefer_understanding_on_conflict:
                # More conflict (cos smaller than tau) -> alpha moves toward understanding.
                logit = gate_scale * (gate_tau - cos)
            else:
                logit = gate_scale * (cos - gate_tau)
            alpha_t = torch.sigmoid(torch.tensor(logit, device=gu_group[0].device, dtype=gu_group[0].dtype))
            alpha = float(alpha_t.clamp(alpha_min, alpha_max).item())
            layer_alpha[group_name] = alpha
            merged_group = [alpha * gu + (1.0 - alpha) * gg for gu, gg in zip(gu_merge, gg_merge)]
        else:
            layer_alpha[group_name] = 0.5
            if cos < conflict_threshold:
                merged_group = _cagrad_like_merge(gu_merge, gg_merge, beta=beta)
            else:
                merged_group = _add(gu_merge, gg_merge)

        for idx, gm in zip(indices, merged_group):
            merged[idx] = gm

    covered = set()
    for idxs in group_to_indices.values():
        covered.update(idxs)
    for idx in range(len(params)):
        if idx not in covered:
            merged[idx] = g_u[idx] + g_g[idx]

    set_grads(params, merged)

    layer_values = list(layer_cos.values())
    if len(layer_values) == 0:
        layer_cos_mean = 0.0
        layer_cos_neg_ratio = 0.0
    else:
        layer_cos_mean = float(sum(layer_values) / len(layer_values))
        layer_cos_neg_ratio = float(sum(1 for c in layer_values if c < conflict_threshold) / len(layer_values))
    alpha_values = list(layer_alpha.values())
    layer_alpha_mean = float(sum(alpha_values) / len(alpha_values)) if len(alpha_values) > 0 else 0.5

    return {
        "global_cos": grad_cosine(g_u, g_g),
        "layer_cos_mean": layer_cos_mean,
        "layer_cos_neg_ratio": layer_cos_neg_ratio,
        "layer_cos": layer_cos,
        "layer_alpha_mean": layer_alpha_mean,
        "layer_alpha": layer_alpha,
    }
