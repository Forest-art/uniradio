from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch

TensorList = List[torch.Tensor]
IndexGroups = Dict[str, List[int]]


@dataclass
class MergeStats:
    strategy: str
    shared_grad_cosine: float
    global_magnitude_gain: float = 1.0


def _zeros_like(param: torch.nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(param, memory_format=torch.preserve_format)


def _materialize_grads(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[torch.nn.Parameter],
) -> TensorList:
    return [_zeros_like(param) if grad is None else grad for grad, param in zip(grads, params)]


def _flatten(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.zeros(1)
    return torch.cat([grad.reshape(-1) for grad in grads], dim=0)


def _add(lhs: Sequence[torch.Tensor], rhs: Sequence[torch.Tensor]) -> TensorList:
    return [g1 + g2 for g1, g2 in zip(lhs, rhs)]


def _scale(grads: Sequence[torch.Tensor], value: float) -> TensorList:
    return [value * grad for grad in grads]


def _group_norm(grads: Sequence[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    flat = _flatten(grads)
    return torch.linalg.norm(flat).clamp_min(eps)


def _set_grads(params: Sequence[torch.nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
    for param, grad in zip(params, grads):
        param.grad = grad


def grad_cosine(grads_u: Sequence[torch.Tensor], grads_g: Sequence[torch.Tensor], eps: float = 1e-12) -> float:
    flat_u = _flatten(grads_u)
    flat_g = _flatten(grads_g)
    denom = (torch.linalg.norm(flat_u) * torch.linalg.norm(flat_g)).clamp_min(eps)
    return float((torch.dot(flat_u, flat_g) / denom).item())


def make_param_groups(
    shared_params: Sequence[torch.nn.Parameter],
    named_param_groups: Mapping[str, Iterable[torch.nn.Parameter]],
) -> IndexGroups:
    """Map module-wise parameter iterables to indices in `shared_params`."""
    index_by_id = {id(param): idx for idx, param in enumerate(shared_params)}
    groups: IndexGroups = {}
    for name, params in named_param_groups.items():
        indices = [index_by_id[id(param)] for param in params if id(param) in index_by_id]
        if indices:
            groups[name] = indices
    return groups


def _project_conflicting_symmetrically(grads_u: Sequence[torch.Tensor], grads_g: Sequence[torch.Tensor]) -> TensorList:
    flat_u = _flatten(grads_u)
    flat_g = _flatten(grads_g)
    dot = torch.dot(flat_u, flat_g)
    if float(dot.item()) >= 0.0:
        return _add(grads_u, grads_g)

    uu = torch.dot(flat_u, flat_u).clamp_min(1e-12)
    gg = torch.dot(flat_g, flat_g).clamp_min(1e-12)
    proj_u = (dot / gg).detach()
    proj_g = (dot / uu).detach()
    clean_u = [u - proj_u * g for u, g in zip(grads_u, grads_g)]
    clean_g = [g - proj_g * u for u, g in zip(grads_u, grads_g)]
    return _add(clean_u, clean_g)


def _mgda_two_task_merge(grads_u: Sequence[torch.Tensor], grads_g: Sequence[torch.Tensor]) -> TensorList:
    flat_u = _flatten(grads_u)
    flat_g = _flatten(grads_g)
    diff = flat_u - flat_g
    denom = torch.dot(diff, diff)
    if float(denom.item()) <= 1e-12:
        alpha = 0.5
    else:
        alpha = float(torch.dot(flat_g, flat_g - flat_u).item() / denom.item())
        alpha = max(0.0, min(1.0, alpha))
    return [alpha * u + (1.0 - alpha) * g for u, g in zip(grads_u, grads_g)]


def _cagrad_like_merge(grads_u: Sequence[torch.Tensor], grads_g: Sequence[torch.Tensor], beta: float) -> TensorList:
    beta = max(0.0, min(1.0, float(beta)))
    average = [0.5 * (u + g) for u, g in zip(grads_u, grads_g)]
    mgda = _mgda_two_task_merge(grads_u, grads_g)
    return [(1.0 - beta) * avg + beta * mg for avg, mg in zip(average, mgda)]


def _default_groups(num_grads: int) -> IndexGroups:
    return {"all": list(range(num_grads))}


def merge_joint(grads_u: Sequence[torch.Tensor], grads_g: Sequence[torch.Tensor]) -> TensorList:
    return _add(grads_u, grads_g)


def merge_pcgrad(grads_u: Sequence[torch.Tensor], grads_g: Sequence[torch.Tensor]) -> TensorList:
    return _project_conflicting_symmetrically(grads_u, grads_g)


def merge_cagrad(grads_u: Sequence[torch.Tensor], grads_g: Sequence[torch.Tensor], beta: float = 0.35) -> TensorList:
    return _cagrad_like_merge(grads_u, grads_g, beta=beta)


def merge_dsga(
    grads_u: Sequence[torch.Tensor],
    grads_g: Sequence[torch.Tensor],
    groups: Optional[IndexGroups] = None,
    lambda_mag: float = 0.2,
    conflict_threshold: float = 0.0,
    conflict_only: bool = False,
    magnitude_scope: str = "global",
    norm_restore: bool = False,
    eps: float = 1e-8,
) -> tuple[TensorList, float]:
    """Merge two task gradients with DSGA-D and DSGA-M.

    DSGA-D:
      For a conflicting group, remove the component of the generation gradient
      that lies along the understanding gradient.

    DSGA-M:
      Scale the generation gradient by
      (||g_u|| / ||g_g||) ** lambda_mag
      either globally or per group.

    The default paper setting is layer-wise DSGA-D with global DSGA-M.
    """
    if len(grads_u) != len(grads_g):
        raise ValueError("grads_u and grads_g must have the same length")
    if not grads_u:
        return [], 1.0

    magnitude_scope = str(magnitude_scope).lower()
    if magnitude_scope not in {"global", "layerwise"}:
        raise ValueError("magnitude_scope must be 'global' or 'layerwise'")

    groups = groups or _default_groups(len(grads_u))
    merged = [torch.zeros_like(grad) for grad in grads_u]
    covered = set()
    lambda_mag = max(0.0, float(lambda_mag))
    eps = max(float(eps), 1e-12)

    global_gain = 1.0
    if magnitude_scope == "global":
        global_gain = float(
            torch.pow(
                (_group_norm(grads_u, eps=eps) / _group_norm(grads_g, eps=eps)).clamp_min(eps),
                lambda_mag,
            ).item()
        )

    for name, raw_indices in groups.items():
        del name
        indices = [int(idx) for idx in raw_indices if 0 <= int(idx) < len(grads_u)]
        if not indices:
            continue
        covered.update(indices)

        group_u = [grads_u[idx] for idx in indices]
        group_g = [grads_g[idx] for idx in indices]
        flat_u = _flatten(group_u)
        flat_g = _flatten(group_g)
        norm_u = torch.linalg.norm(flat_u)
        norm_g = torch.linalg.norm(flat_g)
        dot = torch.dot(flat_u, flat_g)
        cosine = dot / (norm_u * norm_g + eps)
        is_conflict = float(cosine.item()) < float(conflict_threshold)

        if magnitude_scope == "global":
            gain = global_gain
        else:
            gain = float(torch.pow((norm_u / (norm_g + eps)).clamp_min(eps), lambda_mag).item())
        aligned_g = [gain * grad for grad in group_g]

        if (not is_conflict) and conflict_only:
            merged_group = [u + g for u, g in zip(group_u, group_g)]
        elif not is_conflict:
            merged_group = [u + g for u, g in zip(group_u, aligned_g)]
        else:
            aligned_flat = _flatten(aligned_g)
            coeff = torch.dot(aligned_flat, flat_u) / (torch.dot(flat_u, flat_u) + eps)
            perpendicular = [g - coeff * u for u, g in zip(group_u, aligned_g)]
            if norm_restore:
                aligned_norm = torch.linalg.norm(aligned_flat)
                perp_norm = torch.linalg.norm(_flatten(perpendicular))
                restore = aligned_norm / (perp_norm + eps)
                perpendicular = [restore * grad for grad in perpendicular]
            merged_group = [u + g for u, g in zip(group_u, perpendicular)]

        for idx, grad in zip(indices, merged_group):
            merged[idx] = grad

    for idx in range(len(grads_u)):
        if idx not in covered:
            merged[idx] = grads_u[idx] + grads_g[idx]

    return merged, global_gain


def apply_two_task_strategy(
    loss_understanding: torch.Tensor,
    loss_generation: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    strategy: str,
    lambda_understanding: float = 1.0,
    lambda_generation: float = 1.0,
    cagrad_beta: float = 0.35,
    dsga_groups: Optional[IndexGroups] = None,
    dsga_lambda_mag: float = 0.2,
    dsga_conflict_threshold: float = 0.0,
    dsga_conflict_only: bool = False,
    dsga_magnitude_scope: str = "global",
    dsga_norm_restore: bool = False,
) -> MergeStats:
    """Apply a two-task gradient strategy and write `.grad` in-place.

    `shared_params` receive the merged gradient.
    `aux_params` receive gradients from the weighted sum of both losses.
    """
    strategy = str(strategy).lower()
    total_loss = lambda_understanding * loss_understanding + lambda_generation * loss_generation

    if shared_params:
        raw_u = torch.autograd.grad(loss_understanding, shared_params, retain_graph=True, allow_unused=True)
        raw_g = torch.autograd.grad(loss_generation, shared_params, retain_graph=True, allow_unused=True)
        grads_u = _scale(_materialize_grads(raw_u, shared_params), lambda_understanding)
        grads_g = _scale(_materialize_grads(raw_g, shared_params), lambda_generation)
        cosine = grad_cosine(grads_u, grads_g)
        global_gain = 1.0

        if strategy == "joint":
            merged = merge_joint(grads_u, grads_g)
        elif strategy == "pcgrad":
            merged = merge_pcgrad(grads_u, grads_g)
        elif strategy == "cagrad":
            merged = merge_cagrad(grads_u, grads_g, beta=cagrad_beta)
        elif strategy == "dsga":
            merged, global_gain = merge_dsga(
                grads_u,
                grads_g,
                groups=dsga_groups,
                lambda_mag=dsga_lambda_mag,
                conflict_threshold=dsga_conflict_threshold,
                conflict_only=dsga_conflict_only,
                magnitude_scope=dsga_magnitude_scope,
                norm_restore=dsga_norm_restore,
            )
        else:
            raise ValueError("strategy must be one of: joint, pcgrad, cagrad, dsga")

        _set_grads(shared_params, merged)
    else:
        cosine = 0.0
        global_gain = 1.0

    if aux_params:
        aux_grads = torch.autograd.grad(total_loss, aux_params, allow_unused=True)
        aux_grads = _materialize_grads(aux_grads, aux_params)
        _set_grads(aux_params, aux_grads)

    return MergeStats(
        strategy=strategy,
        shared_grad_cosine=cosine,
        global_magnitude_gain=global_gain,
    )
