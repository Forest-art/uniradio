from typing import List, Optional, Sequence, Tuple

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


def set_grads(params: Sequence[torch.nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g


def apply_conflict_aware(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    lora_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
) -> float:
    if len(lora_params) == 0:
        # Fallback to a regular backward path by filling aux grads only.
        total = lambda_txt * loss_txt + lambda_rec * loss_rec
        if len(aux_params) > 0:
            aux_grads = torch.autograd.grad(total, aux_params, allow_unused=True)
            aux_grads = _materialize_grads(aux_grads, aux_params)
            set_grads(aux_params, aux_grads)
        return 0.0

    raw_txt = torch.autograd.grad(loss_txt, lora_params, retain_graph=True, allow_unused=True)
    raw_rec = torch.autograd.grad(loss_rec, lora_params, retain_graph=True, allow_unused=True)

    g_txt = _materialize_grads(raw_txt, lora_params)
    g_rec = _materialize_grads(raw_rec, lora_params)

    cos = grad_cosine(g_txt, g_rec)

    wg_txt = [lambda_txt * g for g in g_txt]
    wg_rec = [lambda_rec * g for g in g_rec]

    if cos < 0:
        wg_txt, wg_rec = project_conflicting(wg_txt, wg_rec)

    merged_lora = [(a + b) * 0.5 for a, b in zip(wg_txt, wg_rec)]

    total = lambda_txt * loss_txt + lambda_rec * loss_rec
    if len(aux_params) > 0:
        aux_grads = torch.autograd.grad(total, aux_params, allow_unused=True)
        aux_grads = _materialize_grads(aux_grads, aux_params)
    else:
        aux_grads = []

    set_grads(lora_params, merged_lora)
    if len(aux_params) > 0:
        set_grads(aux_params, aux_grads)
    return cos


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
