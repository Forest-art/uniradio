from typing import Callable, List, Optional, Sequence, Tuple

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


def _weighted(grads: Sequence[torch.Tensor], weight: float) -> List[torch.Tensor]:
    return [weight * g for g in grads]


def _add(g1: Sequence[torch.Tensor], g2: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [a + b for a, b in zip(g1, g2)]


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
        extra_loss=extra_loss,
    )


def apply_mgda_ub(
    loss_txt: torch.Tensor,
    loss_rec: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    aux_params: Sequence[torch.nn.Parameter],
    lambda_txt: float,
    lambda_rec: float,
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
    extra_loss: Optional[torch.Tensor] = None,
) -> float:
    def _merge(wg_txt: List[torch.Tensor], wg_rec: List[torch.Tensor], _: float) -> List[torch.Tensor]:
        return _cagrad_like_merge(wg_txt, wg_rec, beta=beta)

    return _apply_multi_objective(
        loss_txt=loss_txt,
        loss_rec=loss_rec,
        shared_params=shared_params,
        aux_params=aux_params,
        lambda_txt=lambda_txt,
        lambda_rec=lambda_rec,
        merge_fn=_merge,
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
