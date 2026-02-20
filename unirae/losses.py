from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def align_text_dim(text_emb: torch.Tensor, target_dim: int) -> torch.Tensor:
    if text_emb.shape[-1] == target_dim:
        return text_emb
    if text_emb.shape[-1] > target_dim:
        return text_emb[:, :target_dim]

    pad = torch.zeros(
        text_emb.shape[0],
        target_dim - text_emb.shape[-1],
        device=text_emb.device,
        dtype=text_emb.dtype,
    )
    return torch.cat([text_emb, pad], dim=-1)


def text_classification_loss(
    image_emb: torch.Tensor,
    labels: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    text_emb = align_text_dim(text_emb, image_emb.shape[-1])
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    logits = image_emb @ text_emb.t()
    logits = logits / max(temperature, 1e-6)
    loss = F.cross_entropy(logits, labels)

    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, {"txt_acc": acc}


def feature_recon_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    l2 = F.mse_loss(pred, target)
    cos = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1).mean()
    return l2, {"feature_l2": float(l2.detach().item()), "feature_cos": float(cos.detach().item())}


def pixel_recon_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "mse", huber_delta: float = 1.0):
    if kind == "huber":
        loss = F.huber_loss(pred, target, delta=huber_delta)
    else:
        loss = F.mse_loss(pred, target)

    mse = F.mse_loss(pred, target)
    psnr = -10.0 * torch.log10(mse + 1e-8)
    return loss, {"mse": float(mse.detach().item()), "psnr": float(psnr.detach().item())}


def lpips_score(pred: torch.Tensor, target: torch.Tensor):
    try:
        import lpips
    except ImportError:
        return None

    with torch.no_grad():
        metric = lpips.LPIPS(net="alex").to(pred.device)
        score = metric(pred, target).mean()
    return float(score.item())
