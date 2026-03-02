import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from .data_cifar10 import (
    build_cifar10_loader,
    denormalize_cifar10,
    make_batch_dict,
)
from .grad_conflict import (
    apply_naive,
    apply_cagrad,
    apply_saop,
    apply_laga_objective,
    apply_ma_laga_objective,
    apply_conflict_aware,
    compute_grad_cosine,
)
from .losses import FeatureVarianceLoss
from .models import build_backbone
from .utils import (
    append_jsonl,
    apply_overrides,
    count_parameters,
    cycle_loader,
    ensure_dir,
    load_yaml,
    now_str,
    save_json,
    save_yaml,
    to_device,
)


class CifarTradeoffModel(nn.Module):
    def __init__(self, cfg: Dict, num_classes: int):
        super().__init__()
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})

        image_size = int(data_cfg.get("image_size", 32))
        backbone_name = model_cfg.get("backbone", "resnet18")
        pretrained = bool(model_cfg.get("pretrained", False))

        self.backbone, feat_dim = build_backbone(
            backbone_name=backbone_name,
            image_size=image_size,
            pretrained=pretrained,
        )
        self.feat_dim = feat_dim

        txt_dim = int(model_cfg.get("txt_dim", 256))
        rec_dim = int(model_cfg.get("rec_dim", 256))
        self.recon_size = int(model_cfg.get("recon_size", image_size))
        hidden_dim = int(model_cfg.get("decoder_hidden_dim", 512))

        self.txt_head = nn.Linear(feat_dim, txt_dim)
        self.rec_head = nn.Linear(feat_dim, rec_dim)
        self.decoder = nn.Sequential(
            nn.Linear(rec_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * self.recon_size * self.recon_size),
        )

        self.text_prototypes = nn.Parameter(torch.randn(num_classes, txt_dim) * 0.02)

        if bool(model_cfg.get("freeze_backbone", False)):
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(images)
        z_txt = self.txt_head(feat)
        z_rec = self.rec_head(feat)
        recon = self.decoder(z_rec).view(images.shape[0], 3, self.recon_size, self.recon_size)
        return {
            "feat": feat,
            "z_txt": z_txt,
            "z_rec": z_rec,
            "recon": recon,
        }


def text_prototype_loss(
    z_txt: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    z = F.normalize(z_txt, dim=-1)
    p = F.normalize(prototypes, dim=-1)
    logits = z @ p.t()
    logits = logits / max(temperature, 1e-6)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, {"txt_acc": acc}


def save_checkpoint(path: str, model: CifarTradeoffModel, optimizer: torch.optim.Optimizer, step: int, cfg: Dict) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


def _build_shared_and_aux_params(model: CifarTradeoffModel, shared_mode: str) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    if shared_mode == "all":
        shared = [p for p in model.parameters() if p.requires_grad]
    else:
        shared = [p for p in model.backbone.parameters() if p.requires_grad]

    shared_ids = {id(p) for p in shared}
    aux = [p for p in model.parameters() if p.requires_grad and id(p) not in shared_ids]
    return shared, aux


def _dist_mean_scalar(accelerator: Accelerator, value, device: torch.device) -> float:
    if torch.is_tensor(value):
        t = value.detach().to(device)
        if t.ndim > 0:
            t = t.float().mean()
        else:
            t = t.float()
    else:
        t = torch.tensor(float(value), device=device, dtype=torch.float32)
    reduced = accelerator.reduce(t, reduction="mean")
    return float(reduced.item())


def _save_recon_samples(
    images: torch.Tensor,
    recons: torch.Tensor,
    dataset_name: str,
    out_path: str,
    max_items: int,
) -> None:
    b = min(max_items, images.shape[0])
    gt = torch.clamp(denormalize_cifar10(images[:b], dataset=dataset_name).cpu(), 0.0, 1.0)
    rc = torch.clamp(denormalize_cifar10(recons[:b], dataset=dataset_name).cpu(), 0.0, 1.0)
    if rc.shape[-2:] != gt.shape[-2:]:
        rc = F.interpolate(rc, size=gt.shape[-2:], mode="bilinear", align_corners=False)

    rows = []
    for i in range(b):
        rows.append(gt[i])
        rows.append(rc[i])
    grid = make_grid(rows, nrow=2)
    ensure_dir(str(Path(out_path).parent))
    save_image(grid, out_path)


def _materialize_grads(
    grads: List[Optional[torch.Tensor]],
    params: List[nn.Parameter],
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p, memory_format=torch.preserve_format))
        else:
            out.append(g)
    return out


def _global_grad_l2_norm(grads: List[torch.Tensor], eps: float = 1e-12) -> float:
    if len(grads) == 0:
        return 0.0
    s = 0.0
    for g in grads:
        x = g.detach().float()
        s += float((x * x).sum().item())
    return float((s + eps) ** 0.5)


def _safe_cosine_flat(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    nu = torch.norm(u)
    nv = torch.norm(v)
    if float(nu.item()) < eps or float(nv.item()) < eps:
        return 0.0
    return float(torch.dot(u, v).item() / (nu.item() * nv.item() + eps))


def _neg_ratio_flat(u: torch.Tensor, v: torch.Tensor) -> float:
    return float(((u * v) < 0).float().mean().item())


def _pearson_corr(xs: List[float], ys: List[float], eps: float = 1e-12) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= eps or vy <= eps:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / ((vx * vy) ** 0.5 + eps))


def _rank_values(values: List[float]) -> List[float]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _rank_values(xs)
    ry = _rank_values(ys)
    return _pearson_corr(rx, ry)


def _extract_first_int(parts: List[str], pos: int, default: int = 0) -> int:
    if pos < 0 or pos >= len(parts):
        return default
    token = parts[pos]
    return int(token) if token.isdigit() else default


def _layer_group_from_full_name(name: str) -> Tuple[str, int]:
    n = str(name)
    if n.startswith("backbone."):
        n = n[len("backbone.") :]

    # ResNet18Backbone (torchvision in stem).
    if n.startswith("stem.0.") or n.startswith("stem.1.") or n.startswith("stem.2.") or n.startswith("stem.3."):
        return "stem", 0
    if n.startswith("stem.4."):
        parts = n.split(".")
        b = _extract_first_int(parts, 2, default=0)
        return f"layer1.{b}", 1 + b
    if n.startswith("stem.5."):
        parts = n.split(".")
        b = _extract_first_int(parts, 2, default=0)
        return f"layer2.{b}", 3 + b
    if n.startswith("stem.6."):
        parts = n.split(".")
        b = _extract_first_int(parts, 2, default=0)
        return f"layer3.{b}", 5 + b
    if n.startswith("stem.7."):
        parts = n.split(".")
        b = _extract_first_int(parts, 2, default=0)
        return f"layer4.{b}", 7 + b
    if n.startswith("stem.8."):
        return "avgpool", 9
    if n.startswith("layer1."):
        parts = n.split(".")
        b = _extract_first_int(parts, 1, default=0)
        return f"layer1.{b}", 1 + b
    if n.startswith("layer2."):
        parts = n.split(".")
        b = _extract_first_int(parts, 1, default=0)
        return f"layer2.{b}", 3 + b
    if n.startswith("layer3."):
        parts = n.split(".")
        b = _extract_first_int(parts, 1, default=0)
        return f"layer3.{b}", 5 + b
    if n.startswith("layer4."):
        parts = n.split(".")
        b = _extract_first_int(parts, 1, default=0)
        return f"layer4.{b}", 7 + b

    # ViT/DINO style.
    if ".blocks." in n:
        parts = n.split(".")
        for i in range(len(parts) - 1):
            if parts[i] == "blocks" and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                return f"blocks.{idx}", idx + 1
    if ".layers." in n and ".blocks." in n:
        parts = n.split(".")
        stage = 0
        block = 0
        for i in range(len(parts) - 1):
            if parts[i] == "layers" and parts[i + 1].isdigit():
                stage = int(parts[i + 1])
            if parts[i] == "blocks" and parts[i + 1].isdigit():
                block = int(parts[i + 1])
        depth = stage * 16 + block + 1
        return f"layers.{stage}.blocks.{block}", depth
    if "patch_embed" in n:
        return "patch_embed", 0
    if "cls_token" in n or "pos_embed" in n or "register_tokens" in n:
        return "embeddings", 0
    if n.endswith("norm.weight") or n.endswith("norm.bias") or ".norm." in n or "fc_norm" in n:
        return "norm", 999

    # Non-backbone trainable params (only if shared_params=all).
    if n.startswith("txt_head."):
        return "txt_head", 2000
    if n.startswith("rec_head."):
        return "rec_head", 2001
    if n.startswith("decoder."):
        return "decoder", 2002
    if n.startswith("text_prototypes"):
        return "text_prototypes", 2003
    return "other", 9999


def _build_probe_groups(
    shared_param_names: List[str],
) -> Tuple[Dict[str, List[int]], Dict[str, int], List[str]]:
    group_to_indices: Dict[str, List[int]] = {}
    group_to_depth: Dict[str, int] = {}
    for i, name in enumerate(shared_param_names):
        g, d = _layer_group_from_full_name(name)
        group_to_indices.setdefault(g, []).append(i)
        if g not in group_to_depth:
            group_to_depth[g] = int(d)
        else:
            group_to_depth[g] = min(group_to_depth[g], int(d))
    ordered = sorted(group_to_indices.keys(), key=lambda k: (group_to_depth[k], k))
    return group_to_indices, group_to_depth, ordered


def _collect_layer_probe_rows(
    step: int,
    g_u: List[torch.Tensor],
    g_g: List[torch.Tensor],
    group_to_indices: Dict[str, List[int]],
    group_to_depth: Dict[str, int],
    ordered_groups: List[str],
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    cos_values: List[float] = []
    neg_values: List[float] = []
    depth_values: List[float] = []

    for gname in ordered_groups:
        indices = group_to_indices.get(gname, [])
        if len(indices) == 0:
            continue
        gu_flat = torch.cat([g_u[i].detach().reshape(-1) for i in indices], dim=0)
        gg_flat = torch.cat([g_g[i].detach().reshape(-1) for i in indices], dim=0)
        cos = _safe_cosine_flat(gu_flat, gg_flat)
        neg = _neg_ratio_flat(gu_flat, gg_flat)
        depth = int(group_to_depth[gname])
        row = {
            "step": int(step),
            "layer": gname,
            "depth": depth,
            "cosine_similarity": float(cos),
            "neg_ratio": float(neg),
            "gu_norm": float(torch.norm(gu_flat).item()),
            "gg_norm": float(torch.norm(gg_flat).item()),
            "numel": int(gu_flat.numel()),
        }
        rows.append(row)
        cos_values.append(float(cos))
        neg_values.append(float(neg))
        depth_values.append(float(depth))

    if len(rows) == 0:
        return [], {
            "step": int(step),
            "num_layers": 0,
            "mean_cosine": 0.0,
            "mean_neg_ratio": 0.0,
            "depth_cos_pearson": 0.0,
            "depth_cos_spearman": 0.0,
        }

    depth_cos_pearson = _pearson_corr(depth_values, cos_values)
    depth_cos_spearman = _spearman_corr(depth_values, cos_values)
    stats = {
        "step": int(step),
        "num_layers": int(len(rows)),
        "mean_cosine": float(sum(cos_values) / len(cos_values)),
        "mean_neg_ratio": float(sum(neg_values) / len(neg_values)),
        "depth_cos_pearson": float(depth_cos_pearson),
        "depth_cos_spearman": float(depth_cos_spearman),
    }
    return rows, stats


def _save_layer_probe_csv(path: str, rows: List[Dict[str, float]]) -> None:
    ensure_dir(str(Path(path).parent))
    if len(rows) == 0:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "layer",
                "depth",
                "cosine_similarity",
                "neg_ratio",
                "gu_norm",
                "gg_norm",
                "numel",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_understanding(
    model: CifarTradeoffModel,
    loader,
    device: torch.device,
    class_names: List[str],
    temperature: float,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = make_batch_dict(batch)
            batch = to_device(batch, device)

            out = model(batch["images"])
            loss, _ = text_prototype_loss(
                z_txt=out["z_txt"],
                labels=batch["labels"],
                prototypes=model.text_prototypes,
                temperature=temperature,
            )

            logits = F.normalize(out["z_txt"], dim=-1) @ F.normalize(model.text_prototypes, dim=-1).t()
            pred = logits.argmax(dim=1)

            total += batch["labels"].numel()
            correct += (pred == batch["labels"]).sum().item()
            loss_sum += float(loss.item()) * batch["labels"].shape[0]

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return {
        "acc_txt": acc,
        "zero_shot_acc": acc,
        "zero_shot_loss": avg_loss,
        "num_samples": total,
        "class_names": list(class_names),
    }


def evaluate_generation(
    model: CifarTradeoffModel,
    loader,
    device: torch.device,
    dataset_name: str,
    max_batches: int,
    sample_path: str,
    save_samples: bool,
    sample_images: int,
) -> Dict[str, float]:
    model.eval()

    total_mse = 0.0
    n = 0
    saved = False

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = make_batch_dict(batch)
            batch = to_device(batch, device)

            out = model(batch["images"])
            recon = out["recon"]
            target = batch["images"]
            if recon.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=recon.shape[-2:], mode="bilinear", align_corners=False)

            mse = F.mse_loss(recon, target)
            total_mse += float(mse.item())
            n += 1

            if save_samples and (not saved):
                _save_recon_samples(
                    target,
                    recon,
                    dataset_name=dataset_name,
                    out_path=sample_path,
                    max_items=sample_images,
                )
                saved = True

    recon_mse = total_mse / max(n, 1)
    psnr = -10.0 * torch.log10(torch.tensor(recon_mse + 1e-8)).item()
    return {
        "recon_mode": "pixel_recon",
        "loss": recon_mse,
        "recon_mse": recon_mse,
        "mse": recon_mse,
        "psnr": psnr,
        "num_batches": n,
    }


def run_eval(
    cfg: Dict,
    model: CifarTradeoffModel,
    eval_loader,
    device: torch.device,
    run_dir: str,
    step: int,
    class_names: List[str],
    dataset_name: str,
) -> None:
    eval_cfg = cfg.get("eval", {})
    max_batches = eval_cfg.get("max_batches", None)

    understanding = evaluate_understanding(
        model=model,
        loader=eval_loader,
        device=device,
        class_names=class_names,
        temperature=float(cfg.get("text", {}).get("temperature", 0.07)),
        max_batches=max_batches,
    )

    generation = evaluate_generation(
        model=model,
        loader=eval_loader,
        device=device,
        dataset_name=dataset_name,
        max_batches=max_batches,
        sample_path=os.path.join(run_dir, "samples", f"step_{step:07d}.png"),
        save_samples=bool(eval_cfg.get("save_recon_samples", True)),
        sample_images=int(eval_cfg.get("sample_images", 8)),
    )

    save_json(understanding, os.path.join(run_dir, "understanding.json"))
    save_json(generation, os.path.join(run_dir, "generation.json"))
    save_json({"step": step, "understanding": understanding, "generation": generation}, os.path.join(run_dir, "eval_last.json"))


def _resolve_training_mode(cfg: Dict) -> Tuple[str, int, float, float, str]:
    train_cfg = cfg.get("train", {})

    mode = str(train_cfg.get("mode", "joint")).lower()
    steps = int(train_cfg.get("steps", 1000))
    lambda_txt = float(train_cfg.get("lambda_txt", 1.0))
    lambda_rec = float(train_cfg.get("lambda_rec", 1.0))
    strategy = str(train_cfg.get("grad_strategy", train_cfg.get("strategy", "naive"))).lower()

    if mode == "joint":
        if strategy == "conflict_aware":
            strategy = "pcgrad"
        if strategy not in {"naive", "pcgrad", "cagrad", "saop", "laga", "ma_laga"}:
            raise ValueError(
                f"Unsupported train.strategy={strategy}. "
                "Only naive|pcgrad|cagrad|saop|laga|ma_laga are allowed for joint mode."
            )
        return mode, steps, lambda_txt, lambda_rec, strategy
    if mode == "text_only":
        return mode, steps, lambda_txt, 0.0, "naive"
    if mode == "recon_only":
        return mode, steps, 0.0, lambda_rec, "naive"
    raise ValueError(
        f"Unsupported train.mode={mode}. "
        "Only joint|text_only|recon_only are supported."
    )


def _resolve_grad_norm_mode(cfg: Dict, mode: str) -> str:
    train_cfg = cfg.get("train", {})
    raw = train_cfg.get("grad_norm_mode", train_cfg.get("grad_norm", "none"))

    if isinstance(raw, bool):
        grad_norm_mode = "mean" if raw else "none"
    else:
        grad_norm_mode = str(raw).lower()
        if grad_norm_mode in {"true", "on", "1"}:
            grad_norm_mode = "mean"
        if grad_norm_mode in {"false", "off", "0"}:
            grad_norm_mode = "none"

    if mode != "joint":
        return "none"
    if grad_norm_mode not in {"none", "mean", "geom", "unit"}:
        raise ValueError(
            f"Unsupported train.grad_norm_mode={grad_norm_mode}. "
            "Use one of: none|mean|geom|unit."
        )
    return grad_norm_mode


def _parse_grad_norm_layers(raw_layers) -> List[str]:
    if raw_layers is None:
        return ["layer3", "layer4"]
    if isinstance(raw_layers, str):
        norm = raw_layers.replace("+", ",")
        items = [x.strip().lower() for x in norm.split(",")]
        items = [x for x in items if x]
        return items if items else ["layer3", "layer4"]
    if isinstance(raw_layers, (list, tuple)):
        items = [str(x).strip().lower() for x in raw_layers if str(x).strip()]
        return items if items else ["layer3", "layer4"]
    return ["layer3", "layer4"]


def _layer_tag_from_param_name(name: str) -> str:
    n = str(name)
    if n.startswith("backbone."):
        n = n[len("backbone.") :]
    if n.startswith("model."):
        n = n[len("model.") :]

    # Swin/ViT patch embedding and token embeddings as shallow stem.
    if (
        n.startswith("patch_embed.")
        or n.startswith("embeddings.")
        or n.startswith("cls_token")
        or n.startswith("pos_embed")
        or n.startswith("register_tokens")
    ):
        return "stem"

    # Swin stages: layers.0..3 -> layer1..4.
    if n.startswith("layers."):
        parts = n.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            stage = int(parts[1])
            if 0 <= stage <= 3:
                return f"layer{stage + 1}"

    # Final norm in transformer backbones belongs to deepest stage.
    if n.startswith("norm.") or n.startswith("fc_norm.") or ".norm." in n:
        return "layer4"

    # ResNet18Backbone.
    if n.startswith("stem.0.") or n.startswith("stem.1."):
        return "stem"
    if n.startswith("stem.4.") or n.startswith("layer1."):
        return "layer1"
    if n.startswith("stem.5.") or n.startswith("layer2."):
        return "layer2"
    if n.startswith("stem.6.") or n.startswith("layer3."):
        return "layer3"
    if n.startswith("stem.7.") or n.startswith("layer4."):
        return "layer4"
    return "other"


def _build_shared_param_names(
    model: CifarTradeoffModel,
    shared_params: List[nn.Parameter],
    shared_mode: str,
) -> List[str]:
    if shared_mode == "all":
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    else:
        named = [(f"backbone.{n}", p) for n, p in model.backbone.named_parameters() if p.requires_grad]

    id_to_idx = {id(p): i for i, p in enumerate(shared_params)}
    out = [""] * len(shared_params)
    for n, p in named:
        idx = id_to_idx.get(id(p), None)
        if idx is not None and out[idx] == "":
            out[idx] = n
    for i in range(len(out)):
        if out[i] == "":
            out[i] = f"param_{i}"
    return out


def _resolve_grad_norm_plan(
    cfg: Dict,
    mode: str,
    grad_norm_mode: str,
    model: CifarTradeoffModel,
    shared_params: List[nn.Parameter],
    shared_mode: str,
) -> Tuple[str, Optional[List[int]], bool, List[str]]:
    if mode != "joint" or grad_norm_mode == "none" or len(shared_params) == 0:
        return "all", None, False, []

    train_cfg = cfg.get("train", {})
    scope = str(train_cfg.get("grad_norm_scope", "all")).lower()
    layers = _parse_grad_norm_layers(train_cfg.get("grad_norm_layers", ["layer3", "layer4"]))

    if scope in {"all", "global"}:
        return "all", None, False, layers
    if scope in {"conflict_all", "all_conflict"}:
        return "conflict_all", None, True, layers
    if scope not in {"deep", "conflict_deep"}:
        raise ValueError(
            f"Unsupported train.grad_norm_scope={scope}. "
            "Use one of: all|deep|conflict_all|conflict_deep."
        )

    names = _build_shared_param_names(model=model, shared_params=shared_params, shared_mode=shared_mode)
    indices = [i for i, n in enumerate(names) if _layer_tag_from_param_name(n) in set(layers)]

    if len(indices) == 0:
        # Fallback to all params to avoid silent no-op when naming/layout changes.
        return ("conflict_all", None, True, layers) if scope == "conflict_deep" else ("all", None, False, layers)

    return scope, indices, (scope == "conflict_deep"), layers


def _resolve_cagrad_adaptive_plan(
    cfg: Dict,
    mode: str,
    strategy: str,
    model: CifarTradeoffModel,
    shared_params: List[nn.Parameter],
    shared_mode: str,
) -> Tuple[bool, Dict[str, List[int]], str, List[str]]:
    train_cfg = cfg.get("train", {})
    raw = train_cfg.get("cagrad_adaptive_beta", False)
    if isinstance(raw, bool):
        enabled = raw
    else:
        enabled = str(raw).lower() in {"1", "true", "on", "yes"}

    if (not enabled) or mode != "joint" or strategy != "cagrad" or len(shared_params) == 0:
        return False, {}, "deep", ["layer3", "layer4"]

    scope = str(train_cfg.get("cagrad_adaptive_scope", "deep")).lower()
    if scope not in {"all", "deep"}:
        raise ValueError(
            f"Unsupported train.cagrad_adaptive_scope={scope}. "
            "Use one of: all|deep."
        )
    layers = _parse_grad_norm_layers(train_cfg.get("cagrad_adaptive_layers", ["layer3", "layer4"]))

    names = _build_shared_param_names(model=model, shared_params=shared_params, shared_mode=shared_mode)
    groups: Dict[str, List[int]] = {}
    selected_layers = set(layers)

    for i, n in enumerate(names):
        tag = _layer_tag_from_param_name(n)
        if scope == "deep" and tag not in selected_layers:
            continue
        groups.setdefault(tag, []).append(i)

    if len(groups) == 0:
        return False, {}, scope, layers
    return True, groups, scope, layers


def _resolve_saop_plan(
    cfg: Dict,
    mode: str,
    strategy: str,
    model: CifarTradeoffModel,
    shared_params: List[nn.Parameter],
    shared_mode: str,
) -> Tuple[Dict[str, List[int]], str, List[str]]:
    if mode != "joint" or strategy != "saop" or len(shared_params) == 0:
        return {}, "deep", ["layer3", "layer4"]

    train_cfg = cfg.get("train", {})
    scope = str(train_cfg.get("saop_scope", "deep")).lower()
    if scope not in {"all", "deep"}:
        raise ValueError(
            f"Unsupported train.saop_scope={scope}. "
            "Use one of: all|deep."
        )
    layers = _parse_grad_norm_layers(train_cfg.get("saop_layers", ["layer3", "layer4"]))
    layer_set = set(layers)

    names = _build_shared_param_names(model=model, shared_params=shared_params, shared_mode=shared_mode)
    groups: Dict[str, List[int]] = {}
    for i, n in enumerate(names):
        tag = _layer_tag_from_param_name(n)
        if scope == "deep" and tag not in layer_set:
            continue
        groups.setdefault(tag, []).append(i)

    return groups, scope, layers


def _resolve_laga_plan(
    mode: str,
    strategy: str,
    model: CifarTradeoffModel,
    shared_params: List[nn.Parameter],
    shared_mode: str,
) -> Dict[str, List[int]]:
    if mode != "joint" or strategy not in {"laga", "ma_laga"} or len(shared_params) == 0:
        return {}

    # Fixed 5 groups for fair protocol: stem + layer1..layer4.
    ordered_groups = ["stem", "layer1", "layer2", "layer3", "layer4"]
    groups: Dict[str, List[int]] = {k: [] for k in ordered_groups}
    names = _build_shared_param_names(model=model, shared_params=shared_params, shared_mode=shared_mode)
    for i, n in enumerate(names):
        tag = _layer_tag_from_param_name(n)
        if tag in groups:
            groups[tag].append(i)

    return {k: v for k, v in groups.items() if len(v) > 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    cfg = apply_overrides(base_cfg, args.set)

    mixed_precision = cfg.get("accelerate", {}).get("mixed_precision", "no")
    if isinstance(mixed_precision, bool):
        mixed_precision = "no" if mixed_precision is False else "fp16"
    mixed_precision = str(mixed_precision)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device

    seed = int(cfg.get("seed", 42))
    set_seed(seed, device_specific=True)

    exp_name = args.run_name or cfg.get("experiment", {}).get("name", "cifar_exp")
    run_dir = os.path.join(cfg.get("output", {}).get("root", "runs"), exp_name)

    if accelerator.is_main_process:
        if os.path.exists(run_dir):
            run_dir = f"{run_dir}_{now_str()}"
        ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
        save_yaml(cfg, os.path.join(run_dir, "run_config.yaml"))
    else:
        ckpt_dir = ""

    objs = [run_dir, ckpt_dir]
    broadcast_object_list(objs)
    run_dir, ckpt_dir = objs[0], objs[1]
    accelerator.wait_for_everyone()

    data_cfg = cfg.get("data", {})
    dataset_name = str(data_cfg.get("dataset", "cifar10")).lower()
    train_bs = int(data_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 128)))
    eval_bs = int(cfg.get("eval", {}).get("batch_size", train_bs))

    train_loader, class_names = build_cifar10_loader(
        dataset=dataset_name,
        data_root=data_cfg.get("root", "./data"),
        split="train",
        image_size=int(data_cfg.get("image_size", 32)),
        batch_size=train_bs,
        num_workers=int(data_cfg.get("num_workers", 8)),
        val_from_train=bool(data_cfg.get("val_from_train", False)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        shuffle=True,
        drop_last=True,
        download=bool(data_cfg.get("download", True)),
        use_fake_data=bool(data_cfg.get("use_fake_data", False)),
        fake_train_size=int(data_cfg.get("fake_train_size", 8192)),
        fake_eval_size=int(data_cfg.get("fake_eval_size", 1024)),
        two_view=False,
        aug_strength="light",
        target_source="view1",
        sun_source=str(data_cfg.get("sun_source", "hf")),
        sun_hf_dataset=str(data_cfg.get("sun_hf_dataset", "dpdl-benchmark/sun397")),
        sun_hf_cache_dir=data_cfg.get("sun_hf_cache_dir"),
        sun_hf_image_key=str(data_cfg.get("sun_hf_image_key", "image")),
        sun_hf_label_key=str(data_cfg.get("sun_hf_label_key", "label")),
        sun_max_train_samples=int(data_cfg.get("sun_max_train_samples", 0)),
        sun_max_eval_samples=int(data_cfg.get("sun_max_eval_samples", 0)),
    )

    val_loader, _ = build_cifar10_loader(
        dataset=dataset_name,
        data_root=data_cfg.get("root", "./data"),
        split="val",
        image_size=int(data_cfg.get("image_size", 32)),
        batch_size=eval_bs,
        num_workers=int(data_cfg.get("num_workers", 8)),
        val_from_train=bool(data_cfg.get("val_from_train", False)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        shuffle=False,
        drop_last=False,
        download=bool(data_cfg.get("download", True)),
        use_fake_data=bool(data_cfg.get("use_fake_data", False)),
        fake_train_size=int(data_cfg.get("fake_train_size", 8192)),
        fake_eval_size=int(data_cfg.get("fake_eval_size", 1024)),
        sun_source=str(data_cfg.get("sun_source", "hf")),
        sun_hf_dataset=str(data_cfg.get("sun_hf_dataset", "dpdl-benchmark/sun397")),
        sun_hf_cache_dir=data_cfg.get("sun_hf_cache_dir"),
        sun_hf_image_key=str(data_cfg.get("sun_hf_image_key", "image")),
        sun_hf_label_key=str(data_cfg.get("sun_hf_label_key", "label")),
        sun_max_train_samples=int(data_cfg.get("sun_max_train_samples", 0)),
        sun_max_eval_samples=int(data_cfg.get("sun_max_eval_samples", 0)),
    )

    test_loader, _ = build_cifar10_loader(
        dataset=dataset_name,
        data_root=data_cfg.get("root", "./data"),
        split="test",
        image_size=int(data_cfg.get("image_size", 32)),
        batch_size=eval_bs,
        num_workers=int(data_cfg.get("num_workers", 8)),
        val_from_train=False,
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        shuffle=False,
        drop_last=False,
        download=bool(data_cfg.get("download", True)),
        use_fake_data=bool(data_cfg.get("use_fake_data", False)),
        fake_train_size=int(data_cfg.get("fake_train_size", 8192)),
        fake_eval_size=int(data_cfg.get("fake_eval_size", 1024)),
        sun_source=str(data_cfg.get("sun_source", "hf")),
        sun_hf_dataset=str(data_cfg.get("sun_hf_dataset", "dpdl-benchmark/sun397")),
        sun_hf_cache_dir=data_cfg.get("sun_hf_cache_dir"),
        sun_hf_image_key=str(data_cfg.get("sun_hf_image_key", "image")),
        sun_hf_label_key=str(data_cfg.get("sun_hf_label_key", "label")),
        sun_max_train_samples=int(data_cfg.get("sun_max_train_samples", 0)),
        sun_max_eval_samples=int(data_cfg.get("sun_max_eval_samples", 0)),
    )

    if accelerator.is_main_process:
        save_json(
            {
                "templates": cfg.get("text", {}).get("prompt_templates", ["a photo of a {class}"]),
                "class_names": list(class_names),
                "dataset": dataset_name,
                "note": "CIFAR lightweight text alignment with learnable prototypes.",
            },
            os.path.join(run_dir, "text_prompts.json"),
        )

    model = CifarTradeoffModel(cfg, num_classes=len(class_names)).to(device)
    backbone_name = str(cfg.get("model", {}).get("backbone", "resnet18"))

    shared_mode = str(cfg.get("train", {}).get("shared_params", "backbone"))
    shared_params, aux_params = _build_shared_and_aux_params(model, shared_mode=shared_mode)
    shared_param_names = _build_shared_param_names(
        model=model,
        shared_params=shared_params,
        shared_mode=shared_mode,
    )
    probe_group_to_indices, probe_group_to_depth, probe_ordered_groups = _build_probe_groups(shared_param_names)

    all_trainable = [p for p in model.parameters() if p.requires_grad]
    if len(all_trainable) == 0:
        raise RuntimeError("No trainable parameters in model.")

    optimizer = torch.optim.AdamW(
        all_trainable,
        lr=float(cfg.get("optim", {}).get("lr", 3e-4)),
        weight_decay=float(cfg.get("optim", {}).get("weight_decay", 1e-4)),
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    mode, steps, lambda_txt, lambda_rec, strategy = _resolve_training_mode(cfg)
    grad_norm_mode = _resolve_grad_norm_mode(cfg, mode=mode)
    train_cfg = cfg.get("train", {})
    cagrad_beta = float(cfg.get("train", {}).get("cagrad_beta", 0.5))
    cagrad_conflict_only_raw = train_cfg.get("cagrad_conflict_only", False)
    if isinstance(cagrad_conflict_only_raw, bool):
        cagrad_conflict_only = cagrad_conflict_only_raw
    else:
        cagrad_conflict_only = str(cagrad_conflict_only_raw).lower() in {"1", "true", "on", "yes"}
    cagrad_conflict_threshold = float(train_cfg.get("cagrad_conflict_threshold", 0.0))
    cagrad_nonconflict_merge = str(train_cfg.get("cagrad_nonconflict_merge", "cagrad")).lower()
    if cagrad_nonconflict_merge not in {"cagrad", "sum", "avg", "average"}:
        raise ValueError(
            f"Unsupported train.cagrad_nonconflict_merge={cagrad_nonconflict_merge}. "
            "Use one of: cagrad|sum|avg."
        )
    cagrad_adaptive_nonconflict_merge = str(train_cfg.get("cagrad_adaptive_nonconflict_merge", "sum")).lower()
    if cagrad_adaptive_nonconflict_merge not in {"cagrad", "sum", "avg", "average"}:
        raise ValueError(
            f"Unsupported train.cagrad_adaptive_nonconflict_merge={cagrad_adaptive_nonconflict_merge}. "
            "Use one of: cagrad|sum|avg."
        )
    cagrad_adaptive_conflict_threshold = float(train_cfg.get("cagrad_adaptive_conflict_threshold", 0.0))
    cagrad_adaptive_strength = float(train_cfg.get("cagrad_adaptive_strength", 1.0))
    cagrad_adaptive_power = float(train_cfg.get("cagrad_adaptive_power", 1.0))
    cagrad_adaptive_beta_cap = float(train_cfg.get("cagrad_adaptive_beta_cap", 1.0))
    cagrad_adaptive_online_raw = train_cfg.get("cagrad_adaptive_online_beta", False)
    if isinstance(cagrad_adaptive_online_raw, bool):
        cagrad_adaptive_online_beta = cagrad_adaptive_online_raw
    else:
        cagrad_adaptive_online_beta = str(cagrad_adaptive_online_raw).lower() in {"1", "true", "on", "yes"}
    cagrad_adaptive_online_lr = float(train_cfg.get("cagrad_adaptive_online_lr", 0.1))
    if cagrad_adaptive_strength < 0.0:
        raise ValueError(
            f"Unsupported train.cagrad_adaptive_strength={cagrad_adaptive_strength}. "
            "Must be >= 0."
        )
    if cagrad_adaptive_power <= 0.0:
        raise ValueError(
            f"Unsupported train.cagrad_adaptive_power={cagrad_adaptive_power}. "
            "Must be > 0."
        )
    if not (0.0 <= cagrad_adaptive_beta_cap <= 1.0):
        raise ValueError(
            f"Unsupported train.cagrad_adaptive_beta_cap={cagrad_adaptive_beta_cap}. "
            "Must be in [0, 1]."
        )
    if not (0.0 <= cagrad_adaptive_online_lr <= 1.0):
        raise ValueError(
            f"Unsupported train.cagrad_adaptive_online_lr={cagrad_adaptive_online_lr}. "
            "Must be in [0, 1]."
        )
    grad_norm_scope, grad_norm_indices, grad_norm_conflict_only, grad_norm_layers = _resolve_grad_norm_plan(
        cfg=cfg,
        mode=mode,
        grad_norm_mode=grad_norm_mode,
        model=model,
        shared_params=shared_params,
        shared_mode=shared_mode,
    )
    cagrad_adaptive_beta, cagrad_adaptive_groups, cagrad_adaptive_scope, cagrad_adaptive_layers = _resolve_cagrad_adaptive_plan(
        cfg=cfg,
        mode=mode,
        strategy=strategy,
        model=model,
        shared_params=shared_params,
        shared_mode=shared_mode,
    )
    cagrad_adaptive_state: Optional[Dict[str, float]] = {} if (cagrad_adaptive_beta and cagrad_adaptive_online_beta) else None
    saop_groups, saop_scope, saop_layers = _resolve_saop_plan(
        cfg=cfg,
        mode=mode,
        strategy=strategy,
        model=model,
        shared_params=shared_params,
        shared_mode=shared_mode,
    )
    laga_groups = _resolve_laga_plan(
        mode=mode,
        strategy=strategy,
        model=model,
        shared_params=shared_params,
        shared_mode=shared_mode,
    )
    laga_eps = float(train_cfg.get("laga_eps", 1e-8))
    if laga_eps <= 0.0:
        raise ValueError(f"Unsupported train.laga_eps={laga_eps}. Must be > 0.")
    laga_conflict_threshold = float(train_cfg.get("laga_conflict_threshold", 0.0))
    if not (-1.0 <= laga_conflict_threshold <= 1.0):
        raise ValueError(
            f"Unsupported train.laga_conflict_threshold={laga_conflict_threshold}. Must be in [-1, 1]."
        )
    laga_restore_ratio = float(train_cfg.get("laga_restore_ratio", 0.0))
    if laga_restore_ratio < 0.0:
        raise ValueError(
            f"Unsupported train.laga_restore_ratio={laga_restore_ratio}. Must be >= 0."
        )
    laga_alpha_mode = str(train_cfg.get("laga_alpha_mode", "fixed")).lower()
    if laga_alpha_mode not in {"fixed", "ratio", "adaptive_ratio"}:
        raise ValueError(
            f"Unsupported train.laga_alpha_mode={laga_alpha_mode}. "
            "Use one of: fixed|ratio."
        )
    laga_alpha_power = float(train_cfg.get("laga_alpha_power", 1.0))
    if laga_alpha_power <= 0.0:
        raise ValueError(
            f"Unsupported train.laga_alpha_power={laga_alpha_power}. Must be > 0."
        )
    laga_alpha_min = float(train_cfg.get("laga_alpha_min", 1.0))
    laga_alpha_max = float(train_cfg.get("laga_alpha_max", 1.0))
    if laga_alpha_min < 0.0:
        raise ValueError(
            f"Unsupported train.laga_alpha_min={laga_alpha_min}. Must be >= 0."
        )
    if laga_alpha_max < laga_alpha_min:
        raise ValueError(
            "Unsupported LAGA alpha range: train.laga_alpha_max must be >= train.laga_alpha_min."
        )
    ma_laga_align_gamma = float(train_cfg.get("ma_laga_align_gamma", 0.5))
    if ma_laga_align_gamma < 0.0:
        raise ValueError(
            f"Unsupported train.ma_laga_align_gamma={ma_laga_align_gamma}. Must be >= 0."
        )
    ma_laga_norm_restore_raw = train_cfg.get("ma_laga_norm_restore", True)
    if isinstance(ma_laga_norm_restore_raw, bool):
        ma_laga_norm_restore = ma_laga_norm_restore_raw
    else:
        ma_laga_norm_restore = str(ma_laga_norm_restore_raw).lower() in {"1", "true", "on", "yes"}
    ma_laga_eps = float(train_cfg.get("ma_laga_eps", 1e-8))
    if ma_laga_eps <= 0.0:
        raise ValueError(
            f"Unsupported train.ma_laga_eps={ma_laga_eps}. Must be > 0."
        )

    saop_eps = float(train_cfg.get("saop_eps", 1e-8))
    saop_log_norm_ratio_raw = train_cfg.get("saop_log_norm_ratio", False)
    if isinstance(saop_log_norm_ratio_raw, bool):
        saop_log_norm_ratio = saop_log_norm_ratio_raw
    else:
        saop_log_norm_ratio = str(saop_log_norm_ratio_raw).lower() in {"1", "true", "on", "yes"}
    if saop_eps <= 0.0:
        raise ValueError(
            f"Unsupported train.saop_eps={saop_eps}. Must be > 0."
        )
    lambda_var = float(train_cfg.get("lambda_var", 0.0))
    var_gamma = float(train_cfg.get("var_gamma", 1.0))
    var_eps = float(train_cfg.get("var_eps", 1e-4))
    if lambda_var < 0.0:
        raise ValueError(
            f"Unsupported train.lambda_var={lambda_var}. Must be >= 0."
        )
    if var_eps <= 0.0:
        raise ValueError(
            f"Unsupported train.var_eps={var_eps}. Must be > 0."
        )
    variance_loss_fn = FeatureVarianceLoss(gamma=var_gamma, eps=var_eps)
    lambda_gbvc = float(train_cfg.get("lambda_gbvc", 0.0))
    gbvc_nu = float(train_cfg.get("gbvc_nu", 1.0))
    gbvc_eps = float(train_cfg.get("gbvc_eps", 1e-8))
    if lambda_gbvc < 0.0:
        raise ValueError(
            f"Unsupported train.lambda_gbvc={lambda_gbvc}. Must be >= 0."
        )
    if gbvc_eps <= 0.0:
        raise ValueError(
            f"Unsupported train.gbvc_eps={gbvc_eps}. Must be > 0."
        )

    log_cfg = cfg.get("log", {})
    log_every = int(log_cfg.get("every", 10))
    cos_every = int(log_cfg.get("cos_every", 10))
    save_every = int(log_cfg.get("save_every", 200))
    eval_every = int(log_cfg.get("eval_every", 200))
    probe_every = int(train_cfg.get("probe_every", log_cfg.get("probe_every", 0)))
    probe_until = int(train_cfg.get("probe_until", steps))

    grad_norm_balance_every = int(train_cfg.get("grad_norm_balance_every", 0))
    grad_norm_balance_ema = float(train_cfg.get("grad_norm_balance_ema", 0.9))
    grad_norm_balance_power = float(train_cfg.get("grad_norm_balance_power", 1.0))
    grad_norm_balance_min_scale = float(train_cfg.get("grad_norm_balance_min_scale", 0.1))
    grad_norm_balance_max_scale = float(train_cfg.get("grad_norm_balance_max_scale", 100.0))
    if grad_norm_balance_every < 0:
        raise ValueError("train.grad_norm_balance_every must be >= 0")
    if not (0.0 <= grad_norm_balance_ema < 1.0):
        raise ValueError("train.grad_norm_balance_ema must be in [0, 1).")
    if grad_norm_balance_power <= 0.0:
        raise ValueError("train.grad_norm_balance_power must be > 0.")
    if grad_norm_balance_min_scale <= 0.0 or grad_norm_balance_max_scale <= 0.0:
        raise ValueError("train.grad_norm_balance_min_scale/max_scale must be > 0.")
    if grad_norm_balance_min_scale > grad_norm_balance_max_scale:
        raise ValueError("train.grad_norm_balance_min_scale must be <= max_scale.")

    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    layer_probe_summary_file = os.path.join(run_dir, "layer_probe_summary.jsonl")
    layer_probe_dir = os.path.join(run_dir, "layer_probe")
    cos_curve = []
    temperature = float(cfg.get("text", {}).get("temperature", 0.07))

    dyn_rec_scale = 1.0
    last_target_rec_scale = 1.0
    last_gu_norm = 0.0
    last_gg_norm = 0.0

    if accelerator.is_main_process:
        save_json(
            {
                "strategy": strategy,
                "dataset": dataset_name,
                "backbone": backbone_name,
                "mode": mode,
                "lambda_txt": lambda_txt,
                "lambda_rec": lambda_rec,
                "cagrad_beta": cagrad_beta,
                "cagrad_conflict_only": bool(cagrad_conflict_only),
                "cagrad_conflict_threshold": cagrad_conflict_threshold,
                "cagrad_nonconflict_merge": cagrad_nonconflict_merge,
                "cagrad_adaptive_beta": bool(cagrad_adaptive_beta),
                "cagrad_adaptive_scope": cagrad_adaptive_scope,
                "cagrad_adaptive_layers": cagrad_adaptive_layers,
                "cagrad_adaptive_nonconflict_merge": cagrad_adaptive_nonconflict_merge,
                "cagrad_adaptive_conflict_threshold": cagrad_adaptive_conflict_threshold,
                "cagrad_adaptive_strength": cagrad_adaptive_strength,
                "cagrad_adaptive_power": cagrad_adaptive_power,
                "cagrad_adaptive_beta_cap": cagrad_adaptive_beta_cap,
                "cagrad_adaptive_online_beta": bool(cagrad_adaptive_online_beta),
                "cagrad_adaptive_online_lr": cagrad_adaptive_online_lr,
                "cagrad_adaptive_num_groups": int(len(cagrad_adaptive_groups)),
                "saop_scope": saop_scope,
                "saop_layers": saop_layers,
                "saop_num_groups": int(len(saop_groups)),
                "saop_eps": saop_eps,
                "saop_log_norm_ratio": bool(saop_log_norm_ratio),
                "laga_num_groups": int(len(laga_groups)),
                "laga_eps": laga_eps,
                "laga_conflict_threshold": laga_conflict_threshold,
                "laga_restore_ratio": laga_restore_ratio,
                "laga_alpha_mode": laga_alpha_mode,
                "laga_alpha_power": laga_alpha_power,
                "laga_alpha_min": laga_alpha_min,
                "laga_alpha_max": laga_alpha_max,
                "ma_laga_align_gamma": ma_laga_align_gamma,
                "ma_laga_norm_restore": bool(ma_laga_norm_restore),
                "ma_laga_eps": ma_laga_eps,
                "lambda_var": lambda_var,
                "var_gamma": var_gamma,
                "var_eps": var_eps,
                "lambda_gbvc": lambda_gbvc,
                "gbvc_nu": gbvc_nu,
                "gbvc_eps": gbvc_eps,
                "grad_norm_mode": grad_norm_mode,
                "grad_norm_scope": grad_norm_scope,
                "grad_norm_conflict_only": bool(grad_norm_conflict_only),
                "grad_norm_layers": grad_norm_layers,
                "grad_norm_num_indices": None if grad_norm_indices is None else int(len(grad_norm_indices)),
                "grad_norm_balance_every": grad_norm_balance_every,
                "grad_norm_balance_ema": grad_norm_balance_ema,
                "grad_norm_balance_power": grad_norm_balance_power,
                "grad_norm_balance_min_scale": grad_norm_balance_min_scale,
                "grad_norm_balance_max_scale": grad_norm_balance_max_scale,
                "probe_every": probe_every,
                "probe_until": probe_until,
                "probe_num_groups": int(len(probe_ordered_groups)),
            },
            os.path.join(run_dir, "train_setup.json"),
        )

    eval_split = str(cfg.get("eval", {}).get("split", "val")).lower()
    eval_loader = test_loader if eval_split == "test" else val_loader

    train_start = time.time()

    if steps <= 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                os.path.join(ckpt_dir, "latest.pt"),
                model=accelerator.unwrap_model(model),
                optimizer=optimizer,
                step=0,
                cfg=cfg,
            )
            run_eval(
                cfg=cfg,
                model=accelerator.unwrap_model(model),
                eval_loader=eval_loader,
                device=device,
                run_dir=run_dir,
                step=0,
                class_names=class_names,
                dataset_name=dataset_name,
            )
        accelerator.wait_for_everyone()

    train_iter = cycle_loader(train_loader)
    pbar = tqdm(range(1, steps + 1), desc="train_cifar10", disable=not accelerator.is_local_main_process)

    for step in pbar:
        batch = make_batch_dict(next(train_iter))
        batch = to_device(batch, device)

        model.train()
        images = batch["images"]
        images_target = batch.get("images_target", images)

        out = model(images)
        recon = out["recon"]
        if recon.shape[-2:] != images_target.shape[-2:]:
            images_target = F.interpolate(images_target, size=recon.shape[-2:], mode="bilinear", align_corners=False)

        Lu, txt_extra = text_prototype_loss(
            z_txt=out["z_txt"],
            labels=batch["labels"],
            prototypes=model.text_prototypes,
            temperature=temperature,
        )
        Lg = F.mse_loss(recon, images_target)
        Lvar = variance_loss_fn(out["feat"]) if lambda_var > 0.0 else Lu.new_zeros(())
        feat_for_gbvc = out["feat"]
        if feat_for_gbvc.ndim == 4:
            feat_for_gbvc = F.adaptive_avg_pool2d(feat_for_gbvc, output_size=(1, 1)).flatten(1)
        elif feat_for_gbvc.ndim != 2:
            raise ValueError(
                f"GBVC expects [B, D] or [B, C, H, W], got shape={tuple(feat_for_gbvc.shape)}"
            )
        # L_GBVC = mean_j [max(0, nu - sigma_j)]^2 where sigma_j = sqrt(var_j + eps).
        sigma = torch.sqrt(torch.var(feat_for_gbvc, dim=0, unbiased=False) + gbvc_eps)
        Lgbvc = torch.relu(gbvc_nu - sigma).pow(2).mean() if lambda_gbvc > 0.0 else Lu.new_zeros(())
        var_term = lambda_var * Lvar
        gbvc_term = lambda_gbvc * Lgbvc

        # Dynamic Lu/Lg balancing on shared params:
        # keep lambda_txt fixed and adapt effective lambda_rec by grad-norm ratio.
        if (
            mode == "joint"
            and grad_norm_balance_every > 0
            and len(shared_params) > 0
            and lambda_rec > 0.0
            and (step % grad_norm_balance_every == 0)
        ):
            raw_gu = torch.autograd.grad(Lu, shared_params, retain_graph=True, allow_unused=True)
            raw_gg = torch.autograd.grad(Lg, shared_params, retain_graph=True, allow_unused=True)
            gu_for_balance = _materialize_grads(raw_gu, list(shared_params))
            gg_for_balance = _materialize_grads(raw_gg, list(shared_params))
            gu_norm = _global_grad_l2_norm(gu_for_balance)
            gg_norm = _global_grad_l2_norm(gg_for_balance)
            target_scale = (lambda_txt * gu_norm) / max(lambda_rec * gg_norm, 1e-12)
            target_scale = float(target_scale ** grad_norm_balance_power)
            target_scale = float(
                min(max(target_scale, grad_norm_balance_min_scale), grad_norm_balance_max_scale)
            )
            dyn_rec_scale = float(grad_norm_balance_ema) * dyn_rec_scale + (
                1.0 - float(grad_norm_balance_ema)
            ) * target_scale
            last_target_rec_scale = target_scale
            last_gu_norm = gu_norm
            last_gg_norm = gg_norm

        eff_lambda_rec = float(lambda_rec * dyn_rec_scale)
        total = lambda_txt * Lu + eff_lambda_rec * Lg + var_term + gbvc_term

        should_probe = (
            mode == "joint"
            and probe_every > 0
            and len(shared_params) > 0
            and step <= probe_until
            and (step % probe_every == 0)
        )
        probe_stats: Optional[Dict[str, float]] = None
        probe_csv_path = ""
        if should_probe:
            raw_gu_probe = torch.autograd.grad(Lu, shared_params, retain_graph=True, allow_unused=True)
            raw_gg_probe = torch.autograd.grad(Lg, shared_params, retain_graph=True, allow_unused=True)
            gu_probe = _materialize_grads(raw_gu_probe, list(shared_params))
            gg_probe = _materialize_grads(raw_gg_probe, list(shared_params))
            probe_rows, probe_stats_local = _collect_layer_probe_rows(
                step=step,
                g_u=gu_probe,
                g_g=gg_probe,
                group_to_indices=probe_group_to_indices,
                group_to_depth=probe_group_to_depth,
                ordered_groups=probe_ordered_groups,
            )
            probe_stats_local["global_gu_norm"] = _global_grad_l2_norm(gu_probe)
            probe_stats_local["global_gg_norm"] = _global_grad_l2_norm(gg_probe)
            probe_stats_local["global_gu_over_gg"] = float(
                probe_stats_local["global_gu_norm"] / max(probe_stats_local["global_gg_norm"], 1e-12)
            )
            probe_stats = probe_stats_local

            if accelerator.is_main_process and len(probe_rows) > 0:
                probe_csv_path = os.path.join(layer_probe_dir, f"step_{step:07d}_layerwise.csv")
                _save_layer_probe_csv(probe_csv_path, probe_rows)
                probe_stats["probe_csv"] = probe_csv_path
                append_jsonl(layer_probe_summary_file, probe_stats)
                accelerator.print(
                    f"[probe] step={step} mean_cos={probe_stats['mean_cosine']:.4f} "
                    f"mean_neg={probe_stats['mean_neg_ratio']:.4f} "
                    f"depth_pearson={probe_stats['depth_cos_pearson']:.4f} "
                    f"depth_spearman={probe_stats['depth_cos_spearman']:.4f} "
                    f"csv={probe_csv_path}"
                )

        optimizer.zero_grad(set_to_none=True)
        cos = 0.0
        if strategy == "pcgrad":
            cos = apply_conflict_aware(
                loss_txt=Lu,
                loss_rec=Lg,
                lora_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=eff_lambda_rec,
                grad_norm_mode=grad_norm_mode,
                grad_norm_indices=grad_norm_indices,
                grad_norm_conflict_only=grad_norm_conflict_only,
                extra_loss=(var_term + gbvc_term) if (lambda_var > 0.0 or lambda_gbvc > 0.0) else None,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        elif strategy == "cagrad":
            cos = apply_cagrad(
                loss_txt=Lu,
                loss_rec=Lg,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=eff_lambda_rec,
                beta=cagrad_beta,
                conflict_only=cagrad_conflict_only,
                conflict_threshold=cagrad_conflict_threshold,
                nonconflict_merge=cagrad_nonconflict_merge,
                adaptive_beta=cagrad_adaptive_beta,
                adaptive_group_to_indices=cagrad_adaptive_groups if cagrad_adaptive_beta else None,
                adaptive_nonconflict_merge=cagrad_adaptive_nonconflict_merge,
                adaptive_conflict_threshold=cagrad_adaptive_conflict_threshold,
                adaptive_strength=cagrad_adaptive_strength,
                adaptive_power=cagrad_adaptive_power,
                adaptive_beta_cap=cagrad_adaptive_beta_cap,
                adaptive_online_beta=cagrad_adaptive_online_beta,
                adaptive_online_lr=cagrad_adaptive_online_lr,
                adaptive_state=cagrad_adaptive_state,
                grad_norm_mode=grad_norm_mode,
                grad_norm_indices=grad_norm_indices,
                grad_norm_conflict_only=grad_norm_conflict_only,
                extra_loss=(var_term + gbvc_term) if (lambda_var > 0.0 or lambda_gbvc > 0.0) else None,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        elif strategy == "saop":
            cos = apply_saop(
                loss_txt=Lu,
                loss_rec=Lg,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=eff_lambda_rec,
                deep_group_to_indices=saop_groups,
                eps=saop_eps,
                log_norm_ratio=saop_log_norm_ratio,
                grad_norm_mode=grad_norm_mode,
                grad_norm_indices=grad_norm_indices,
                grad_norm_conflict_only=grad_norm_conflict_only,
                extra_loss=(var_term + gbvc_term) if (lambda_var > 0.0 or lambda_gbvc > 0.0) else None,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        elif strategy == "laga":
            cos = apply_laga_objective(
                loss_txt=Lu,
                loss_rec=Lg,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=eff_lambda_rec,
                group_to_indices=laga_groups,
                eps=laga_eps,
                conflict_threshold=laga_conflict_threshold,
                restore_ratio=laga_restore_ratio,
                alpha_mode=laga_alpha_mode,
                alpha_power=laga_alpha_power,
                alpha_min=laga_alpha_min,
                alpha_max=laga_alpha_max,
                grad_norm_mode=grad_norm_mode,
                grad_norm_indices=grad_norm_indices,
                grad_norm_conflict_only=grad_norm_conflict_only,
                extra_loss=(var_term + gbvc_term) if (lambda_var > 0.0 or lambda_gbvc > 0.0) else None,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        elif strategy == "ma_laga":
            cos = apply_ma_laga_objective(
                loss_txt=Lu,
                loss_rec=Lg,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=eff_lambda_rec,
                group_to_indices=laga_groups,
                align_gamma=ma_laga_align_gamma,
                norm_restore=ma_laga_norm_restore,
                eps=ma_laga_eps,
                grad_norm_mode=grad_norm_mode,
                grad_norm_indices=grad_norm_indices,
                grad_norm_conflict_only=grad_norm_conflict_only,
                extra_loss=(var_term + gbvc_term) if (lambda_var > 0.0 or lambda_gbvc > 0.0) else None,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        else:
            manual_naive = (mode == "joint") and (grad_norm_mode != "none") and (len(shared_params) > 0)
            if manual_naive:
                cos = apply_naive(
                    loss_txt=Lu,
                    loss_rec=Lg,
                    shared_params=shared_params,
                    aux_params=aux_params,
                    lambda_txt=lambda_txt,
                    lambda_rec=eff_lambda_rec,
                    grad_norm_mode=grad_norm_mode,
                    grad_norm_indices=grad_norm_indices,
                    grad_norm_conflict_only=grad_norm_conflict_only,
                    extra_loss=(var_term + gbvc_term) if (lambda_var > 0.0 or lambda_gbvc > 0.0) else None,
                )
                if accelerator.num_processes > 1:
                    for p in list(shared_params) + list(aux_params):
                        if p.grad is not None:
                            p.grad = accelerator.reduce(p.grad, reduction="mean")
                optimizer.step()
            else:
                if step % cos_every == 0 and len(shared_params) > 0:
                    cos = compute_grad_cosine(Lu, Lg, shared_params)
                accelerator.backward(total)
                optimizer.step()

        cos_global = _dist_mean_scalar(accelerator, cos, device)
        Lu_global = _dist_mean_scalar(accelerator, Lu, device)
        Lg_global = _dist_mean_scalar(accelerator, Lg, device)
        Lvar_global = _dist_mean_scalar(accelerator, Lvar, device)
        Lgbvc_global = _dist_mean_scalar(accelerator, Lgbvc, device)
        total_global = _dist_mean_scalar(accelerator, total, device)
        txt_acc_global = _dist_mean_scalar(accelerator, txt_extra.get("txt_acc", 0.0), device)

        if step % cos_every == 0 and accelerator.is_main_process:
            cos_curve.append({"step": step, "cos": cos_global})

        row = {
            "step": step,
            "Lu": Lu_global,
            "Lg": Lg_global,
            "Lvar": Lvar_global,
            "Lgbvc": Lgbvc_global,
            "loss_txt": Lu_global,
            "loss_g": Lg_global,
            "loss_var": Lvar_global,
            "loss_gbvc": Lgbvc_global,
            "loss_total": total_global,
            "total": total_global,
            "cos": cos_global,
            "txt_acc": txt_acc_global,
            "recon_mse": Lg_global,
            "strategy": strategy,
            "mode": mode,
            "lambda_txt": float(lambda_txt),
            "lambda_rec": float(lambda_rec),
            "effective_lambda_rec": float(eff_lambda_rec),
            "dyn_rec_scale": float(dyn_rec_scale),
            "target_rec_scale": float(last_target_rec_scale),
            "raw_lu_over_lg": float(Lu_global / max(Lg_global, 1e-12)),
            "weighted_lu_over_lg": float((lambda_txt * Lu_global) / max(eff_lambda_rec * Lg_global, 1e-12)),
            "balance_gu_norm": float(last_gu_norm),
            "balance_gg_norm": float(last_gg_norm),
            "balance_gu_over_gg": float(last_gu_norm / max(last_gg_norm, 1e-12)),
        }
        if probe_stats is not None:
            row["probe_mean_cosine"] = float(probe_stats["mean_cosine"])
            row["probe_mean_neg_ratio"] = float(probe_stats["mean_neg_ratio"])
            row["probe_depth_cos_pearson"] = float(probe_stats["depth_cos_pearson"])
            row["probe_depth_cos_spearman"] = float(probe_stats["depth_cos_spearman"])
            row["probe_global_gu_norm"] = float(probe_stats["global_gu_norm"])
            row["probe_global_gg_norm"] = float(probe_stats["global_gg_norm"])
            row["probe_global_gu_over_gg"] = float(probe_stats["global_gu_over_gg"])
            row["probe_weighted_gu_over_gg"] = float(
                (float(lambda_txt) * float(probe_stats["global_gu_norm"]))
                / max(float(eff_lambda_rec) * float(probe_stats["global_gg_norm"]), 1e-12)
            )
            row["probe_csv"] = str(probe_csv_path)

        if accelerator.is_main_process and (step % log_every == 0 or step == 1 or step == steps):
            append_jsonl(metrics_file, row)
            pbar.set_postfix(
                Lu=f"{row['Lu']:.4f}",
                Lg=f"{row['Lg']:.4f}",
                Lvar=f"{row['Lvar']:.4f}",
                Lgbvc=f"{row['Lgbvc']:.4f}",
                total=f"{row['total']:.4f}",
                cos=f"{row['cos']:.4f}",
                lrec=f"{row['effective_lambda_rec']:.3f}",
                w_ratio=f"{row['weighted_lu_over_lg']:.2f}",
            )

        if step % save_every == 0 or step == steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    os.path.join(ckpt_dir, "latest.pt"),
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    step=step,
                    cfg=cfg,
                )
            accelerator.wait_for_everyone()

        if step % eval_every == 0 or step == steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                run_eval(
                    cfg=cfg,
                    model=accelerator.unwrap_model(model),
                    eval_loader=eval_loader,
                    device=device,
                    run_dir=run_dir,
                    step=step,
                    class_names=class_names,
                    dataset_name=dataset_name,
                )
            accelerator.wait_for_everyone()

    elapsed_sec = float(time.time() - train_start)

    if accelerator.is_main_process:
        cos_values = [x["cos"] for x in cos_curve]
        if len(cos_values) == 0:
            cos_mean = 0.0
            cos_neg_ratio = 0.0
        else:
            cos_mean = float(sum(cos_values) / len(cos_values))
            cos_neg_ratio = float(sum(1 for c in cos_values if c < 0) / len(cos_values))

        summary = {
            "run_dir": run_dir,
            "dataset": cfg.get("data", {}).get("dataset", "cifar10"),
            "backbone": backbone_name,
            "strategy": strategy,
            "grad_strategy": strategy,
            "mode": mode,
            "seed": seed,
            "lambda_txt": lambda_txt,
            "lambda_rec": lambda_rec,
            "num_trainable": count_parameters([p for p in model.parameters() if p.requires_grad]),
            "num_shared": count_parameters(shared_params),
            "num_aux": count_parameters(aux_params),
            "shared_params_mode": shared_mode,
            "cagrad_beta": cagrad_beta,
            "cagrad_conflict_only": bool(cagrad_conflict_only),
            "cagrad_conflict_threshold": cagrad_conflict_threshold,
            "cagrad_nonconflict_merge": cagrad_nonconflict_merge,
            "cagrad_adaptive_beta": bool(cagrad_adaptive_beta),
            "cagrad_adaptive_scope": cagrad_adaptive_scope,
            "cagrad_adaptive_layers": cagrad_adaptive_layers,
            "cagrad_adaptive_nonconflict_merge": cagrad_adaptive_nonconflict_merge,
            "cagrad_adaptive_conflict_threshold": cagrad_adaptive_conflict_threshold,
            "cagrad_adaptive_strength": cagrad_adaptive_strength,
            "cagrad_adaptive_power": cagrad_adaptive_power,
            "cagrad_adaptive_beta_cap": cagrad_adaptive_beta_cap,
            "cagrad_adaptive_online_beta": bool(cagrad_adaptive_online_beta),
            "cagrad_adaptive_online_lr": cagrad_adaptive_online_lr,
            "cagrad_adaptive_state": {} if cagrad_adaptive_state is None else {k: float(v) for k, v in cagrad_adaptive_state.items()},
            "cagrad_adaptive_num_groups": int(len(cagrad_adaptive_groups)),
            "saop_scope": saop_scope,
            "saop_layers": saop_layers,
            "saop_num_groups": int(len(saop_groups)),
            "saop_eps": saop_eps,
            "saop_log_norm_ratio": bool(saop_log_norm_ratio),
            "laga_num_groups": int(len(laga_groups)),
            "laga_eps": laga_eps,
            "laga_conflict_threshold": laga_conflict_threshold,
            "laga_restore_ratio": laga_restore_ratio,
            "laga_alpha_mode": laga_alpha_mode,
            "laga_alpha_power": laga_alpha_power,
            "laga_alpha_min": laga_alpha_min,
            "laga_alpha_max": laga_alpha_max,
            "ma_laga_align_gamma": ma_laga_align_gamma,
            "ma_laga_norm_restore": bool(ma_laga_norm_restore),
            "ma_laga_eps": ma_laga_eps,
            "lambda_var": lambda_var,
            "var_gamma": var_gamma,
            "var_eps": var_eps,
            "lambda_gbvc": lambda_gbvc,
            "gbvc_nu": gbvc_nu,
            "gbvc_eps": gbvc_eps,
            "grad_norm_mode": grad_norm_mode,
            "grad_norm_scope": grad_norm_scope,
            "grad_norm_conflict_only": bool(grad_norm_conflict_only),
            "grad_norm_layers": grad_norm_layers,
            "grad_norm_num_indices": None if grad_norm_indices is None else int(len(grad_norm_indices)),
            "grad_norm_balance_every": grad_norm_balance_every,
            "grad_norm_balance_ema": grad_norm_balance_ema,
            "grad_norm_balance_power": grad_norm_balance_power,
            "grad_norm_balance_min_scale": grad_norm_balance_min_scale,
            "grad_norm_balance_max_scale": grad_norm_balance_max_scale,
            "final_dyn_rec_scale": float(dyn_rec_scale),
            "final_effective_lambda_rec": float(lambda_rec * dyn_rec_scale),
            "probe_every": probe_every,
            "probe_until": probe_until,
            "probe_num_groups": int(len(probe_ordered_groups)),
            "cos_mean": cos_mean,
            "cos_neg_ratio": cos_neg_ratio,
            "world_size": accelerator.num_processes,
            "train_steps": steps,
            "walltime_sec": elapsed_sec,
        }
        save_json(summary, os.path.join(run_dir, "cos_summary.json"))
        save_json({"curve": cos_curve}, os.path.join(run_dir, "cos_curve.json"))

    accelerator.wait_for_everyone()
    accelerator.print(f"[train_cifar10] done. run_dir={run_dir}")


if __name__ == "__main__":
    main()
