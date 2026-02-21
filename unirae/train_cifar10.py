import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from .data_cifar10 import (
    CIFAR10_CLASS_NAMES,
    build_cifar10_loader,
    denormalize_cifar10,
    make_batch_dict,
)
from .grad_conflict import apply_cagrad, apply_conflict_aware, apply_mgda_ub, compute_grad_cosine
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


def consistency_loss(z1: torch.Tensor, z2: torch.Tensor, loss_type: str = "cosine") -> torch.Tensor:
    loss_type = str(loss_type).lower()
    if loss_type == "mse":
        return F.mse_loss(z1, z2)

    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    return 1.0 - F.cosine_similarity(z1n, z2n, dim=-1).mean()


def select_consistency_embedding(out: Dict[str, torch.Tensor], target: str) -> torch.Tensor:
    t = str(target).lower()
    if t in {"txt_head", "txt", "clip"}:
        return out["z_txt"]
    if t in {"dino", "dino_feat"}:
        return out["feat"]
    if t in {"rec", "z_rec"}:
        return out["z_rec"]
    raise ValueError(f"Unsupported consistency.target={target}. Use txt_head|clip|dino.")


def select_supcon_embedding(out: Dict[str, torch.Tensor], embed: str) -> torch.Tensor:
    e = str(embed).lower()
    if e in {"txt", "clip"}:
        return out["z_txt"]
    if e in {"shared", "feat"}:
        return out["feat"]
    raise ValueError(f"Unsupported supcon.embed={embed}. Use txt|clip|shared.")


def supervised_contrastive_loss(
    z_view1: torch.Tensor,
    z_view2: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    # 2-view SupCon: anchors are [v1; v2], positives are same-label samples except self.
    tau = max(float(tau), 1e-6)
    b = z_view1.shape[0]
    z = torch.cat([z_view1, z_view2], dim=0)
    z = F.normalize(z, dim=-1)

    y = labels.reshape(-1)
    y = torch.cat([y, y], dim=0)

    logits = torch.matmul(z, z.t()) / tau
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    valid_mask = ~eye
    pos_mask = y.unsqueeze(0).eq(y.unsqueeze(1)) & valid_mask

    exp_logits = torch.exp(logits) * valid_mask
    denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = logits - torch.log(denom)

    pos_count = pos_mask.sum(dim=1)
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_count.clamp_min(1)
    valid = pos_count > 0
    if valid.any():
        return -mean_log_prob_pos[valid].mean()
    return torch.zeros((), device=z.device, dtype=z.dtype)


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
    out_path: str,
    max_items: int,
) -> None:
    b = min(max_items, images.shape[0])
    gt = torch.clamp(denormalize_cifar10(images[:b]).cpu(), 0.0, 1.0)
    rc = torch.clamp(denormalize_cifar10(recons[:b]).cpu(), 0.0, 1.0)

    rows = []
    for i in range(b):
        rows.append(gt[i])
        rows.append(rc[i])
    grid = make_grid(rows, nrow=2)
    ensure_dir(str(Path(out_path).parent))
    save_image(grid, out_path)


def evaluate_understanding(
    model: CifarTradeoffModel,
    loader,
    device: torch.device,
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
        "class_names": list(CIFAR10_CLASS_NAMES),
    }


def evaluate_generation(
    model: CifarTradeoffModel,
    loader,
    device: torch.device,
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

            mse = F.mse_loss(recon, target)
            total_mse += float(mse.item())
            n += 1

            if save_samples and (not saved):
                _save_recon_samples(target, recon, out_path=sample_path, max_items=sample_images)
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
) -> None:
    eval_cfg = cfg.get("eval", {})
    max_batches = eval_cfg.get("max_batches", None)

    understanding = evaluate_understanding(
        model=model,
        loader=eval_loader,
        device=device,
        temperature=float(cfg.get("text", {}).get("temperature", 0.07)),
        max_batches=max_batches,
    )

    generation = evaluate_generation(
        model=model,
        loader=eval_loader,
        device=device,
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
    strategy = str(train_cfg.get("strategy", "naive"))

    if mode == "baseline":
        return mode, 0, 0.0, 0.0, "naive"
    if mode == "text_only":
        return mode, steps, lambda_txt, 0.0, "naive"
    if mode == "recon_only":
        return mode, steps, 0.0, lambda_rec, "naive"
    return mode, steps, lambda_txt, lambda_rec, strategy


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

    exp_name = args.run_name or cfg.get("experiment", {}).get("name", "cifar10_exp")
    run_dir = os.path.join(cfg.get("output", {}).get("root", "runs"), exp_name)

    if accelerator.is_main_process:
        if os.path.exists(run_dir):
            run_dir = f"{run_dir}_{now_str()}"
        ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
        save_yaml(cfg, os.path.join(run_dir, "run_config.yaml"))
        save_json(
            {
                "templates": cfg.get("text", {}).get("prompt_templates", ["a photo of a {class}"]),
                "class_names": list(CIFAR10_CLASS_NAMES),
                "note": "CIFAR-10 uses learnable text prototypes for lightweight text alignment.",
            },
            os.path.join(run_dir, "text_prompts.json"),
        )
    else:
        ckpt_dir = ""

    objs = [run_dir, ckpt_dir]
    broadcast_object_list(objs)
    run_dir, ckpt_dir = objs[0], objs[1]
    accelerator.wait_for_everyone()

    data_cfg = cfg.get("data", {})
    train_bs = int(data_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 128)))
    eval_bs = int(cfg.get("eval", {}).get("batch_size", train_bs))
    cons_cfg = cfg.get("consistency", {})
    sup_cfg = cfg.get("supcon", {})
    cons_enabled = bool(cons_cfg.get("enabled", False))
    sup_enabled = bool(sup_cfg.get("enabled", False))
    cons_two_view = bool(cons_cfg.get("two_view", cons_enabled))
    sup_two_view = bool(sup_cfg.get("two_view", sup_enabled))
    two_view_train = bool(cons_two_view or sup_two_view)
    aug_strength = str(sup_cfg.get("aug", cons_cfg.get("aug_strength", "medium"))).lower()
    if aug_strength == "weak":
        aug_strength = "light"
    recon_target_source = str(cons_cfg.get("recon_target_source", "view1"))

    train_loader, class_names = build_cifar10_loader(
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
        two_view=two_view_train,
        aug_strength=aug_strength,
        target_source=recon_target_source,
    )

    val_loader, _ = build_cifar10_loader(
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
    )

    test_loader, _ = build_cifar10_loader(
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
    )

    model = CifarTradeoffModel(cfg, num_classes=len(class_names)).to(device)

    shared_mode = str(cfg.get("train", {}).get("shared_params", "backbone"))
    shared_params, aux_params = _build_shared_and_aux_params(model, shared_mode=shared_mode)

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
    lambda_cons = float(cfg.get("train", {}).get("lambda_cons", cons_cfg.get("lambda_cons", 0.1)))
    if not cons_enabled:
        lambda_cons = 0.0
    cons_loss_type = str(cons_cfg.get("loss_type", "cosine"))
    cons_target = str(cons_cfg.get("target", "txt_head"))
    merge_cons_into_u = bool(cons_cfg.get("merge_into_understanding", True))
    lambda_sup = float(cfg.get("train", {}).get("lambda_sup", sup_cfg.get("lambda", 0.5)))
    if not sup_enabled:
        lambda_sup = 0.0
    sup_tau = float(sup_cfg.get("tau", 0.1))
    sup_embed = str(sup_cfg.get("embed", "txt"))
    merge_sup_into_u = bool(sup_cfg.get("merge_into_understanding", True))
    if sup_enabled and lambda_sup > 0.0 and not two_view_train:
        raise ValueError("SupCon requires two-view training. Set supcon.two_view=true.")

    log_cfg = cfg.get("log", {})
    log_every = int(log_cfg.get("every", 10))
    cos_every = int(log_cfg.get("cos_every", 10))
    save_every = int(log_cfg.get("save_every", 200))
    eval_every = int(log_cfg.get("eval_every", 200))

    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    cos_curve = []
    temperature = float(cfg.get("text", {}).get("temperature", 0.07))

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
            )
        accelerator.wait_for_everyone()

    train_iter = cycle_loader(train_loader)
    pbar = tqdm(range(1, steps + 1), desc="train_cifar10", disable=not accelerator.is_local_main_process)

    for step in pbar:
        batch = make_batch_dict(next(train_iter))
        batch = to_device(batch, device)

        model.train()
        images = batch["images"]
        images_view2 = batch.get("images_view2", images)
        images_target = batch.get("images_target", images)

        out = model(images)

        Lu, txt_extra = text_prototype_loss(
            z_txt=out["z_txt"],
            labels=batch["labels"],
            prototypes=model.text_prototypes,
            temperature=temperature,
        )
        Lg = F.mse_loss(out["recon"], images_target)

        need_view2 = (cons_enabled and lambda_cons > 0.0) or (sup_enabled and lambda_sup > 0.0)
        out_view2 = model(images_view2) if need_view2 else None

        if cons_enabled and lambda_cons > 0.0 and out_view2 is not None:
            z1 = select_consistency_embedding(out, cons_target)
            z2 = select_consistency_embedding(out_view2, cons_target)
            Lcons = consistency_loss(z1, z2, loss_type=cons_loss_type)
        else:
            Lcons = torch.zeros((), device=device, dtype=Lu.dtype)

        if sup_enabled and lambda_sup > 0.0 and out_view2 is not None:
            zsup1 = select_supcon_embedding(out, sup_embed)
            zsup2 = select_supcon_embedding(out_view2, sup_embed)
            Lsup = supervised_contrastive_loss(
                z_view1=zsup1,
                z_view2=zsup2,
                labels=batch["labels"],
                tau=sup_tau,
            )
        else:
            Lsup = torch.zeros((), device=device, dtype=Lu.dtype)

        Lu_for_objective = Lu
        extra_terms = []
        if lambda_cons > 0.0:
            if merge_cons_into_u and lambda_txt > 0:
                Lu_for_objective = Lu_for_objective + (lambda_cons / max(lambda_txt, 1e-12)) * Lcons
            else:
                extra_terms.append(lambda_cons * Lcons)
        if lambda_sup > 0.0:
            if merge_sup_into_u and lambda_txt > 0:
                Lu_for_objective = Lu_for_objective + (lambda_sup / max(lambda_txt, 1e-12)) * Lsup
            else:
                extra_terms.append(lambda_sup * Lsup)

        total = lambda_txt * Lu + lambda_rec * Lg + lambda_cons * Lcons + lambda_sup * Lsup

        optimizer.zero_grad(set_to_none=True)
        cos = 0.0
        extra_loss = None
        if extra_terms:
            extra_loss = extra_terms[0]
            for e in extra_terms[1:]:
                extra_loss = extra_loss + e

        if strategy in {"conflict_aware", "pcgrad"}:
            cos = apply_conflict_aware(
                loss_txt=Lu_for_objective,
                loss_rec=Lg,
                lora_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=lambda_rec,
                extra_loss=extra_loss,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        elif strategy == "mgda_ub":
            cos = apply_mgda_ub(
                loss_txt=Lu_for_objective,
                loss_rec=Lg,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=lambda_rec,
                extra_loss=extra_loss,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        elif strategy == "cagrad":
            cos = apply_cagrad(
                loss_txt=Lu_for_objective,
                loss_rec=Lg,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=lambda_rec,
                beta=float(cfg.get("train", {}).get("cagrad_beta", 0.5)),
                extra_loss=extra_loss,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        else:
            if step % cos_every == 0 and len(shared_params) > 0:
                cos = compute_grad_cosine(Lu_for_objective, Lg, shared_params)
            accelerator.backward(total)
            optimizer.step()

        cos_global = _dist_mean_scalar(accelerator, cos, device)
        Lu_global = _dist_mean_scalar(accelerator, Lu, device)
        Lg_global = _dist_mean_scalar(accelerator, Lg, device)
        Lcons_global = _dist_mean_scalar(accelerator, Lcons, device)
        Lsup_global = _dist_mean_scalar(accelerator, Lsup, device)
        total_global = _dist_mean_scalar(accelerator, total, device)
        txt_acc_global = _dist_mean_scalar(accelerator, txt_extra.get("txt_acc", 0.0), device)

        if step % cos_every == 0 and accelerator.is_main_process:
            cos_curve.append({"step": step, "cos": cos_global})

        row = {
            "step": step,
            "Lu": Lu_global,
            "Lg": Lg_global,
            "L_cons": Lcons_global,
            "L_supcon": Lsup_global,
            "loss_cons": Lcons_global,
            "loss_supcon": Lsup_global,
            "loss_txt": Lu_global,
            "loss_g": Lg_global,
            "total": total_global,
            "cos": cos_global,
            "txt_acc": txt_acc_global,
            "recon_mse": Lg_global,
        }

        if accelerator.is_main_process and (step % log_every == 0 or step == 1 or step == steps):
            append_jsonl(metrics_file, row)
            pbar.set_postfix(
                Lu=f"{row['Lu']:.4f}",
                Lg=f"{row['Lg']:.4f}",
                Lc=f"{row['L_cons']:.4f}",
                Lsup=f"{row['L_supcon']:.4f}",
                total=f"{row['total']:.4f}",
                cos=f"{row['cos']:.4f}",
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
            "backbone": cfg.get("model", {}).get("backbone", "resnet18"),
            "strategy": strategy,
            "mode": mode,
            "seed": seed,
            "lambda_txt": lambda_txt,
            "lambda_rec": lambda_rec,
            "lambda_cons": lambda_cons,
            "lambda_sup": lambda_sup,
            "consistency_enabled": cons_enabled,
            "consistency_two_view": two_view_train,
            "consistency_target": cons_target,
            "consistency_loss_type": cons_loss_type,
            "consistency_merge_into_understanding": merge_cons_into_u,
            "supcon_enabled": sup_enabled,
            "supcon_tau": sup_tau,
            "supcon_embed": sup_embed,
            "supcon_two_view": sup_two_view,
            "supcon_merge_into_understanding": merge_sup_into_u,
            "num_trainable": count_parameters([p for p in model.parameters() if p.requires_grad]),
            "num_shared": count_parameters(shared_params),
            "num_aux": count_parameters(aux_params),
            "shared_params_mode": shared_mode,
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
