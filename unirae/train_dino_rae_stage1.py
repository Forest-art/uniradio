import argparse
import copy
import importlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from tqdm import tqdm

from .data_imagenet import build_imagenet_loader, make_batch_dict
from .models import UniRAEDinoLoRA
from .utils import (
    append_jsonl,
    apply_overrides,
    ensure_dir,
    load_yaml,
    now_str,
    save_json,
    save_yaml,
    to_device,
)


def _build_rae_transform(image_size: int, split: str):
    first_crop = 384 if image_size == 256 else int(image_size * 1.5)
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(first_crop, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(first_crop, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def _build_model(cfg: Dict) -> UniRAEDinoLoRA:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    num_classes = int(model_cfg.get("num_classes", 1000))
    return UniRAEDinoLoRA(
        num_classes=num_classes,
        dino_variant=str(model_cfg.get("dino_variant", "vit_base")),
        image_size=int(data_cfg.get("image_size", 224)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        lora_rank=int(model_cfg.get("lora_rank", 8)),
        lora_alpha=float(model_cfg.get("lora_alpha", 16.0)),
        lora_last_n_blocks=int(model_cfg.get("lora_last_n_blocks", 0)),
        decoder_dim=int(model_cfg.get("decoder_dim", 512)),
        decoder_depth=int(model_cfg.get("decoder_depth", 8)),
        decoder_heads=int(model_cfg.get("decoder_heads", 16)),
        decoder_mlp_ratio=float(model_cfg.get("decoder_mlp_ratio", 4.0)),
        decoder_drop_rate=float(model_cfg.get("decoder_drop_rate", 0.0)),
        decoder_noise_tau=float(model_cfg.get("decoder_noise_tau", 0.8)),
        use_grad_checkpointing=bool(model_cfg.get("use_grad_checkpointing", False)),
    )


def _resolve_encoder_norm(model: UniRAEDinoLoRA, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = getattr(model.encoder, "pretrained_cfg", {}) or {}
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean_t, std_t


def _forward_recon(
    model,
    patch_helper: UniRAEDinoLoRA,
    images_01: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    enc_in = (images_01 - mean.to(images_01)) / std.to(images_01)
    out = model(enc_in)
    pred_img = patch_helper.unpatchify(out["pred_patches"])
    return pred_img


def _compute_recon_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "l1", huber_delta: float = 1.0) -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "huber":
        return F.huber_loss(pred, target, delta=huber_delta)
    return F.l1_loss(pred, target)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    sch_cfg: Dict,
    total_steps: int,
    steps_per_epoch: int,
    base_lr: float,
) -> Optional[LambdaLR]:
    if str(sch_cfg.get("type", "none")).lower() != "cosine":
        return None

    warmup_steps = sch_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_epochs = float(sch_cfg.get("warmup_epochs", 0))
        warmup_steps = int(round(warmup_epochs * steps_per_epoch))
    warmup_steps = max(0, int(warmup_steps))

    decay_end_steps = sch_cfg.get("decay_end_steps")
    if decay_end_steps is None:
        decay_end_epoch = sch_cfg.get("decay_end_epoch")
        if decay_end_epoch is not None:
            decay_end_steps = int(round(float(decay_end_epoch) * steps_per_epoch))
        else:
            decay_end_steps = total_steps
    decay_end_steps = max(1, int(decay_end_steps))

    final_lr = float(sch_cfg.get("final_lr", base_lr))
    min_ratio = final_lr / max(base_lr, 1e-12)

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if step >= decay_end_steps:
            return min_ratio
        denom = max(1, decay_end_steps - warmup_steps)
        prog = float(step - warmup_steps) / float(denom)
        prog = min(max(prog, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _select_gan_losses(disc_kind: str, gen_kind: str, hinge_d_loss, vanilla_d_loss, vanilla_g_loss):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")
    if gen_kind != "vanilla":
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, vanilla_g_loss


def _calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def _autocast_kwargs_from_accelerator(accelerator: Accelerator) -> Dict:
    mp = str(accelerator.mixed_precision)
    if mp == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    if mp == "fp16":
        return {"enabled": True, "dtype": torch.float16}
    return {"enabled": False}


def _resolve_rae_src(rae_code_root: str) -> str:
    root = Path(rae_code_root)
    if (root / "src").exists():
        return str(root / "src")
    return str(root)


@dataclass
class RAEModules:
    LPIPS: object
    build_discriminator: object
    hinge_d_loss: object
    vanilla_d_loss: object
    vanilla_g_loss: object
    evaluate_reconstruction_distributed: object | None


def _import_rae_modules(rae_code_root: str) -> RAEModules:
    src_dir = _resolve_rae_src(rae_code_root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    disc_mod = importlib.import_module("disc")
    try:
        eval_mod = importlib.import_module("eval")
        eval_fn = eval_mod.evaluate_reconstruction_distributed
    except Exception:
        eval_fn = None
    return RAEModules(
        LPIPS=disc_mod.LPIPS,
        build_discriminator=disc_mod.build_discriminator,
        hinge_d_loss=disc_mod.hinge_d_loss,
        vanilla_d_loss=disc_mod.vanilla_d_loss,
        vanilla_g_loss=disc_mod.vanilla_g_loss,
        evaluate_reconstruction_distributed=eval_fn,
    )


class _ReconModelAdapter(nn.Module):
    def __init__(self, model, patch_helper: UniRAEDinoLoRA, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.patch_helper = patch_helper
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _forward_recon(self.model, self.patch_helper, images, self.mean, self.std)


@torch.no_grad()
def _update_ema(ema_model: UniRAEDinoLoRA, src_model: UniRAEDinoLoRA, decay: float) -> None:
    d = float(decay)
    for p_ema, p_src in zip(ema_model.parameters(), src_model.parameters()):
        p_ema.data.mul_(d).add_(p_src.data, alpha=1.0 - d)
    for b_ema, b_src in zip(ema_model.buffers(), src_model.buffers()):
        b_ema.copy_(b_src)


@torch.no_grad()
def _evaluate_simple(
    accelerator: Accelerator,
    model,
    patch_helper: UniRAEDinoLoRA,
    loader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_batches: Optional[int],
) -> Dict[str, float]:
    model.eval()
    sum_mse = torch.zeros(1, device=device)
    sum_psnr = torch.zeros(1, device=device)
    sum_count = torch.zeros(1, device=device)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = to_device(make_batch_dict(batch), device)
        images = batch["images"]
        with accelerator.autocast():
            pred = _forward_recon(model, patch_helper, images, mean, std)
            if pred.shape[-2:] != images.shape[-2:]:
                images = F.interpolate(images, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            mse = F.mse_loss(pred, images)
            psnr = -10.0 * torch.log10(mse + 1e-8)
        bs = float(images.shape[0])
        sum_mse += mse * bs
        sum_psnr += psnr * bs
        sum_count += bs
    red_mse = accelerator.reduce(sum_mse, reduction="sum")
    red_psnr = accelerator.reduce(sum_psnr, reduction="sum")
    red_count = accelerator.reduce(sum_count, reduction="sum").clamp_min(1.0)
    return {
        "mse": float((red_mse / red_count).item()),
        "psnr": float((red_psnr / red_count).item()),
        "num_samples": int(red_count.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.set)
    accelerator = Accelerator(mixed_precision=str(cfg.get("accelerate", {}).get("mixed_precision", "bf16")))
    device = accelerator.device
    set_seed(int(cfg.get("seed", 42)), device_specific=True)

    exp_name = args.run_name or cfg.get("experiment", {}).get("name", "dino_rae_stage1")
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
    run_dir, ckpt_dir = objs
    accelerator.wait_for_everyone()

    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    recon_cfg = cfg.get("recon", {})
    gan_cfg = cfg.get("gan", {})
    rae_cfg = cfg.get("rae_integration", {})

    image_size = int(data_cfg.get("image_size", 224))
    train_loader, _ = build_imagenet_loader(
        data_root=data_cfg["data_root"],
        split="train",
        image_size=image_size,
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 8)),
        class_names_file=data_cfg.get("class_names_file"),
        shuffle=True,
        drop_last=True,
        data_format=data_cfg.get("data_format", "auto"),
        hf_load_from_disk=data_cfg.get("hf_load_from_disk"),
        hf_split_override=data_cfg.get("hf_split_train"),
        hf_image_key=data_cfg.get("hf_image_key", "image"),
        hf_label_key=data_cfg.get("hf_label_key", "label"),
        custom_transform=_build_rae_transform(image_size, "train"),
    )
    val_loader, _ = build_imagenet_loader(
        data_root=data_cfg["data_root"],
        split="val",
        image_size=image_size,
        batch_size=int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 64))),
        num_workers=int(data_cfg.get("num_workers", 8)),
        class_names_file=data_cfg.get("class_names_file"),
        shuffle=False,
        drop_last=False,
        data_format=data_cfg.get("data_format", "auto"),
        hf_load_from_disk=data_cfg.get("hf_load_from_disk"),
        hf_split_override=data_cfg.get("hf_split_val"),
        hf_image_key=data_cfg.get("hf_image_key", "image"),
        hf_label_key=data_cfg.get("hf_label_key", "label"),
        custom_transform=_build_rae_transform(image_size, "val"),
    )
    val_dataset = val_loader.dataset

    model = _build_model(cfg).to(device)
    mean, std = _resolve_encoder_norm(model, device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.generation_parameters():
        p.requires_grad = True
    decoder_params = [p for p in model.generation_parameters() if p.requires_grad]
    if not decoder_params:
        raise RuntimeError("No trainable decoder parameters found.")

    ema_decay = float(train_cfg.get("ema_decay", 0.9978))
    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    optim_cfg = cfg.get("optim", {})
    gen_lr = float(optim_cfg.get("lr", 2e-4))
    optimizer = torch.optim.AdamW(
        decoder_params,
        lr=gen_lr,
        betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    steps = int(train_cfg.get("steps", 20000))
    steps_per_epoch = max(1, len(train_loader))
    scheduler = _build_scheduler(
        optimizer=optimizer,
        sch_cfg=cfg.get("scheduler", {}),
        total_steps=steps,
        steps_per_epoch=steps_per_epoch,
        base_lr=gen_lr,
    )

    use_full_stage1 = bool(gan_cfg.get("enable", False))
    modules = None
    discriminator = None
    disc_aug = None
    disc_optimizer = None
    disc_scheduler = None
    lpips_metric = None
    disc_loss_fn = None
    gen_loss_fn = None

    if use_full_stage1:
        modules = _import_rae_modules(str(rae_cfg.get("code_root", "/project/peilab/luxiaocheng/projects/RAE")))
        discriminator, disc_aug = modules.build_discriminator(gan_cfg.get("disc", {}), device)
        disc_params = [p for p in discriminator.parameters() if p.requires_grad]
        disc_optim_cfg = gan_cfg.get("disc", {}).get("optimizer", {})
        disc_lr = float(disc_optim_cfg.get("lr", 2e-4))
        disc_optimizer = torch.optim.AdamW(
            disc_params,
            lr=disc_lr,
            betas=tuple(disc_optim_cfg.get("betas", [0.9, 0.95])),
            weight_decay=float(disc_optim_cfg.get("weight_decay", 0.0)),
        )
        disc_scheduler = _build_scheduler(
            optimizer=disc_optimizer,
            sch_cfg=gan_cfg.get("disc", {}).get("scheduler", {}),
            total_steps=steps,
            steps_per_epoch=steps_per_epoch,
            base_lr=disc_lr,
        )
        disc_loss_fn, gen_loss_fn = _select_gan_losses(
            str(gan_cfg.get("loss", {}).get("disc_loss", "hinge")),
            str(gan_cfg.get("loss", {}).get("gen_loss", "vanilla")),
            modules.hinge_d_loss,
            modules.vanilla_d_loss,
            modules.vanilla_g_loss,
        )
        lpips_metric = modules.LPIPS().to(device).eval()
        for p in lpips_metric.parameters():
            p.requires_grad = False

    if use_full_stage1:
        model, optimizer, discriminator, disc_optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, discriminator, disc_optimizer, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    patch_helper = accelerator.unwrap_model(model)
    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    log_every = int(cfg.get("log", {}).get("every", 20))
    eval_every = int(cfg.get("log", {}).get("eval_every", 1000))
    save_every = int(cfg.get("log", {}).get("save_every", 1000))
    clip_grad = float(train_cfg.get("clip_grad", 0.0))
    loss_kind = str(recon_cfg.get("pixel_loss", "l1"))
    huber_delta = float(recon_cfg.get("huber_delta", 1.0))
    eval_on_ema = bool(eval_cfg.get("use_ema", True))

    loss_cfg = gan_cfg.get("loss", {})
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 1.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.75))
    disc_start = int(loss_cfg.get("disc_start", 0))
    disc_upd_start = int(loss_cfg.get("disc_upd_start", 0))
    lpips_start = int(loss_cfg.get("lpips_start", 0))
    if "disc_start_epoch" in loss_cfg:
        disc_start = int(round(float(loss_cfg["disc_start_epoch"]) * steps_per_epoch))
    if "disc_upd_start_epoch" in loss_cfg:
        disc_upd_start = int(round(float(loss_cfg["disc_upd_start_epoch"]) * steps_per_epoch))
    if "lpips_start_epoch" in loss_cfg:
        lpips_start = int(round(float(loss_cfg["lpips_start_epoch"]) * steps_per_epoch))
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    last_layer = patch_helper.decoder.pred.weight

    if accelerator.is_main_process:
        print(
            f"[train_dino_rae_stage1] full_stage1={use_full_stage1} "
            f"num_trainable_decoder={sum(p.numel() for p in decoder_params):,}"
        )

    train_iter = iter(train_loader)
    pbar = tqdm(range(1, steps + 1), desc="train_dino_rae_stage1", disable=not accelerator.is_local_main_process)
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = to_device(make_batch_dict(batch), device)
        images = batch["images"]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        if use_full_stage1:
            discriminator.eval()

        with accelerator.autocast():
            pred_img = _forward_recon(model, patch_helper, images, mean, std)
            target_img = images
            if pred_img.shape[-2:] != target_img.shape[-2:]:
                target_img = F.interpolate(target_img, size=pred_img.shape[-2:], mode="bilinear", align_corners=False)

            rec_loss = _compute_recon_loss(pred_img, target_img, kind=loss_kind, huber_delta=huber_delta)
            mse = F.mse_loss(pred_img, target_img)
            psnr = -10.0 * torch.log10(mse + 1e-8)

            lpips_loss = rec_loss.new_zeros(())
            gan_loss = rec_loss.new_zeros(())
            adaptive_weight = rec_loss.new_zeros(())
            total_loss = rec_loss

            if use_full_stage1:
                real_normed = target_img * 2.0 - 1.0
                recon_normed = pred_img * 2.0 - 1.0
                use_lpips = step >= lpips_start and perceptual_weight > 0
                use_gan = step >= disc_start and disc_weight > 0
                if use_lpips:
                    lpips_loss = lpips_metric(real_normed, recon_normed)
                recon_total = rec_loss + perceptual_weight * lpips_loss
                total_loss = recon_total
                if use_gan:
                    fake_aug = disc_aug.aug(recon_normed)
                    logits_fake, _ = discriminator(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                    adaptive_weight = _calculate_adaptive_weight(recon_total, gan_loss, last_layer, max_d_weight=max_d_weight)
                    total_loss = recon_total + disc_weight * adaptive_weight * gan_loss

        accelerator.backward(total_loss)
        if clip_grad > 0:
            accelerator.clip_grad_norm_(decoder_params, clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        _update_ema(ema_model, patch_helper, decay=ema_decay)

        disc_loss_val = 0.0
        disc_acc_val = 0.0
        if use_full_stage1 and step >= disc_upd_start and disc_weight > 0:
            model.eval()
            discriminator.train()
            real_normed = (images * 2.0 - 1.0).detach()
            for _ in range(disc_updates):
                disc_optimizer.zero_grad(set_to_none=True)
                with accelerator.autocast():
                    with torch.no_grad():
                        recon_disc = _forward_recon(model, patch_helper, images, mean, std)
                        recon_disc = recon_disc * 2.0 - 1.0
                    fake_detached = recon_disc.clamp(-1.0, 1.0)
                    fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                    fake_input = disc_aug.aug(fake_detached)
                    real_input = disc_aug.aug(real_normed)
                    logits_fake, logits_real = discriminator(fake_input, real_input)
                    d_loss = disc_loss_fn(logits_real, logits_fake)
                    d_acc = (logits_real > logits_fake).float().mean()
                accelerator.backward(d_loss)
                disc_optimizer.step()
                if disc_scheduler is not None:
                    disc_scheduler.step()
                disc_loss_val = float(d_loss.detach().item())
                disc_acc_val = float(d_acc.detach().item())
            discriminator.eval()
            model.train()

        row = {
            "step": step,
            "total_loss": float(accelerator.reduce(total_loss.detach(), reduction="mean").item()),
            "recon_l1": float(accelerator.reduce(rec_loss.detach(), reduction="mean").item()),
            "mse": float(accelerator.reduce(mse.detach(), reduction="mean").item()),
            "psnr": float(accelerator.reduce(psnr.detach(), reduction="mean").item()),
            "lr_gen": float(optimizer.param_groups[0]["lr"]),
        }
        if use_full_stage1:
            row.update(
                {
                    "lpips": float(accelerator.reduce(lpips_loss.detach(), reduction="mean").item()),
                    "gan_g": float(accelerator.reduce(gan_loss.detach(), reduction="mean").item()),
                    "d_weight": float(accelerator.reduce(adaptive_weight.detach(), reduction="mean").item()),
                    "disc_loss": disc_loss_val,
                    "disc_acc": disc_acc_val,
                    "lr_disc": float(disc_optimizer.param_groups[0]["lr"]),
                }
            )
        if accelerator.is_main_process and (step == 1 or step % log_every == 0 or step == steps):
            append_jsonl(metrics_file, row)
            pbar.set_postfix(loss=f"{row['total_loss']:.4f}", mse=f"{row['mse']:.4f}", psnr=f"{row['psnr']:.2f}")

        if step % eval_every == 0 or step == steps:
            accelerator.wait_for_everyone()
            eval_model_obj = ema_model if eval_on_ema else patch_helper
            use_official_eval = bool(eval_cfg.get("use_official_rae_eval", False))
            has_dist = accelerator.num_processes > 1 and torch.distributed.is_available() and torch.distributed.is_initialized()
            if use_official_eval and modules is not None and has_dist:
                wrapper = _ReconModelAdapter(eval_model_obj, eval_model_obj, mean, std).to(device).eval()
                eval_metrics = modules.evaluate_reconstruction_distributed(
                    model=wrapper,
                    val_dataset=val_dataset,
                    num_samples=int(eval_cfg.get("num_samples", len(val_dataset))),
                    batch_size=int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 64))),
                    rank=accelerator.process_index,
                    world_size=accelerator.num_processes,
                    device=device,
                    experiment_dir=run_dir,
                    global_step=step,
                    autocast_kwargs=_autocast_kwargs_from_accelerator(accelerator),
                    metric_batch_size=int(eval_cfg.get("metric_batch_size", 128)),
                    reference_npz_path=eval_cfg.get("reference_npz_path"),
                    metrics_to_compute=eval_cfg.get("metrics", ["psnr", "ssim", "rfid"]),
                )
                if accelerator.is_main_process:
                    eval_row = {"step": step, "eval_on": "ema" if eval_on_ema else "model"}
                    if eval_metrics is not None:
                        eval_row.update(eval_metrics)
                    save_json(eval_row, os.path.join(run_dir, "eval_last.json"))
                    print(f"[eval][step={step}] {eval_row}")
            else:
                eval_metrics = _evaluate_simple(
                    accelerator=accelerator,
                    model=eval_model_obj,
                    patch_helper=eval_model_obj,
                    loader=val_loader,
                    device=device,
                    mean=mean,
                    std=std,
                    max_batches=eval_cfg.get("max_batches"),
                )
                if accelerator.is_main_process:
                    eval_row = {"step": step, "eval_on": "ema" if eval_on_ema else "model", **eval_metrics}
                    save_json(eval_row, os.path.join(run_dir, "eval_last.json"))
                    print(f"[eval][step={step}] {eval_row}")
            accelerator.wait_for_everyone()

        if step % save_every == 0 or step == steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt = {
                    "model": patch_helper.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "step": step,
                    "config": cfg,
                }
                if use_full_stage1:
                    ckpt["discriminator"] = accelerator.unwrap_model(discriminator).state_dict()
                    ckpt["disc_optimizer"] = disc_optimizer.state_dict()
                    ckpt["disc_scheduler"] = disc_scheduler.state_dict() if disc_scheduler is not None else None
                torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"[train_dino_rae_stage1] done. run_dir={run_dir}")


if __name__ == "__main__":
    main()
