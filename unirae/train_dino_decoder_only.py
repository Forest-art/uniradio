import argparse
import copy
import math
import os
from typing import Dict, Optional, Tuple

import torch
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


def _compute_recon_loss(pred: torch.Tensor, target: torch.Tensor, kind: str, huber_delta: float) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "huber":
        return F.huber_loss(pred, target, delta=huber_delta)
    return F.mse_loss(pred, target)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict, total_steps: int) -> Optional[LambdaLR]:
    sch_cfg = cfg.get("scheduler", {})
    if str(sch_cfg.get("type", "none")).lower() != "cosine":
        return None

    optim_cfg = cfg.get("optim", {})
    base_lr = float(optim_cfg.get("lr", 2e-4))
    final_lr = float(sch_cfg.get("final_lr", 2e-5))
    warmup_steps = int(sch_cfg.get("warmup_steps", 1000))
    min_ratio = final_lr / max(base_lr, 1e-12)

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return min_ratio
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        prog = min(max(prog, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


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


@torch.no_grad()
def _update_ema(ema_model: UniRAEDinoLoRA, src_model: UniRAEDinoLoRA, decay: float) -> None:
    d = float(decay)
    for p_ema, p_src in zip(ema_model.parameters(), src_model.parameters()):
        p_ema.data.mul_(d).add_(p_src.data, alpha=1.0 - d)
    for b_ema, b_src in zip(ema_model.buffers(), src_model.buffers()):
        b_ema.copy_(b_src)


@torch.no_grad()
def evaluate_decoder_only(
    accelerator: Accelerator,
    model: UniRAEDinoLoRA,
    loader,
    device: torch.device,
    loss_kind: str,
    huber_delta: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    sum_loss = torch.zeros(1, device=device)
    sum_mse = torch.zeros(1, device=device)
    sum_psnr = torch.zeros(1, device=device)
    sum_count = torch.zeros(1, device=device)

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        batch = make_batch_dict(batch)
        batch = to_device(batch, device)
        images = batch["images"]  # [0,1]

        with accelerator.autocast():
            pred_img = _forward_recon(
                model=model,
                patch_helper=model,
                images_01=images,
                mean=mean,
                std=std,
            )
            target_img = images
            if pred_img.shape[-2:] != target_img.shape[-2:]:
                target_img = F.interpolate(target_img, size=pred_img.shape[-2:], mode="bilinear", align_corners=False)
            loss = _compute_recon_loss(pred_img, target_img, kind=loss_kind, huber_delta=huber_delta)
            mse = F.mse_loss(pred_img, target_img)
            psnr = -10.0 * torch.log10(mse + 1e-8)

        bs = float(images.shape[0])
        sum_loss += loss.detach() * bs
        sum_mse += mse.detach() * bs
        sum_psnr += psnr.detach() * bs
        sum_count += bs

    red_loss = accelerator.reduce(sum_loss, reduction="sum")
    red_mse = accelerator.reduce(sum_mse, reduction="sum")
    red_psnr = accelerator.reduce(sum_psnr, reduction="sum")
    red_count = accelerator.reduce(sum_count, reduction="sum").clamp_min(1.0)

    return {
        "recon_loss": float((red_loss / red_count).item()),
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

    base_cfg = load_yaml(args.config)
    cfg = apply_overrides(base_cfg, args.set)

    mixed_precision = cfg.get("accelerate", {}).get("mixed_precision", "bf16")
    if isinstance(mixed_precision, bool):
        mixed_precision = "no" if mixed_precision is False else "fp16"
    accelerator = Accelerator(mixed_precision=str(mixed_precision))
    device = accelerator.device

    seed = int(cfg.get("seed", 42))
    set_seed(seed, device_specific=True)

    exp_name = args.run_name or cfg.get("experiment", {}).get("name", "dino_decoder_only")
    run_dir = os.path.join(cfg.get("output", {}).get("root", "runs"), exp_name)
    if accelerator.is_main_process:
        if os.path.exists(run_dir):
            run_dir = f"{run_dir}_{now_str()}"
        ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
        save_yaml(cfg, os.path.join(run_dir, "run_config.yaml"))
    else:
        ckpt_dir = ""

    obj_list = [run_dir, ckpt_dir]
    broadcast_object_list(obj_list)
    run_dir, ckpt_dir = obj_list[0], obj_list[1]
    accelerator.wait_for_everyone()

    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    recon_cfg = cfg.get("recon", {})

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
        custom_transform=_build_rae_transform(image_size=image_size, split="train"),
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
        custom_transform=_build_rae_transform(image_size=image_size, split="val"),
    )

    model = _build_model(cfg).to(device)
    mean, std = _resolve_encoder_norm(model, device=device)

    # RAE stage-1 对齐：冻结 encoder/LoRA/理解头，仅训练生成 decoder。
    for p in model.parameters():
        p.requires_grad = False
    for p in model.generation_parameters():
        p.requires_grad = True

    non_decoder_trainable = [n for n, p in model.named_parameters() if p.requires_grad and not n.startswith("decoder.")]
    if non_decoder_trainable:
        raise RuntimeError(f"Found non-decoder trainable params: {non_decoder_trainable[:10]}")

    decoder_params = [p for p in model.generation_parameters() if p.requires_grad]
    if not decoder_params:
        raise RuntimeError("No trainable decoder parameters found.")

    ema_decay = float(train_cfg.get("ema_decay", 0.9978))
    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.AdamW(
        decoder_params,
        lr=float(optim_cfg.get("lr", 2e-4)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    steps = int(train_cfg.get("steps", 20000))
    scheduler = _build_scheduler(optimizer, cfg=cfg, total_steps=steps)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    if accelerator.is_main_process:
        num_total = sum(p.numel() for p in accelerator.unwrap_model(model).parameters())
        num_train = sum(p.numel() for p in decoder_params)
        print(
            f"[train_decoder_only] total_params={num_total:,} "
            f"decoder_trainable={num_train:,} "
            f"ema_decay={ema_decay}"
        )

    log_every = int(cfg.get("log", {}).get("every", 20))
    eval_every = int(cfg.get("log", {}).get("eval_every", 1000))
    save_every = int(cfg.get("log", {}).get("save_every", 1000))
    clip_grad = float(train_cfg.get("clip_grad", 0.0))
    loss_kind = str(recon_cfg.get("pixel_loss", "l1"))
    huber_delta = float(recon_cfg.get("huber_delta", 1.0))
    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    eval_on_ema = bool(eval_cfg.get("use_ema", True))

    train_iter = iter(train_loader)
    pbar = tqdm(range(1, steps + 1), desc="train_decoder_only", disable=not accelerator.is_local_main_process)
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = make_batch_dict(batch)
        batch = to_device(batch, device)
        images = batch["images"]  # [0,1]

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with accelerator.autocast():
            pred_img = _forward_recon(
                model=model,
                patch_helper=accelerator.unwrap_model(model),
                images_01=images,
                mean=mean,
                std=std,
            )
            target_img = images
            if pred_img.shape[-2:] != target_img.shape[-2:]:
                target_img = F.interpolate(target_img, size=pred_img.shape[-2:], mode="bilinear", align_corners=False)
            loss = _compute_recon_loss(pred_img, target_img, kind=loss_kind, huber_delta=huber_delta)
            mse = F.mse_loss(pred_img, target_img)
            psnr = -10.0 * torch.log10(mse + 1e-8)

        accelerator.backward(loss)
        if clip_grad > 0:
            accelerator.clip_grad_norm_(decoder_params, clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        _update_ema(ema_model, accelerator.unwrap_model(model), decay=ema_decay)

        loss_mean = float(accelerator.reduce(loss.detach(), reduction="mean").item())
        mse_mean = float(accelerator.reduce(mse.detach(), reduction="mean").item())
        psnr_mean = float(accelerator.reduce(psnr.detach(), reduction="mean").item())

        row = {
            "step": step,
            "loss": loss_mean,
            "mse": mse_mean,
            "psnr": psnr_mean,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if accelerator.is_main_process and (step == 1 or step % log_every == 0 or step == steps):
            append_jsonl(metrics_file, row)
            pbar.set_postfix(loss=f"{loss_mean:.4f}", mse=f"{mse_mean:.4f}", psnr=f"{psnr_mean:.2f}")

        if step % eval_every == 0 or step == steps:
            accelerator.wait_for_everyone()
            eval_model = ema_model if eval_on_ema else accelerator.unwrap_model(model)
            eval_metrics = evaluate_decoder_only(
                accelerator=accelerator,
                model=eval_model,
                loader=val_loader,
                device=device,
                loss_kind=loss_kind,
                huber_delta=huber_delta,
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
                torch.save(
                    {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "step": step,
                        "config": cfg,
                    },
                    os.path.join(ckpt_dir, "latest.pt"),
                )
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"[train_decoder_only] done. run_dir={run_dir}")


if __name__ == "__main__":
    main()
