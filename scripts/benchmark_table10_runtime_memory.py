#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import yaml

from unirae.data_cifar10 import build_cifar10_loader, make_batch_dict
from unirae.grad_conflict import apply_cagrad, apply_conflict_aware, apply_ma_laga_objective
from unirae.train_cifar10 import (
    CifarTradeoffModel,
    _build_shared_and_aux_params,
    _resolve_laga_plan,
    text_prototype_loss,
)
from unirae.utils import apply_overrides, cycle_loader, ensure_dir, load_yaml, seed_everything, to_device

ARCH_SPECS = {
    "vit_small_patch16": {"backbone": "vit_small", "label": "vit_small_patch16"},
    "swin_tiny_patch4": {"backbone": "swin_tiny_patch4", "label": "swin_tiny_patch4"},
}
METHODS = ["joint", "pcgrad", "cagrad", "dsga"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runtime and memory overhead for CIFAR100 gradient strategies.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "cifar100_baseline_joint_naive.yaml"))
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data" / "cifar100"))
    parser.add_argument("--out_root", default="")
    parser.add_argument("--arches", nargs="+", default=["vit_small_patch16", "swin_tiny_patch4"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--measure_iters", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cagrad_beta", type=float, default=0.35)
    parser.add_argument("--lambda_mag", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_cfg(base_cfg: Dict, backbone_name: str, batch_size: int, data_root: str, lr: float, weight_decay: float, strategy: str, cagrad_beta: float, lambda_mag: float) -> Dict:
    overrides = [
        f"seed=0",
        "data.dataset=cifar100",
        f"data.root={data_root}",
        "data.image_size=32",
        f"data.batch_size={batch_size}",
        "data.download=true",
        "data.val_from_train=false",
        "data.val_ratio=0.1",
        f"model.backbone={backbone_name}",
        "model.pretrained=false",
        "model.freeze_backbone=false",
        "train.mode=joint",
        "train.steps=1",
        "train.lambda_txt=1.0",
        "train.lambda_rec=1.0",
        f"train.grad_strategy={'naive' if strategy == 'joint' else strategy}",
        "train.shared_params=backbone",
        "train.recon_loss=rmse",
        f"optim.lr={lr}",
        f"optim.weight_decay={weight_decay}",
        "accelerate.mixed_precision=no",
    ]
    if strategy == "cagrad":
        overrides.append(f"train.cagrad_beta={cagrad_beta}")
    if strategy == "dsga":
        overrides.extend(
            [
                "train.laga_grouping=layerwise",
                "train.dsga_d_mode=full",
                "train.dsga_d_conflict_threshold=0.0",
                "train.dsga_d_conflict_only=false",
                "train.dsga_m_scope=global",
                "train.dsga_m_norm_restore=false",
                f"train.dsga_m_align_gamma={lambda_mag}",
            ]
        )
    return apply_overrides(base_cfg, overrides)


def prepare_state(cfg: Dict, device: torch.device, num_workers: int) -> Tuple[CifarTradeoffModel, torch.optim.Optimizer, object, List[str], List[torch.nn.Parameter], List[torch.nn.Parameter], Dict[str, List[int]], float, float]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model = CifarTradeoffModel(cfg, num_classes=100).to(device)
    loader, class_names = build_cifar10_loader(
        dataset=str(data_cfg.get("dataset", "cifar100")),
        data_root=str(data_cfg.get("root")),
        split="train",
        image_size=int(data_cfg.get("image_size", 32)),
        batch_size=int(data_cfg.get("batch_size", 256)),
        num_workers=int(num_workers),
        val_from_train=bool(data_cfg.get("val_from_train", False)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=int(cfg.get("seed", 0)),
        shuffle=True,
        drop_last=True,
        download=bool(data_cfg.get("download", True)),
        use_fake_data=bool(data_cfg.get("use_fake_data", False)),
        fake_train_size=int(data_cfg.get("fake_train_size", 4096)),
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
    shared_mode = str(train_cfg.get("shared_params", "backbone"))
    shared_params, aux_params = _build_shared_and_aux_params(model, shared_mode=shared_mode)
    laga_groups = _resolve_laga_plan(
        cfg=cfg,
        mode="joint",
        strategy=str(train_cfg.get("grad_strategy", "naive")),
        model=model,
        shared_params=shared_params,
        shared_mode=shared_mode,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("optim", {}).get("lr", 5e-4)),
        weight_decay=float(cfg.get("optim", {}).get("weight_decay", 1e-4)),
    )
    temperature = float(cfg.get("text", {}).get("temperature", 0.07))
    recon_loss_eps = float(train_cfg.get("recon_loss_eps", 1e-8))
    return model, optimizer, loader, class_names, shared_params, aux_params, laga_groups, temperature, recon_loss_eps


def one_step(model, optimizer, batch, strategy: str, shared_params, aux_params, laga_groups, temperature: float, recon_loss_eps: float, cagrad_beta: float, lambda_mag: float) -> None:
    model.train()
    batch = make_batch_dict(batch)
    batch = to_device(batch, next(model.parameters()).device)
    images = batch["images"]
    images_target = batch.get("images_target", images)

    optimizer.zero_grad(set_to_none=True)
    out = model(images)
    recon = out["recon"]
    if recon.shape[-2:] != images_target.shape[-2:]:
        images_target = F.interpolate(images_target, size=recon.shape[-2:], mode="bilinear", align_corners=False)

    Lu, _ = text_prototype_loss(
        z_txt=out["z_txt"],
        labels=batch["labels"],
        prototypes=model.text_prototypes,
        temperature=temperature,
    )
    Lg_mse = F.mse_loss(recon, images_target)
    Lg = torch.sqrt(Lg_mse + recon_loss_eps)

    if strategy == "joint":
        (Lu + Lg).backward()
    elif strategy == "pcgrad":
        apply_conflict_aware(
            loss_txt=Lu,
            loss_rec=Lg,
            lora_params=shared_params,
            aux_params=aux_params,
            lambda_txt=1.0,
            lambda_rec=1.0,
        )
    elif strategy == "cagrad":
        apply_cagrad(
            loss_txt=Lu,
            loss_rec=Lg,
            shared_params=shared_params,
            aux_params=aux_params,
            lambda_txt=1.0,
            lambda_rec=1.0,
            beta=float(cagrad_beta),
        )
    elif strategy == "dsga":
        apply_ma_laga_objective(
            loss_txt=Lu,
            loss_rec=Lg,
            shared_params=shared_params,
            aux_params=aux_params,
            lambda_txt=1.0,
            lambda_rec=1.0,
            group_to_indices=laga_groups,
            align_gamma=float(lambda_mag),
            norm_restore=False,
            mode="full",
            conflict_threshold=0.0,
            conflict_only=False,
            eps=1e-8,
            magnitude_scope="global",
        )
    else:
        raise ValueError(f"Unsupported strategy={strategy}")

    optimizer.step()


def benchmark_method(base_cfg: Dict, arch_name: str, strategy: str, args: argparse.Namespace, device: torch.device) -> Tuple[float, float, float]:
    arch_spec = ARCH_SPECS[arch_name]
    seed_everything(int(args.seed))
    torch.cuda.empty_cache()

    cfg = build_cfg(
        base_cfg=base_cfg,
        backbone_name=arch_spec["backbone"],
        batch_size=int(args.batch_size),
        data_root=str(args.data_root),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        strategy=strategy,
        cagrad_beta=float(args.cagrad_beta),
        lambda_mag=float(args.lambda_mag),
    )
    cfg["seed"] = int(args.seed)
    model, optimizer, loader, _, shared_params, aux_params, laga_groups, temperature, recon_loss_eps = prepare_state(
        cfg=cfg,
        device=device,
        num_workers=int(args.num_workers),
    )
    train_iter = cycle_loader(loader)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(int(args.warmup_iters)):
        batch = next(train_iter)
        one_step(model, optimizer, batch, strategy, shared_params, aux_params, laga_groups, temperature, recon_loss_eps, args.cagrad_beta, args.lambda_mag)

    times_ms: List[float] = []
    peak_gb = 0.0
    for _ in range(int(args.measure_iters)):
        batch = next(train_iter)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        one_step(model, optimizer, batch, strategy, shared_params, aux_params, laga_groups, temperature, recon_loss_eps, args.cagrad_beta, args.lambda_mag)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_gb = max(peak_gb, float(torch.cuda.max_memory_allocated(device)) / 1e9)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(statistics.median(times_ms))
    mean_ms = float(statistics.fmean(times_ms))
    return median_ms, mean_ms, peak_gb


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    base_cfg = load_yaml(args.config)

    if args.out_root:
        out_root = Path(args.out_root)
    else:
        out_root = REPO_ROOT / "results" / f"table10_runtime_memory_{time.strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(str(out_root))

    combined_rows: List[Dict[str, object]] = []
    for arch_name in args.arches:
        if arch_name not in ARCH_SPECS:
            raise ValueError(f"Unsupported arch={arch_name}. Use one of: {sorted(ARCH_SPECS)}")
        arch_rows = []
        time_medians: Dict[str, float] = {}
        time_means: Dict[str, float] = {}
        peak_gpu: Dict[str, float] = {}

        for method in METHODS:
            median_ms, mean_ms, peak_gb = benchmark_method(base_cfg, arch_name, method, args, device)
            time_medians[method] = round(median_ms, 3)
            time_means[method] = round(mean_ms, 3)
            peak_gpu[method] = round(peak_gb, 3)

        joint_time = time_medians["joint"]
        overhead = {
            method: round(100.0 * (time_medians[method] - joint_time) / max(joint_time, 1e-12), 2)
            for method in METHODS
            if method != "joint"
        }

        payload = {
            "arch": arch_name,
            "batch_size": int(args.batch_size),
            "window": int(args.measure_iters),
            "warmup": int(args.warmup_iters),
            "time_ms_per_iter": time_medians,
            "time_ms_per_iter_mean": time_means,
            "peak_gpu_gb": peak_gpu,
            "overhead_pct": overhead,
        }
        with open(out_root / f"{arch_name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

        for method in METHODS:
            row = {
                "Method": method,
                "Time_ms_per_iter": time_medians[method],
                "Peak_GPU_GB": peak_gpu[method],
                "Overhead_vs_Joint_pct": 0.0 if method == "joint" else overhead[method],
            }
            arch_rows.append(row)
            combined_rows.append({"Arch": arch_name, **row})

        with open(out_root / f"{arch_name}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Method", "Time_ms_per_iter", "Peak_GPU_GB", "Overhead_vs_Joint_pct"])
            writer.writeheader()
            writer.writerows(arch_rows)

    with open(out_root / "table10_runtime_memory_all.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Arch", "Method", "Time_ms_per_iter", "Peak_GPU_GB", "Overhead_vs_Joint_pct"])
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"[done] Table 10 benchmark written to {out_root}")


if __name__ == "__main__":
    main()
