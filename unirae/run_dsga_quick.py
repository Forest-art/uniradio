from __future__ import annotations

import argparse
import json
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW

from .train_imagenet100_dynamics import build_imagenet100_dataloaders
from .dsga_pipeline import (
    DSGAModel,
    evaluate_linear_probing,
    evaluate_reconstruction_rmse_rfid,
    imagenet_norm_tensors,
    train_step_gradient_decompose,
    train_step_joint_naive,
)
from .utils import ensure_dir, seed_everything


def _run_joint_naive(
    *,
    model: DSGAModel,
    train_loader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    steps: int,
    lr: float,
    wd: float,
    lambda_cls: float,
    lambda_recon: float,
    recon_loss: str,
    recon_rmse_eps: float,
    log_every: int,
) -> Dict[str, float]:
    optimizer = AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    train_iter = cycle(train_loader)

    t0 = time.time()
    last = {}
    for step in range(1, int(steps) + 1):
        batch = next(train_iter)
        last = train_step_joint_naive(
            model,
            batch,
            optimizer,
            mean=mean,
            std=std,
            lambda_cls=float(lambda_cls),
            lambda_recon=float(lambda_recon),
            recon_loss=str(recon_loss),
            recon_rmse_eps=float(recon_rmse_eps),
            input_is_normalized=True,
        )
        if step % max(1, int(log_every)) == 0:
            print(
                f"[joint-naive] step={step}/{steps} "
                f"loss_total={last['loss_total']:.4f} acc={last['acc']:.4f} "
                f"rmse={last['rmse']:.5f} cos={last['grad_cosine']:.4f}"
            )
    last["train_wall_sec"] = float(time.time() - t0)
    return last


def _run_grad_decompose(
    *,
    model: DSGAModel,
    train_loader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    method: str,
    steps: int,
    lr: float,
    wd: float,
    lambda_cls: float,
    lambda_recon: float,
    recon_loss: str,
    recon_rmse_eps: float,
    cagrad_beta: float,
    ma_align_gamma: float,
    ma_norm_restore: bool,
    log_every: int,
) -> Dict[str, float]:
    optimizer = AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    train_iter = cycle(train_loader)

    t0 = time.time()
    last = {}
    for step in range(1, int(steps) + 1):
        batch = next(train_iter)
        last = train_step_gradient_decompose(
            model,
            batch,
            optimizer,
            mean=mean,
            std=std,
            method=str(method),
            lambda_cls=float(lambda_cls),
            lambda_recon=float(lambda_recon),
            recon_loss=str(recon_loss),
            recon_rmse_eps=float(recon_rmse_eps),
            input_is_normalized=True,
            cagrad_beta=float(cagrad_beta),
            ma_align_gamma=float(ma_align_gamma),
            ma_norm_restore=bool(ma_norm_restore),
        )
        if step % max(1, int(log_every)) == 0:
            print(
                f"[{method}] step={step}/{steps} "
                f"loss_total={last['loss_total']:.4f} acc={last['acc']:.4f} "
                f"rmse={last['rmse']:.5f} cos={last['grad_cosine']:.4f}"
            )
    last["train_wall_sec"] = float(time.time() - t0)
    return last


def _evaluate_phase4(
    *,
    model: DSGAModel,
    train_loader,
    val_loader,
    num_classes: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    probe_steps: int,
    probe_lr: float,
    probe_wd: float,
    eval_max_batches: int,
    rfid_num_samples: int,
    rfid_batch_size: int,
    rfid_tmp_dir: str,
) -> Dict[str, float]:
    lp = evaluate_linear_probing(
        model.encoder,
        train_loader,
        val_loader,
        num_classes=int(num_classes),
        device=device,
        max_steps=int(probe_steps),
        lr=float(probe_lr),
        weight_decay=float(probe_wd),
    )
    rec = evaluate_reconstruction_rmse_rfid(
        model,
        val_loader,
        device=device,
        mean=mean,
        std=std,
        input_is_normalized=True,
        max_batches=int(eval_max_batches),
        rfid_num_samples=int(rfid_num_samples),
        rfid_batch_size=int(rfid_batch_size),
        rfid_tmp_dir=str(rfid_tmp_dir),
    )
    out = {}
    out.update(lp)
    out.update(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick DSGA phased experiment on ImageNet-100.")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--output_root", type=str, default="/scratch/peilab/xlubl/unirae_runs/dsga_quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--hf_dataset_id", type=str, default="clane9/imagenet-100")
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--label_key", type=str, default="label")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--joint_steps", type=int, default=2000)
    parser.add_argument("--decomp_steps", type=int, default=2000)
    parser.add_argument("--probe_steps", type=int, default=800)
    parser.add_argument("--eval_max_batches", type=int, default=30)
    parser.add_argument("--rfid_num_samples", type=int, default=512)
    parser.add_argument("--rfid_batch_size", type=int, default=64)
    parser.add_argument("--rfid_tmp_dir", type=str, default="/tmp")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)

    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--recon_loss", type=str, default="rmse", choices=["mse", "rmse"])
    parser.add_argument("--recon_rmse_eps", type=float, default=1e-8)

    parser.add_argument(
        "--decomp_methods",
        type=str,
        default="cagrad,ma_laga_global",
        help="comma-separated: cagrad,ma_laga_global,gma_laga",
    )
    parser.add_argument("--cagrad_beta", type=float, default=0.5)
    parser.add_argument("--ma_align_gamma", type=float, default=0.5)
    parser.add_argument("--ma_norm_restore", action="store_true", default=False)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    seed_everything(int(args.seed))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    mean, std = imagenet_norm_tensors(device=device)

    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"dsga_quick_s{int(args.seed)}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_name
    ensure_dir(str(run_dir))

    train_loader, val_loader, data_meta = build_imagenet100_dataloaders(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        dataset_path=(args.dataset_path.strip() or None),
        hf_dataset_id=(args.hf_dataset_id.strip() or None),
        image_key=str(args.image_key),
        label_key=str(args.label_key),
        distributed=False,
        rank=0,
        world_size=1,
    )
    num_classes = int(data_meta["num_classes"])
    print(f"[setup] run_dir={run_dir}")
    print(f"[setup] data_meta={json.dumps(data_meta, ensure_ascii=False)}")

    # Phase 1: frozen DINO linear probing baseline.
    phase1_model = DSGAModel(
        num_classes=num_classes,
        image_size=int(args.image_size),
        encoder_pretrained=True,
    ).to(device)
    phase1_model.freeze_encoder()
    phase1_ret = evaluate_linear_probing(
        phase1_model.encoder,
        train_loader,
        val_loader,
        num_classes=num_classes,
        device=device,
        max_steps=int(args.probe_steps),
        lr=float(args.probe_lr),
        weight_decay=float(args.probe_weight_decay),
    )
    print(f"[phase1] {phase1_ret}")

    # Phase 2: joint naive.
    phase2_model = DSGAModel(
        num_classes=num_classes,
        image_size=int(args.image_size),
        encoder_pretrained=True,
    ).to(device)
    phase2_model.unfreeze_encoder()
    phase2_last = _run_joint_naive(
        model=phase2_model,
        train_loader=train_loader,
        mean=mean,
        std=std,
        device=device,
        steps=int(args.joint_steps),
        lr=float(args.lr),
        wd=float(args.weight_decay),
        lambda_cls=float(args.lambda_cls),
        lambda_recon=float(args.lambda_recon),
        recon_loss=str(args.recon_loss),
        recon_rmse_eps=float(args.recon_rmse_eps),
        log_every=int(args.log_every),
    )
    phase2_eval = _evaluate_phase4(
        model=phase2_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        mean=mean,
        std=std,
        device=device,
        probe_steps=int(args.probe_steps),
        probe_lr=float(args.probe_lr),
        probe_wd=float(args.probe_weight_decay),
        eval_max_batches=int(args.eval_max_batches),
        rfid_num_samples=int(args.rfid_num_samples),
        rfid_batch_size=int(args.rfid_batch_size),
        rfid_tmp_dir=str(args.rfid_tmp_dir),
    )
    print(f"[phase2] train_last={phase2_last}")
    print(f"[phase2] eval={phase2_eval}")

    # Phase 3: gradient decomposition methods.
    methods = [m.strip() for m in str(args.decomp_methods).split(",") if m.strip()]
    phase3_results: Dict[str, Dict[str, float]] = {}
    for method in methods:
        m = DSGAModel(
            num_classes=num_classes,
            image_size=int(args.image_size),
            encoder_pretrained=True,
        ).to(device)
        m.unfreeze_encoder()

        train_last = _run_grad_decompose(
            model=m,
            train_loader=train_loader,
            mean=mean,
            std=std,
            device=device,
            method=method,
            steps=int(args.decomp_steps),
            lr=float(args.lr),
            wd=float(args.weight_decay),
            lambda_cls=float(args.lambda_cls),
            lambda_recon=float(args.lambda_recon),
            recon_loss=str(args.recon_loss),
            recon_rmse_eps=float(args.recon_rmse_eps),
            cagrad_beta=float(args.cagrad_beta),
            ma_align_gamma=float(args.ma_align_gamma),
            ma_norm_restore=bool(args.ma_norm_restore),
            log_every=int(args.log_every),
        )
        eval_ret = _evaluate_phase4(
            model=m,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            mean=mean,
            std=std,
            device=device,
            probe_steps=int(args.probe_steps),
            probe_lr=float(args.probe_lr),
            probe_wd=float(args.probe_weight_decay),
            eval_max_batches=int(args.eval_max_batches),
            rfid_num_samples=int(args.rfid_num_samples),
            rfid_batch_size=int(args.rfid_batch_size),
            rfid_tmp_dir=str(args.rfid_tmp_dir),
        )
        phase3_results[method] = {
            "train_last": train_last,
            "eval": eval_ret,
        }
        print(f"[phase3:{method}] train_last={train_last}")
        print(f"[phase3:{method}] eval={eval_ret}")

    summary = {
        "args": vars(args),
        "data_meta": data_meta,
        "phase1_linear_probe": phase1_ret,
        "phase2_joint_naive": {
            "train_last": phase2_last,
            "eval": phase2_eval,
        },
        "phase3_decompose": phase3_results,
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # lightweight table for quick comparison
    rows: List[Dict[str, object]] = []
    rows.append(
        {
            "phase": "phase1_lp",
            "method": "frozen_dino_lp",
            "top1": float(phase1_ret["linear_probe_top1"]),
            "rmse": None,
            "rfid": None,
        }
    )
    rows.append(
        {
            "phase": "phase2_joint",
            "method": "naive",
            "top1": float(phase2_eval["linear_probe_top1"]),
            "rmse": float(phase2_eval["val_rmse"]),
            "rfid": float(phase2_eval["val_rfid"]) if phase2_eval["val_rfid"] == phase2_eval["val_rfid"] else None,
        }
    )
    for method, ret in phase3_results.items():
        ev = ret["eval"]
        rows.append(
            {
                "phase": "phase3_decomp",
                "method": method,
                "top1": float(ev["linear_probe_top1"]),
                "rmse": float(ev["val_rmse"]),
                "rfid": float(ev["val_rfid"]) if ev["val_rfid"] == ev["val_rfid"] else None,
            }
        )

    with open(run_dir / "summary_table.csv", "w", encoding="utf-8") as f:
        f.write("phase,method,top1,rmse,rfid\n")
        for r in rows:
            f.write(f"{r['phase']},{r['method']},{r['top1']},{r['rmse']},{r['rfid']}\n")

    print("[done] summary:", run_dir / "summary.json")
    print("[done] table:", run_dir / "summary_table.csv")
    for r in rows:
        print(
            f"[result] phase={r['phase']} method={r['method']} "
            f"top1={r['top1']} rmse={r['rmse']} rfid={r['rfid']}"
        )


if __name__ == "__main__":
    main()

