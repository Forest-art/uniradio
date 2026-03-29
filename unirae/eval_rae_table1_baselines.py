from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


class HFImageDataset(Dataset):
    def __init__(self, hf_ds, transform) -> None:
        self.ds = hf_ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        image = item["image"]
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        label = int(item.get("label", -1))
        return self.transform(image), label


def _import_rae_class(rae_code_root: str):
    root = Path(rae_code_root)
    src = root / "src" if (root / "src").exists() else root
    if not src.exists():
        raise FileNotFoundError(f"Cannot find RAE source dir under: {rae_code_root}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from stage1.rae import RAE  # type: ignore

    return RAE


def _resolve_stage1_params(stage1_config: str, rae_code_root: str) -> Dict:
    cfg_path = Path(stage1_config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"stage1 config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "stage_1" not in cfg or "params" not in cfg["stage_1"]:
        raise ValueError(f"Invalid stage1 config: missing stage_1.params: {cfg_path}")

    params = dict(cfg["stage_1"]["params"])
    root = Path(rae_code_root).expanduser().resolve()
    for key in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        v = params.get(key, "")
        if isinstance(v, str) and v:
            p = Path(v)
            if not p.is_absolute():
                p = root / p
            params[key] = str(p.resolve())
    return params


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


@torch.no_grad()
def _extract_features(rae_model: nn.Module, images_01: torch.Tensor) -> torch.Tensor:
    z = rae_model.encode(images_01)
    if z.ndim == 4:
        return z.mean(dim=(2, 3))
    if z.ndim == 3:
        return z.mean(dim=1)
    if z.ndim == 2:
        return z
    raise RuntimeError(f"Unsupported latent shape: {tuple(z.shape)}")


def evaluate_linear_probe(
    rae_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_classes: int,
    device: torch.device,
    max_steps: int,
    lr: float,
    weight_decay: float,
    probe_log_interval: int,
) -> Dict[str, float]:
    rae_model.eval()
    for p in rae_model.parameters():
        p.requires_grad = False

    first_images, _ = next(iter(train_loader))
    first_images = first_images.to(device, non_blocking=True)
    feat_dim = int(_extract_features(rae_model, first_images).shape[-1])

    head = nn.Linear(feat_dim, int(num_classes)).to(device)
    optimizer = AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_iter = cycle(train_loader)

    head.train()
    train_t0 = time.time()
    seen_samples = 0
    max_steps = int(max_steps)
    probe_log_interval = int(probe_log_interval)
    for step in range(1, max_steps + 1):
        images, labels = next(train_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        seen_samples += int(images.shape[0])

        with torch.no_grad():
            feat = _extract_features(rae_model, images)
        logits = head(feat)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_log = probe_log_interval > 0 and (
            step == 1 or step % probe_log_interval == 0 or step == max_steps
        )
        if should_log:
            elapsed = time.time() - train_t0
            step_per_sec = step / max(elapsed, 1e-8)
            img_per_sec = seen_samples / max(elapsed, 1e-8)
            eta_sec = (max_steps - step) / max(step_per_sec, 1e-8)
            print(
                f"[linear_probe][train] step={step}/{max_steps} "
                f"loss={loss.item():.6f} step_per_sec={step_per_sec:.2f} "
                f"img_per_sec={img_per_sec:.1f} eta={eta_sec:.1f}s",
                flush=True,
            )

    head.eval()
    total = 0
    correct = 0
    val_total_batches = int(len(val_loader))
    val_log_interval = max(1, val_total_batches // 10)
    with torch.no_grad():
        for bi, (images, labels) in enumerate(val_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feat = _extract_features(rae_model, images)
            pred = head(feat).argmax(dim=-1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())
            if bi == 1 or bi % val_log_interval == 0 or bi == val_total_batches:
                print(
                    f"[linear_probe][val] batch={bi}/{val_total_batches} seen={total}",
                    flush=True,
                )

    top1 = float(correct / max(total, 1))
    return {
        "linear_probe_top1": top1,
        "linear_probe_top1_percent": 100.0 * top1,
        "probe_steps": int(max_steps),
        "num_val_samples": int(total),
    }


@torch.no_grad()
def evaluate_reconstruction(
    rae_model: nn.Module,
    val_loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    recon_log_interval: int,
) -> Dict[str, float]:
    rae_model.eval()
    sum_mse = 0.0
    sum_psnr = 0.0
    count = 0

    max_batches = int(max_batches)
    recon_log_interval = int(recon_log_interval)
    planned_batches = int(len(val_loader)) if max_batches <= 0 else min(int(len(val_loader)), max_batches)
    for bi, (images, _) in enumerate(val_loader):
        if max_batches > 0 and bi >= int(max_batches):
            break
        images = images.to(device, non_blocking=True)
        rec = rae_model(images).clamp(0.0, 1.0)
        target = images
        if rec.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=rec.shape[-2:], mode="bilinear", align_corners=False)

        mse = F.mse_loss(rec, target)
        psnr = -10.0 * torch.log10(mse + 1e-8)
        bs = int(images.shape[0])
        sum_mse += float(mse.item()) * bs
        sum_psnr += float(psnr.item()) * bs
        count += bs
        batch_i = bi + 1
        if recon_log_interval > 0 and (
            batch_i == 1 or batch_i % recon_log_interval == 0 or batch_i == planned_batches
        ):
            print(
                f"[reconstruction][progress] batch={batch_i}/{planned_batches} "
                f"seen={count}",
                flush=True,
            )

    denom = max(count, 1)
    mse_mean = sum_mse / denom
    rmse_mean = mse_mean**0.5
    psnr_mean = sum_psnr / denom
    return {
        "recon_mse": float(mse_mean),
        "recon_rmse": float(rmse_mean),
        "recon_psnr": float(psnr_mean),
        "recon_num_samples": int(count),
    }


@torch.no_grad()
def evaluate_rfid(
    rae_model: nn.Module,
    val_loader: DataLoader,
    *,
    device: torch.device,
    rfid_num_samples: int,
    rfid_batch_size: int,
    rfid_tmp_dir: str,
    rfid_log_interval: int,
) -> Tuple[float, int, str]:
    if int(rfid_num_samples) <= 1:
        return float("nan"), 0, "rfid_num_samples<=1"

    tmp_root = tempfile.TemporaryDirectory(prefix="rae_table1_rfid_", dir=(rfid_tmp_dir or None))
    root = Path(tmp_root.name)
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    rfid_log_interval = int(rfid_log_interval)
    for images, _ in val_loader:
        if saved >= int(rfid_num_samples):
            break
        images = images.to(device, non_blocking=True)
        rec = rae_model(images).clamp(0.0, 1.0)
        target = images
        if rec.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=rec.shape[-2:], mode="bilinear", align_corners=False)
        bsz = int(images.shape[0])
        for i in range(bsz):
            if saved >= int(rfid_num_samples):
                break
            save_image(target[i].cpu(), str(real_dir / f"{saved:07d}.png"))
            save_image(rec[i].cpu(), str(fake_dir / f"{saved:07d}.png"))
            saved += 1
            if rfid_log_interval > 0 and (
                saved == 1 or saved % rfid_log_interval == 0 or saved == int(rfid_num_samples)
            ):
                print(f"[rfid][save] {saved}/{int(rfid_num_samples)}", flush=True)

    rfid = float("nan")
    rfid_error = ""
    if saved > 1:
        try:
            from torch_fidelity import calculate_metrics

            print(f"[rfid] computing FID with {saved} image pairs...", flush=True)
            ret = calculate_metrics(
                input1=str(real_dir),
                input2=str(fake_dir),
                cuda=(device.type == "cuda"),
                isc=False,
                fid=True,
                kid=False,
                prc=False,
                verbose=False,
                batch_size=int(rfid_batch_size),
            )
            rfid = float(ret.get("frechet_inception_distance", float("nan")))
        except Exception as e:  # noqa: BLE001
            rfid_error = str(e)
            if isinstance(e, ModuleNotFoundError) and "torch_fidelity" in str(e):
                rfid_error = f"{rfid_error}; install with: pip install torch-fidelity"
    else:
        rfid_error = "not enough samples for rfid"
    tmp_root.cleanup()
    return rfid, int(saved), rfid_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAE pretrained baseline: linear probe + reconstruction rMSE + rFID."
    )
    parser.add_argument("--rae_code_root", default="/project/peilab/luxiaocheng/projects/RAE")
    parser.add_argument("--stage1_config", required=True, help="RAE stage1 pretrained yaml, e.g. DINOv2-B.yaml")

    parser.add_argument("--hf_dataset", default="clane9/imagenet-100")
    parser.add_argument("--hf_config", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")

    parser.add_argument("--image_size", type=int, default=0, help="0 means use encoder_input_size from config.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--probe_steps", type=int, default=10000)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)
    parser.add_argument("--probe_log_interval", type=int, default=100)

    parser.add_argument("--max_eval_batches", type=int, default=0, help="0 means full val set for rMSE/MSE.")
    parser.add_argument("--recon_log_interval", type=int, default=20)
    parser.add_argument("--rfid_num_samples", type=int, default=5000)
    parser.add_argument("--rfid_batch_size", type=int, default=64)
    parser.add_argument("--rfid_tmp_dir", default="/scratch/peilab/xlubl/tmp_rfid")
    parser.add_argument("--rfid_log_interval", type=int, default=500)

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    t0 = time.time()
    stage1_params = _resolve_stage1_params(args.stage1_config, args.rae_code_root)
    image_size = int(args.image_size) if int(args.image_size) > 0 else int(stage1_params.get("encoder_input_size", 256))
    tfm = _build_transform(image_size=image_size)

    hf_cfg = str(args.hf_config).strip()
    if hf_cfg:
        train_hf = load_dataset(args.hf_dataset, hf_cfg, split=args.train_split)
        val_hf = load_dataset(args.hf_dataset, hf_cfg, split=args.val_split)
        dataset_tag = f"{args.hf_dataset}:{hf_cfg}"
    else:
        train_hf = load_dataset(args.hf_dataset, split=args.train_split)
        val_hf = load_dataset(args.hf_dataset, split=args.val_split)
        dataset_tag = str(args.hf_dataset)

    train_ds = HFImageDataset(train_hf, transform=tfm)
    val_ds = HFImageDataset(val_hf, transform=tfm)
    if hasattr(train_hf.features.get("label", None), "names") and train_hf.features["label"].names is not None:
        num_classes = int(len(train_hf.features["label"].names))
    else:
        num_classes = int(len(train_hf.unique("label")))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    RAE = _import_rae_class(args.rae_code_root)
    model = RAE(**stage1_params).to(device)
    model.eval()

    print(f"[setup] device={device} dataset={dataset_tag} image_size={image_size} num_classes={num_classes}")
    print(f"[setup] stage1_config={Path(args.stage1_config).resolve()}")
    print(f"[setup] stage1_params.encoder_cls={stage1_params.get('encoder_cls', '')}")
    print(
        f"[setup] train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"probe_steps={int(args.probe_steps)}",
        flush=True,
    )

    lp_t0 = time.time()
    probe_ret = evaluate_linear_probe(
        rae_model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=int(num_classes),
        device=device,
        max_steps=int(args.probe_steps),
        lr=float(args.probe_lr),
        weight_decay=float(args.probe_weight_decay),
        probe_log_interval=int(args.probe_log_interval),
    )
    probe_wall = time.time() - lp_t0
    print(f"[linear_probe] top1={probe_ret['linear_probe_top1']:.4f} wall={probe_wall:.1f}s")

    rec_t0 = time.time()
    recon_ret = evaluate_reconstruction(
        rae_model=model,
        val_loader=val_loader,
        device=device,
        max_batches=int(args.max_eval_batches),
        recon_log_interval=int(args.recon_log_interval),
    )
    rec_wall = time.time() - rec_t0
    print(
        f"[reconstruction] rmse={recon_ret['recon_rmse']:.6f} "
        f"mse={recon_ret['recon_mse']:.6f} wall={rec_wall:.1f}s"
    )

    rfid_t0 = time.time()
    rfid, rfid_saved, rfid_error = evaluate_rfid(
        rae_model=model,
        val_loader=val_loader,
        device=device,
        rfid_num_samples=int(args.rfid_num_samples),
        rfid_batch_size=int(args.rfid_batch_size),
        rfid_tmp_dir=str(args.rfid_tmp_dir),
        rfid_log_interval=int(args.rfid_log_interval),
    )
    rfid_wall = time.time() - rfid_t0
    if rfid == rfid:
        print(f"[rfid] rfid={rfid:.4f} n={rfid_saved} wall={rfid_wall:.1f}s")
    else:
        print(f"[rfid] rfid=nan n={rfid_saved} err={rfid_error}")

    out = {
        "stage1_config": str(Path(args.stage1_config).resolve()),
        "dataset": dataset_tag,
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "image_size": int(image_size),
        "num_classes": int(num_classes),
        "probe": probe_ret,
        "reconstruction": recon_ret,
        "rfid": {
            "value": float(rfid),
            "num_samples": int(rfid_saved),
            "error": str(rfid_error),
        },
        "timing_sec": {
            "linear_probe": float(probe_wall),
            "reconstruction": float(rec_wall),
            "rfid": float(rfid_wall),
            "total": float(time.time() - t0),
        },
    }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote={out_path}")


if __name__ == "__main__":
    main()
