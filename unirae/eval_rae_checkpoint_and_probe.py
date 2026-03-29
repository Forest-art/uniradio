from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image


class HFImageDataset(Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), int(item.get("label", -1))


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def _import_rae_class(rae_code_root: str):
    root = Path(rae_code_root)
    src = root / "src" if (root / "src").exists() else root
    if not src.exists():
        raise FileNotFoundError(f"Cannot find RAE source dir under: {rae_code_root}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from stage1.rae import RAE  # type: ignore

    return RAE


def _resolve_defaults(
    rae_code_root: str,
    decoder_config_path: Optional[str],
    pretrained_decoder_path: Optional[str],
    normalization_stat_path: Optional[str],
) -> Tuple[str, str, str]:
    root = Path(rae_code_root)
    dec_cfg = decoder_config_path or str(root / "configs" / "decoder" / "ViTXL")
    dec_ckpt = pretrained_decoder_path or str(
        root / "models" / "decoders" / "dinov2" / "wReg_base" / "ViTXL_n08" / "model.pt"
    )
    stat_ckpt = normalization_stat_path or str(
        root / "models" / "stats" / "dinov2" / "wReg_base" / "imagenet1k" / "stat.pt"
    )
    return dec_cfg, dec_ckpt, stat_ckpt


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


@torch.no_grad()
def evaluate_reconstruction(
    rae_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int,
    sample_path: str,
    sample_items: int,
) -> Dict[str, float]:
    rae_model.eval()

    sum_mse = 0.0
    sum_psnr = 0.0
    count = 0
    saved = False
    sample_file = Path(sample_path)
    sample_file.parent.mkdir(parents=True, exist_ok=True)

    for bi, (images, _) in enumerate(val_loader):
        if max_batches > 0 and bi >= max_batches:
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

        if (not saved) and sample_items > 0:
            n = min(int(sample_items), bs)
            pair = torch.cat([target[:n].cpu(), rec[:n].cpu()], dim=0)
            grid = make_grid(pair, nrow=n, padding=2)
            save_image(grid, str(sample_file))
            saved = True

    denom = max(count, 1)
    mse_mean = sum_mse / denom
    rmse_mean = mse_mean**0.5
    psnr_mean = sum_psnr / denom
    return {
        "mse": float(mse_mean),
        "rmse": float(rmse_mean),
        "psnr": float(psnr_mean),
        "num_samples": int(count),
        "sample_image": str(sample_file),
    }


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
    for _ in range(int(max_steps)):
        images, labels = next(train_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            feat = _extract_features(rae_model, images)
        logits = head(feat)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    head.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feat = _extract_features(rae_model, images)
            pred = head(feat).argmax(dim=-1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())

    top1 = float(correct / max(total, 1))
    return {
        "linear_probe_top1": top1,
        "linear_probe_top1_percent": 100.0 * top1,
        "num_val_samples": int(total),
        "probe_steps": int(max_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAE checkpoint: reconstruction + sample images + linear probe.")
    parser.add_argument("--checkpoint", required=True, help="Path to DSGA/RAE training checkpoint (expects key: model).")
    parser.add_argument("--rae_code_root", default="/project/peilab/luxiaocheng/projects/RAE")
    parser.add_argument("--decoder_config_path", default="")
    parser.add_argument("--pretrained_decoder_path", default="")
    parser.add_argument("--normalization_stat_path", default="")

    parser.add_argument("--hf_dataset", default="frgfm/imagenette")
    parser.add_argument("--hf_config", default="160px")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_eval_batches", type=int, default=50)
    parser.add_argument("--sample_items", type=int, default=8)

    parser.add_argument("--probe_steps", type=int, default=1200)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)

    parser.add_argument("--out_dir", default="results/rae_ckpt_eval")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = _build_transform(int(args.image_size))
    hf_config = str(args.hf_config).strip()
    if hf_config:
        train_hf = load_dataset(args.hf_dataset, hf_config, split=args.train_split)
        val_hf = load_dataset(args.hf_dataset, hf_config, split=args.val_split)
        ds_tag = f"{args.hf_dataset}:{hf_config}"
    else:
        train_hf = load_dataset(args.hf_dataset, split=args.train_split)
        val_hf = load_dataset(args.hf_dataset, split=args.val_split)
        ds_tag = str(args.hf_dataset)
    train_ds = HFImageDataset(train_hf, transform=tfm)
    val_ds = HFImageDataset(val_hf, transform=tfm)
    num_classes = int(len(train_hf.features["label"].names))

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

    dec_cfg, dec_ckpt, stat_ckpt = _resolve_defaults(
        rae_code_root=args.rae_code_root,
        decoder_config_path=(args.decoder_config_path or None),
        pretrained_decoder_path=(args.pretrained_decoder_path or None),
        normalization_stat_path=(args.normalization_stat_path or None),
    )
    for p in [dec_cfg, dec_ckpt, stat_ckpt, args.checkpoint]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file/path not found: {p}")

    RAE = _import_rae_class(args.rae_code_root)
    model = RAE(
        encoder_cls="Dinov2withNorm",
        encoder_config_path="facebook/dinov2-with-registers-base",
        encoder_input_size=224,
        encoder_params={"dinov2_path": "facebook/dinov2-with-registers-base", "normalize": True},
        decoder_config_path=dec_cfg,
        pretrained_decoder_path=dec_ckpt,
        noise_tau=0.0,
        reshape_to_2d=True,
        normalization_stat_path=stat_ckpt,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_ret = model.load_state_dict(state_model, strict=False)

    print(
        f"[load] ckpt={args.checkpoint} missing={len(load_ret.missing_keys)} unexpected={len(load_ret.unexpected_keys)}"
    )

    recon_ret = evaluate_reconstruction(
        rae_model=model,
        val_loader=val_loader,
        device=device,
        max_batches=int(args.max_eval_batches),
        sample_path=str(out_dir / "recon_samples.png"),
        sample_items=int(args.sample_items),
    )
    print(f"[reconstruction] {recon_ret}")

    probe_ret = evaluate_linear_probe(
        rae_model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        max_steps=int(args.probe_steps),
        lr=float(args.probe_lr),
        weight_decay=float(args.probe_weight_decay),
    )
    print(f"[linear_probe] {probe_ret}")

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset": ds_tag,
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "num_classes": int(num_classes),
        "device": str(device),
        "load_missing_count": int(len(load_ret.missing_keys)),
        "load_unexpected_count": int(len(load_ret.unexpected_keys)),
        "reconstruction": recon_ret,
        "linear_probe": probe_ret,
    }
    out_json = out_dir / "eval_summary.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote={out_json}")


if __name__ == "__main__":
    main()
