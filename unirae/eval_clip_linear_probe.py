from __future__ import annotations

import argparse
import json
import time
from itertools import cycle
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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


def _build_clip_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


@torch.no_grad()
def _extract_features(clip_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    out = clip_model(pixel_values=images)
    if out.pooler_output is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 0, :]


def evaluate_linear_probe(
    *,
    clip_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    max_steps: int,
    lr: float,
    weight_decay: float,
    log_interval: int,
) -> Dict[str, float]:
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    first_images, _ = next(iter(train_loader))
    first_images = first_images.to(device, non_blocking=True)
    feat_dim = int(_extract_features(clip_model, first_images).shape[-1])

    head = nn.Linear(feat_dim, int(num_classes)).to(device)
    optimizer = AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_iter = cycle(train_loader)

    train_t0 = time.time()
    seen_samples = 0
    max_steps = int(max_steps)
    log_interval = int(log_interval)

    head.train()
    for step in range(1, max_steps + 1):
        images, labels = next(train_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        seen_samples += int(images.shape[0])

        with torch.no_grad():
            feat = _extract_features(clip_model, images)
        logits = head(feat)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == max_steps):
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
            feat = _extract_features(clip_model, images)
            pred = head(feat).argmax(dim=-1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())
            if bi == 1 or bi % val_log_interval == 0 or bi == val_total_batches:
                print(f"[linear_probe][val] batch={bi}/{val_total_batches} seen={total}", flush=True)

    top1 = float(correct / max(total, 1))
    return {
        "linear_probe_top1": top1,
        "linear_probe_top1_percent": 100.0 * top1,
        "probe_steps": int(max_steps),
        "num_val_samples": int(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP vision encoder with linear probe on ImageNet-100.")
    parser.add_argument("--clip_vision_model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--hf_dataset", default="clane9/imagenet-100")
    parser.add_argument("--hf_config", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--probe_steps", type=int, default=2000)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)
    parser.add_argument("--probe_log_interval", type=int, default=100)

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    t0 = time.time()
    tfm = _build_clip_transform(image_size=int(args.image_size))

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

    from transformers import CLIPVisionModel

    clip_model = CLIPVisionModel.from_pretrained(str(args.clip_vision_model)).to(device)
    clip_model.eval()

    print(f"[setup] device={device} dataset={dataset_tag} image_size={int(args.image_size)} num_classes={num_classes}")
    print(f"[setup] clip_vision_model={args.clip_vision_model}")
    print(
        f"[setup] train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"probe_steps={int(args.probe_steps)}",
        flush=True,
    )

    lp_t0 = time.time()
    probe_ret = evaluate_linear_probe(
        clip_model=clip_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=int(num_classes),
        device=device,
        max_steps=int(args.probe_steps),
        lr=float(args.probe_lr),
        weight_decay=float(args.probe_weight_decay),
        log_interval=int(args.probe_log_interval),
    )
    probe_wall = time.time() - lp_t0
    print(f"[linear_probe] top1={probe_ret['linear_probe_top1']:.4f} wall={probe_wall:.1f}s")

    out = {
        "clip_vision_model": str(args.clip_vision_model),
        "dataset": dataset_tag,
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
        "num_classes": int(num_classes),
        "probe": probe_ret,
        "timing_sec": {
            "linear_probe": float(probe_wall),
            "total": float(time.time() - t0),
        },
    }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote={out_path}")


if __name__ == "__main__":
    main()
