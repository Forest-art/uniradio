import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_radio_import(radio_code_root: str) -> None:
    root = Path(radio_code_root)
    if not (root / "hubconf.py").exists():
        raise FileNotFoundError(f"Cannot find hubconf.py in radio_code_root={radio_code_root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_radio_model(args, device: torch.device):
    kwargs = {
        "version": args.model_version,
        "adaptor_names": args.adaptor_name,
        "vitdet_window_size": args.vitdet_window_size,
    }

    if not (os.path.isfile(args.model_version) or "radio" in args.model_version):
        raise ValueError(
            f"Unsupported model_version={args.model_version}. "
            "This script is for RADIO checkpoints/versions."
        )

    if args.use_huggingface:
        from transformers import AutoConfig, AutoModel

        hf_repo = "E-RADIO" if "eradio" in args.model_version else "RADIO"
        hf_repo = f"nvidia/{hf_repo}"
        config = AutoConfig.from_pretrained(
            hf_repo,
            trust_remote_code=True,
            version=args.model_version,
            adaptor_names=args.adaptor_name,
            vitdet_window_size=args.vitdet_window_size,
        )
        model = AutoModel.from_pretrained(hf_repo, config=config, trust_remote_code=True)
    elif args.use_local_lib:
        if not args.radio_code_root:
            raise ValueError("--radio_code_root is required when --use_local_lib is enabled.")
        _prepare_radio_import(args.radio_code_root)
        from hubconf import radio_model

        model = radio_model(**kwargs)
    else:
        model = torch.hub.load(
            args.torchhub_repo,
            "radio_model",
            progress=True,
            force_reload=args.force_reload,
            **kwargs,
        )

    model = model.to(device).eval()
    return model


def _build_transform(resolution: Tuple[int, int], preprocessor=None):
    tf = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ]
    if preprocessor is not None:
        tf.append(preprocessor)
    return transforms.Compose(tf)


def _summary_from_output(output, adaptor_name: Optional[str]):
    if isinstance(output, dict):
        key = adaptor_name if adaptor_name else "backbone"
        if key not in output:
            raise KeyError(f"Output key '{key}' missing. Available keys: {list(output.keys())}")
        feat = output[key].summary
    else:
        feat = output.summary
    return feat


@torch.no_grad()
def _extract_feature_batch(model, images: torch.Tensor, adaptor_name: Optional[str]) -> torch.Tensor:
    out = model(images)
    feat = _summary_from_output(out, adaptor_name)
    return feat.float()


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _reduce_mean(accelerator: Accelerator, x: torch.Tensor) -> float:
    return float(accelerator.reduce(x.detach().float(), reduction="mean").item())


@torch.no_grad()
def eval_linear_probe(
    accelerator: Accelerator,
    model,
    classifier: nn.Module,
    val_loader,
    adaptor_name: Optional[str],
) -> float:
    model.eval()
    classifier.eval()

    total = 0
    correct = 0

    for images, labels in val_loader:
        images = images.to(accelerator.device, non_blocking=True)
        labels = labels.to(accelerator.device, non_blocking=True)
        feat = _extract_feature_batch(model, images, adaptor_name)
        logits = classifier(feat)
        pred = logits.argmax(dim=1)

        pred_g = accelerator.gather_for_metrics(pred)
        label_g = accelerator.gather_for_metrics(labels)

        if accelerator.is_main_process:
            total += int(label_g.numel())
            correct += int((pred_g == label_g).sum().item())

    acc = (100.0 * correct / max(total, 1)) if accelerator.is_main_process else 0.0
    obj = [acc]
    broadcast_object_list(obj)
    return float(obj[0])


def maybe_run_knn(
    accelerator: Accelerator,
    model,
    train_loader,
    val_loader,
    adaptor_name: Optional[str],
    k: int,
    max_knn_train: int,
    max_knn_val: int,
) -> Optional[float]:
    model.eval()

    def collect(loader, limit: int):
        feats = []
        labels = []
        seen = 0

        for images, y in loader:
            images = images.to(accelerator.device, non_blocking=True)
            y = y.to(accelerator.device, non_blocking=True)
            f = _extract_feature_batch(model, images, adaptor_name)

            fg = accelerator.gather_for_metrics(f)
            yg = accelerator.gather_for_metrics(y)

            stop = 0
            if accelerator.is_main_process:
                feats.append(fg.cpu())
                labels.append(yg.cpu())
                seen += int(yg.numel())
                if limit > 0 and seen >= limit:
                    stop = 1

            stop_t = torch.tensor(stop, device=accelerator.device, dtype=torch.int32)
            stop_t = accelerator.reduce(stop_t, reduction="max")
            if int(stop_t.item()) > 0:
                break

        if not accelerator.is_main_process:
            return None, None

        if not feats:
            return torch.empty(0), torch.empty(0, dtype=torch.long)

        xf = torch.cat(feats, dim=0)
        yf = torch.cat(labels, dim=0)
        if limit > 0 and xf.shape[0] > limit:
            xf = xf[:limit]
            yf = yf[:limit]
        return xf, yf

    train_x, train_y = collect(train_loader, max_knn_train)
    val_x, val_y = collect(val_loader, max_knn_val)

    if accelerator.is_main_process:
        if train_x is None or val_x is None or train_x.numel() == 0 or val_x.numel() == 0:
            knn_acc = None
        else:
            k_use = min(k, train_x.shape[0])
            train_n = F.normalize(train_x, dim=1)
            val_n = F.normalize(val_x, dim=1)
            sim = val_n @ train_n.T
            idx = sim.topk(k=k_use, dim=1).indices

            preds = []
            for i in range(idx.shape[0]):
                cls = train_y[idx[i]].tolist()
                preds.append(Counter(cls).most_common(1)[0][0])

            pred = torch.tensor(preds)
            knn_acc = float((pred == val_y).float().mean().item() * 100.0)
    else:
        knn_acc = None

    obj = [knn_acc]
    broadcast_object_list(obj)
    return obj[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate RADIO representation with accelerate multi-GPU linear probe")
    parser.add_argument(
        "--radio_code_root",
        default=None,
        help="Path to RADIO repo root (contains hubconf.py). Required when --use_local_lib.",
    )
    parser.add_argument("--data_root", required=True, help="ImageNet root that contains train/ and val/")
    parser.add_argument("--model_version", default="radio_v2", help="RADIO model version key or checkpoint path")
    parser.add_argument("--adaptor_name", default=None, help="Optional adaptor name, default uses backbone summary")
    parser.add_argument("--use_huggingface", action="store_true", help="Match RADIO official HF loading path")
    parser.add_argument("--use_local_lib", dest="use_local_lib", action="store_true", default=True)
    parser.add_argument("--no_use_local_lib", dest="use_local_lib", action="store_false")
    parser.add_argument("--torchhub_repo", default="NVlabs/RADIO", help="Used when --no_use_local_lib")
    parser.add_argument("--force_reload", action="store_true", help="torch.hub force reload")
    parser.add_argument("--vitdet_window_size", type=int, default=None)

    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--run_knn", action="store_true", help="Run kNN on gathered features")
    parser.add_argument("--max_knn_train", type=int, default=50000)
    parser.add_argument("--max_knn_val", type=int, default=10000)

    parser.add_argument("--linear_steps", type=int, default=2000)
    parser.add_argument("--linear_lr", type=float, default=0.05)
    parser.add_argument("--linear_wd", type=float, default=1e-4)

    parser.add_argument("--mixed_precision", default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    seed_everything(args.seed)
    set_seed(args.seed, device_specific=True)

    model = _load_radio_model(args, accelerator.device)
    for p in model.parameters():
        p.requires_grad = False

    res = (model.preferred_resolution.height, model.preferred_resolution.width)
    preprocessor = model.make_preprocessor_external() if hasattr(model, "make_preprocessor_external") else None
    transform = _build_transform(res, preprocessor=preprocessor)

    train_dir = Path(args.data_root) / args.train_split
    val_dir = Path(args.data_root) / args.val_split
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected directories not found: {train_dir} and/or {val_dir}. data_root must contain train/ and val/."
        )

    train_ds = datasets.ImageFolder(str(train_dir), transform=transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=transform)
    if train_ds.classes != val_ds.classes:
        raise ValueError("train/val class folders mismatch")

    num_classes = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = accelerator.prepare_model(model, evaluation_mode=True)

    sample_batch = next(iter(train_loader))[0].to(accelerator.device)
    with torch.no_grad():
        feat_dim = int(_extract_feature_batch(model, sample_batch, args.adaptor_name).shape[-1])

    classifier = nn.Linear(feat_dim, num_classes).to(accelerator.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.linear_lr, weight_decay=args.linear_wd)

    classifier, optimizer, train_loader, val_loader = accelerator.prepare(classifier, optimizer, train_loader, val_loader)

    train_iter = _cycle(train_loader)
    progress = tqdm(range(1, args.linear_steps + 1), disable=not accelerator.is_local_main_process, desc="linear_probe")

    for step in progress:
        images, labels = next(train_iter)
        images = images.to(accelerator.device, non_blocking=True)
        labels = labels.to(accelerator.device, non_blocking=True)

        with torch.no_grad():
            feat = _extract_feature_batch(model, images, args.adaptor_name)

        logits = classifier(feat)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        optimizer.step()

        if step == 1 or step == args.linear_steps or step % max(1, args.linear_steps // 20) == 0:
            loss_mean = _reduce_mean(accelerator, loss)
            progress.set_postfix(loss=f"{loss_mean:.4f}")

    linear_acc = eval_linear_probe(
        accelerator=accelerator,
        model=model,
        classifier=classifier,
        val_loader=val_loader,
        adaptor_name=args.adaptor_name,
    )

    knn_acc = None
    if args.run_knn:
        knn_acc = maybe_run_knn(
            accelerator=accelerator,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            adaptor_name=args.adaptor_name,
            k=args.k,
            max_knn_train=args.max_knn_train,
            max_knn_val=args.max_knn_val,
        )

    result: Dict = {
        "model_version": args.model_version,
        "adaptor_name": args.adaptor_name,
        "use_huggingface": bool(args.use_huggingface),
        "use_local_lib": bool(args.use_local_lib),
        "torchhub_repo": args.torchhub_repo,
        "resolution": list(res),
        "feature_dim": feat_dim,
        "num_classes": num_classes,
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "k": int(args.k),
        "run_knn": bool(args.run_knn),
        "knn_top1": knn_acc,
        "linear_probe_top1": float(linear_acc),
        "linear_steps": int(args.linear_steps),
        "world_size": int(accelerator.num_processes),
        "seed": int(args.seed),
    }

    if accelerator.is_main_process:
        print(json.dumps(result, indent=2))
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
