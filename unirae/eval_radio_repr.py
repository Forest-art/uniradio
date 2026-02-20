import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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

    # Align with RADIO official logic in examples/common/model_loader.py
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


@torch.no_grad()
def extract_features(model, loader, device, adaptor_name: str = None):
    feats = []
    labels = []
    model.eval()

    for images, y in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)

        if isinstance(out, dict):
            key = adaptor_name if adaptor_name else "backbone"
            if key not in out:
                raise KeyError(f"Output key '{key}' missing. Available keys: {list(out.keys())}")
            feat = out[key].summary
        else:
            feat = out.summary

        feats.append(feat.float().cpu())
        labels.append(y)

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def knn_top1(train_x, train_y, val_x, val_y, k: int) -> float:
    k = min(k, train_x.shape[0])
    train_n = F.normalize(train_x, dim=1)
    val_n = F.normalize(val_x, dim=1)

    sim = val_n @ train_n.T
    idx = sim.topk(k=k, dim=1).indices

    preds = []
    for i in range(idx.shape[0]):
        cls = train_y[idx[i]].tolist()
        preds.append(Counter(cls).most_common(1)[0][0])

    pred = torch.tensor(preds)
    return float((pred == val_y).float().mean().item() * 100.0)


def linear_probe_top1(
    train_x,
    train_y,
    val_x,
    val_y,
    num_classes: int,
    steps: int,
    lr: float,
    weight_decay: float,
) -> float:
    clf = nn.Linear(train_x.shape[1], num_classes)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(steps):
        logits = clf(train_x)
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        pred = clf(val_x).argmax(dim=1)
        return float((pred == val_y).float().mean().item() * 100.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RADIO representation with kNN + linear probe")
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
    parser.add_argument("--linear_steps", type=int, default=2000)
    parser.add_argument("--linear_lr", type=float, default=0.05)
    parser.add_argument("--linear_wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_radio_model(args, device)

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
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    train_x, train_y = extract_features(model, train_loader, device, adaptor_name=args.adaptor_name)
    val_x, val_y = extract_features(model, val_loader, device, adaptor_name=args.adaptor_name)

    knn_acc = knn_top1(train_x, train_y, val_x, val_y, k=args.k)
    lin_acc = linear_probe_top1(
        train_x,
        train_y,
        val_x,
        val_y,
        num_classes=num_classes,
        steps=args.linear_steps,
        lr=args.linear_lr,
        weight_decay=args.linear_wd,
    )

    result: Dict = {
        "model_version": args.model_version,
        "adaptor_name": args.adaptor_name,
        "use_huggingface": bool(args.use_huggingface),
        "use_local_lib": bool(args.use_local_lib),
        "torchhub_repo": args.torchhub_repo,
        "resolution": list(res),
        "feature_dim": int(train_x.shape[1]),
        "num_classes": num_classes,
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "k": int(args.k),
        "knn_top1": knn_acc,
        "linear_probe_top1": lin_acc,
        "feature_mean": float(train_x.mean().item()),
        "feature_std": float(train_x.std().item()),
        "seed": int(args.seed),
    }

    print(json.dumps(result, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
