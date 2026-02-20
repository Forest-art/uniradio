from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _resolve_split_root(data_root: str, split: str) -> str:
    root = Path(data_root)
    candidates = [
        root / split,
        root / "imagenet" / split,
        root / "ILSVRC" / "Data" / "CLS-LOC" / split,
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"Cannot find ImageNet split={split} under {data_root}. "
        f"Tried: {', '.join(str(x) for x in candidates)}"
    )


def _load_class_names(class_names_file: Optional[str], default_names: List[str]) -> List[str]:
    if not class_names_file:
        return default_names
    path = Path(class_names_file)
    if not path.exists():
        raise FileNotFoundError(f"class_names_file not found: {class_names_file}")

    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(default_names):
        raise ValueError(
            f"class_names_file length mismatch. expected={len(default_names)}, got={len(names)}"
        )
    return names


def build_imagenet_transforms(image_size: int, split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_imagenet_dataset(
    data_root: str,
    split: str,
    image_size: int,
    class_names_file: Optional[str] = None,
):
    split_root = _resolve_split_root(data_root, split)
    tfm = build_imagenet_transforms(image_size=image_size, split=split)
    ds = datasets.ImageFolder(root=split_root, transform=tfm)
    class_names = _load_class_names(class_names_file=class_names_file, default_names=list(ds.classes))
    return ds, class_names


def build_imagenet_loader(
    data_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    class_names_file: Optional[str] = None,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
) -> Tuple[DataLoader, List[str]]:
    ds, class_names = build_imagenet_dataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        class_names_file=class_names_file,
    )

    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, class_names


def make_batch_dict(batch) -> Dict[str, torch.Tensor]:
    images, labels = batch
    return {"images": images, "labels": labels}
