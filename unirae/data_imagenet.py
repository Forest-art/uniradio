from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
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
    if len(default_names) > 0 and len(names) != len(default_names):
        raise ValueError(
            f"class_names_file length mismatch. expected={len(default_names)}, got={len(names)}"
        )
    return names


def _looks_like_hf_disk_dataset(path: str) -> bool:
    p = Path(path)
    return any((p / name).exists() for name in ("dataset_info.json", "dataset_dict.json", "state.json"))


def _resolve_hf_split(ds, split: str):
    if not hasattr(ds, "keys"):
        return ds

    keys = set(ds.keys())
    if split in keys:
        return ds[split]

    aliases = {
        "val": "validation",
        "validation": "val",
        "test": "validation",
    }
    alt = aliases.get(split)
    if alt and alt in keys:
        return ds[alt]

    raise ValueError(f"Split '{split}' not found in HF dataset. Available splits: {sorted(keys)}")


def _infer_hf_class_names(hf_ds, label_key: str, class_names_file: Optional[str]) -> List[str]:
    names = []
    feat = None
    if hasattr(hf_ds, "features"):
        feat = hf_ds.features.get(label_key)

    if feat is not None and hasattr(feat, "names") and feat.names is not None:
        names = list(feat.names)
    else:
        # Fallback: infer label range from dataset unique labels.
        if hasattr(hf_ds, "unique"):
            uniq = sorted(int(x) for x in hf_ds.unique(label_key))
            names = [str(i) for i in uniq]

    return _load_class_names(class_names_file, names)


class HFDiskVisionDataset(Dataset):
    def __init__(self, hf_dataset, transform, image_key: str = "image", label_key: str = "label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_key]
        label = int(item[self.label_key])

        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class HFImageNetDataset(Dataset):
    """HuggingFace ImageNet dataset wrapper with resilient sample loading.

    行为与用户给出的逻辑一致：
    - 读取 item['image'] 与 item['label']
    - 灰度图转 RGB
    - 如果样本损坏/处理失败，打印错误并跳到下一个样本
    """

    def __init__(
        self,
        hf_dataset,
        transform=None,
        image_key: str = "image",
        label_key: str = "label",
        max_retry: Optional[int] = None,
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.max_retry = max_retry

    def __len__(self):
        return len(self.dataset)

    def _process_item(self, item):
        image = item[self.image_key]
        label = int(item[self.label_key])

        # ImageNet 存在少量灰度图，统一转成 RGB 便于后续 transforms。
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __getitem__(self, idx):
        n = len(self.dataset)
        if n <= 0:
            raise IndexError("HFImageNetDataset is empty.")

        start = int(idx) % n
        max_retry = n if self.max_retry is None else max(1, int(self.max_retry))
        max_retry = min(max_retry, n)

        last_error: Optional[Exception] = None
        for offset in range(max_retry):
            cur = (start + offset) % n
            try:
                item = self.dataset[cur]
                return self._process_item(item)
            except Exception as e:  # noqa: BLE001
                print(f"Error processing sample at index {cur}: {e}")
                last_error = e

        raise RuntimeError(
            f"Failed to load a valid HF ImageNet sample after {max_retry} retries "
            f"(start index: {start})."
        ) from last_error


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
    data_format: str = "auto",  # auto | imagefolder | hf_disk
    hf_load_from_disk: Optional[str] = None,
    hf_split_override: Optional[str] = None,
    hf_image_key: str = "image",
    hf_label_key: str = "label",
    custom_transform=None,
):
    ds_format = data_format
    if ds_format == "auto":
        if hf_load_from_disk is not None or _looks_like_hf_disk_dataset(data_root):
            ds_format = "hf_disk"
        else:
            ds_format = "imagefolder"

    tfm = custom_transform if custom_transform is not None else build_imagenet_transforms(image_size=image_size, split=split)

    if ds_format == "hf_disk":
        try:
            from datasets import load_from_disk
        except ImportError as e:
            raise ImportError("datasets package is required for hf_disk data format") from e

        path = hf_load_from_disk or data_root
        hf_split = hf_split_override or split
        ds_all = load_from_disk(path)
        ds_split = _resolve_hf_split(ds_all, hf_split)

        ds = HFImageNetDataset(ds_split, transform=tfm, image_key=hf_image_key, label_key=hf_label_key)
        class_names = _infer_hf_class_names(ds_split, label_key=hf_label_key, class_names_file=class_names_file)
        return ds, class_names

    if ds_format == "imagefolder":
        split_root = _resolve_split_root(data_root, split)
        ds = datasets.ImageFolder(root=split_root, transform=tfm)
        class_names = _load_class_names(class_names_file=class_names_file, default_names=list(ds.classes))
        return ds, class_names

    raise ValueError(f"Unsupported data_format={data_format}. Use one of: auto, imagefolder, hf_disk")


def build_imagenet_loader(
    data_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    class_names_file: Optional[str] = None,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    data_format: str = "auto",
    hf_load_from_disk: Optional[str] = None,
    hf_split_override: Optional[str] = None,
    hf_image_key: str = "image",
    hf_label_key: str = "label",
    custom_transform=None,
) -> Tuple[DataLoader, List[str]]:
    ds, class_names = build_imagenet_dataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        class_names_file=class_names_file,
        data_format=data_format,
        hf_load_from_disk=hf_load_from_disk,
        hf_split_override=hf_split_override,
        hf_image_key=hf_image_key,
        hf_label_key=hf_label_key,
        custom_transform=custom_transform,
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
