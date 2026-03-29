from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalize_dataset_name(dataset: str) -> str:
    d = str(dataset).lower()
    if d in {"cifar10", "cifar100", "sun397"}:
        return d
    raise ValueError(f"Unsupported dataset={dataset}. Use cifar10|cifar100|sun397.")


def _dataset_stats(dataset: str) -> Tuple[List[float], List[float]]:
    d = _normalize_dataset_name(dataset)
    if d == "sun397":
        return IMAGENET_MEAN, IMAGENET_STD
    if d == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    return CIFAR10_MEAN, CIFAR10_STD


def _dataset_num_classes(dataset: str) -> int:
    d = _normalize_dataset_name(dataset)
    if d == "sun397":
        return 397
    return 100 if d == "cifar100" else 10


def _dataset_torchvision_cls(dataset: str):
    d = _normalize_dataset_name(dataset)
    if d == "sun397":
        return datasets.SUN397
    return datasets.CIFAR100 if d == "cifar100" else datasets.CIFAR10


def _resize_ops(image_size: int) -> List:
    if image_size == 32:
        return []
    return [transforms.Resize((image_size, image_size), antialias=True)]


def build_cifar10_transforms(
    image_size: int,
    split: str,
    aug_strength: str = "medium",
    dataset: str = "cifar10",
) -> transforms.Compose:
    mean, std = _dataset_stats(dataset)
    dataset_name = _normalize_dataset_name(dataset)

    if dataset_name == "sun397":
        if split == "train":
            aug_strength = str(aug_strength).lower()
            if aug_strength == "light":
                aug = [
                    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                ]
            elif aug_strength == "strong":
                aug = [
                    transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                ]
            else:
                aug = [
                    transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                ]
            return transforms.Compose(
                aug
                + [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14), antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    resize_ops = []
    if image_size != 32:
        resize_ops = _resize_ops(image_size)

    if split == "train":
        aug_strength = str(aug_strength).lower()
        if aug_strength == "light":
            aug = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        elif aug_strength == "strong":
            aug = [
                transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            ]
        else:
            aug = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            ]
        return transforms.Compose(
            aug
            + resize_ops
            + [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return transforms.Compose(
        resize_ops
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_cifar10_multi_view_transforms(
    image_size: int,
    aug_strength: str = "medium",
    dataset: str = "cifar10",
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    view1 = build_cifar10_transforms(dataset=dataset, image_size=image_size, split="train", aug_strength=aug_strength)
    view2 = build_cifar10_transforms(dataset=dataset, image_size=image_size, split="train", aug_strength=aug_strength)
    target = build_cifar10_transforms(dataset=dataset, image_size=image_size, split="test", aug_strength=aug_strength)
    return view1, view2, target


class MultiViewCIFARWrapper(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        view1_transform,
        view2_transform,
        target_transform,
        target_source: str = "view1",
    ):
        self.base_dataset = base_dataset
        self.view1_transform = view1_transform
        self.view2_transform = view2_transform
        self.target_transform = target_transform
        self.target_source = str(target_source).lower()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image, label = self.base_dataset[index]
        img1 = self.view1_transform(image)
        img2 = self.view2_transform(image)

        if self.target_source == "view1":
            target = img1
        elif self.target_source == "view2":
            target = img2
        else:
            target = self.target_transform(image)

        return {
            "images": img1,
            "images_view2": img2,
            "images_target": target,
            "labels": int(label),
        }


class HFImageLabelDataset(Dataset):
    def __init__(self, hf_dataset, transform, image_key: str = "image", label_key: str = "label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        image = item[self.image_key]
        label = int(item[self.label_key])
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _make_split_indices(length: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    n_val = int(round(length * val_ratio))
    n_val = max(1, min(length - 1, n_val))

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(length, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _build_fake_dataset(
    split: str,
    image_size: int,
    size_train: int,
    size_eval: int,
    num_classes: int,
    transform=None,
) -> Dataset:
    size = size_train if split == "train" else size_eval
    return datasets.FakeData(
        size=size,
        image_size=(3, image_size, image_size),
        num_classes=num_classes,
        transform=transform,
    )


def _extract_class_names(ds: Dataset, num_classes: int) -> List[str]:
    base = ds
    if isinstance(base, Subset):
        base = base.dataset
    if hasattr(base, "base_dataset"):
        base = base.base_dataset

    classes = getattr(base, "classes", None)
    if isinstance(classes, list) and len(classes) == num_classes:
        return [str(x) for x in classes]

    if num_classes == len(CIFAR10_CLASS_NAMES):
        return list(CIFAR10_CLASS_NAMES)
    return [f"class_{i}" for i in range(num_classes)]


def build_cifar10_dataset(
    data_root: str,
    split: str,
    image_size: int = 32,
    val_from_train: bool = False,
    val_ratio: float = 0.1,
    seed: int = 42,
    download: bool = True,
    use_fake_data: bool = False,
    fake_train_size: int = 8192,
    fake_eval_size: int = 1024,
    custom_transform=None,
    two_view: bool = False,
    aug_strength: str = "medium",
    target_source: str = "view1",
    dataset: str = "cifar10",
    sun_source: str = "hf",
    sun_hf_dataset: str = "dpdl-benchmark/sun397",
    sun_hf_cache_dir: Optional[str] = None,
    sun_hf_image_key: str = "image",
    sun_hf_label_key: str = "label",
    sun_max_train_samples: int = 0,
    sun_max_eval_samples: int = 0,
) -> Tuple[Dataset, List[str]]:
    dataset_name = _normalize_dataset_name(dataset)
    num_classes = _dataset_num_classes(dataset_name)
    dataset_cls = _dataset_torchvision_cls(dataset_name)

    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split={split}, expected train|val|test")

    tfm = custom_transform if custom_transform is not None else build_cifar10_transforms(
        dataset=dataset_name,
        image_size=image_size,
        split=split,
        aug_strength=aug_strength,
    )

    if use_fake_data:
        base = _build_fake_dataset(
            split=split,
            image_size=image_size,
            size_train=fake_train_size,
            size_eval=fake_eval_size,
            num_classes=num_classes,
            transform=None if (split == "train" and two_view) else tfm,
        )
        if split == "train" and two_view:
            view1_tfm, view2_tfm, target_tfm = build_cifar10_multi_view_transforms(
                dataset=dataset_name,
                image_size=image_size,
                aug_strength=aug_strength,
            )
            ds = MultiViewCIFARWrapper(
                base_dataset=base,
                view1_transform=view1_tfm,
                view2_transform=view2_tfm,
                target_transform=target_tfm,
                target_source=target_source,
            )
        else:
            ds = base
        return ds, _extract_class_names(ds, num_classes=num_classes)

    data_root = str(Path(data_root).expanduser())

    if dataset_name == "sun397":
        if two_view:
            raise ValueError("two_view is not supported for sun397 in this training path.")

        sun_source_norm = str(sun_source).lower()
        if sun_source_norm in {"hf", "huggingface"}:
            try:
                from datasets import load_dataset
            except ImportError as e:
                raise ImportError("datasets package is required for sun397 with sun_source=hf") from e

            hf_split = {"train": "train", "val": "validation", "test": "test"}[split]
            ds_hf = load_dataset(sun_hf_dataset, split=hf_split, cache_dir=sun_hf_cache_dir)

            max_samples = int(sun_max_train_samples if split == "train" else sun_max_eval_samples)
            if max_samples > 0:
                n = min(max_samples, len(ds_hf))
                # Keep contiguous head subset to avoid forcing full-shard shuffle downloads in quick experiments.
                ds_hf = ds_hf.select(range(n))

            ds = HFImageLabelDataset(
                hf_dataset=ds_hf,
                transform=tfm,
                image_key=sun_hf_image_key,
                label_key=sun_hf_label_key,
            )
            feat = ds_hf.features.get(sun_hf_label_key) if hasattr(ds_hf, "features") else None
            names = list(feat.names) if (feat is not None and hasattr(feat, "names") and feat.names is not None) else []
            if len(names) == num_classes:
                return ds, [str(x) for x in names]
            return ds, [f"class_{i}" for i in range(num_classes)]

        if sun_source_norm in {"torchvision", "tv"}:
            full = dataset_cls(root=data_root, transform=tfm, download=download)
            n = len(full)
            n_val = int(round(n * val_ratio))
            n_test = n_val
            n_train = max(1, n - n_val - n_test)
            g = torch.Generator()
            g.manual_seed(seed)
            perm = torch.randperm(n, generator=g).tolist()
            train_idx = perm[:n_train]
            val_idx = perm[n_train : n_train + n_val]
            test_idx = perm[n_train + n_val :]
            if split == "train":
                ds = Subset(full, train_idx)
            elif split == "val":
                ds = Subset(full, val_idx)
            else:
                ds = Subset(full, test_idx)
            return ds, _extract_class_names(full, num_classes=num_classes)

        raise ValueError(f"Unsupported sun_source={sun_source}. Use hf|torchvision.")

    if split == "train":
        if two_view:
            base: Dataset = dataset_cls(root=data_root, train=True, transform=None, download=download)
            if val_from_train:
                train_idx, _ = _make_split_indices(len(base), val_ratio=val_ratio, seed=seed)
                base = Subset(base, train_idx)
            view1_tfm, view2_tfm, target_tfm = build_cifar10_multi_view_transforms(
                dataset=dataset_name,
                image_size=image_size,
                aug_strength=aug_strength,
            )
            ds = MultiViewCIFARWrapper(
                base_dataset=base,
                view1_transform=view1_tfm,
                view2_transform=view2_tfm,
                target_transform=target_tfm,
                target_source=target_source,
            )
        else:
            base = dataset_cls(root=data_root, train=True, transform=tfm, download=download)
            if val_from_train:
                train_idx, _ = _make_split_indices(len(base), val_ratio=val_ratio, seed=seed)
                ds = Subset(base, train_idx)
            else:
                ds = base
        return ds, _extract_class_names(ds, num_classes=num_classes)

    if split == "test":
        ds = dataset_cls(root=data_root, train=False, transform=tfm, download=download)
        return ds, _extract_class_names(ds, num_classes=num_classes)

    # split == "val"
    if val_from_train:
        full_train = dataset_cls(root=data_root, train=True, transform=tfm, download=download)
        train_idx, val_idx = _make_split_indices(len(full_train), val_ratio=val_ratio, seed=seed)
        _ = train_idx  # documented: val comes from train holdout
        ds = Subset(full_train, val_idx)
    else:
        ds = dataset_cls(root=data_root, train=False, transform=tfm, download=download)

    return ds, _extract_class_names(ds, num_classes=num_classes)


def build_cifar10_loader(
    data_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_from_train: bool = False,
    val_ratio: float = 0.1,
    seed: int = 42,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    download: bool = True,
    use_fake_data: bool = False,
    fake_train_size: int = 8192,
    fake_eval_size: int = 1024,
    custom_transform=None,
    two_view: bool = False,
    aug_strength: str = "medium",
    target_source: str = "view1",
    dataset: str = "cifar10",
    sun_source: str = "hf",
    sun_hf_dataset: str = "dpdl-benchmark/sun397",
    sun_hf_cache_dir: Optional[str] = None,
    sun_hf_image_key: str = "image",
    sun_hf_label_key: str = "label",
    sun_max_train_samples: int = 0,
    sun_max_eval_samples: int = 0,
) -> Tuple[DataLoader, List[str]]:
    ds, class_names = build_cifar10_dataset(
        dataset=dataset,
        data_root=data_root,
        split=split,
        image_size=image_size,
        val_from_train=val_from_train,
        val_ratio=val_ratio,
        seed=seed,
        download=download,
        use_fake_data=use_fake_data,
        fake_train_size=fake_train_size,
        fake_eval_size=fake_eval_size,
        custom_transform=custom_transform,
        two_view=two_view,
        aug_strength=aug_strength,
        target_source=target_source,
        sun_source=sun_source,
        sun_hf_dataset=sun_hf_dataset,
        sun_hf_cache_dir=sun_hf_cache_dir,
        sun_hf_image_key=sun_hf_image_key,
        sun_hf_label_key=sun_hf_label_key,
        sun_max_train_samples=sun_max_train_samples,
        sun_max_eval_samples=sun_max_eval_samples,
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
    if isinstance(batch, dict):
        out = {
            "images": batch["images"],
            "labels": batch["labels"],
        }
        if "images_view2" in batch:
            out["images_view2"] = batch["images_view2"]
        if "images_target" in batch:
            out["images_target"] = batch["images_target"]
        return out

    images, labels = batch
    return {"images": images, "labels": labels}


def denormalize_cifar10(images: torch.Tensor, dataset: str = "cifar10") -> torch.Tensor:
    mean_vals, std_vals = _dataset_stats(dataset)
    mean = torch.tensor(mean_vals, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std_vals, device=images.device).view(1, 3, 1, 1)
    return images * std + mean
