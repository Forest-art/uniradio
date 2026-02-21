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


def _resize_ops(image_size: int) -> List:
    if image_size == 32:
        return []
    return [transforms.Resize((image_size, image_size), antialias=True)]


def build_cifar10_transforms(
    image_size: int,
    split: str,
    aug_strength: str = "medium",
) -> transforms.Compose:
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
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    return transforms.Compose(
        resize_ops
        + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_cifar10_multi_view_transforms(
    image_size: int,
    aug_strength: str = "medium",
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    view1 = build_cifar10_transforms(image_size=image_size, split="train", aug_strength=aug_strength)
    view2 = build_cifar10_transforms(image_size=image_size, split="train", aug_strength=aug_strength)
    target = build_cifar10_transforms(image_size=image_size, split="test", aug_strength=aug_strength)
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
    transform=None,
) -> Dataset:
    size = size_train if split == "train" else size_eval
    return datasets.FakeData(
        size=size,
        image_size=(3, image_size, image_size),
        num_classes=10,
        transform=transform,
    )


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
) -> Tuple[Dataset, List[str]]:
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split={split}, expected train|val|test")

    tfm = custom_transform if custom_transform is not None else build_cifar10_transforms(
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
            transform=None if (split == "train" and two_view) else tfm,
        )
        if split == "train" and two_view:
            view1_tfm, view2_tfm, target_tfm = build_cifar10_multi_view_transforms(
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
        return ds, list(CIFAR10_CLASS_NAMES)

    data_root = str(Path(data_root).expanduser())

    if split == "train":
        if two_view:
            base = datasets.CIFAR10(root=data_root, train=True, transform=None, download=download)
            view1_tfm, view2_tfm, target_tfm = build_cifar10_multi_view_transforms(
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
            ds = datasets.CIFAR10(root=data_root, train=True, transform=tfm, download=download)
        return ds, list(CIFAR10_CLASS_NAMES)

    if split == "test":
        ds = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=download)
        return ds, list(CIFAR10_CLASS_NAMES)

    # split == "val"
    if val_from_train:
        full_train = datasets.CIFAR10(root=data_root, train=True, transform=tfm, download=download)
        train_idx, val_idx = _make_split_indices(len(full_train), val_ratio=val_ratio, seed=seed)
        _ = train_idx  # documented: val comes from train holdout
        ds = Subset(full_train, val_idx)
    else:
        ds = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=download)

    return ds, list(CIFAR10_CLASS_NAMES)


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
) -> Tuple[DataLoader, List[str]]:
    ds, class_names = build_cifar10_dataset(
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


def denormalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return images * std + mean
