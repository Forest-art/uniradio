import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from .grad_conflict import apply_cagrad, apply_conflict_aware, apply_naive
from .utils import append_jsonl, ensure_dir, save_json, seed_everything


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HFImageNet100TorchDataset(torch.utils.data.Dataset):
    """将 Hugging Face Dataset 包装成 PyTorch Dataset，容错处理坏样本。"""

    def __init__(
        self,
        hf_dataset: Dataset,
        transform: transforms.Compose,
        image_key: str = "image",
        label_key: str = "label",
        max_retry: Optional[int] = None,
    ) -> None:
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.max_retry = max_retry

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        n = len(self.dataset)
        if n <= 0:
            raise IndexError("HFImageNet100TorchDataset is empty.")

        start = int(idx) % n
        max_retry = n if self.max_retry is None else max(1, int(self.max_retry))
        max_retry = min(max_retry, n)
        last_error: Optional[Exception] = None

        for offset in range(max_retry):
            cur = (start + offset) % n
            try:
                item = self.dataset[cur]
                image = item[self.image_key]
                label = int(item[self.label_key])
                # ImageNet 存在少量灰度图，统一转 RGB，避免 transform 报错。
                if hasattr(image, "mode") and image.mode != "RGB":
                    image = image.convert("RGB")
                image = self.transform(image)
                return image, label
            except Exception as e:  # noqa: BLE001
                print(f"[warn] failed to process sample idx={cur}: {e}")
                last_error = e

        raise RuntimeError(
            f"failed to load valid sample after {max_retry} retries, start_idx={start}"
        ) from last_error


def _pick_split_name(ds: DatasetDict, candidates: Sequence[str]) -> Optional[str]:
    keys = list(ds.keys())
    for name in candidates:
        if name in ds:
            return name
    # 兼容写法：优先用第一个 split。
    return keys[0] if len(keys) > 0 else None


def _looks_like_hf_disk_dataset(path: Path) -> bool:
    return any((path / name).exists() for name in ("dataset_info.json", "dataset_dict.json", "state.json"))


def _resolve_imagefolder_split_root(data_root: Path, split: str) -> Path:
    split_alias = {
        "train": ["train"],
        "val": ["val", "validation", "valid", "test"],
        "validation": ["validation", "val", "valid", "test"],
        "test": ["test", "val", "validation", "valid"],
    }
    candidates = split_alias.get(split, [split])

    roots = [data_root]
    # 兼容常见层级: <root>/imagenet, <root>/ILSVRC/Data/CLS-LOC
    roots.extend(
        [
            data_root / "imagenet",
            data_root / "ILSVRC" / "Data" / "CLS-LOC",
        ]
    )

    for root in roots:
        for cand in candidates:
            p = root / cand
            if p.exists() and p.is_dir():
                return p

    tried = [str(root / cand) for root in roots for cand in candidates]
    raise FileNotFoundError(
        f"Cannot find ImageFolder split='{split}' under data_root={data_root}. "
        f"Tried: {', '.join(tried)}"
    )


def _infer_num_classes(split_ds: Dataset, label_key: str = "label") -> int:
    feat = getattr(split_ds, "features", {}).get(label_key, None)
    if feat is not None and hasattr(feat, "names") and feat.names is not None:
        return int(len(feat.names))
    if hasattr(split_ds, "unique"):
        return int(len(split_ds.unique(label_key)))
    raise RuntimeError("Cannot infer number of classes from HF dataset.")


def build_imagenet100_dataloaders(
    batch_size: int = 128,
    num_workers: int = 8,
    dataset_path: Optional[str] = None,
    hf_dataset_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    image_key: str = "image",
    label_key: str = "label",
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, Dict[str, object]]:
    """构建 IN100 dataloader。

    支持两种来源：
    1) `dataset_path` 本地路径（HF load_from_disk 或 ImageFolder）
    2) `hf_dataset_id` 远程 HuggingFace 数据集
    """
    candidates = [hf_dataset_id] if hf_dataset_id else []
    # 以 clane9/imagenet-100 为首选，其他作为回退。
    candidates.extend(
        [
            "clane9/imagenet-100",
            "jokerak/imagenet100",
            "randall-lab/imagenet100",
            "ilee0022/ImageNet100",
        ]
    )
    # 去重并保持顺序。
    seen = set()
    ordered_candidates = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered_candidates.append(c)

    train_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # 优先使用本地路径，方便跨集群一键复现实验。
    if dataset_path:
        ds_root = Path(dataset_path).expanduser()
        if not ds_root.exists():
            raise FileNotFoundError(f"dataset_path not found: {ds_root}")

        # A) HF load_from_disk 目录
        if _looks_like_hf_disk_dataset(ds_root):
            ds_all = load_from_disk(str(ds_root))
            if not isinstance(ds_all, DatasetDict):
                raise TypeError(
                    f"{ds_root} is not a DatasetDict from load_from_disk. got={type(ds_all)}"
                )

            train_split = _pick_split_name(ds_all, ["train"])
            val_split = _pick_split_name(ds_all, ["validation", "val", "test"])
            if train_split is None or val_split is None:
                raise RuntimeError(f"{ds_root} has no valid splits. available={list(ds_all.keys())}")

            train_ds = HFImageNet100TorchDataset(
                ds_all[train_split],
                transform=train_transform,
                image_key=image_key,
                label_key=label_key,
            )
            val_ds = HFImageNet100TorchDataset(
                ds_all[val_split],
                transform=val_transform,
                image_key=image_key,
                label_key=label_key,
            )
            num_classes = _infer_num_classes(ds_all[train_split], label_key=label_key)
            source = "local_hf_disk"
        else:
            # B) torchvision ImageFolder 目录
            train_root = _resolve_imagefolder_split_root(ds_root, "train")
            val_root = _resolve_imagefolder_split_root(ds_root, "val")
            train_ds = tv_datasets.ImageFolder(str(train_root), transform=train_transform)
            val_ds = tv_datasets.ImageFolder(str(val_root), transform=val_transform)
            num_classes = int(len(train_ds.classes))
            source = "local_imagefolder"
            train_split = str(train_root)
            val_split = str(val_root)

        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=int(world_size),
                rank=int(rank),
                shuffle=True,
                drop_last=True,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(batch_size),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=int(num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=int(num_workers) > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=True,
            drop_last=False,
            persistent_workers=int(num_workers) > 0,
        )

        meta = {
            "dataset_source": source,
            "dataset_path": str(ds_root),
            "hf_dataset_id": None,
            "train_split": train_split,
            "val_split": val_split,
            "num_classes": int(num_classes),
            "train_size": int(len(train_ds)),
            "val_size": int(len(val_ds)),
            "distributed": bool(distributed),
            "rank": int(rank),
            "world_size": int(world_size),
        }
        return train_loader, val_loader, meta

    errors: List[str] = []
    for ds_name in ordered_candidates:
        try:
            ds_all = load_dataset(ds_name, cache_dir=cache_dir)
            if not isinstance(ds_all, DatasetDict):
                raise TypeError(f"{ds_name} is not DatasetDict. got={type(ds_all)}")

            train_split = _pick_split_name(ds_all, ["train"])
            val_split = _pick_split_name(ds_all, ["validation", "val", "test"])
            if train_split is None or val_split is None:
                raise RuntimeError(f"{ds_name} has no valid splits. available={list(ds_all.keys())}")

            train_ds = HFImageNet100TorchDataset(
                ds_all[train_split],
                transform=train_transform,
                image_key=image_key,
                label_key=label_key,
            )
            val_ds = HFImageNet100TorchDataset(
                ds_all[val_split],
                transform=val_transform,
                image_key=image_key,
                label_key=label_key,
            )
            num_classes = _infer_num_classes(ds_all[train_split], label_key=label_key)

            train_sampler = None
            if distributed:
                train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=int(world_size),
                    rank=int(rank),
                    shuffle=True,
                    drop_last=True,
                )

            train_loader = DataLoader(
                train_ds,
                batch_size=int(batch_size),
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=int(num_workers),
                pin_memory=True,
                drop_last=True,
                persistent_workers=int(num_workers) > 0,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=int(num_workers),
                pin_memory=True,
                drop_last=False,
                persistent_workers=int(num_workers) > 0,
            )
            meta = {
                "hf_dataset_id": ds_name,
                "train_split": train_split,
                "val_split": val_split,
                "num_classes": int(num_classes),
                "train_size": int(len(train_ds)),
                "val_size": int(len(val_ds)),
                "distributed": bool(distributed),
                "rank": int(rank),
                "world_size": int(world_size),
            }
            return train_loader, val_loader, meta
        except Exception as e:  # noqa: BLE001
            errors.append(f"{ds_name}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Failed to load any ImageNet-100 HF dataset candidate.\n" + "\n".join(errors)
    )


class LightweightTransformerDecoder(nn.Module):
    """轻量级 Patch Decoder（MAE/RAE 风格）。"""

    def __init__(
        self,
        in_dim: int,
        num_patches: int,
        patch_dim: int,
        decoder_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.input_proj = nn.Linear(in_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        hidden_dim = int(decoder_dim * float(mlp_ratio))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=decoder_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=drop_rate,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_dim)

    def _resize_pos_embed(self, n_tokens: int) -> torch.Tensor:
        if n_tokens == self.pos_embed.shape[1]:
            return self.pos_embed
        # token 数变化时做一维插值，保证 decoder 仍可运行。
        x = self.pos_embed.transpose(1, 2)
        x = F.interpolate(x, size=n_tokens, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(patch_tokens)
        x = x + self._resize_pos_embed(x.shape[1]).to(device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.pred(x)


class ViTBridgeModel(nn.Module):
    """Bridge 实验模型: ViT encoder + 分类头 + 重建 decoder。"""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        image_size: int = 224,
        decoder_dim: int = 384,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        decoder_mlp_ratio: float = 4.0,
        decoder_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.image_size = int(image_size)
        self.embed_dim = int(self.encoder.embed_dim)
        p = self.encoder.patch_embed.patch_size
        self.patch_size = int(p[0] if isinstance(p, tuple) else p)
        self.num_patches = int(self.encoder.patch_embed.num_patches)
        self.patch_dim = 3 * self.patch_size * self.patch_size

        self.cls_head = nn.Linear(self.embed_dim, int(num_classes))
        nn.init.trunc_normal_(self.cls_head.weight, std=0.02)
        if self.cls_head.bias is not None:
            nn.init.zeros_(self.cls_head.bias)

        self.decoder = LightweightTransformerDecoder(
            in_dim=self.embed_dim,
            num_patches=self.num_patches,
            patch_dim=self.patch_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            drop_rate=decoder_drop_rate,
        )

    def encode_tokens(self, images_norm: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(images_norm)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[-1]
        if isinstance(tokens, dict):
            # 兼容部分 timm/hf 模型返回 dict 的情况。
            for k in ("x_prenorm", "x", "last_hidden_state"):
                if k in tokens:
                    tokens = tokens[k]
                    break
        if tokens.ndim == 2:
            # 兼容返回 pooled 特征的实现，改走 token 路径。
            x = self.encoder.patch_embed(images_norm)
            x = self.encoder._pos_embed(x)
            x = self.encoder.patch_drop(x)
            x = self.encoder.norm_pre(x)
            for blk in self.encoder.blocks:
                x = blk(x)
            x = self.encoder.norm(x)
            tokens = x
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected token tensor [B,T,C], got shape={tuple(tokens.shape)}")
        return tokens

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.shape
        p = self.patch_size
        if (h % p) != 0 or (w % p) != 0:
            raise ValueError(f"image shape {(h, w)} is not divisible by patch size {p}")
        gh, gw = h // p, w // p
        # [B,C,H,W] -> [B,gh*gw,p*p*C]
        x = images.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, p * p * c)
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, d = patches.shape
        p = self.patch_size
        expected = 3 * p * p
        if d != expected:
            raise ValueError(f"patch dim mismatch: got {d}, expected {expected}")
        gh = gw = int(n**0.5)
        if gh * gw != n:
            raise ValueError(f"number of patches {n} is not a perfect square")
        x = patches.reshape(b, gh, gw, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, gh * p, gw * p)
        return x

    def forward(self, images_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.encode_tokens(images_norm)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, -self.num_patches :]
        logits = self.cls_head(cls_token)
        pred_patches = self.decoder(patch_tokens)
        return {
            "tokens": tokens,
            "logits": logits,
            "pred_patches": pred_patches,
        }


def _extract_state_dict_from_checkpoint(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model", "teacher", "student", "network", "module"):
            v = ckpt_obj.get(key)
            if isinstance(v, dict):
                return v
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj  # 直接就是 state_dict
    raise RuntimeError("Cannot extract state_dict from checkpoint object.")


def _best_prefix_stripped_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_state_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    model_key_set = set(model_state_keys)
    prefixes = [
        "",
        "module.",
        "model.",
        "teacher.",
        "student.",
        "backbone.",
        "encoder.",
        "base_model.",
        "trunk.",
    ]
    best_sd = state_dict
    best_score = -1
    for prefix in prefixes:
        if prefix:
            cand = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in state_dict.items()
            }
        else:
            cand = state_dict
        score = sum(1 for k in cand.keys() if k in model_key_set)
        if score > best_score:
            best_score = score
            best_sd = cand
    return best_sd


def _load_local_encoder_checkpoint(encoder: nn.Module, ckpt_path: str) -> None:
    path = Path(ckpt_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"encoder_ckpt not found: {path}")

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    raw_sd = _extract_state_dict_from_checkpoint(ckpt)
    sd = _best_prefix_stripped_state_dict(raw_sd, list(encoder.state_dict().keys()))
    load_ret = encoder.load_state_dict(sd, strict=False)

    matched = 0
    for k, v in sd.items():
        if k in encoder.state_dict() and tuple(v.shape) == tuple(encoder.state_dict()[k].shape):
            matched += 1
    if matched <= 0:
        raise RuntimeError(
            f"Loaded 0 matched keys from local checkpoint: {path}. "
            "Please check model variant and checkpoint key format."
        )
    print(
        f"[encoder_ckpt] loaded={path} matched={matched} "
        f"missing={len(load_ret.missing_keys)} unexpected={len(load_ret.unexpected_keys)}"
    )


def build_encoder(encoder_init: str, image_size: int = 224, encoder_ckpt: str = "") -> nn.Module:
    mode = str(encoder_init).lower()
    if mode not in {"scratch", "dinov2"}:
        raise ValueError(f"encoder_init must be scratch|dinov2, got: {encoder_init}")

    # 同一架构下控制变量：scratch 与 dinov2 仅初始化权重来源不同。
    encoder = timm.create_model(
        "vit_small_patch14_dinov2",
        pretrained=(mode == "dinov2" and not bool(str(encoder_ckpt).strip())),
        num_classes=0,
        global_pool="",
        img_size=int(image_size),
    )
    if mode == "dinov2" and bool(str(encoder_ckpt).strip()):
        _load_local_encoder_checkpoint(encoder, str(encoder_ckpt).strip())
    # 本实验要求全量微调，因此 encoder 参数全部 trainable。
    for p in encoder.parameters():
        p.requires_grad = True
    return encoder


def _group_encoder_params(encoder: nn.Module) -> List[Tuple[str, int, List[Tuple[str, nn.Parameter]]]]:
    """按层分组，用于统计每层 gu/gg 冲突。"""
    groups: Dict[str, List[Tuple[str, nn.Parameter]]] = {}
    depth_map: Dict[str, int] = {}

    for name, p in encoder.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("patch_embed"):
            gname, depth = "patch_embed", 0
        elif name.startswith("cls_token") or name.startswith("pos_embed") or name.startswith("register_tokens"):
            gname, depth = "embeddings", 0
        elif name.startswith("blocks."):
            try:
                idx = int(name.split(".")[1])
            except Exception:
                idx = 0
            gname, depth = f"blocks.{idx}", idx + 1
        elif name.startswith("norm"):
            gname, depth = "norm", 999
        else:
            gname, depth = "other", 1000

        if gname not in groups:
            groups[gname] = []
            depth_map[gname] = depth
        else:
            depth_map[gname] = min(depth_map[gname], depth)
        groups[gname].append((name, p))

    merged = [(k, depth_map[k], groups[k]) for k in groups.keys()]
    merged.sort(key=lambda x: (x[1], x[0]))
    return merged


def _autograd_grads_by_name(
    loss: torch.Tensor,
    named_params: List[Tuple[str, nn.Parameter]],
    retain_graph: bool,
) -> Dict[str, torch.Tensor]:
    params = [p for _, p in named_params]
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    out: Dict[str, torch.Tensor] = {}
    for (name, p), g in zip(named_params, grads):
        if g is None:
            out[name] = torch.zeros_like(p, memory_format=torch.preserve_format).detach().clone()
        else:
            out[name] = g.detach().clone()
    return out


def _safe_cosine(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    nu = torch.norm(u)
    nv = torch.norm(v)
    if float(nu.item()) < eps or float(nv.item()) < eps:
        return 0.0
    return float(torch.dot(u, v).item() / (nu.item() * nv.item() + eps))


def _neg_ratio(u: torch.Tensor, v: torch.Tensor) -> float:
    # 按符号相反元素占比统计冲突强度。
    return float(((u * v) < 0).float().mean().item())


def save_layerwise_conflict_csv(
    *,
    step: int,
    dict_gu: Dict[str, torch.Tensor],
    dict_gg: Dict[str, torch.Tensor],
    encoder_groups: List[Tuple[str, int, List[Tuple[str, nn.Parameter]]]],
    out_dir: Path,
) -> Dict[str, float]:
    ensure_dir(str(out_dir))
    csv_path = out_dir / f"step_{int(step):04d}.csv"

    rows: List[Dict[str, float]] = []
    cos_values: List[float] = []
    neg_values: List[float] = []

    for layer_name, depth, layer_params in encoder_groups:
        gu_parts = []
        gg_parts = []
        numel = 0
        for pname, p in layer_params:
            gu = dict_gu.get(pname)
            gg = dict_gg.get(pname)
            if gu is None:
                gu = torch.zeros_like(p, memory_format=torch.preserve_format)
            if gg is None:
                gg = torch.zeros_like(p, memory_format=torch.preserve_format)
            gu_parts.append(gu.reshape(-1))
            gg_parts.append(gg.reshape(-1))
            numel += int(p.numel())

        gu_flat = torch.cat(gu_parts, dim=0)
        gg_flat = torch.cat(gg_parts, dim=0)
        cos = _safe_cosine(gu_flat, gg_flat)
        neg = _neg_ratio(gu_flat, gg_flat)
        cos_values.append(cos)
        neg_values.append(neg)
        rows.append(
            {
                "step": int(step),
                "layer": layer_name,
                "depth": int(depth),
                "cosine_similarity": float(cos),
                "neg_ratio": float(neg),
                "gu_norm": float(torch.norm(gu_flat).item()),
                "gg_norm": float(torch.norm(gg_flat).item()),
                "numel": int(numel),
            }
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "layer",
                "depth",
                "cosine_similarity",
                "neg_ratio",
                "gu_norm",
                "gg_norm",
                "numel",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "step": int(step),
        "mean_cosine": float(sum(cos_values) / max(1, len(cos_values))),
        "mean_neg_ratio": float(sum(neg_values) / max(1, len(neg_values))),
        "csv_path": str(csv_path),
    }


def _global_grad_norm(grads: Dict[str, torch.Tensor]) -> float:
    s = 0.0
    for g in grads.values():
        x = g.detach()
        # 统一到 fp32 统计，避免 bf16/fp16 下数值抖动。
        s += float((x.float() * x.float()).sum().item())
    return float(s**0.5)


@torch.no_grad()
def evaluate_recon_and_understanding(
    *,
    model: ViTBridgeModel,
    val_loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_batches: int,
    compute_rfid: bool,
    rfid_num_samples: int,
    rfid_batch_size: int,
    rfid_tmp_dir: str,
) -> Dict[str, float]:
    model.eval()

    sum_mse = 0.0
    sum_acc = 0.0
    n_samples = 0

    # rFID: 使用 GT vs Recon 的 FID，作为重建质量 proxy（数值越小越好）。
    saved = 0
    rfid = float("nan")
    rfid_error = ""
    tmp_root = None
    real_dir = None
    fake_dir = None
    if compute_rfid:
        tmp_root = tempfile.TemporaryDirectory(prefix="in100_rfid_", dir=(rfid_tmp_dir or None))
        root = Path(tmp_root.name)
        real_dir = root / "real"
        fake_dir = root / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

    for bi, (images, labels) in enumerate(val_loader):
        if bi >= int(max_batches):
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images_01 = (images * std + mean).clamp(0.0, 1.0)

        out = model(images)
        logits = out["logits"]
        pred_img = model.unpatchify(out["pred_patches"]).clamp(0.0, 1.0)
        if pred_img.shape[-2:] != images_01.shape[-2:]:
            pred_img = F.interpolate(pred_img, size=images_01.shape[-2:], mode="bilinear", align_corners=False)

        mse_per = ((pred_img - images_01) ** 2).flatten(1).mean(dim=1)
        acc_per = (logits.argmax(dim=-1) == labels).float()

        sum_mse += float(mse_per.sum().item())
        sum_acc += float(acc_per.sum().item())
        n_samples += int(images.shape[0])

        if compute_rfid and saved < int(rfid_num_samples):
            bsz = int(images.shape[0])
            for i in range(bsz):
                if saved >= int(rfid_num_samples):
                    break
                save_image(images_01[i].cpu(), str(real_dir / f"{saved:07d}.png"))
                save_image(pred_img[i].cpu(), str(fake_dir / f"{saved:07d}.png"))
                saved += 1

    if n_samples <= 0:
        return {
            "val_top1_acc": 0.0,
            "val_mse": 0.0,
            "val_rmse": 0.0,
            "val_rfid": float("nan"),
            "val_num_samples": 0,
            "rfid_num_samples": 0,
            "rfid_error": "no validation samples",
        }

    val_mse = sum_mse / n_samples
    val_rmse = val_mse**0.5
    val_acc = sum_acc / n_samples

    if compute_rfid and saved > 1:
        try:
            from torch_fidelity import calculate_metrics

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
                rfid_error = (
                    f"{rfid_error}; install with: pip install torch-fidelity"
                )

    if tmp_root is not None:
        tmp_root.cleanup()

    return {
        "val_top1_acc": float(val_acc),
        "val_mse": float(val_mse),
        "val_rmse": float(val_rmse),
        "val_rfid": float(rfid),
        "val_num_samples": int(n_samples),
        "rfid_num_samples": int(saved),
        "rfid_error": rfid_error,
    }


def _init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    return is_distributed, rank, world_size, local_rank


def _reduce_mean_scalar(value: float, device: torch.device, is_distributed: bool, world_size: int) -> float:
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    if is_distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= float(world_size)
    return float(t.item())


def main() -> None:
    parser = argparse.ArgumentParser("ImageNet-100 early gradient dynamics bridge experiment")
    parser.add_argument("--encoder_init", type=str, default="scratch", choices=["scratch", "dinov2"])
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default="",
        help="Optional local .pth/.pt for encoder init. "
        "When set with --encoder_init dinov2, load local checkpoint instead of timm online pretrained.",
    )
    parser.add_argument("--hf_dataset_id", type=str, default="clane9/imagenet-100")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--label_key", type=str, default="label")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--probe_until", type=int, default=1000)
    parser.add_argument("--probe_every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_g", type=float, default=1.0)
    parser.add_argument(
        "--grad_strategy",
        type=str,
        default="naive",
        choices=["naive", "pcgrad", "cagrad", "conflict_aware"],
        help="Gradient merge strategy on shared encoder params for joint optimization.",
    )
    parser.add_argument(
        "--cagrad_beta",
        type=float,
        default=0.5,
        help="CAGrad interpolation strength in [0,1]. Used when --grad_strategy=cagrad.",
    )
    parser.add_argument(
        "--grad_norm_balance_every",
        type=int,
        default=0,
        help="If >0, every N steps estimate encoder gu/gg norms and dynamically scale lambda_g.",
    )
    parser.add_argument(
        "--grad_norm_balance_ema",
        type=float,
        default=0.9,
        help="EMA momentum for dynamic lambda_g scaling when grad_norm_balance_every>0.",
    )
    parser.add_argument(
        "--grad_norm_balance_power",
        type=float,
        default=1.0,
        help="Exponent on target dynamic scale; <1.0 makes balancing updates more conservative.",
    )
    parser.add_argument(
        "--grad_norm_balance_min_scale",
        type=float,
        default=0.1,
        help="Min clamp for dynamic scale factor on lambda_g.",
    )
    parser.add_argument(
        "--grad_norm_balance_max_scale",
        type=float,
        default=30.0,
        help="Max clamp for dynamic scale factor on lambda_g.",
    )
    parser.add_argument("--decoder_dim", type=int, default=384)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=6)
    parser.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--decoder_drop_rate", type=float, default=0.0)
    parser.add_argument("--output_root", type=str, default="results")
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional explicit run directory name under output_root. "
        "If empty, defaults to in100_grad_dynamics_{encoder_init}.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_max_batches", type=int, default=50)
    parser.add_argument("--eval_rfid_num_samples", type=int, default=512)
    parser.add_argument("--eval_rfid_batch_size", type=int, default=64)
    parser.add_argument("--eval_rfid_tmp_dir", type=str, default="/tmp")
    parser.add_argument(
        "--skip_rfid",
        action="store_true",
        help="Disable rFID computation during periodic eval.",
    )
    args = parser.parse_args()

    is_distributed, rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

    seed_everything(int(args.seed) + int(rank))

    if is_distributed and args.device == "cpu" and torch.cuda.is_available():
        raise RuntimeError(
            "Distributed launch with CUDA available does not support --device=cpu. "
            "Use --device=auto or --device=cuda."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda is set but CUDA is not available.")
    if is_distributed and torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if is_distributed and is_main_process and device.type == "cpu":
        print("[warn] running distributed training on CPU backend, this is usually very slow.")
    grad_strategy = str(args.grad_strategy).lower()
    if grad_strategy == "conflict_aware":
        grad_strategy = "pcgrad"
    if is_distributed and grad_strategy != "naive":
        raise RuntimeError(
            "DDP + non-naive grad strategy is not supported in this script yet. "
            "Please run single GPU for pcgrad/cagrad fair compare."
        )

    train_loader, val_loader, data_meta = build_imagenet100_dataloaders(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        hf_dataset_id=str(args.hf_dataset_id),
        cache_dir=(str(args.cache_dir) if args.cache_dir else None),
        image_key=str(args.image_key),
        label_key=str(args.label_key),
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    encoder = build_encoder(
        args.encoder_init,
        image_size=224,
        encoder_ckpt=str(args.encoder_ckpt),
    )
    model = ViTBridgeModel(
        encoder=encoder,
        num_classes=int(data_meta["num_classes"]),
        image_size=224,
        decoder_dim=int(args.decoder_dim),
        decoder_depth=int(args.decoder_depth),
        decoder_heads=int(args.decoder_heads),
        decoder_mlp_ratio=float(args.decoder_mlp_ratio),
        decoder_drop_rate=float(args.decoder_drop_rate),
    ).to(device)
    if is_distributed:
        if device.type == "cuda":
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        else:
            model = DDP(model, broadcast_buffers=False)
    base_model = model.module if isinstance(model, DDP) else model
    model.train()

    # 统一输入已在 dataloader 做过 Normalize，这里仅用于必要时重建 target 反归一化。
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )

    run_name = str(args.run_name).strip() or f"in100_grad_dynamics_{args.encoder_init}"
    run_dir = Path(args.output_root) / run_name
    ensure_dir(str(run_dir))
    if is_main_process:
        save_json(
            {
                "args": vars(args),
                "data_meta": data_meta,
                "device": str(device),
                "rank": int(rank),
                "world_size": int(world_size),
                "num_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "num_train_samples": int(len(train_loader.dataset)),
                "num_val_samples": int(len(val_loader.dataset)),
            },
            str(run_dir / "run_setup.json"),
        )

    metrics_path = run_dir / "train_metrics.jsonl"
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    encoder_groups = _group_encoder_params(base_model.encoder)
    shared_params = [p for p in base_model.encoder.parameters() if p.requires_grad]
    shared_ids = {id(p) for p in shared_params}
    aux_params = [
        p
        for p in base_model.parameters()
        if p.requires_grad and (id(p) not in shared_ids)
    ]
    global_step = 0
    train_epoch = 0
    train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    if train_sampler is not None:
        train_sampler.set_epoch(train_epoch)
    train_iter = iter(train_loader)
    pbar = (
        tqdm(total=int(args.max_steps), desc=f"in100-{args.encoder_init}", dynamic_ncols=True)
        if is_main_process
        else None
    )
    encoder_named_params = [(n, p) for n, p in base_model.encoder.named_parameters() if p.requires_grad]
    dyn_scale = 1.0
    dyn_gu_norm = float("nan")
    dyn_gg_norm = float("nan")

    while global_step < int(args.max_steps):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(train_epoch)
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        global_step += 1
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # dataloader 输出是归一化后的图像；重建目标改回 [0,1] 空间，避免 MSE 量纲异常。
        images_01 = (images * std + mean).clamp_(0.0, 1.0)
        outputs = model(images)
        logits = outputs["logits"]
        pred_patches = outputs["pred_patches"]
        gt_patches = base_model.patchify(images_01)

        loss_u = F.cross_entropy(logits, labels)
        loss_g = F.mse_loss(pred_patches, gt_patches)
        balance_every = max(0, int(args.grad_norm_balance_every))
        if balance_every > 0 and (global_step % balance_every == 0):
            # 动态估计共享 encoder 上两任务梯度幅度，用于平衡 lambda_g。
            gu_dyn = _autograd_grads_by_name(
                loss_u,
                encoder_named_params,
                retain_graph=True,
            )
            gg_dyn = _autograd_grads_by_name(
                loss_g,
                encoder_named_params,
                retain_graph=True,
            )
            dyn_gu_norm = _global_grad_norm(gu_dyn)
            dyn_gg_norm = _global_grad_norm(gg_dyn)
            base_weight_ratio = float(args.lambda_u) / max(float(args.lambda_g), 1e-12)
            target_scale = float(base_weight_ratio * dyn_gu_norm / max(dyn_gg_norm, 1e-12))
            target_scale = float(target_scale ** float(args.grad_norm_balance_power))
            target_scale = float(
                min(
                    max(target_scale, float(args.grad_norm_balance_min_scale)),
                    float(args.grad_norm_balance_max_scale),
                )
            )
            dyn_scale = float(args.grad_norm_balance_ema) * dyn_scale + (
                1.0 - float(args.grad_norm_balance_ema)
            ) * target_scale
        eff_lambda_g = float(args.lambda_g) * float(dyn_scale)
        total_loss = float(args.lambda_u) * loss_u + eff_lambda_g * loss_g

        should_probe = (
            global_step <= int(args.probe_until)
            and (global_step % max(1, int(args.probe_every)) == 0)
        )

        optimizer.zero_grad(set_to_none=True)
        probe_stats: Optional[Dict[str, float]] = None
        train_grad_cos = 0.0

        if should_probe and is_main_process:
            # DDP 下探针梯度不能用多次 backward（会触发 mark ready twice）。
            # 这里单独走 base_model 前向 + autograd.grad，只做梯度观测，不写入 .grad。
            probe_outputs = base_model(images)
            probe_logits = probe_outputs["logits"]
            probe_pred_patches = probe_outputs["pred_patches"]
            probe_gt_patches = base_model.patchify(images_01)
            probe_loss_u = F.cross_entropy(probe_logits, labels)
            probe_loss_g = F.mse_loss(probe_pred_patches, probe_gt_patches)
            enc_named_params = [
                (n, p) for n, p in base_model.encoder.named_parameters() if p.requires_grad
            ]
            dict_gu = _autograd_grads_by_name(
                probe_loss_u,
                enc_named_params,
                retain_graph=True,
            )
            dict_gg = _autograd_grads_by_name(
                probe_loss_g,
                enc_named_params,
                retain_graph=False,
            )
            probe_stats = save_layerwise_conflict_csv(
                step=global_step,
                dict_gu=dict_gu,
                dict_gg=dict_gg,
                encoder_groups=encoder_groups,
                out_dir=run_dir,
            )
            gu_norm = _global_grad_norm(dict_gu)
            gg_norm = _global_grad_norm(dict_gg)
            probe_stats["global_gu_norm"] = float(gu_norm)
            probe_stats["global_gg_norm"] = float(gg_norm)
            probe_stats["global_gu_over_gg"] = float(gu_norm / max(gg_norm, 1e-12))

            # 探针使用独立前向，不影响训练图；训练仍在当前图上做一步更新。

        if grad_strategy == "naive":
            train_grad_cos = apply_naive(
                loss_txt=loss_u,
                loss_rec=loss_g,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(eff_lambda_g),
            )
        elif grad_strategy == "pcgrad":
            train_grad_cos = apply_conflict_aware(
                loss_txt=loss_u,
                loss_rec=loss_g,
                lora_params=shared_params,
                aux_params=aux_params,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(eff_lambda_g),
            )
        elif grad_strategy == "cagrad":
            train_grad_cos = apply_cagrad(
                loss_txt=loss_u,
                loss_rec=loss_g,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(eff_lambda_g),
                beta=float(args.cagrad_beta),
            )
        else:
            raise ValueError(f"Unsupported grad_strategy={grad_strategy}")

        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            rmse = torch.sqrt(loss_g.detach().clamp_min(0.0))

        loss_u_log = _reduce_mean_scalar(float(loss_u.detach().item()), device, is_distributed, world_size)
        loss_g_log = _reduce_mean_scalar(float(loss_g.detach().item()), device, is_distributed, world_size)
        loss_total_log = _reduce_mean_scalar(float(total_loss.detach().item()), device, is_distributed, world_size)
        loss_u_weighted_log = float(args.lambda_u) * loss_u_log
        loss_g_weighted_log = float(eff_lambda_g) * loss_g_log
        acc_log = _reduce_mean_scalar(float(acc.item()), device, is_distributed, world_size)
        rmse_log = _reduce_mean_scalar(float(rmse.item()), device, is_distributed, world_size)
        raw_ratio = loss_u_log / max(loss_g_log, 1e-12)
        weighted_ratio = loss_u_weighted_log / max(loss_g_weighted_log, 1e-12)

        row = {
            "step": int(global_step),
            "loss_u": float(loss_u_log),
            "loss_g": float(loss_g_log),
            "loss_u_weighted": float(loss_u_weighted_log),
            "loss_g_weighted": float(loss_g_weighted_log),
            "loss_total": float(loss_total_log),
            "loss_ratio_u_over_g": float(raw_ratio),
            "loss_ratio_weighted_u_over_g": float(weighted_ratio),
            "effective_lambda_g": float(eff_lambda_g),
            "grad_strategy": grad_strategy,
            "train_grad_cosine": float(train_grad_cos),
            "acc": float(acc_log),
            "rmse": float(rmse_log),
        }
        if balance_every > 0:
            row["dyn_scale"] = float(dyn_scale)
            row["dyn_gu_norm"] = float(dyn_gu_norm)
            row["dyn_gg_norm"] = float(dyn_gg_norm)
            row["dyn_gu_over_gg"] = float(dyn_gu_norm / max(dyn_gg_norm, 1e-12))
            row["dyn_weighted_gu_over_gg"] = float(
                (float(args.lambda_u) * dyn_gu_norm) / max(float(eff_lambda_g) * dyn_gg_norm, 1e-12)
            )
        if is_main_process and probe_stats is not None:
            row["probe_mean_cosine"] = float(probe_stats["mean_cosine"])
            row["probe_mean_neg_ratio"] = float(probe_stats["mean_neg_ratio"])
            row["probe_global_gu_norm"] = float(probe_stats["global_gu_norm"])
            row["probe_global_gg_norm"] = float(probe_stats["global_gg_norm"])
            row["probe_global_gu_over_gg"] = float(probe_stats["global_gu_over_gg"])
            row["probe_weighted_gu_over_gg"] = float(
                (float(args.lambda_u) * probe_stats["global_gu_norm"])
                / max(float(eff_lambda_g) * probe_stats["global_gg_norm"], 1e-12)
            )
            row["probe_csv"] = str(probe_stats["csv_path"])
            print(
                f"[probe] step={global_step} mean_cos={probe_stats['mean_cosine']:.4f} "
                f"mean_neg_ratio={probe_stats['mean_neg_ratio']:.4f} "
                f"gu/gg={probe_stats['global_gu_over_gg']:.2f} "
                f"weighted_gu/weighted_gg={row['probe_weighted_gu_over_gg']:.2f} "
                f"csv={probe_stats['csv_path']}"
            )

        if is_main_process:
            append_jsonl(str(metrics_path), row)
            if pbar is not None:
                pbar.update(1)
                if (global_step == 1) or (global_step % int(args.log_every) == 0) or (
                    global_step == int(args.max_steps)
                ):
                    pbar.set_postfix(
                        lu=f"{row['loss_u']:.4f}",
                        lg=f"{row['loss_g']:.4f}",
                        luw=f"{row['loss_u_weighted']:.4f}",
                        lgw=f"{row['loss_g_weighted']:.4f}",
                        lg_eff=f"{row['effective_lambda_g']:.1f}",
                        acc=f"{row['acc']:.3f}",
                        rmse=f"{row['rmse']:.4f}",
                    )

        if int(args.eval_every) > 0 and (
            global_step % int(args.eval_every) == 0 or global_step == int(args.max_steps)
        ):
            if is_main_process:
                eval_ret = evaluate_recon_and_understanding(
                    model=base_model,
                    val_loader=val_loader,
                    device=device,
                    mean=mean,
                    std=std,
                    max_batches=int(args.eval_max_batches),
                    compute_rfid=(not bool(args.skip_rfid)),
                    rfid_num_samples=int(args.eval_rfid_num_samples),
                    rfid_batch_size=int(args.eval_rfid_batch_size),
                    rfid_tmp_dir=str(args.eval_rfid_tmp_dir),
                )
                eval_row = {"step": int(global_step), **eval_ret}
                append_jsonl(str(eval_metrics_path), eval_row)
                print(
                    "[eval] step={} acc={:.4f} rMSE={:.6f} rFID={:.4f} n={} rfid_n={}".format(
                        int(global_step),
                        float(eval_ret["val_top1_acc"]),
                        float(eval_ret["val_rmse"]),
                        float(eval_ret["val_rfid"]),
                        int(eval_ret["val_num_samples"]),
                        int(eval_ret["rfid_num_samples"]),
                    )
                )
                if eval_ret.get("rfid_error"):
                    print(f"[eval][rfid_error] {eval_ret['rfid_error']}")
            if is_distributed:
                dist.barrier()
            model.train()

    if pbar is not None:
        pbar.close()
    if is_main_process:
        torch.save(
            {
                "model": base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": int(global_step),
                "args": vars(args),
                "data_meta": data_meta,
            },
            str(run_dir / "latest.pt"),
        )
        print(f"[done] output={run_dir}")
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
