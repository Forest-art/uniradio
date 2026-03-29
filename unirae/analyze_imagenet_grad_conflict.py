import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .losses import FeatureVarianceLoss
from .models import UniRAEDinoLoRA
from .utils import ensure_dir, save_json


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
        x = self.transform(image)
        y = int(item.get("label", -1))
        return x, y


def _layer_group_vit(name: str) -> Tuple[str, int]:
    if name.startswith("patch_embed."):
        return "patch_embed", 0
    if name.startswith("blocks."):
        parts = name.split(".")
        try:
            idx = int(parts[1])
        except Exception:
            idx = -1
        return f"block{idx:02d}", idx + 1
    if name.startswith("norm."):
        return "norm", 999
    return "other", 1000


def _normalize_for_dino(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _tensor_to_flat(g: torch.Tensor, p: torch.nn.Parameter) -> torch.Tensor:
    if g is None:
        return torch.zeros_like(p, memory_format=torch.preserve_format).reshape(-1)
    return g.detach().reshape(-1)


def _cosine(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> float:
    n1 = torch.norm(g1)
    n2 = torch.norm(g2)
    if float(n1.item()) < eps or float(n2.item()) < eps:
        return 0.0
    return float(torch.dot(g1, g2).item() / (n1.item() * n2.item() + eps))


def _build_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 轻量增强：第二个视角做水平翻转 + 亮度抖动。
    x2 = torch.flip(x, dims=[-1])
    b = x.shape[0]
    scale = 0.6 + 0.8 * torch.rand((b, 1, 1, 1), device=x.device, dtype=x.dtype)
    x2 = (x2 * scale).clamp(0.0, 1.0)
    return x, x2


def _make_plots(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    ordered = summary["layer"].tolist()
    heat = df.pivot(index="layer", columns="batch_idx", values="cos").reindex(ordered)

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.7, 1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax0.set_yticks(np.arange(len(ordered)))
    ax0.set_yticklabels(ordered)
    xticks = np.linspace(0, heat.shape[1] - 1, num=min(8, heat.shape[1]), dtype=int)
    ax0.set_xticks(xticks)
    ax0.set_xticklabels([str(int(heat.columns[i])) for i in xticks])
    ax0.set_xlabel("Batch Index")
    ax0.set_ylabel("Encoder Layer")
    ax0.set_title("cos(grad_understanding, grad_generation)")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity")

    ax1 = fig.add_subplot(gs[0, 1])
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in summary["cos_mean"]]
    ax1.barh(summary["layer"], summary["cos_mean"], color=colors, alpha=0.9)
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean cosine")
    ax1.set_title("Mean by Layer")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(summary["depth"], summary["cos_mean"], marker="o", label="mean cos")
    ax2b = ax2.twinx()
    ax2b.plot(summary["depth"], summary["cos_neg_ratio"], marker="s", color="#d62728", label="neg ratio")
    ax2.set_xlabel("Depth Index")
    ax2.set_ylabel("Mean cosine")
    ax2b.set_ylabel("Negative ratio")
    ax2.set_ylim(-1.0, 1.0)
    ax2b.set_ylim(0.0, 1.0)
    ax2.set_title("Depth Profile")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)

    fig.suptitle("ImageNet-like (Imagenette Val) Gradient Conflict", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "grad_conflict_layers.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze layer-wise gradient conflict: DINO regularization vs reconstruction MSE.")
    parser.add_argument("--out_dir", default="results/grad_conflict_imagenette")
    parser.add_argument("--hf_dataset", default="frgfm/imagenette")
    parser.add_argument("--hf_config", default="160px")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=40)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dino_variant", default="vit_small_patch14_reg4_dinov2")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lambda_var", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    tfm = transforms.Compose(
        [
            transforms.Resize(int(args.image_size * 1.14)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )

    hf_ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
    ds = HFImageDataset(hf_ds, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = UniRAEDinoLoRA(
        num_classes=10,
        dino_variant=args.dino_variant,
        image_size=args.image_size,
        pretrained=bool(args.pretrained),
        lora_last_n_blocks=0,
        decoder_dim=384,
        decoder_depth=4,
        decoder_heads=8,
        decoder_noise_tau=0.0,  # 分析时关闭噪声，避免额外方差
    ).to(device)

    # 只做梯度分析：打开 encoder 梯度，方便比较不同层冲突模式。
    for p in model.encoder.parameters():
        p.requires_grad = True
    model.train()

    enc_named_params = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
    if len(enc_named_params) == 0:
        raise RuntimeError("No encoder params found for gradient analysis.")

    params = []
    param_to_group: List[str] = []
    group_depth: Dict[str, int] = {}
    for n, p in enc_named_params:
        g, d = _layer_group_vit(n)
        params.append(p)
        param_to_group.append(g)
        if g not in group_depth:
            group_depth[g] = d
        else:
            group_depth[g] = min(group_depth[g], d)

    var_loss_fn = FeatureVarianceLoss(gamma=1.0, eps=1e-4)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    rows = []
    for bi, (images, _) in enumerate(loader):
        if bi >= args.num_batches:
            break
        images = images.to(device, non_blocking=True)
        x1, x2 = _build_views(images)
        x1n = _normalize_for_dino(x1, mean, std)
        x2n = _normalize_for_dino(x2, mean, std)

        out1 = model(x1n)
        out2 = model(x2n)

        # Understanding: DINO-style regularization (view consistency + variance regularization).
        cls1 = F.normalize(out1["cls_token"], dim=-1)
        cls2 = F.normalize(out2["cls_token"], dim=-1)
        lu_align = 1.0 - (cls1 * cls2).sum(dim=-1).mean()
        lu_var = var_loss_fn(out1["cls_token"]) + var_loss_fn(out2["cls_token"])
        lu = lu_align + float(args.lambda_var) * lu_var

        # Generation: reconstruction MSE on patch pixels.
        target_patches = model.patchify(x1)
        lg = F.mse_loss(out1["pred_patches"], target_patches)

        gu = torch.autograd.grad(lu, params, retain_graph=True, allow_unused=True)
        gg = torch.autograd.grad(lg, params, retain_graph=False, allow_unused=True)

        by_group_u = defaultdict(list)
        by_group_g = defaultdict(list)
        for gname, p, g1, g2 in zip(param_to_group, params, gu, gg):
            by_group_u[gname].append(_tensor_to_flat(g1, p))
            by_group_g[gname].append(_tensor_to_flat(g2, p))

        for gname, depth in sorted(group_depth.items(), key=lambda x: (x[1], x[0])):
            gu_flat = torch.cat(by_group_u[gname], dim=0)
            gg_flat = torch.cat(by_group_g[gname], dim=0)
            rows.append(
                {
                    "batch_idx": bi,
                    "layer": gname,
                    "depth": depth,
                    "cos": _cosine(gu_flat, gg_flat),
                    "gu_norm": float(torch.norm(gu_flat).item()),
                    "gg_norm": float(torch.norm(gg_flat).item()),
                    "lu": float(lu.detach().item()),
                    "lg": float(lg.detach().item()),
                }
            )

    if len(rows) == 0:
        raise RuntimeError("No gradient rows collected.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "layerwise_grad_cos.csv", index=False)

    summary = (
        df.groupby(["layer", "depth"], as_index=False)
        .agg(
            cos_mean=("cos", "mean"),
            cos_std=("cos", "std"),
            cos_neg_ratio=("cos", lambda x: float((x < 0).mean())),
            gu_norm_mean=("gu_norm", "mean"),
            gg_norm_mean=("gg_norm", "mean"),
            lu_mean=("lu", "mean"),
            lg_mean=("lg", "mean"),
            n=("cos", "count"),
        )
        .sort_values(["depth", "layer"])
    )
    summary["cos_std"] = summary["cos_std"].fillna(0.0)
    summary.to_csv(out_dir / "layerwise_grad_summary.csv", index=False)

    global_cos_mean = float(df["cos"].mean())
    global_neg_ratio = float((df["cos"] < 0).mean())
    deep = summary[summary["depth"] >= summary["depth"].median()]
    shallow = summary[summary["depth"] < summary["depth"].median()]
    stats = {
        "dataset": f"{args.hf_dataset}:{args.hf_config}:{args.split}",
        "num_batches": int(df["batch_idx"].nunique()),
        "num_layers": int(summary.shape[0]),
        "global_cos_mean": global_cos_mean,
        "global_neg_ratio": global_neg_ratio,
        "shallow_mean_cos": float(shallow["cos_mean"].mean()) if len(shallow) else float("nan"),
        "deep_mean_cos": float(deep["cos_mean"].mean()) if len(deep) else float("nan"),
        "shallow_neg_ratio": float(shallow["cos_neg_ratio"].mean()) if len(shallow) else float("nan"),
        "deep_neg_ratio": float(deep["cos_neg_ratio"].mean()) if len(deep) else float("nan"),
    }
    save_json(stats, str(out_dir / "conflict_stats.json"))

    _make_plots(df=df, summary=summary, out_dir=out_dir)

    print(f"[grad_conflict] out_dir={out_dir}")
    print(
        "[grad_conflict] global_cos_mean={:.4f}, global_neg_ratio={:.3f}, shallow_mean={:.4f}, deep_mean={:.4f}".format(
            stats["global_cos_mean"],
            stats["global_neg_ratio"],
            stats["shallow_mean_cos"],
            stats["deep_mean_cos"],
        )
    )


if __name__ == "__main__":
    main()
