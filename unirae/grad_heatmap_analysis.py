import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data_cifar10 import build_cifar10_loader, make_batch_dict
from .train_cifar10 import CifarTradeoffModel, text_prototype_loss
from .utils import ensure_dir, load_yaml, save_json, to_device


def _layer_group_resnet18(param_name: str) -> Tuple[str, int]:
    if param_name.startswith("stem.0.") or param_name.startswith("stem.1."):
        return "stem", 0
    if param_name.startswith("stem.4."):
        return "layer1", 1
    if param_name.startswith("stem.5."):
        return "layer2", 2
    if param_name.startswith("stem.6."):
        return "layer3", 3
    if param_name.startswith("stem.7."):
        return "layer4", 4
    return "other", 99


def _layer_group_vit_small(param_name: str) -> Tuple[str, int]:
    if param_name.startswith("model.patch_embed."):
        return "patch_embed", 0
    if param_name.startswith("model.blocks."):
        parts = param_name.split(".")
        try:
            idx = int(parts[2])
        except Exception:
            idx = 0
        return f"block{idx:02d}", idx + 1
    if param_name.startswith("model.norm."):
        return "norm", 999
    return "other", 1000


def _layer_group(backbone_name: str, param_name: str) -> Tuple[str, int]:
    name = backbone_name.lower()
    if name == "resnet18":
        return _layer_group_resnet18(param_name)
    if name == "vit_small":
        return _layer_group_vit_small(param_name)
    return "other", 9999


def _grad_to_tensor(g, p: torch.nn.Parameter) -> torch.Tensor:
    if g is None:
        return torch.zeros_like(p).reshape(-1)
    return g.reshape(-1)


def _cosine(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> float:
    n1 = torch.norm(g1)
    n2 = torch.norm(g2)
    if float(n1.item()) < eps or float(n2.item()) < eps:
        return 0.0
    return float(torch.dot(g1, g2).item() / (n1.item() * n2.item() + eps))


def _build_layer_map(model: CifarTradeoffModel, backbone_name: str):
    named_params = [(n, p) for n, p in model.backbone.named_parameters() if p.requires_grad]
    if len(named_params) == 0:
        named_params = [(f"all.{n}", p) for n, p in model.named_parameters() if p.requires_grad]

    group_meta: Dict[str, int] = {}
    param_to_group: List[str] = []
    params: List[torch.nn.Parameter] = []
    for n, p in named_params:
        g, depth = _layer_group(backbone_name, n)
        params.append(p)
        param_to_group.append(g)
        if g not in group_meta:
            group_meta[g] = depth
        else:
            group_meta[g] = min(group_meta[g], depth)

    ordered_groups = sorted(group_meta.keys(), key=lambda x: (group_meta[x], x))
    return params, param_to_group, group_meta, ordered_groups


def _shallow_deep_split(ordered_groups: List[str]) -> Tuple[List[str], List[str]]:
    if len(ordered_groups) <= 1:
        return ordered_groups, []
    mid = (len(ordered_groups) + 1) // 2
    return ordered_groups[:mid], ordered_groups[mid:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-wise gradient cosine heatmap for CIFAR-10 understanding vs generation.")
    parser.add_argument("--run_dir", required=True, help="Path to run directory containing run_config.yaml and checkpoints/")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path; default is <run_dir>/checkpoints/latest.pt")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_batches", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_root", default="", help="Override data.root from run config")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--out_dir", default="", help="Output folder, default <run_dir>/analysis/grad_heatmap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.yaml not found: {cfg_path}")
    cfg = load_yaml(str(cfg_path))

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (run_dir / "checkpoints" / "latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "analysis" / "grad_heatmap")
    ensure_dir(str(out_dir))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_cfg = cfg.get("data", {})
    data_root = args.data_root or data_cfg.get("root", "")
    if not data_root:
        raise ValueError("data.root is empty; pass --data_root explicitly.")

    split = args.split
    if split == "val":
        split = "test"

    dataset_name = str(data_cfg.get("dataset", "cifar10")).lower()
    loader, class_names = build_cifar10_loader(
        dataset=dataset_name,
        data_root=data_root,
        split=split,
        image_size=int(data_cfg.get("image_size", 32)),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        val_from_train=bool(data_cfg.get("val_from_train", False)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=int(cfg.get("seed", 42)),
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        download=bool(data_cfg.get("download", True)),
        use_fake_data=bool(data_cfg.get("use_fake_data", False)),
        fake_train_size=int(data_cfg.get("fake_train_size", 8192)),
        fake_eval_size=int(data_cfg.get("fake_eval_size", 1024)),
        two_view=False,
        aug_strength=str(cfg.get("consistency", {}).get("aug_strength", "medium")),
        target_source="view1",
    )

    model = CifarTradeoffModel(cfg=cfg, num_classes=len(class_names)).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    temperature = float(cfg.get("text", {}).get("temperature", 0.07))
    backbone_name = str(cfg.get("model", {}).get("backbone", "resnet18"))
    params, param_to_group, group_meta, ordered_groups = _build_layer_map(model, backbone_name=backbone_name)
    if len(params) == 0:
        raise RuntimeError("No trainable params found for gradient heatmap.")

    rows = []
    max_batches = max(1, int(args.num_batches))
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = to_device(make_batch_dict(batch), device)
        images = batch["images"]
        labels = batch["labels"]

        out = model(images)
        lu, _ = text_prototype_loss(
            z_txt=out["z_txt"],
            labels=labels,
            prototypes=model.text_prototypes,
            temperature=temperature,
        )
        lg = F.mse_loss(out["recon"], images)

        gu = torch.autograd.grad(lu, params, retain_graph=True, allow_unused=True)
        gg = torch.autograd.grad(lg, params, retain_graph=False, allow_unused=True)

        by_group_u = defaultdict(list)
        by_group_g = defaultdict(list)
        for gname, p, g1, g2 in zip(param_to_group, params, gu, gg):
            by_group_u[gname].append(_grad_to_tensor(g1, p))
            by_group_g[gname].append(_grad_to_tensor(g2, p))

        for gname in ordered_groups:
            gu_flat = torch.cat(by_group_u[gname], dim=0)
            gg_flat = torch.cat(by_group_g[gname], dim=0)
            cos = _cosine(gu_flat, gg_flat)
            rows.append(
                {
                    "batch_idx": bi,
                    "layer": gname,
                    "depth": int(group_meta[gname]),
                    "cos": cos,
                    "gu_norm": float(torch.norm(gu_flat).item()),
                    "gg_norm": float(torch.norm(gg_flat).item()),
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
            n=("cos", "count"),
        )
        .sort_values(["depth", "layer"])
    )
    summary["cos_std"] = summary["cos_std"].fillna(0.0)
    summary.to_csv(out_dir / "layerwise_grad_summary.csv", index=False)

    ordered_groups = summary["layer"].tolist()
    heat = df.pivot(index="layer", columns="batch_idx", values="cos").reindex(ordered_groups)

    shallow_layers, deep_layers = _shallow_deep_split(ordered_groups)
    shallow_mask = summary["layer"].isin(shallow_layers)
    deep_mask = summary["layer"].isin(deep_layers)

    shallow_mean_cos = float(summary.loc[shallow_mask, "cos_mean"].mean()) if shallow_layers else float("nan")
    deep_mean_cos = float(summary.loc[deep_mask, "cos_mean"].mean()) if deep_layers else float("nan")
    shallow_neg_ratio = float(summary.loc[shallow_mask, "cos_neg_ratio"].mean()) if shallow_layers else float("nan")
    deep_neg_ratio = float(summary.loc[deep_mask, "cos_neg_ratio"].mean()) if deep_layers else float("nan")

    stats = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_batches": int(df["batch_idx"].nunique()),
        "num_layers": len(ordered_groups),
        "backbone": backbone_name,
        "shallow_layers": shallow_layers,
        "deep_layers": deep_layers,
        "shallow_mean_cos": shallow_mean_cos,
        "deep_mean_cos": deep_mean_cos,
        "shallow_neg_ratio": shallow_neg_ratio,
        "deep_neg_ratio": deep_neg_ratio,
    }
    save_json(stats, str(out_dir / "grad_depth_stats.json"))

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.7, 1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax0.set_yticks(np.arange(len(ordered_groups)))
    ax0.set_yticklabels(ordered_groups)
    xticks = np.linspace(0, heat.shape[1] - 1, num=min(8, heat.shape[1]), dtype=int)
    ax0.set_xticks(xticks)
    ax0.set_xticklabels([str(int(heat.columns[i])) for i in xticks])
    ax0.set_xlabel("Batch Index")
    ax0.set_ylabel("Backbone Layer Group (shallow -> deep)")
    ax0.set_title("Layer-wise cos(g_understanding, g_generation)")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity")

    ax1 = fig.add_subplot(gs[0, 1])
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in summary["cos_mean"]]
    ax1.barh(summary["layer"], summary["cos_mean"], color=colors, alpha=0.85)
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean cosine")
    ax1.set_title("Mean by Layer")

    ax2 = fig.add_subplot(gs[0, 2])
    group_names = ["shallow", "deep"]
    mean_vals = [shallow_mean_cos, deep_mean_cos]
    neg_vals = [shallow_neg_ratio, deep_neg_ratio]
    xpos = np.arange(2)
    bw = 0.38
    ax2.bar(xpos - bw / 2, mean_vals, width=bw, color=["#1f77b4", "#ff7f0e"], label="mean cos")
    ax2b = ax2.twinx()
    ax2b.bar(xpos + bw / 2, neg_vals, width=bw, color=["#17becf", "#9467bd"], alpha=0.75, label="neg ratio")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(group_names)
    ax2.set_ylim(-1.0, 1.0)
    ax2b.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Mean cosine")
    ax2b.set_ylabel("Negative ratio")
    ax2.set_title("Depth Summary")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)

    fig.suptitle(f"Gradient Cooperation/Conflict by Depth ({backbone_name})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "grad_layer_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[grad_heatmap] out_dir: {out_dir}")
    print(
        "[grad_heatmap] shallow_mean_cos={:.4f}, deep_mean_cos={:.4f}, shallow_neg_ratio={:.3f}, deep_neg_ratio={:.3f}".format(
            shallow_mean_cos,
            deep_mean_cos,
            shallow_neg_ratio,
            deep_neg_ratio,
        )
    )


if __name__ == "__main__":
    main()
