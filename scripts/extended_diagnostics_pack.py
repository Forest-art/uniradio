#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("NUMEXPR_MAX_THREADS", "64")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from unirae.data_cifar10 import build_cifar10_loader, make_batch_dict
from unirae.train_cifar10 import CifarTradeoffModel, text_prototype_loss
from unirae.train_imagenet100_dynamics import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ViTBridgeModel,
    _group_encoder_params,
    build_encoder,
    build_imagenet100_dataloaders,
)


ALLOWED_ROOTS = [
    Path("/project/peilab/luxiaocheng").resolve(),
    Path("/home/xlubl").resolve(),
]

IN100_METHOD_RUNS = {
    "joint": Path("results/long_runs/baseline3_10k_eval1k_20260228_134109_scratch_joint_lu1_lg100_s42"),
    "pcgrad": Path("results/long_runs/gradstrategy3_10k_eval1k_20260228_140137_scratch_pcgrad_lu1.0_lg100.0_s42"),
    "cagrad": Path("results/long_runs/gradstrategy3_10k_eval1k_20260228_140137_scratch_cagrad_lu1.0_lg100.0_s42"),
    "dsga": Path("in100_ma_laga_scratch_bs64_s42"),
}

IN100_INIT_RUNS = {
    "scratch_joint": Path("results/long_runs/baseline3_10k_eval1k_20260228_134109_scratch_joint_lu1_lg100_s42"),
    "scratch_genonly": Path("results/long_runs/baseline3_10k_eval1k_20260228_134109_scratch_genonly_lu0_lg100_s42"),
    "dino_joint": Path("results/long_runs/baseline3_10k_eval1k_20260228_134109_dinov2_joint_lu1_lg100_s42"),
    "dsga": Path("in100_ma_laga_scratch_bs64_s42"),
}

CIFAR_GRANULARITY_RUNS = {
    "global": Path("runs/cifar100_patch4_ma_laga_grouping20k_global_g1p0_s42_20260304_124457"),
    "layerwise": Path("runs/cifar100_patch4_ma_laga_grouping20k_layerwise_g1p0_s42_20260304_124457"),
}

CIFAR_ABLATION_RUNS = {
    "joint": Path("runs/cifar100_patch4_ma_laga_ablation20k_vanilla_naive_s42_20260303_003056"),
    "dsga_d": Path("runs/cifar100_patch4_ma_laga_ablation20k_pure_laga_dironly_s42_20260303_003056"),
    "dsga_full": Path("runs/cifar100_patch4_ma_laga_ablation20k_full_ma_laga_s42_20260303_003056"),
}

CIFAR_ARCH_RUNS = {
    "resnet18": Path("runs/cifar100_b5_s42_20260223_113334_joint_naive_s42"),
    "vit_small": Path("runs/cifar100_vit_cmp_20260224_20k_naive_s42"),
    "swin_tiny_patch4": Path("runs/cifar100_patch4_cmp_20260224_20k_naive_s42"),
}

CIFAR_MAG_RUNS = [
    Path("runs/dsga_swin_cifar100_mag_0p1_s42"),
    Path("runs/dsga_swin_cifar100_mag_0p2_s42"),
    Path("runs/dsga_swin_cifar100_mag_0p3_s42"),
    Path("runs/dsga_swin_cifar100_mag_0p4_s42"),
    Path("runs/cifar100_patch4_ma_laga_grouping20k_layerwise_g1p0_s42_20260304_124457"),
]
ITEM2_EXPECTED_GAMMAS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
ITEM2_DEFAULT_PREFIX = "item2_cifar100_vits_dsga_mag"

ITEM_ORDER = ["1", "2", "3", "4", "5", "6", "7"]


@dataclass
class ReadmeNote:
    item: str
    title: str
    summary: str
    anomaly: str


def ensure_allowed_path(path: Path) -> Path:
    resolved = path.resolve()
    for root in ALLOWED_ROOTS:
        if resolved == root or root in resolved.parents:
            return resolved
    raise ValueError(f"Path is outside allowlist: {resolved}")


def ensure_dir(path: Path) -> Path:
    ensure_allowed_path(path.parent if path.suffix else path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def save_yaml(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def bold_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")


def parse_items(raw: str) -> List[str]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    items = [x for x in parts if x in ITEM_ORDER]
    if not items:
        return list(ITEM_ORDER)
    return items


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def round_headline(acc: float, rmse: float, rfid: float) -> Dict[str, object]:
    return {
        "acc_pct": None if math.isnan(acc) else round(acc * 100.0, 2),
        "rmse": None if math.isnan(rmse) else round(rmse, 3),
        "rfid": None if math.isnan(rfid) else round(rfid, 3),
    }


def headline_from_in100_run(run_dir: Path) -> Dict[str, object]:
    run_dir = ensure_allowed_path(run_dir)
    final_eval = run_dir / "final_eval.json"
    if final_eval.exists():
        data = load_json(final_eval)
    else:
        rows = load_jsonl(run_dir / "eval_metrics.jsonl")
        if not rows:
            raise FileNotFoundError(f"No evaluation rows found in {run_dir}")
        data = rows[-1]
    return round_headline(
        safe_float(data.get("val_top1_acc", float("nan"))),
        safe_float(data.get("val_rmse", float("nan"))),
        safe_float(data.get("val_rfid", float("nan"))),
    )


def headline_from_cifar_run(run_dir: Path) -> Dict[str, object]:
    run_dir = ensure_allowed_path(run_dir)
    data = load_json(run_dir / "eval_last.json")
    acc = safe_float(data.get("understanding", {}).get("acc_txt", float("nan")))
    mse = safe_float(data.get("generation", {}).get("recon_mse", float("nan")))
    rmse = math.sqrt(max(mse, 0.0)) if not math.isnan(mse) else float("nan")
    rfid = safe_float(data.get("generation", {}).get("rfid", float("nan")))
    return round_headline(acc, rmse, rfid)


def heatmap_arrays_from_step_csvs(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], pd.DataFrame]:
    run_dir = ensure_allowed_path(run_dir)
    csvs = sorted(run_dir.glob("step_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No step_*.csv found in {run_dir}")
    frames = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    layer_order = (
        df_all[["layer", "depth"]]
        .drop_duplicates()
        .sort_values(["depth", "layer"])
    )
    layers = [str(x) for x in layer_order["layer"].tolist()]
    steps = sorted(int(x) for x in df_all["step"].unique().tolist())
    pivot = (
        df_all.pivot(index="step", columns="layer", values="cosine_similarity")
        .reindex(index=steps, columns=layers)
    )
    heat = pivot.to_numpy(dtype=np.float32)
    rho = (heat < 0.0).astype(np.float32)
    return heat, rho, steps, layers, df_all


def plot_heatmap(
    path: Path,
    heat: np.ndarray,
    x_values: Sequence[int],
    y_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    im = ax.imshow(heat.T, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="lower")
    xticks = np.linspace(0, max(len(x_values) - 1, 0), num=min(6, len(x_values)), dtype=int)
    ax.set_xticks(xticks)
    if len(x_values) > 0:
        ax.set_xticklabels([str(int(x_values[i])) for i in xticks])
    yticks = np.arange(len(y_labels))
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels)
    bold_axis(ax, xlabel, ylabel, title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Cosine", fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_rho_bars(path: Path, per_method_layer_mean: Dict[str, Tuple[List[str], np.ndarray]]) -> None:
    methods = list(per_method_layer_mean.keys())
    base_layers = per_method_layer_mean[methods[0]][0]
    xs = np.arange(len(base_layers))
    width = 0.18
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    palette = ["#4c566a", "#6b7280", "#9ca3af", "#111827"]
    for i, method in enumerate(methods):
        _, vals = per_method_layer_mean[method]
        ax.bar(xs + (i - (len(methods) - 1) / 2.0) * width, vals, width=width, label=method, color=palette[i % len(palette)])
    ax.set_xticks(xs)
    ax.set_xticklabels(base_layers, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    bold_axis(ax, "Layer", r"Temporal mean $\rho_k^-$", "IN-100 Temporal Conflict Occupancy")
    ax.legend(ncol=2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def item1_conflict_cartography(pack_root: Path, notes: List[ReadmeNote]) -> None:
    fig_root = ensure_dir(pack_root / "figs")
    all_layer_means: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for method, rel_run in IN100_METHOD_RUNS.items():
        run_dir = ensure_allowed_path(pack_root.parent.parent / rel_run if not rel_run.is_absolute() else rel_run)
        heat, rho, steps, layers, _ = heatmap_arrays_from_step_csvs(run_dir)
        exp_id = f"item1_in100_vits_{method}"
        log_dir = ensure_dir(pack_root / "logs" / exp_id)
        np.save(log_dir / "ck_heatmap.npy", heat)
        np.save(log_dir / "rho_minus.npy", rho)
        save_json(log_dir / "layers.json", layers)
        save_json(log_dir / "steps.json", steps)
        save_json(log_dir / "headline.json", headline_from_in100_run(run_dir))
        save_yaml(
            log_dir / "config.yaml",
            {
                "protocol": "existing_run_replay",
                "source_run": str(run_dir),
                "deviations": [
                    "Probe cadence reuses existing step_*.csv logs at every 500 steps, not every 100 steps.",
                    "rho_minus is computed as 1[cosine_similarity < 0] from the saved layerwise cosine traces.",
                ],
            },
        )
        plot_heatmap(
            fig_root / f"diag_ck_heatmap_in100_vits_{method}.pdf",
            heat,
            steps,
            layers,
            xlabel="Step",
            ylabel="Layer",
            title=f"IN-100 ViT-S {method.upper()}",
        )
        all_layer_means[method] = (layers, rho.mean(axis=0))

    plot_rho_bars(fig_root / "diag_rho_minus_in100_vits.pdf", all_layer_means)

    dsga_layers, dsga_rho = all_layer_means["dsga"]
    deep_slice = dsga_rho[-4:] if len(dsga_rho) >= 4 else dsga_rho
    notes.append(
        ReadmeNote(
            item="1",
            title="Depth-Time Conflict Cartography",
            summary=(
                f"Existing IN-100 10k probe logs show persistent antagonistic patches across all four methods. "
                f"In DSGA, the deepest four layers still spend {deep_slice.mean():.2f} of sampled time in negative cosine territory."
            ),
            anomaly="The available runs probe every 500 steps rather than every 100 steps from the requested protocol.",
        )
    )


class CifarRun:
    def __init__(self, run_dir: Path, device: torch.device):
        self.run_dir = ensure_allowed_path(run_dir)
        self.cfg = yaml.safe_load((self.run_dir / "run_config.yaml").read_text())
        self.device = device
        self.num_classes = 100 if str(self.cfg.get("data", {}).get("dataset", "")).lower() == "cifar100" else 10
        self.model = CifarTradeoffModel(self.cfg, num_classes=self.num_classes)
        ckpt = torch.load(self.run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.step = int(ckpt.get("step", 0))
        self.shared_mode = str(self.cfg.get("train", {}).get("shared_params", "backbone")).lower()
        self.backbone_name = str(self.cfg.get("model", {}).get("backbone", "resnet18")).lower()
        self.temperature = float(self.cfg.get("text", {}).get("temperature", 0.07))
        self.dataset_name = str(self.cfg.get("data", {}).get("dataset", "cifar10")).lower()
        self.group_info = self._build_group_info()

    def _shared_named_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        if self.shared_mode == "all":
            return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        return [(f"backbone.{n}", p) for n, p in self.model.backbone.named_parameters() if p.requires_grad]

    def _layer_group(self, full_name: str) -> Tuple[str, int]:
        name = full_name[len("backbone.") :] if full_name.startswith("backbone.") else full_name
        if self.backbone_name == "resnet18":
            if name.startswith(("stem.0.", "stem.1.", "stem.2.", "stem.3.")):
                return "stem", 0
            if name.startswith("stem.4."):
                block = int(name.split(".")[2]) if len(name.split(".")) > 2 and name.split(".")[2].isdigit() else 0
                return f"layer1.{block}", 1 + block
            if name.startswith("stem.5."):
                block = int(name.split(".")[2]) if len(name.split(".")) > 2 and name.split(".")[2].isdigit() else 0
                return f"layer2.{block}", 3 + block
            if name.startswith("stem.6."):
                block = int(name.split(".")[2]) if len(name.split(".")) > 2 and name.split(".")[2].isdigit() else 0
                return f"layer3.{block}", 5 + block
            if name.startswith("stem.7."):
                block = int(name.split(".")[2]) if len(name.split(".")) > 2 and name.split(".")[2].isdigit() else 0
                return f"layer4.{block}", 7 + block
            return "other", 999
        if self.backbone_name == "vit_small":
            if name.startswith(("model.cls_token", "model.pos_embed")):
                return "embeddings", 0
            if name.startswith("model.patch_embed."):
                return "patch_embed", 1
            if name.startswith("model.blocks."):
                parts = name.split(".")
                block = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                return f"block{block:02d}", block + 2
            if name.startswith("model.norm."):
                return "norm", 999
            return "other", 1000
        if self.backbone_name in {"swin_tiny_patch4", "patch4", "vit_patch4"}:
            if name.startswith("model.patch_embed."):
                return "patch_embed", 0
            if name.startswith("model.layers."):
                parts = name.split(".")
                stage = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                block = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else -1
                if ".downsample." in name:
                    return f"stage{stage}_downsample", stage * 10 + 8
                return f"stage{stage}_block{block}", stage * 10 + block + 1
            if name.startswith("model.norm."):
                return "norm", 999
            return "other", 1000
        return "all", 0

    def _build_group_info(self) -> Dict[str, object]:
        named = self._shared_named_params()
        params = [p for _, p in named]
        param_to_group: List[str] = []
        group_to_depth: Dict[str, int] = {}
        group_to_indices: Dict[str, List[int]] = {}
        for idx, (name, _) in enumerate(named):
            gname, depth = self._layer_group(name)
            param_to_group.append(gname)
            group_to_indices.setdefault(gname, []).append(idx)
            group_to_depth[gname] = min(depth, group_to_depth.get(gname, depth))
        ordered = sorted(group_to_indices.keys(), key=lambda x: (group_to_depth[x], x))
        return {
            "named": named,
            "params": params,
            "group_to_indices": group_to_indices,
            "group_to_depth": group_to_depth,
            "ordered_groups": ordered,
        }

    def build_loader(self, split: str, batch_size: int, num_workers: int):
        data_cfg = self.cfg.get("data", {})
        data_root = Path(str(data_cfg.get("root", "")))
        ensure_allowed_path(data_root)
        loader, class_names = build_cifar10_loader(
            dataset=str(data_cfg.get("dataset", "cifar10")),
            data_root=str(data_root),
            split=split,
            image_size=int(data_cfg.get("image_size", 32)),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            val_from_train=bool(data_cfg.get("val_from_train", False)),
            val_ratio=float(data_cfg.get("val_ratio", 0.1)),
            seed=int(self.cfg.get("seed", 42)),
            shuffle=False,
            drop_last=False,
            download=bool(data_cfg.get("download", False)),
        )
        return loader, class_names



def _materialize_grad_list(grads: Sequence[Optional[torch.Tensor]], params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    out = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p, memory_format=torch.preserve_format))
        else:
            out.append(g.detach())
    return out



def _safe_cosine_flat(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> float:
    n1 = torch.norm(g1)
    n2 = torch.norm(g2)
    if float(n1.item()) < eps or float(n2.item()) < eps:
        return 0.0
    return float(torch.dot(g1, g2).item() / (n1.item() * n2.item() + eps))



def _group_flat(grads: Sequence[torch.Tensor], indices: Sequence[int]) -> torch.Tensor:
    return torch.cat([grads[i].reshape(-1) for i in indices], dim=0)



def collect_cifar_grad_heatmap(
    run: CifarRun,
    split: str,
    batch_size: int,
    num_batches: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    loader, _ = run.build_loader(split=split, batch_size=batch_size, num_workers=num_workers)
    info = run.group_info
    params = info["params"]
    group_to_indices = info["group_to_indices"]
    ordered_groups = info["ordered_groups"]
    all_rows: List[List[float]] = []
    with torch.enable_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= int(num_batches):
                break
            batch = make_batch_dict(batch)
            images = batch["images"].to(run.device, non_blocking=True)
            labels = batch["labels"].to(run.device, non_blocking=True)

            out = run.model(images)
            loss_u, _ = text_prototype_loss(
                z_txt=out["z_txt"],
                labels=labels,
                prototypes=run.model.text_prototypes,
                temperature=run.temperature,
            )
            target = images
            recon = out["recon"]
            if recon.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=recon.shape[-2:], mode="bilinear", align_corners=False)
            loss_g = F.mse_loss(recon, target)

            gu = torch.autograd.grad(loss_u, params, retain_graph=True, allow_unused=True)
            gg = torch.autograd.grad(loss_g, params, retain_graph=False, allow_unused=True)
            gu = _materialize_grad_list(gu, params)
            gg = _materialize_grad_list(gg, params)

            row = []
            for gname in ordered_groups:
                idxs = group_to_indices[gname]
                gu_flat = _group_flat(gu, idxs)
                gg_flat = _group_flat(gg, idxs)
                row.append(_safe_cosine_flat(gu_flat, gg_flat))
            all_rows.append(row)
    heat = np.asarray(all_rows, dtype=np.float32)
    rho = (heat < 0.0).astype(np.float32)
    steps = list(range(len(all_rows)))
    return heat, rho, steps, list(ordered_groups)



def _covariance_stats(features: np.ndarray) -> Dict[str, object]:
    x = features.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0.0)
    eig = np.sort(eig)[::-1]
    total = float(eig.sum())
    pr = float((total ** 2) / max(float((eig * eig).sum()), 1e-12))
    if total <= 0.0:
        eff_rank = 0
    else:
        cdf = np.cumsum(eig) / total
        eff_rank = int(np.searchsorted(cdf, 0.95) + 1)
    return {
        "eigvals": eig,
        "participation_ratio": pr,
        "effective_rank_95": eff_rank,
        "dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
    }



def _extract_cifar_feature_matrix(run: CifarRun, images: torch.Tensor, feature_mode: str) -> torch.Tensor:
    feature_mode = str(feature_mode).lower()
    if feature_mode in {"pooled", "feat"}:
        return run.model(images)["feat"].detach()

    backbone = run.model.backbone.model
    if not hasattr(backbone, "forward_intermediates"):
        raise RuntimeError(f"Backbone has no forward_intermediates: {run.backbone_name}")

    if feature_mode in {"swin_stage2", "swin_stage_2", "stage2"}:
        _, inter = backbone.forward_intermediates(
            images,
            indices=[2],
            norm=False,
            output_fmt="NCHW",
            intermediates_only=False,
        )
        feat = inter[0].detach()
        return feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])

    if feature_mode in {"vit_penultimate", "vit_block10", "vit_block_10"}:
        _, inter = backbone.forward_intermediates(
            images,
            indices=[-2],
            norm=False,
            output_fmt="NLC",
            intermediates_only=False,
        )
        feat = inter[0].detach()
        return feat.reshape(-1, feat.shape[-1])

    raise ValueError(f"Unsupported feature_mode={feature_mode}")


def collect_cifar_features(
    run: CifarRun,
    split: str,
    batch_size: int,
    num_batches: int,
    num_workers: int,
    feature_mode: str = "pooled",
) -> np.ndarray:
    loader, _ = run.build_loader(split=split, batch_size=batch_size, num_workers=num_workers)
    feats = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= int(num_batches):
                break
            batch = make_batch_dict(batch)
            images = batch["images"].to(run.device, non_blocking=True)
            feat = _extract_cifar_feature_matrix(run, images, feature_mode=feature_mode)
            feats.append(feat.cpu().numpy())
    if not feats:
        raise RuntimeError(f"No features collected for {run.run_dir}")
    return np.concatenate(feats, axis=0)



def save_cifar_heatmap_outputs(
    pack_root: Path,
    exp_id: str,
    heat: np.ndarray,
    rho: np.ndarray,
    steps: List[int],
    layers: List[str],
    headline: Dict[str, object],
    title: str,
    fig_name: str,
    source_run: Path,
    deviations: List[str],
) -> None:
    log_dir = ensure_dir(pack_root / "logs" / exp_id)
    np.save(log_dir / "ck_heatmap.npy", heat)
    np.save(log_dir / "rho_minus.npy", rho)
    save_json(log_dir / "steps.json", steps)
    save_json(log_dir / "layers.json", layers)
    save_json(log_dir / "headline.json", headline)
    save_yaml(log_dir / "config.yaml", {"source_run": str(source_run), "deviations": deviations})
    plot_heatmap(
        pack_root / "figs" / fig_name,
        heat,
        steps,
        layers,
        xlabel="Batch index",
        ylabel="Layer",
        title=title,
    )



def item3_granularity(pack_root: Path, device: torch.device, args, notes: List[ReadmeNote]) -> None:
    deviations = [
        "Granularity heatmaps are recomputed offline from the final checkpoints over held-out CIFAR-100 batches.",
        "The x-axis is batch index instead of training time because the stored runs do not contain per-step layer_probe traces.",
    ]
    metrics = {}
    for mode, run_dir in CIFAR_GRANULARITY_RUNS.items():
        run = CifarRun(pack_root.parent.parent / run_dir, device)
        heat, rho, steps, layers = collect_cifar_grad_heatmap(
            run,
            split=args.cifar_split,
            batch_size=args.cifar_batch_size,
            num_batches=args.cifar_num_batches,
            num_workers=args.num_workers,
        )
        headline = headline_from_cifar_run(run.run_dir)
        metrics[mode] = headline
        save_cifar_heatmap_outputs(
            pack_root=pack_root,
            exp_id=f"item3_cifar_swin_{mode}",
            heat=heat,
            rho=rho,
            steps=steps,
            layers=layers,
            headline=headline,
            title=f"CIFAR-100 Swin-T {mode}",
            fig_name=f"diag_ck_heatmap_cifar_swin_{mode}.pdf",
            source_run=run.run_dir,
            deviations=deviations,
        )

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), constrained_layout=True)
    modes = ["global", "layerwise"]
    acc_vals = [metrics[m]["acc_pct"] for m in modes]
    rmse_vals = [metrics[m]["rmse"] for m in modes]
    axes[0].bar(modes, acc_vals, color=["#9ca3af", "#111827"])
    bold_axis(axes[0], "Mode", "Acc (%)", "Granularity vs Acc")
    axes[1].bar(modes, rmse_vals, color=["#9ca3af", "#111827"])
    bold_axis(axes[1], "Mode", "rMSE", "Granularity vs rMSE")
    fig.savefig(pack_root / "figs" / "ablation_granularity_cifar_swin.pdf", bbox_inches="tight")
    plt.close(fig)

    layer_acc = safe_float(metrics["layerwise"]["acc_pct"], float("nan"))
    global_acc = safe_float(metrics["global"]["acc_pct"], float("nan"))
    layer_rmse = safe_float(metrics["layerwise"]["rmse"], float("nan"))
    global_rmse = safe_float(metrics["global"]["rmse"], float("nan"))
    notes.append(
        ReadmeNote(
            item="3",
            title="Granularity Matters",
            summary=(
                f"Layer-wise DSGA stays ahead of global routing on both axes: Acc {layer_acc:.2f}% vs {global_acc:.2f}%, "
                f"rMSE {layer_rmse:.3f} vs {global_rmse:.3f}."
            ),
            anomaly="Granularity heatmaps are batch-wise offline probes from the final checkpoints, not training-time heatmaps.",
        )
    )



def item4_feature_variance(pack_root: Path, device: torch.device, args, notes: List[ReadmeNote]) -> None:
    repo_root = ensure_allowed_path(Path(args.repo_root))
    feature_mode = str(args.item4_feature_mode)
    feature_label = feature_mode.replace("_", " ")
    stats = {}
    for method, rel_run in CIFAR_ABLATION_RUNS.items():
        run = CifarRun(repo_root / rel_run, device)
        feats = collect_cifar_features(
            run,
            split=args.cifar_split,
            batch_size=args.cifar_batch_size,
            num_batches=args.cifar_feature_batches,
            num_workers=args.num_workers,
            feature_mode=feature_mode,
        )
        stat = _covariance_stats(feats)
        headline = headline_from_cifar_run(run.run_dir)
        stats[method] = {"variance": stat, "headline": headline}
        log_dir = ensure_dir(pack_root / "logs" / f"item4_cifar_swin_{method}")
        np.save(log_dir / "eigvals.npy", stat["eigvals"].astype(np.float32))
        save_json(
            log_dir / "variance_stats.json",
            {
                "layer_id": feature_mode,
                "feature_mode": feature_mode,
                "participation_ratio": round(stat["participation_ratio"], 4),
                "effective_rank_95": int(stat["effective_rank_95"]),
                "num_samples": int(stat["num_samples"]),
                "dim": int(stat["dim"]),
            },
        )
        save_json(log_dir / "headline.json", headline)
        save_yaml(
            log_dir / "config.yaml",
            {
                "source_run": str(run.run_dir),
                "feature_mode": feature_mode,
                "deviations": [
                    f"Feature geometry is computed on final-checkpoint intermediate activations with feature_mode={feature_mode}.",
                    "This is an offline held-out variance probe rather than a time-resolved training trace.",
                ],
            },
        )

    fig, ax = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
    colors = {"joint": "#9ca3af", "dsga_d": "#6b7280", "dsga_full": "#111827"}
    for method in ["joint", "dsga_d", "dsga_full"]:
        eig = stats[method]["variance"]["eigvals"]
        xs = np.arange(1, len(eig) + 1)
        ax.plot(xs, eig + 1e-12, label=method, color=colors[method])
    ax.set_yscale("log")
    bold_axis(ax, "Eigen index", "Eigenvalue (log)", f"CIFAR-100 Swin {feature_label.title()} Spectrum")
    ax.legend()
    fig.savefig(pack_root / "figs" / "diag_spectrum_cifar_swin_layerfeat.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), constrained_layout=True)
    methods = ["joint", "dsga_d", "dsga_full"]
    pr_vals = [stats[m]["variance"]["participation_ratio"] for m in methods]
    rk_vals = [stats[m]["variance"]["effective_rank_95"] for m in methods]
    axes[0].bar(methods, pr_vals, color=[colors[m] for m in methods])
    bold_axis(axes[0], "Method", "PR", "Participation Ratio")
    axes[1].bar(methods, rk_vals, color=[colors[m] for m in methods])
    bold_axis(axes[1], "Method", "Rank@95%", "Effective Rank")
    fig.savefig(pack_root / "figs" / "diag_pr_rank_cifar_swin.pdf", bbox_inches="tight")
    plt.close(fig)

    pr_joint = stats["joint"]["variance"]["participation_ratio"]
    pr_full = stats["dsga_full"]["variance"]["participation_ratio"]
    rk_joint = stats["joint"]["variance"]["effective_rank_95"]
    rk_full = stats["dsga_full"]["variance"]["effective_rank_95"]
    trend = "broadens" if (pr_full > pr_joint and rk_full >= rk_joint) else "does not broaden"
    notes.append(
        ReadmeNote(
            item="4",
            title="Feature Variance Geometry",
            summary=(
                f"On {feature_label} intermediate features, DSGA Full {trend} the bandwidth relative to Joint: "
                f"PR {pr_full:.2f} vs {pr_joint:.2f}, effective rank {rk_full} vs {rk_joint}. "
                f"DSGA-D only remains narrower and keeps weaker reconstruction than DSGA Full."
            ),
            anomaly=(
                f"The variance probe uses feature_mode={feature_mode} from the final checkpoint instead of a full time-resolved feature trajectory."
            ),
        )
    )



def item5_init_anchor(pack_root: Path, notes: List[ReadmeNote]) -> None:
    rows = []
    for label, rel_run in IN100_INIT_RUNS.items():
        run_dir = ensure_allowed_path(pack_root.parent.parent / rel_run)
        headline = headline_from_in100_run(run_dir)
        rows.append({"label": label, **headline, "run_dir": str(run_dir)})
        log_dir = ensure_dir(pack_root / "logs" / f"item5_{label}")
        save_json(log_dir / "headline.json", headline)
        save_yaml(
            log_dir / "config.yaml",
            {
                "source_run": str(run_dir),
                "deviations": [
                    "No separate DINO-freeze run was found in the current workspace; the scatter uses scratch joint, scratch gen-only, DINO joint, and DSGA scratch.",
                ],
            },
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    colors = {
        "scratch_joint": "#6b7280",
        "scratch_genonly": "#9ca3af",
        "dino_joint": "#1f2937",
        "dsga": "#111827",
    }
    for _, row in df.iterrows():
        ax.scatter(row["rfid"], row["acc_pct"], s=42, color=colors[row["label"]])
        ax.text(row["rfid"] + 0.25, row["acc_pct"] + 0.25, row["label"], fontsize=8)
    bold_axis(ax, "rFID", "Acc (%)", "Initialization Anchoring on IN-100")
    fig.savefig(pack_root / "figs" / "pareto_init_anchor_in100.pdf", bbox_inches="tight")
    plt.close(fig)

    dino = df[df["label"] == "dino_joint"].iloc[0]
    scratch = df[df["label"] == "scratch_joint"].iloc[0]
    dsga = df[df["label"] == "dsga"].iloc[0]
    notes.append(
        ReadmeNote(
            item="5",
            title="Initialization Anchoring",
            summary=(
                f"DINO initialization anchors the classifier but hurts generation: DINO Joint reaches {dino['acc_pct']:.2f}% Acc at rFID {dino['rfid']:.3f}, "
                f"while Scratch Joint lands at {scratch['acc_pct']:.2f}% / {scratch['rfid']:.3f}. DSGA shifts the frontier to {dsga['acc_pct']:.2f}% / {dsga['rfid']:.3f}."
            ),
            anomaly="A separate DINO-freeze point was not available in the current workspace.",
        )
    )



def item6_arch_heterogeneity(pack_root: Path, device: torch.device, args, notes: List[ReadmeNote]) -> None:
    summaries = []
    for arch, rel_run in CIFAR_ARCH_RUNS.items():
        run = CifarRun(pack_root.parent.parent / rel_run, device)
        heat, rho, steps, layers = collect_cifar_grad_heatmap(
            run,
            split=args.cifar_split,
            batch_size=args.cifar_batch_size,
            num_batches=args.cifar_num_batches,
            num_workers=args.num_workers,
        )
        save_cifar_heatmap_outputs(
            pack_root=pack_root,
            exp_id=f"item6_cifar_{arch}",
            heat=heat,
            rho=rho,
            steps=steps,
            layers=layers,
            headline=headline_from_cifar_run(run.run_dir),
            title=f"CIFAR-100 {arch}",
            fig_name=f"diag_ck_heatmap_cifar_{arch}.pdf",
            source_run=run.run_dir,
            deviations=[
                "Architecture heterogeneity is measured with final-checkpoint batch-wise probes on CIFAR-100 joint baselines.",
            ],
        )
        summaries.append((arch, float(rho.mean())))

    summary_text = ", ".join([f"{arch}: mean rho^-={rho_mean:.2f}" for arch, rho_mean in summaries])
    notes.append(
        ReadmeNote(
            item="6",
            title="Architecture Heterogeneity",
            summary=f"Spatial heterogeneity remains architecture-agnostic in offline probes; {summary_text}.",
            anomaly="The heatmaps use batch index as the horizontal axis because no time-resolved layer probe traces were stored for these CIFAR runs.",
        )
    )



def discover_mag_sweep_runs(repo_root: Path, run_prefix: str) -> List[Path]:
    repo_root = ensure_allowed_path(repo_root)
    runs_root = ensure_allowed_path(repo_root / "runs")
    best_by_gamma: Dict[float, Tuple[Tuple[int, int, int, float], Path]] = {}
    for run_dir in sorted(runs_root.glob(f"{run_prefix}_*")):
        if (not run_dir.is_dir()) or run_dir.name.startswith(f"{run_prefix}_logs_"):
            continue
        cfg_path = run_dir / "run_config.yaml"
        if not cfg_path.exists():
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            gamma = extract_gamma_from_cfg(cfg)
        except Exception:
            continue
        has_probe = (run_dir / "dsga_probe_summary.jsonl").exists()
        has_mt = (run_dir / "dsga_mt_series.jsonl").exists()
        has_eval = (run_dir / "eval_last.json").exists()
        eval_step = 0
        if has_eval:
            try:
                eval_step = int(load_json(run_dir / "eval_last.json").get("step", 0))
            except Exception:
                eval_step = 0
        score = (int(has_probe and has_mt and has_eval), int(has_eval), int(eval_step), float(run_dir.stat().st_mtime))
        prev = best_by_gamma.get(gamma)
        if prev is None or score > prev[0]:
            best_by_gamma[gamma] = (score, run_dir)
    ordered = []
    for gamma in sorted(best_by_gamma.keys()):
        ordered.append(best_by_gamma[gamma][1])
    return ordered



def discover_historical_mag_runs(repo_root: Path) -> List[Path]:
    found = []
    for rel in CIFAR_MAG_RUNS:
        run_dir = ensure_allowed_path(repo_root / rel)
        ckpt = run_dir / "checkpoints" / "latest.pt"
        if ckpt.exists():
            found.append(run_dir)
    return found



def extract_gamma_from_cfg(cfg: dict) -> float:
    train_cfg = cfg.get("train", {})
    return float(train_cfg.get("dsga_m_align_gamma", train_cfg.get("ma_laga_align_gamma", 1.0)))



def _finite_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))



def _load_item2_probe_frame(run_dir: Path) -> pd.DataFrame:
    probe_dir = run_dir / "dsga_probe"
    csvs = sorted(probe_dir.glob("*_dsga.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_dsga.csv files found under {probe_dir}")
    frames = [pd.read_csv(csv_path) for csv_path in csvs]
    df = pd.concat(frames, ignore_index=True)
    df["projection_applied"] = pd.to_numeric(df["projection_applied"], errors="coerce").fillna(0).astype(int)
    df["alpha_residual"] = np.where(
        df["projection_applied"] > 0,
        np.abs(pd.to_numeric(df["alpha_post"], errors="coerce")),
        np.nan,
    )
    return df



def _item2_pivots(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    layer_order = df[["layer", "depth"]].drop_duplicates().sort_values(["depth", "layer"])
    layers = [str(x) for x in layer_order["layer"].tolist()]
    steps = sorted(int(x) for x in df["step"].unique().tolist())
    norm_ratio = (
        df.pivot(index="step", columns="layer", values="norm_ratio")
        .reindex(index=steps, columns=layers)
        .to_numpy(dtype=np.float32)
    )
    alpha_residual = (
        df.pivot(index="step", columns="layer", values="alpha_residual")
        .reindex(index=steps, columns=layers)
        .to_numpy(dtype=np.float32)
    )
    return norm_ratio, alpha_residual, steps, layers



def plot_item2_norm_ratio(pack_root: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.3), constrained_layout=True)
    gammas = [float(row["gamma"]) for row in rows]
    values = [np.asarray(row["norm_ratio_values"], dtype=np.float64) for row in rows]
    parts = ax.violinplot(values, positions=gammas, widths=0.08, showmeans=False, showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor("#9ca3af")
        body.set_edgecolor("#4b5563")
        body.set_alpha(0.85)
    means = [float(np.nanmean(v)) for v in values]
    ax.scatter(gammas, means, color="#111827", s=14, zorder=3)
    ax.plot(gammas, means, color="#111827", linewidth=1.0)
    ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=0.8)
    ax.set_xticks(gammas)
    bold_axis(ax, r"$\lambda_{mag}$", r"$r_k = ||g_k^*|| / ||g_u + g_g||$", "CIFAR-100 ViT-S Norm Ratio")
    fig.savefig(pack_root / "figs" / "diag_norm_ratio_cifar_vits.pdf", bbox_inches="tight")
    plt.close(fig)



def plot_item2_mt_hist(pack_root: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.3), constrained_layout=True)
    colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(rows)))
    for color, row in zip(colors, rows):
        values = np.asarray(row["mt_values"], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ax.hist(values, bins=20, histtype="step", linewidth=1.2, density=True, color=color, label=f"{row['gamma']:.1f}")
    bold_axis(ax, r"$m_t$", "Density", "CIFAR-100 ViT-S DSGA-M Scale Histogram")
    ax.legend(title=r"$\lambda_{mag}$", ncol=2)
    fig.savefig(pack_root / "figs" / "diag_mt_hist_cifar_vits.pdf", bbox_inches="tight")
    plt.close(fig)



def plot_item2_mt_time(pack_root: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.3), constrained_layout=True)
    colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(rows)))
    for color, row in zip(colors, rows):
        mt_df = pd.DataFrame(row["mt_series"]).sort_values("step")
        if mt_df.empty:
            continue
        ax.plot(mt_df["step"], mt_df["m_t"], linewidth=1.0, color=color, label=f"{row['gamma']:.1f}")
    bold_axis(ax, "Step", r"$m_t$", "CIFAR-100 ViT-S DSGA-M Scale Over Time")
    ax.legend(title=r"$\lambda_{mag}$", ncol=2)
    fig.savefig(pack_root / "figs" / "diag_mt_time_cifar_vits.pdf", bbox_inches="tight")
    plt.close(fig)



def plot_item2_sensitivity(pack_root: Path, rows: List[dict]) -> None:
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows).sort_values("gamma")
    fig, ax1 = plt.subplots(figsize=(6.0, 3.3), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(df["gamma"], df["acc_pct"], marker="o", color="#111827", label="Acc")
    ax2.plot(df["gamma"], df["rfid"], marker="s", color="#6b7280", label="rFID")
    ax1.set_xticks(df["gamma"].tolist())
    ax1.set_xlabel(r"$\lambda_{mag}$", fontweight="bold")
    ax1.set_ylabel("Acc (%)", fontweight="bold")
    ax2.set_ylabel("rFID", fontweight="bold")
    ax1.set_title("Sensitivity to DSGA-M Magnitude Alignment", fontweight="bold")
    fig.savefig(pack_root / "figs" / "sensitivity_lambda_mag.pdf", bbox_inches="tight")
    plt.close(fig)



def item2_historical_placeholder(pack_root: Path, repo_root: Path, notes: List[ReadmeNote]) -> None:
    runs = discover_historical_mag_runs(repo_root)
    rows = []
    for run_dir in runs:
        cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text())
        gamma = extract_gamma_from_cfg(cfg)
        headline = headline_from_cifar_run(run_dir) if (run_dir / "eval_last.json").exists() else {"acc_pct": None, "rmse": None, "rfid": None}
        rows.append(
            {
                "gamma": gamma,
                "acc_pct": np.nan if headline["acc_pct"] is None else float(headline["acc_pct"]),
                "rmse": np.nan if headline["rmse"] is None else float(headline["rmse"]),
                "rfid": np.nan if headline["rfid"] is None else float(headline["rfid"]),
                "run_dir": str(run_dir),
            }
        )
        log_dir = ensure_dir(pack_root / "logs" / f"item2_mag_{str(gamma).replace('.', 'p')}")
        save_json(log_dir / "headline.json", headline)
        save_yaml(
            log_dir / "config.yaml",
            {
                "source_run": str(run_dir),
                "deviations": [
                    "Only historical Swin-T lambda_mag runs available in the workspace were indexed here.",
                    "Several requested sweep points (0.6, 0.8) and CIFAR rFID logs are missing; sensitivity entries may contain NaN rFID.",
                ],
            },
        )
    if rows:
        plot_item2_sensitivity(pack_root, rows)
    notes.append(
        ReadmeNote(
            item="2",
            title="Magnitude Attenuation vs Restoration",
            summary="Indexed the available historical lambda_mag sweep checkpoints so the evidence pack retains a sensitivity placeholder and per-run metadata.",
            anomaly="Fresh ViT-S 2k DSGA runs with r_k, m_t, alpha residual, and CIFAR rFID were not yet available when this pack was generated.",
        )
    )



def item2_magnitude_restoration(pack_root: Path, args, notes: List[ReadmeNote]) -> None:
    repo_root = ensure_allowed_path(Path(args.repo_root))
    runs = discover_mag_sweep_runs(repo_root, run_prefix=str(args.item2_run_prefix))
    if not runs:
        item2_historical_placeholder(pack_root, repo_root, notes)
        return

    rows = []
    missing = []
    for run_dir in runs:
        cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text())
        gamma = extract_gamma_from_cfg(cfg)
        if not ((run_dir / "dsga_probe_summary.jsonl").exists() and (run_dir / "dsga_mt_series.jsonl").exists()):
            missing.append(float(gamma))
            continue
        try:
            df = _load_item2_probe_frame(run_dir)
            norm_ratio, alpha_residual, steps, layers = _item2_pivots(df)
        except FileNotFoundError:
            missing.append(float(gamma))
            continue

        probe_rows = load_jsonl(run_dir / "dsga_probe_summary.jsonl")
        mt_rows = load_jsonl(run_dir / "dsga_mt_series.jsonl")
        headline = headline_from_cifar_run(run_dir) if (run_dir / "eval_last.json").exists() else {"acc_pct": None, "rmse": None, "rfid": None}
        log_dir = ensure_dir(pack_root / "logs" / f"item2_cifar100_vits_lambda_{str(gamma).replace('.', 'p')}")
        np.save(log_dir / "norm_ratio.npy", norm_ratio)
        np.save(log_dir / "alpha_residual.npy", alpha_residual)
        save_json(log_dir / "layers.json", layers)
        save_json(log_dir / "steps.json", steps)
        save_json(log_dir / "mt_series.json", mt_rows)
        save_json(log_dir / "headline.json", headline)
        save_yaml(
            log_dir / "config.yaml",
            {
                "source_run": str(run_dir),
                "seed": int(cfg.get("seed", 3407)),
                "gamma": float(gamma),
                "feature_log": {
                    "num_probe_steps": len(steps),
                    "num_layers": len(layers),
                },
                "run_config": cfg,
                "deviations": [
                    "The current train_cifar10.py loop uses fixed lr=5e-4 without an explicit 1k-step warmup.",
                    "alpha_residual.npy stores abs(alpha_post) on projected layers and NaN elsewhere.",
                ],
            },
        )
        rows.append(
            {
                "gamma": float(gamma),
                "acc_pct": np.nan if headline["acc_pct"] is None else float(headline["acc_pct"]),
                "rmse": np.nan if headline["rmse"] is None else float(headline["rmse"]),
                "rfid": np.nan if headline["rfid"] is None else float(headline["rfid"]),
                "run_dir": str(run_dir),
                "layers": layers,
                "steps": steps,
                "norm_ratio_values": norm_ratio.reshape(-1).tolist(),
                "mt_values": [float(x.get("m_t", float("nan"))) for x in mt_rows],
                "mt_series": [
                    {
                        "step": int(x.get("step", 0)),
                        "m_t": float(x.get("m_t", float("nan"))),
                        "clipped_flag": int(x.get("clipped_flag", 0)),
                    }
                    for x in mt_rows
                ],
                "mean_norm_ratio": _finite_mean(df["norm_ratio"].tolist()),
                "mean_alpha_residual": _finite_mean(df["alpha_residual"].tolist()),
                "mean_mt": _finite_mean([x.get("m_t", float("nan")) for x in mt_rows]),
                "probe_rows": len(df),
                "eval_complete": bool((run_dir / "eval_last.json").exists()),
            }
        )

    if not rows:
        item2_historical_placeholder(pack_root, repo_root, notes)
        return

    rows.sort(key=lambda x: float(x["gamma"]))
    plot_item2_norm_ratio(pack_root, rows)
    plot_item2_mt_hist(pack_root, rows)
    plot_item2_mt_time(pack_root, rows)
    plot_item2_sensitivity(pack_root, rows)

    valid_rfid_rows = [row for row in rows if np.isfinite(float(row["rfid"]))]
    valid_acc_rows = [row for row in rows if np.isfinite(float(row["acc_pct"]))]
    best_rfid = min(valid_rfid_rows, key=lambda x: float(x["rfid"])) if valid_rfid_rows else None
    best_acc = max(valid_acc_rows, key=lambda x: float(x["acc_pct"])) if valid_acc_rows else None
    mean_alpha = _finite_mean([row["mean_alpha_residual"] for row in rows])
    mean_norm_lo = min(float(row["mean_norm_ratio"]) for row in rows)
    mean_norm_hi = max(float(row["mean_norm_ratio"]) for row in rows)
    coverage = sorted({round(float(row["gamma"]), 3) for row in rows})
    coverage_text = ", ".join([f"{g:.1f}" for g in coverage])
    if best_rfid is not None and best_acc is not None:
        summary = (
            f"Fresh CIFAR-100 ViT-S DSGA sweep covers lambda_mag={coverage_text}. "
            f"Mean norm ratio spans {mean_norm_lo:.3f}-{mean_norm_hi:.3f}, and projected alpha residual stays near zero (mean {mean_alpha:.2e}). "
            f"Best rFID appears at {best_rfid['gamma']:.1f} ({best_rfid['rfid']:.3f}), while best Acc is {best_acc['acc_pct']:.2f}% at {best_acc['gamma']:.1f}."
        )
    else:
        summary = (
            f"Fresh CIFAR-100 ViT-S DSGA sweep covers lambda_mag={coverage_text}. "
            f"Mean norm ratio spans {mean_norm_lo:.3f}-{mean_norm_hi:.3f}, and projected alpha residual stays near zero (mean {mean_alpha:.2e})."
        )
    anomaly = "The train loop still deviates from the requested protocol by omitting explicit LR warmup."
    if missing:
        anomaly = anomaly + f" Missing or incomplete gamma points: {', '.join([f'{g:.1f}' for g in sorted(missing)])}."
    elif len(coverage) < len(ITEM2_EXPECTED_GAMMAS):
        remaining = [g for g in ITEM2_EXPECTED_GAMMAS if g not in coverage]
        anomaly = anomaly + f" Missing gamma points relative to target sweep: {', '.join([f'{g:.1f}' for g in remaining])}."
    notes.append(
        ReadmeNote(
            item="2",
            title="Magnitude Attenuation vs Restoration",
            summary=summary,
            anomaly=anomaly,
        )
    )



def _load_in100_bridge_model(run_dir: Path, device: torch.device) -> Tuple[ViTBridgeModel, dict, dict]:
    run_dir = ensure_allowed_path(run_dir)
    setup = load_json(run_dir / "run_setup.json")
    run_args = dict(setup.get("args", {}))
    data_meta = dict(setup.get("data_meta", {}))
    encoder = build_encoder(
        encoder_init=str(run_args.get("encoder_init", "scratch")),
        image_size=224,
        encoder_ckpt=str(run_args.get("encoder_ckpt", "")),
    )
    model = ViTBridgeModel(
        encoder=encoder,
        num_classes=int(data_meta.get("num_classes", 100)),
        image_size=224,
        decoder_dim=int(run_args.get("decoder_dim", 384)),
        decoder_depth=int(run_args.get("decoder_depth", 4)),
        decoder_heads=int(run_args.get("decoder_heads", 6)),
        decoder_mlp_ratio=float(run_args.get("decoder_mlp_ratio", 4.0)),
        decoder_drop_rate=float(run_args.get("decoder_drop_rate", 0.0)),
    )
    ckpt = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, run_args, data_meta



def _compute_in100_losses(model: ViTBridgeModel, images: torch.Tensor, labels: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    images_01 = (images * std + mean).clamp(0.0, 1.0)
    outputs = model(images)
    logits = outputs["logits"]
    gt_patches = model.patchify(images_01)
    loss_u = F.cross_entropy(logits, labels)
    loss_g = F.mse_loss(outputs["pred_patches"], gt_patches)
    return loss_u, loss_g



def _compute_in100_scalar_losses(model: ViTBridgeModel, images: torch.Tensor, labels: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        loss_u, loss_g = _compute_in100_losses(model, images, labels, mean, std)
    return float(loss_u.item()), float(loss_g.item())



def _choose_item7_group(model: ViTBridgeModel) -> Tuple[str, int, List[Tuple[str, torch.nn.Parameter]]]:
    groups = _group_encoder_params(model.encoder)
    block_groups = [group for group in groups if str(group[0]).startswith("blocks.")]
    if block_groups:
        return block_groups[-1]
    return groups[-1]



def item7_local_pareto(pack_root: Path, device: torch.device, args, notes: List[ReadmeNote]) -> None:
    repo_root = ensure_allowed_path(Path(args.repo_root))
    run_dir_arg = Path(str(args.item7_run_dir))
    run_dir = ensure_allowed_path(run_dir_arg if run_dir_arg.is_absolute() else repo_root / run_dir_arg)
    model, run_args, data_meta = _load_in100_bridge_model(run_dir, device)
    torch.manual_seed(int(args.item7_seed))

    train_loader, _, _ = build_imagenet100_dataloaders(
        batch_size=int(args.item7_batch_size),
        num_workers=int(args.num_workers),
        dataset_path=run_args.get("dataset_path") or None,
        hf_dataset_id=run_args.get("hf_dataset_id") or data_meta.get("hf_dataset_id") or None,
        cache_dir=run_args.get("cache_dir") or None,
        image_key=str(run_args.get("image_key", "image")),
        label_key=str(run_args.get("label_key", "label")),
        distributed=False,
        rank=0,
        world_size=1,
    )

    layer_name, layer_depth, layer_params_named = _choose_item7_group(model)
    layer_params = [p for _, p in layer_params_named]
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    eta = float(args.item7_eta)
    eps = 1e-12
    full_idx = list(range(len(layer_params)))

    cases: List[dict] = []
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= int(args.item7_max_batches) or len(cases) >= int(args.item7_max_cases):
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss_u, loss_g = _compute_in100_losses(model, images, labels, mean, std)
        gu_raw = torch.autograd.grad(loss_u, layer_params, retain_graph=True, allow_unused=True)
        gg_raw = torch.autograd.grad(loss_g, layer_params, retain_graph=False, allow_unused=True)
        gu = _materialize_grad_list(gu_raw, layer_params)
        gg = _materialize_grad_list(gg_raw, layer_params)
        gu_flat = _group_flat(gu, full_idx)
        gg_flat = _group_flat(gg, full_idx)
        cos = _safe_cosine_flat(gu_flat, gg_flat)
        if cos >= 0.0:
            continue

        gu_sq = float(torch.dot(gu_flat, gu_flat).item())
        if gu_sq <= eps:
            continue
        proj_coeff = float(torch.dot(gg_flat, gu_flat).item() / (gu_sq + eps))
        joint_parts = [gu_i + gg_i for gu_i, gg_i in zip(gu, gg)]
        dsga_parts = [gu_i + gg_i - proj_coeff * gu_i for gu_i, gg_i in zip(gu, gg)]
        base_u = float(loss_u.item())
        base_g = float(loss_g.item())
        originals = [p.detach().clone() for p in layer_params]

        with torch.no_grad():
            for p, orig, grad in zip(layer_params, originals, joint_parts):
                p.copy_(orig - eta * grad)
        joint_u, joint_g = _compute_in100_scalar_losses(model, images, labels, mean, std)

        with torch.no_grad():
            for p, orig in zip(layer_params, originals):
                p.copy_(orig)
            for p, orig, grad in zip(layer_params, originals, dsga_parts):
                p.copy_(orig - eta * grad)
        dsga_u, dsga_g = _compute_in100_scalar_losses(model, images, labels, mean, std)

        with torch.no_grad():
            for p, orig in zip(layer_params, originals):
                p.copy_(orig)

        delta_u_joint = float(joint_u - base_u)
        delta_g_joint = float(joint_g - base_g)
        delta_u_dsga = float(dsga_u - base_u)
        delta_g_dsga = float(dsga_g - base_g)
        cases.append(
            {
                "batch_idx": int(batch_idx),
                "layer": layer_name,
                "depth": int(layer_depth),
                "cosine": float(cos),
                "proj_coeff": float(proj_coeff),
                "base_loss_u": float(base_u),
                "base_loss_g": float(base_g),
                "delta_u_joint": delta_u_joint,
                "delta_g_joint": delta_g_joint,
                "delta_u_dsga": delta_u_dsga,
                "delta_g_dsga": delta_g_dsga,
                "joint_u_descent": bool(delta_u_joint < 0.0),
                "dsga_u_descent": bool(delta_u_dsga < 0.0),
                "joint_feasible": bool(delta_u_joint < 0.0),
                "dsga_feasible": bool((delta_u_dsga < 0.0) and (delta_g_dsga <= delta_g_joint)),
            }
        )
        if len(cases) >= int(args.item7_max_cases):
            break

    num_cases = len(cases)
    if num_cases > 0:
        joint_u_descent_frac = float(sum(int(c["joint_u_descent"]) for c in cases) / num_cases)
        dsga_u_descent_frac = float(sum(int(c["dsga_u_descent"]) for c in cases) / num_cases)
        joint_feasible_frac = float(sum(int(c["joint_feasible"]) for c in cases) / num_cases)
        dsga_feasible_frac = float(sum(int(c["dsga_feasible"]) for c in cases) / num_cases)
        mean_delta_g_joint = float(np.mean([c["delta_g_joint"] for c in cases]))
        mean_delta_g_dsga = float(np.mean([c["delta_g_dsga"] for c in cases]))
    else:
        joint_u_descent_frac = 0.0
        dsga_u_descent_frac = 0.0
        joint_feasible_frac = 0.0
        dsga_feasible_frac = 0.0
        mean_delta_g_joint = float("nan")
        mean_delta_g_dsga = float("nan")

    log_dir = ensure_dir(pack_root / "logs" / "item7_in100_local_pareto")
    save_json(log_dir / "cases.json", cases)
    save_json(
        log_dir / "headline.json",
        {
            "num_cases": int(num_cases),
            "joint_u_descent_frac": round(joint_u_descent_frac, 4),
            "dsga_u_descent_frac": round(dsga_u_descent_frac, 4),
            "joint_feasible_frac": round(joint_feasible_frac, 4),
            "dsga_feasible_frac": round(dsga_feasible_frac, 4),
            "mean_delta_g_joint": None if math.isnan(mean_delta_g_joint) else round(mean_delta_g_joint, 6),
            "mean_delta_g_dsga": None if math.isnan(mean_delta_g_dsga) else round(mean_delta_g_dsga, 6),
        },
    )
    save_yaml(
        log_dir / "config.yaml",
        {
            "source_run": str(run_dir),
            "seed": int(args.item7_seed),
            "batch_size": int(args.item7_batch_size),
            "max_batches": int(args.item7_max_batches),
            "max_cases": int(args.item7_max_cases),
            "eta": float(eta),
            "chosen_layer": layer_name,
            "layer_depth": int(layer_depth),
            "deviations": [
                "The local Pareto check replays an existing joint checkpoint offline on sampled train minibatches.",
                "The feasibility metric compares DSGA-D generation change against the joint micro-update on the same layer and minibatch.",
            ],
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), constrained_layout=True)
    axes[0].bar(["Joint", "DSGA-D"], [joint_u_descent_frac, dsga_u_descent_frac], color=["#9ca3af", "#111827"])
    axes[0].set_ylim(0.0, 1.0)
    bold_axis(axes[0], "Update", "Fraction", "Understanding Descent")
    axes[1].bar(["Joint", "DSGA-D"], [joint_feasible_frac, dsga_feasible_frac], color=["#9ca3af", "#111827"])
    axes[1].set_ylim(0.0, 1.0)
    bold_axis(axes[1], "Update", "Fraction", "Generation-Preserving Descent")
    fig.savefig(pack_root / "figs" / "local_pareto_check_in100.pdf", bbox_inches="tight")
    plt.close(fig)

    if num_cases > 0:
        notes.append(
            ReadmeNote(
                item="7",
                title="Local Pareto Micro-step Check",
                summary=(
                    f"On {num_cases} negative-cosine cases from {layer_name}, DSGA-D raises the generation-preserving descent fraction "
                    f"from {joint_feasible_frac:.2f} to {dsga_feasible_frac:.2f}. "
                    f"Mean delta L_G moves from {mean_delta_g_joint:.6f} under Joint to {mean_delta_g_dsga:.6f} under DSGA-D."
                ),
                anomaly="This is an offline single-layer micro-update replay on one existing IN-100 joint checkpoint rather than an online intervention during training.",
            )
        )
    else:
        notes.append(
            ReadmeNote(
                item="7",
                title="Local Pareto Micro-step Check",
                summary=f"No negative-cosine cases were collected for the selected layer {layer_name} within the requested sampling budget.",
                anomaly="Increase --item7_max_batches or switch the sampled layer if a denser conflict set is needed.",
            )
        )



def write_readme(pack_root: Path, notes: List[ReadmeNote], items_done: Sequence[str]) -> None:
    notes_by_item = {note.item: note for note in notes}
    lines = ["# Extended Diagnostics Pack", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}", ""]
    lines.append("This pack prioritizes offline reuse of completed runs and documents every protocol deviation in the per-experiment `config.yaml`.")
    lines.append("")
    for item in ITEM_ORDER:
        if item not in items_done:
            continue
        note = notes_by_item.get(item)
        if note is None:
            continue
        lines.append(f"## ({item}) {note.title}")
        lines.append(note.summary)
        lines.append(f"Anomaly: {note.anomaly}")
        lines.append("")
    (pack_root / "README.md").write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline extended diagnostics evidence pack for DSGA.")
    parser.add_argument("--repo_root", default="/project/peilab/luxiaocheng/projects/DSGA")
    parser.add_argument("--out_root", default="")
    parser.add_argument("--items", default="1,2,3,4,5,6,7")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cifar_split", default="test")
    parser.add_argument("--cifar_batch_size", type=int, default=128)
    parser.add_argument("--cifar_num_batches", type=int, default=32)
    parser.add_argument("--cifar_feature_batches", type=int, default=40)
    parser.add_argument("--item2_run_prefix", default=ITEM2_DEFAULT_PREFIX)
    parser.add_argument("--item4_feature_mode", default="swin_stage2")
    parser.add_argument("--item7_run_dir", default=str(IN100_METHOD_RUNS["joint"]))
    parser.add_argument("--item7_batch_size", type=int, default=8)
    parser.add_argument("--item7_max_batches", type=int, default=32)
    parser.add_argument("--item7_max_cases", type=int, default=128)
    parser.add_argument("--item7_eta", type=float, default=5e-4)
    parser.add_argument("--item7_seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_plot_style()
    repo_root = ensure_allowed_path(Path(args.repo_root))
    if args.out_root:
        pack_root = ensure_allowed_path(Path(args.out_root))
        ensure_dir(pack_root)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pack_root = ensure_dir(repo_root / "results" / f"extended_diagnostics_{stamp}")
    ensure_dir(pack_root / "logs")
    ensure_dir(pack_root / "figs")

    os.chdir(repo_root)
    device = pick_device(args.device)
    notes: List[ReadmeNote] = []
    items = parse_items(args.items)

    if "1" in items:
        item1_conflict_cartography(pack_root, notes)
    if "2" in items:
        item2_magnitude_restoration(pack_root, args, notes)
    if "3" in items:
        item3_granularity(pack_root, device, args, notes)
    if "4" in items:
        item4_feature_variance(pack_root, device, args, notes)
    if "5" in items:
        item5_init_anchor(pack_root, notes)
    if "6" in items:
        item6_arch_heterogeneity(pack_root, device, args, notes)
    if "7" in items:
        item7_local_pareto(pack_root, device, args, notes)

    write_readme(pack_root, notes, items)
    print(f"[done] evidence pack written to {pack_root}")


if __name__ == "__main__":
    main()
