import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


def _configure_style(use_latex: bool = False) -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 500,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "text.usetex": bool(use_latex),
            "axes.titlesize": 14,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def _load_cifar_layerwise(cifar_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cifar_csv)
    req = {"layer", "depth", "cos_mean", "cos_std", "cos_neg_ratio"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {cifar_csv}: {sorted(missing)}")
    df = df.sort_values(["depth", "layer"]).reset_index(drop=True)
    return df


def _list_step_csv(run_dir: Path) -> List[Path]:
    files = []
    for p in run_dir.glob("step_*.csv"):
        m = re.search(r"step_(\d+)\.csv", p.name)
        if m:
            files.append((int(m.group(1)), p))
    files = sorted(files, key=lambda x: x[0])
    return [p for _, p in files]


def _parse_step_from_name(p: Path) -> int:
    m = re.search(r"step_(\d+)\.csv", p.name)
    if not m:
        raise ValueError(f"Invalid step csv filename: {p.name}")
    return int(m.group(1))


def _compute_deep_stats_from_run(
    run_dir: Path,
    deep_block_start: int = 8,
    deep_block_end: int = 11,
) -> pd.DataFrame:
    rows = []
    step_files = _list_step_csv(run_dir)
    if not step_files:
        raise FileNotFoundError(f"No step_*.csv found under {run_dir}")

    deep_names = [f"blocks.{i}" for i in range(deep_block_start, deep_block_end + 1)]
    for p in step_files:
        step = _parse_step_from_name(p)
        df = pd.read_csv(p)
        if "layer" not in df.columns or "neg_ratio" not in df.columns:
            raise ValueError(f"{p} missing required columns.")
        deep = df[df["layer"].isin(deep_names)]
        if len(deep) == 0:
            # 若没有分块层，退化为 depth>=8 的层
            if "depth" in df.columns:
                deep = df[df["depth"] >= deep_block_start]
        if len(deep) == 0:
            continue
        rows.append(
            {
                "step": step,
                "deep_neg_ratio": float(deep["neg_ratio"].mean()),
                "deep_cos_mean": float(deep["cosine_similarity"].mean()),
                "run": run_dir.name,
            }
        )
    if not rows:
        raise RuntimeError(f"No deep rows computed from {run_dir}")
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def _aggregate_multi_runs(frames: Sequence[pd.DataFrame], value_col: str) -> pd.DataFrame:
    merged = None
    for i, df in enumerate(frames):
        cur = df[["step", value_col]].rename(columns={value_col: f"v{i}"})
        merged = cur if merged is None else pd.merge(merged, cur, on="step", how="outer")
    merged = merged.sort_values("step").reset_index(drop=True)
    arr = merged.drop(columns=["step"]).to_numpy(dtype=np.float64)
    merged[f"{value_col}_mean"] = np.nanmean(arr, axis=1)
    merged[f"{value_col}_std"] = np.nanstd(arr, axis=1)
    return merged[["step", f"{value_col}_mean", f"{value_col}_std"]]


def plot_figure1_cifar_spatial(
    cifar_df: pd.DataFrame,
    out_png: Path,
    depth_split: int = 2,
) -> Dict[str, float]:
    x = np.arange(len(cifar_df))
    y = cifar_df["cos_mean"].to_numpy(dtype=np.float64)
    e = cifar_df["cos_std"].to_numpy(dtype=np.float64)
    labels = cifar_df["layer"].tolist()

    y_min = float(np.min(y - e) - 0.05)
    y_max = float(np.max(y + e) + 0.05)

    fig, ax = plt.subplots(figsize=(8.6, 4.9))

    ax.axhspan(0.0, y_max, color="#E8F5E9", alpha=0.55, zorder=0)
    ax.axhspan(y_min, 0.0, color="#FFEBEE", alpha=0.55, zorder=0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, zorder=2)

    ax.plot(
        x,
        y,
        color="#1f77b4",
        linewidth=2.3,
        marker="o",
        markersize=5.0,
        label="Mean cosine (real CIFAR run)",
        zorder=3,
    )
    ax.fill_between(x, y - e, y + e, color="#1f77b4", alpha=0.20, label=r"$\pm$1 std", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Network Depth / Layer Group")
    ax.set_ylabel(r"Cosine Similarity $\,\cos(g_u, g_g)$")
    ax.set_title("Figure 1: Layer-wise Spatial Heterogeneity (Real CIFAR)")

    ax.text(0.05, y_max - 0.03, "Synergistic (Shallow)", color="#2E7D32", fontsize=11, va="top")
    ax.text(len(x) - 2.4, y_min + 0.03, "Conflicting (Deep)", color="#C62828", fontsize=11, va="bottom")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    plt.close(fig)

    shallow = cifar_df[cifar_df["depth"] <= depth_split]
    deep = cifar_df[cifar_df["depth"] > depth_split]
    return {
        "shallow_cos_mean": float(shallow["cos_mean"].mean()) if len(shallow) else float("nan"),
        "deep_cos_mean": float(deep["cos_mean"].mean()) if len(deep) else float("nan"),
        "shallow_neg_ratio_mean": float(shallow["cos_neg_ratio"].mean()) if len(shallow) else float("nan"),
        "deep_neg_ratio_mean": float(deep["cos_neg_ratio"].mean()) if len(deep) else float("nan"),
    }


def plot_figure2_imagenet_temporal(
    run_dirs: Sequence[Path],
    out_png: Path,
    deep_block_start: int = 8,
    deep_block_end: int = 11,
) -> Dict[str, float]:
    frames = [
        _compute_deep_stats_from_run(
            run_dir=p,
            deep_block_start=deep_block_start,
            deep_block_end=deep_block_end,
        )
        for p in run_dirs
    ]

    agg = _aggregate_multi_runs(frames, value_col="deep_neg_ratio")
    steps = agg["step"].to_numpy(dtype=np.int64)
    mean = agg["deep_neg_ratio_mean"].to_numpy(dtype=np.float64)
    std = agg["deep_neg_ratio_std"].to_numpy(dtype=np.float64)

    plateau_mask = steps >= 2000
    plateau = float(np.mean(mean[plateau_mask])) if np.any(plateau_mask) else float(np.mean(mean))

    fig, ax = plt.subplots(figsize=(8.9, 5.0))
    ax.plot(steps, mean, color="#d62728", linewidth=2.4, label="Deep-layer negative-cosine ratio")
    ax.fill_between(steps, mean - std, mean + std, color="#d62728", alpha=0.18, label=r"$\pm$1 std")

    ax.axvspan(2000, int(steps.max()), color="#E8EAF6", alpha=0.24, zorder=0)
    ax.axhline(plateau, color="#2F4F4F", linestyle="--", linewidth=1.4)
    ax.annotate(
        "Structural Conflict Plateau",
        xy=(int(steps.max() * 0.62), plateau),
        xytext=(int(steps.max() * 0.70), plateau + 0.04),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#2F4F4F"),
        fontsize=10.5,
        color="#2F4F4F",
    )

    ax.set_xlim(0, int(steps.max()))
    ax.set_ylim(0.40, 0.62)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Proportion of Negative Cosine Similarity")
    ax.set_title("Figure 2: Temporal Persistent Conflict (Real ImageNet-100)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    plt.close(fig)

    return {
        "num_runs": int(len(run_dirs)),
        "plateau_mean_step_ge_2000": plateau,
        "global_mean": float(np.mean(mean)),
        "global_std_over_steps": float(np.std(mean)),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Plot real CIFAR + ImageNet conflict figures (no simulation).")
    parser.add_argument(
        "--cifar_csv",
        type=str,
        default="results/cifar100_real_gradheatmap_naive_normnone_s42_b60/layerwise_grad_summary.csv",
        help="Real CIFAR layerwise summary csv.",
    )
    parser.add_argument(
        "--imagenet_runs",
        type=str,
        default="results/long_runs/gradstrategy3_10k_eval1k_20260228_140137_scratch_naive_lu1.0_lg100.0_s42",
        help="Comma-separated run dirs, each containing step_*.csv",
    )
    parser.add_argument("--out_dir", type=str, default="results/paper_figs_real_conflict")
    parser.add_argument("--deep_block_start", type=int, default=8)
    parser.add_argument("--deep_block_end", type=int, default=11)
    parser.add_argument("--use_latex", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_style(use_latex=bool(args.use_latex))

    cifar_csv = Path(args.cifar_csv)
    cifar_df = _load_cifar_layerwise(cifar_csv)
    fig1_stats = plot_figure1_cifar_spatial(
        cifar_df=cifar_df,
        out_png=out_dir / "figure1_cifar_spatial_real.png",
        depth_split=2,
    )

    run_dirs = [Path(x.strip()) for x in str(args.imagenet_runs).split(",") if x.strip()]
    fig2_stats = plot_figure2_imagenet_temporal(
        run_dirs=run_dirs,
        out_png=out_dir / "figure2_imagenet_temporal_real.png",
        deep_block_start=int(args.deep_block_start),
        deep_block_end=int(args.deep_block_end),
    )

    report = {
        "cifar_csv": str(cifar_csv),
        "imagenet_runs": [str(p) for p in run_dirs],
        "figure1_stats": fig1_stats,
        "figure2_stats": fig2_stats,
    }
    with open(out_dir / "real_conflict_stats.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[done] out_dir: {out_dir}")
    print(f" - {out_dir / 'figure1_cifar_spatial_real.png'}")
    print(f" - {out_dir / 'figure2_imagenet_temporal_real.png'}")
    print(f" - {out_dir / 'real_conflict_stats.json'}")


if __name__ == "__main__":
    main()
