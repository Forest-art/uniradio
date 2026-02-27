import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_run_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "train_metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing metrics: {path}")
    df = pd.read_json(path, lines=True)
    if "step" not in df.columns:
        raise RuntimeError(f"invalid metrics format: {path}")
    return df.sort_values("step").reset_index(drop=True)


def _load_probe_summary(run_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(run_dir.glob("step_*.csv")):
        step = int(p.stem.split("_")[-1])
        df = pd.read_csv(p)
        rows.append(
            {
                "step": step,
                "mean_cos": float(df["cosine_similarity"].mean()),
                "mean_neg": float(df["neg_ratio"].mean()),
            }
        )
    if not rows:
        raise RuntimeError(f"no probe csv found in {run_dir}")
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def _stack_with_fill(frames: List[pd.DataFrame], value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 把多条曲线按 step 对齐，不同 run 缺失 step 用 NaN 补齐，再计算均值/std。
    merged = None
    for i, df in enumerate(frames):
        cur = df[["step", value_col]].rename(columns={value_col: f"v{i}"})
        merged = cur if merged is None else pd.merge(merged, cur, on="step", how="outer")
    merged = merged.sort_values("step").reset_index(drop=True)
    arr = merged.drop(columns=["step"]).to_numpy(dtype=np.float64)
    mean = np.nanmean(arr, axis=1)
    std = np.nanstd(arr, axis=1)
    steps = merged["step"].to_numpy(dtype=np.int64)
    return steps, mean, std


def main() -> None:
    parser = argparse.ArgumentParser("Plot publication-style figures for IN100 dynamics runs")
    parser.add_argument("--root", required=True, help="Root folder containing run dirs.")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--modes", default="scratch,dinov2")
    parser.add_argument("--run_name_template", default="in100_grad_dynamics_{mode}_s{seed}")
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "analysis_pub")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    color_map = {"scratch": "#1f77b4", "dinov2": "#d62728"}

    mode_metrics: Dict[str, List[pd.DataFrame]] = {}
    mode_probe: Dict[str, List[pd.DataFrame]] = {}
    summary_rows = []

    for mode in modes:
        mode_metrics[mode] = []
        mode_probe[mode] = []
        for seed in seeds:
            run_name = args.run_name_template.format(mode=mode, seed=seed)
            run_dir = root / run_name
            if not run_dir.exists():
                raise FileNotFoundError(f"run not found: {run_dir}")
            mdf = _load_run_metrics(run_dir)
            pdf = _load_probe_summary(run_dir)
            mode_metrics[mode].append(mdf)
            mode_probe[mode].append(pdf)

            summary_rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "run_name": run_name,
                    "final_loss_u": float(mdf["loss_u"].iloc[-1]),
                    "final_loss_g": float(mdf["loss_g"].iloc[-1]),
                    "final_acc": float(mdf["acc"].iloc[-1]),
                    "final_rmse": float(mdf["rmse"].iloc[-1]),
                    "probe_mean_cos": float(pdf["mean_cos"].mean()),
                    "probe_mean_neg": float(pdf["mean_neg"].mean()),
                }
            )

    pd.DataFrame(summary_rows).to_csv(out_dir / "per_run_summary.csv", index=False)

    # Figure 1: 理解/生成 loss 曲线（均值±std）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for mode in modes:
        c = color_map.get(mode, None)
        steps_u, mean_u, std_u = _stack_with_fill(mode_metrics[mode], "loss_u")
        steps_g, mean_g, std_g = _stack_with_fill(mode_metrics[mode], "loss_g")

        axes[0].plot(steps_u, mean_u, label=mode, color=c)
        axes[0].fill_between(steps_u, mean_u - std_u, mean_u + std_u, alpha=0.18, color=c)
        axes[1].plot(steps_g, mean_g, label=mode, color=c)
        axes[1].fill_between(steps_g, mean_g - std_g, mean_g + std_g, alpha=0.18, color=c)

    axes[0].set_title("Understanding Loss (Cross Entropy)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss_U")
    axes[0].legend()
    axes[1].set_title("Generation Loss (Patch MSE)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss_G")
    axes[1].legend()
    fig.savefig(out_dir / "loss_curves_understanding_generation.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: 冲突动力学（mean cosine / mean neg ratio）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for mode in modes:
        c = color_map.get(mode, None)
        s1, m1, d1 = _stack_with_fill(mode_probe[mode], "mean_cos")
        s2, m2, d2 = _stack_with_fill(mode_probe[mode], "mean_neg")
        axes[0].plot(s1, m1, label=mode, color=c)
        axes[0].fill_between(s1, m1 - d1, m1 + d1, alpha=0.18, color=c)
        axes[1].plot(s2, m2, label=mode, color=c)
        axes[1].fill_between(s2, m2 - d2, m2 + d2, alpha=0.18, color=c)

    axes[0].axhline(0.0, linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("Conflict Dynamics: Mean Cosine(gu, gg)")
    axes[0].set_xlabel("Probe Step")
    axes[0].set_ylabel("Cosine")
    axes[0].legend()
    axes[1].axhline(0.5, linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("Conflict Dynamics: Mean Negative Ratio")
    axes[1].set_xlabel("Probe Step")
    axes[1].set_ylabel("Neg Ratio")
    axes[1].legend()
    fig.savefig(out_dir / "conflict_dynamics_mean_cos_neg_ratio.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Block-wise 平均冲突柱状对比
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.2), constrained_layout=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        all_blocks = []
        for seed in seeds:
            run_name = args.run_name_template.format(mode=mode, seed=seed)
            run_dir = root / run_name
            for p in sorted(run_dir.glob("step_*.csv")):
                df = pd.read_csv(p)
                b = df[df["layer"].str.startswith("blocks.")].copy()
                b["block_idx"] = b["layer"].str.extract(r"blocks\.(\d+)").astype(int)
                b["seed"] = seed
                all_blocks.append(b[["block_idx", "cosine_similarity", "seed"]])
        bdf = pd.concat(all_blocks, ignore_index=True)
        grp = bdf.groupby("block_idx", as_index=False)["cosine_similarity"].mean().sort_values("block_idx")
        ax.bar(grp["block_idx"], grp["cosine_similarity"], color=color_map.get(mode, "#444444"), alpha=0.85)
        ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"Block-wise Mean Cosine ({mode})")
        ax.set_xlabel("Block Index")
        ax.set_ylabel("Cosine")
    fig.savefig(out_dir / "blockwise_mean_cosine_compare.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[done] publication figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
