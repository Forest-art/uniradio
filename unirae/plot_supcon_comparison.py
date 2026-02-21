import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def infer_variant(exp_name: str) -> str:
    n = exp_name.lower()
    if "supcon0_cagrad0" in n:
        return "baseline"
    if "supcon1_cagrad0" in n:
        return "supcon-naive"
    if "supcon1_cagrad1" in n:
        return "supcon-cagrad"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/cifar10_supcon_full_results.csv")
    parser.add_argument("--out", default="results/cifar10_supcon_zoom_delta.png")
    parser.add_argument("--out_table", default="results/cifar10_supcon_delta_table.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise RuntimeError(f"Empty csv: {args.csv}")

    df["variant"] = df["exp_name"].astype(str).map(infer_variant)
    df = df[df["variant"].isin(["baseline", "supcon-naive", "supcon-cagrad"])].copy()
    if df.empty:
        raise RuntimeError("No supcon rows found in csv.")

    agg = (
        df.groupby(["variant", "lambda_sup"], as_index=False)
        .agg(
            acc_mean=("acc_txt", "mean"),
            acc_std=("acc_txt", "std"),
            mse_mean=("recon_mse", "mean"),
            mse_std=("recon_mse", "std"),
            n=("exp_name", "count"),
        )
        .sort_values(["variant", "lambda_sup"])
    )
    for c in ["acc_std", "mse_std"]:
        agg[c] = agg[c].fillna(0.0)

    base = agg[(agg["variant"] == "baseline") & (agg["lambda_sup"] == 0.0)]
    if base.empty:
        raise RuntimeError("Baseline row not found (variant=baseline, lambda_sup=0).")
    base_acc = float(base["acc_mean"].iloc[0])
    base_mse = float(base["mse_mean"].iloc[0])

    delta = agg.copy()
    delta["delta_acc_pp"] = (delta["acc_mean"] - base_acc) * 100.0
    delta["delta_mse_pct"] = (delta["mse_mean"] / base_mse - 1.0) * 100.0
    delta.to_csv(args.out_table, index=False)

    colors = {"baseline": "#2f2f2f", "supcon-naive": "#1f77b4", "supcon-cagrad": "#d62728"}
    markers = {"baseline": "*", "supcon-naive": "o", "supcon-cagrad": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # (A) Zoomed Pareto with error bars and lambda labels
    ax = axes[0]
    for v in ["supcon-naive", "supcon-cagrad"]:
        sub = agg[agg["variant"] == v].sort_values("lambda_sup")
        if sub.empty:
            continue
        ax.plot(sub["mse_mean"], sub["acc_mean"], color=colors[v], lw=1.6, alpha=0.9)
        ax.errorbar(
            sub["mse_mean"],
            sub["acc_mean"],
            xerr=sub["mse_std"],
            yerr=sub["acc_std"],
            fmt=markers[v],
            color=colors[v],
            capsize=3,
            label=v,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                f"λ={r['lambda_sup']:.1f}",
                (r["mse_mean"], r["acc_mean"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

    ax.scatter([base_mse], [base_acc], c=colors["baseline"], marker=markers["baseline"], s=180, label="baseline")
    ax.annotate("baseline", (base_mse, base_acc), textcoords="offset points", xytext=(7, -10), fontsize=8)
    ax.set_xlabel("Recon MSE (lower better)")
    ax.set_ylabel("Acc (higher better)")
    ax.set_title("SupCon Trade-off (Zoomed)")
    ax.grid(alpha=0.25)
    ax.legend()

    x_min = float(agg["mse_mean"].min()) - 0.01
    x_max = float(agg["mse_mean"].max()) + 0.01
    y_min = float(agg["acc_mean"].min()) - 0.01
    y_max = float(agg["acc_mean"].max()) + 0.01
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # (B) Delta-vs-baseline plane
    ax = axes[1]
    ax.axhline(0.0, color="black", lw=1.0, linestyle="--")
    ax.axvline(0.0, color="black", lw=1.0, linestyle="--")
    for v in ["supcon-naive", "supcon-cagrad"]:
        sub = delta[delta["variant"] == v].sort_values("lambda_sup")
        ax.plot(sub["delta_mse_pct"], sub["delta_acc_pp"], color=colors[v], lw=1.6, alpha=0.9)
        ax.scatter(
            sub["delta_mse_pct"],
            sub["delta_acc_pp"],
            marker=markers[v],
            color=colors[v],
            s=65,
            label=v,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                f"λ={r['lambda_sup']:.1f}",
                (r["delta_mse_pct"], r["delta_acc_pp"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

    ax.set_xlabel("ΔRecon MSE vs baseline (%)  (left is better)")
    ax.set_ylabel("ΔAcc vs baseline (percentage points)")
    ax.set_title("Relative Gain/Loss vs Baseline")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("CIFAR-10 SupCon Comparison", fontsize=13)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_supcon] figure: {args.out}")
    print(f"[plot_supcon] delta table: {args.out_table}")


if __name__ == "__main__":
    main()

