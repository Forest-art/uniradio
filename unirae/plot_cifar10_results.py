import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt


GROUP_ORDER = [
    "baseline",
    "text-only",
    "recon-only",
    "joint-naive",
    "joint-conflict",
    "joint-cagrad",
    "joint-mgda",
]
GROUP_COLOR = {
    "baseline": "#6e6e6e",
    "text-only": "#1f77b4",
    "recon-only": "#ff7f0e",
    "joint-naive": "#2ca02c",
    "joint-conflict": "#d62728",
    "joint-cagrad": "#9467bd",
    "joint-mgda": "#8c564b",
}


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def infer_group(exp_name: str, strategy: str, lambda_txt: float, lambda_rec: float) -> str:
    n = (exp_name or "").lower()
    s = (strategy or "").lower()
    lt = lambda_txt if lambda_txt is not None else 0.0
    lr = lambda_rec if lambda_rec is not None else 0.0

    if "baseline" in n:
        return "baseline"
    if lt > 0 and lr == 0:
        return "text-only"
    if lr > 0 and lt == 0:
        return "recon-only"
    if "cagrad" in n or s == "cagrad":
        return "joint-cagrad"
    if "mgda" in n or s in {"mgda", "mgda_ub"}:
        return "joint-mgda"
    if "pcgrad" in n or "conflict" in n or s in {"conflict", "pcgrad"}:
        return "joint-conflict"
    if "naive" in n or s == "naive":
        return "joint-naive"
    if lt > 0 and lr > 0:
        return "joint-naive"
    return s or "unknown"


def load_rows(csv_path: str, exp_prefix: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("dataset", "") != "cifar10":
                continue
            if exp_prefix and not r.get("exp_name", "").startswith(exp_prefix):
                continue
            rows.append(
                {
                    "exp_name": r.get("exp_name", ""),
                    "group": "",
                    "strategy": r.get("strategy", ""),
                    "seed": r.get("seed", ""),
                    "lambda_txt": _to_float(r.get("lambda_txt", "")),
                    "lambda_rec": _to_float(r.get("lambda_rec", "")),
                    "acc_txt": _to_float(r.get("acc_txt", "")),
                    "recon_mse": _to_float(r.get("recon_mse", "")),
                    "cos_mean": _to_float(r.get("cos_mean", "")),
                    "cos_neg_ratio": _to_float(r.get("cos_neg_ratio", "")),
                }
            )

    for r in rows:
        r["group"] = infer_group(
            exp_name=r.get("exp_name", ""),
            strategy=r.get("strategy", ""),
            lambda_txt=r.get("lambda_txt"),
            lambda_rec=r.get("lambda_rec"),
        )

    return rows


def group_stats(rows: List[Dict], key: str) -> Dict[str, Dict[str, float]]:
    out = {}
    present_groups = GROUP_ORDER + sorted({r["group"] for r in rows if r["group"] not in GROUP_ORDER})
    for g in present_groups:
        vals = [r[key] for r in rows if r["group"] == g and r[key] is not None]
        if not vals:
            continue
        out[g] = {
            "mean": mean(vals),
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def plot_tradeoff(rows: List[Dict], out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    present_groups = GROUP_ORDER + sorted({r["group"] for r in rows if r["group"] not in GROUP_ORDER})

    # A) Pareto scatter
    ax = axes[0]
    for g in present_groups:
        pts = [r for r in rows if r["group"] == g and r["acc_txt"] is not None and r["recon_mse"] is not None]
        if not pts:
            continue
        xs = [p["recon_mse"] for p in pts]
        ys = [p["acc_txt"] for p in pts]
        color = GROUP_COLOR.get(g, "#17becf")
        ax.scatter(xs, ys, s=60, alpha=0.85, c=color, label=f"{g} (n={len(pts)})")
        ax.scatter([mean(xs)], [mean(ys)], marker="X", s=180, c=color, edgecolor="black", linewidth=0.7)

    jn = [r for r in rows if r["group"] == "joint-naive" and r["acc_txt"] is not None and r["recon_mse"] is not None]
    jc = [r for r in rows if r["group"] == "joint-conflict" and r["acc_txt"] is not None and r["recon_mse"] is not None]
    if jn and jc:
        x1, y1 = mean([r["recon_mse"] for r in jn]), mean([r["acc_txt"] for r in jn])
        x2, y2 = mean([r["recon_mse"] for r in jc]), mean([r["acc_txt"] for r in jc])
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.7, color="black"),
        )

    ax.set_xlabel("Recon MSE (lower is better)")
    ax.set_ylabel("Understanding Acc (higher is better)")
    ax.set_title("Pareto View")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    # B) mean acc
    ax = axes[1]
    acc_stats = group_stats(rows, "acc_txt")
    groups = [g for g in present_groups if g in acc_stats]
    means = [acc_stats[g]["mean"] for g in groups]
    stds = [acc_stats[g]["std"] for g in groups]
    colors = [GROUP_COLOR.get(g, "#17becf") for g in groups]
    ax.bar(range(len(groups)), means, yerr=stds, color=colors, alpha=0.9, capsize=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel("Acc")
    ax.set_title("Understanding (mean ± std)")
    ax.grid(axis="y", alpha=0.25)

    # C) mean recon
    ax = axes[2]
    rec_stats = group_stats(rows, "recon_mse")
    groups = [g for g in present_groups if g in rec_stats]
    means = [rec_stats[g]["mean"] for g in groups]
    stds = [rec_stats[g]["std"] for g in groups]
    colors = [GROUP_COLOR.get(g, "#17becf") for g in groups]
    ax.bar(range(len(groups)), means, yerr=stds, color=colors, alpha=0.9, capsize=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel("Recon MSE")
    ax.set_title("Generation Proxy (mean ± std)")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("CIFAR-10 Understanding vs Generation Trade-off", fontsize=13)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cos_curve(cos_json_path: str, out_path: str, exp_prefix: str) -> None:
    obj = json.load(open(cos_json_path, "r", encoding="utf-8"))
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))

    plotted = 0
    for exp_name, payload in obj.items():
        if exp_prefix and not exp_name.startswith(exp_prefix):
            continue
        group = payload.get("group", "")
        if group not in {"joint-naive", "joint-conflict"}:
            continue
        curve = payload.get("curve", [])
        if not curve:
            continue
        xs = [int(p["step"]) for p in curve]
        ys = [float(p["cos"]) for p in curve]
        ax.plot(xs, ys, lw=1.6, alpha=0.9, label=f"{exp_name}")
        plotted += 1

    ax.axhline(0.0, color="black", lw=1.0, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("grad cosine")
    ax.set_title("Gradient Conflict Curve (joint runs)")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/cifar10_results.csv")
    parser.add_argument("--cos", default="results/cifar10_cos_curve.json")
    parser.add_argument("--out", default="results/cifar10_tradeoff.png")
    parser.add_argument("--out_cos", default="results/cifar10_grad_cos_curve.png")
    parser.add_argument("--exp_prefix", default="study_cifar10_")
    args = parser.parse_args()

    rows = load_rows(args.csv, args.exp_prefix)
    if not rows:
        raise RuntimeError(f"No rows found in {args.csv} for exp_prefix={args.exp_prefix}")

    plot_tradeoff(rows, args.out)
    plot_cos_curve(args.cos, args.out_cos, args.exp_prefix)

    print(f"[plot] tradeoff figure: {args.out}")
    print(f"[plot] grad-cos figure: {args.out_cos}")


if __name__ == "__main__":
    main()
