import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: str, rows: List[Dict], keys: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _parse_variant(exp_name: str, warmup_steps: int) -> str:
    token = f"warmup{warmup_steps}"
    return "warmup_then_joint" if token in exp_name else "joint"


def _parse_strategy(exp_name: str) -> str:
    n = exp_name.lower()
    if "_pcgrad_" in n:
        return "pcgrad"
    if "_cagrad_" in n:
        return "cagrad"
    return "naive"


def _read_rows(runs_glob: str, warmup_steps: int) -> List[Dict]:
    rows = []
    for rd in sorted(glob.glob(runs_glob)):
        if not os.path.isdir(rd):
            continue
        u = _load_json(os.path.join(rd, "understanding.json"))
        g = _load_json(os.path.join(rd, "generation.json"))
        t = _load_json(os.path.join(rd, "train_setup.json"))
        c = _load_json(os.path.join(rd, "cos_summary.json"))
        if not u or not g or not t or not c:
            continue
        exp = os.path.basename(rd)
        rows.append(
            {
                "exp_name": exp,
                "run_dir": rd,
                "strategy": str(t.get("strategy", _parse_strategy(exp))),
                "variant": _parse_variant(exp, warmup_steps=warmup_steps),
                "text_warmup_steps": int(t.get("text_warmup_steps", 0)),
                "acc": float(u.get("zero_shot_acc", u.get("acc_txt"))),
                "mse": float(g.get("mse", g.get("recon_mse"))),
                "cos_mean": float(c.get("cos_mean", 0.0)),
                "cos_neg_ratio": float(c.get("cos_neg_ratio", 0.0)),
                "train_steps": int(c.get("train_steps", 0)),
            }
        )
    return rows


def _pair_rows(rows: List[Dict]) -> List[Tuple[Dict, Dict]]:
    out: List[Tuple[Dict, Dict]] = []
    for strategy in ["naive", "pcgrad", "cagrad"]:
        base = next((r for r in rows if r["strategy"] == strategy and r["variant"] == "joint"), None)
        warm = next((r for r in rows if r["strategy"] == strategy and r["variant"] == "warmup_then_joint"), None)
        if base and warm:
            out.append((base, warm))
    return out


def _save_plot(rows: List[Dict], out_png: str) -> None:
    colors = {"naive": "#111111", "pcgrad": "#1f77b4", "cagrad": "#d62728"}
    markers = {"joint": "o", "warmup_then_joint": "s"}
    labels_done = set()

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.8))
    for r in rows:
        key = f"{r['strategy']}-{r['variant']}"
        label = key if key not in labels_done else None
        labels_done.add(key)
        ax.scatter(
            [r["mse"]],
            [r["acc"]],
            c=colors.get(r["strategy"], "#555555"),
            marker=markers.get(r["variant"], "o"),
            s=64,
            alpha=0.9,
            label=label,
        )
        ax.text(r["mse"] + 0.001, r["acc"] + 0.0002, r["strategy"], fontsize=8, alpha=0.85)

    for base, warm in _pair_rows(rows):
        ax.annotate(
            "",
            xy=(warm["mse"], warm["acc"]),
            xytext=(base["mse"], base["acc"]),
            arrowprops={"arrowstyle": "->", "lw": 1.6, "alpha": 0.8, "color": colors.get(base["strategy"], "#555555")},
        )

    ax.set_xlabel("Generation MSE (lower is better)")
    ax.set_ylabel("Understanding Acc (higher is better)")
    ax.set_title("Gradient Intervention: Joint vs Warmup-Then-Joint (CIFAR100)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", default="runs/cifar100_textwarmup_cmp_*")
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    rows = _read_rows(args.runs_glob, warmup_steps=args.warmup_steps)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = str(out_dir / "summary.csv")
    _write_csv(
        summary_csv,
        sorted(rows, key=lambda x: (x["strategy"], x["variant"])),
        ["exp_name", "strategy", "variant", "text_warmup_steps", "acc", "mse", "cos_mean", "cos_neg_ratio", "train_steps", "run_dir"],
    )

    deltas: List[Dict] = []
    for base, warm in _pair_rows(rows):
        deltas.append(
            {
                "strategy": base["strategy"],
                "joint_acc": base["acc"],
                "joint_mse": base["mse"],
                "warmup_acc": warm["acc"],
                "warmup_mse": warm["mse"],
                "delta_acc": warm["acc"] - base["acc"],
                "delta_mse": warm["mse"] - base["mse"],
                "win_both": (warm["acc"] > base["acc"]) and (warm["mse"] < base["mse"]),
                "joint_run_dir": base["run_dir"],
                "warmup_run_dir": warm["run_dir"],
            }
        )
    delta_csv = str(out_dir / "delta_warmup_minus_joint.csv")
    _write_csv(
        delta_csv,
        sorted(deltas, key=lambda x: x["strategy"]),
        ["strategy", "joint_acc", "joint_mse", "warmup_acc", "warmup_mse", "delta_acc", "delta_mse", "win_both", "joint_run_dir", "warmup_run_dir"],
    )

    fig_path = str(out_dir / "pareto_warmup_compare.png")
    _save_plot(rows, out_png=fig_path)

    print(f"[summarize_textwarmup_compare] runs={len(rows)}")
    print(f"[summarize_textwarmup_compare] summary={summary_csv}")
    print(f"[summarize_textwarmup_compare] delta={delta_csv}")
    print(f"[summarize_textwarmup_compare] fig={fig_path}")


if __name__ == "__main__":
    main()
