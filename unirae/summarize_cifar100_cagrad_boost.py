import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _pareto_front(points: List[Dict]) -> List[Dict]:
    out = []
    for p in points:
        dominated = False
        for q in points:
            if p is q:
                continue
            better_or_equal = (_float(q["acc"]) >= _float(p["acc"])) and (_float(q["mse"]) <= _float(p["mse"]))
            strict = (_float(q["acc"]) > _float(p["acc"])) or (_float(q["mse"]) < _float(p["mse"]))
            if better_or_equal and strict:
                dominated = True
                break
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda x: (_float(x["mse"]), -_float(x["acc"])))


def _write_csv(path: str, rows: List[Dict], keys: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", default="runs/cifar100_cagrad_pareto_boost*")
    parser.add_argument(
        "--naive_run",
        default="runs/cifar100_gradnorm_conflictdeep_cmp_20260223_20k_v2_joint_naive_normmean_conflictdeep_s42",
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    run_dirs = [p for p in sorted(glob.glob(args.runs_glob)) if os.path.isdir(p)]
    rows: List[Dict] = []
    for rd in run_dirs:
        u = _load_json(os.path.join(rd, "understanding.json"))
        g = _load_json(os.path.join(rd, "generation.json"))
        t = _load_json(os.path.join(rd, "train_setup.json"))
        c = _load_json(os.path.join(rd, "cos_summary.json"))
        if not u or not g:
            continue
        row = {
            "run_dir": rd,
            "exp_name": os.path.basename(rd),
            "strategy": t.get("strategy", c.get("strategy", "cagrad")),
            "acc": _float(u.get("zero_shot_acc", u.get("acc_txt"))),
            "mse": _float(g.get("mse", g.get("recon_mse"))),
            "beta": t.get("cagrad_beta", c.get("cagrad_beta")),
            "lambda_txt": t.get("lambda_txt", c.get("lambda_txt")),
            "lambda_rec": t.get("lambda_rec", c.get("lambda_rec")),
            "cagrad_conflict_only": t.get("cagrad_conflict_only", c.get("cagrad_conflict_only", False)),
            "cagrad_conflict_threshold": t.get("cagrad_conflict_threshold", c.get("cagrad_conflict_threshold", 0.0)),
            "cagrad_nonconflict_merge": t.get("cagrad_nonconflict_merge", c.get("cagrad_nonconflict_merge", "cagrad")),
            "grad_norm_mode": t.get("grad_norm_mode", c.get("grad_norm_mode")),
            "grad_norm_scope": t.get("grad_norm_scope", c.get("grad_norm_scope")),
            "grad_norm_layers": "|".join(t.get("grad_norm_layers", c.get("grad_norm_layers", [])))
            if isinstance(t.get("grad_norm_layers", c.get("grad_norm_layers", [])), list)
            else t.get("grad_norm_layers", c.get("grad_norm_layers", "")),
            "train_steps": c.get("train_steps", ""),
            "cos_mean": c.get("cos_mean", ""),
            "cos_neg_ratio": c.get("cos_neg_ratio", ""),
        }
        rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    naive_u = _load_json(os.path.join(args.naive_run, "understanding.json"))
    naive_g = _load_json(os.path.join(args.naive_run, "generation.json"))
    naive_acc = _float(naive_u.get("zero_shot_acc", naive_u.get("acc_txt")))
    naive_mse = _float(naive_g.get("mse", naive_g.get("recon_mse")))

    rows_sorted = sorted(rows, key=lambda x: (-_float(x["acc"]), _float(x["mse"])))
    summary_csv = str(out_dir / "summary.csv")
    _write_csv(
        summary_csv,
        rows_sorted,
        [
            "exp_name",
            "strategy",
            "acc",
            "mse",
            "beta",
            "lambda_txt",
            "lambda_rec",
            "cagrad_conflict_only",
            "cagrad_conflict_threshold",
            "cagrad_nonconflict_merge",
            "grad_norm_mode",
            "grad_norm_scope",
            "grad_norm_layers",
            "cos_mean",
            "cos_neg_ratio",
            "train_steps",
            "run_dir",
        ],
    )

    delta_rows = []
    for r in rows_sorted:
        delta_rows.append(
            {
                **r,
                "naive_acc": naive_acc,
                "naive_mse": naive_mse,
                "delta_acc": _float(r["acc"]) - naive_acc,
                "delta_mse": _float(r["mse"]) - naive_mse,
                "win_both": (_float(r["acc"]) > naive_acc) and (_float(r["mse"]) < naive_mse),
            }
        )
    delta_rows = sorted(delta_rows, key=lambda x: (not bool(x["win_both"]), -_float(x["delta_acc"]), _float(x["delta_mse"])))
    delta_csv = str(out_dir / "delta_vs_naive.csv")
    _write_csv(
        delta_csv,
        delta_rows,
        [
            "exp_name",
            "acc",
            "mse",
            "naive_acc",
            "naive_mse",
            "delta_acc",
            "delta_mse",
            "win_both",
            "beta",
            "lambda_txt",
            "cagrad_conflict_only",
            "cagrad_conflict_threshold",
            "cagrad_nonconflict_merge",
            "run_dir",
        ],
    )

    pareto_rows = _pareto_front(rows_sorted)
    pareto_csv = str(out_dir / "pareto_front.csv")
    _write_csv(
        pareto_csv,
        pareto_rows,
        ["exp_name", "acc", "mse", "beta", "lambda_txt", "cagrad_conflict_only", "cagrad_conflict_threshold", "cagrad_nonconflict_merge", "run_dir"],
    )

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.8))
    xs = [_float(r["mse"]) for r in rows_sorted]
    ys = [_float(r["acc"]) for r in rows_sorted]
    ax.scatter(xs, ys, s=40, alpha=0.85, c="#1f77b4", label="cagrad variants")

    if pareto_rows:
        px = [_float(r["mse"]) for r in pareto_rows]
        py = [_float(r["acc"]) for r in pareto_rows]
        ax.plot(px, py, c="#d62728", lw=1.6, marker="o", ms=4, label="pareto front")

    ax.scatter([naive_mse], [naive_acc], s=160, marker="*", c="#111111", label="naive reference")
    ax.axhline(naive_acc, color="#111111", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(naive_mse, color="#111111", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Generation MSE (lower is better)")
    ax.set_ylabel("Understanding Acc (higher is better)")
    ax.set_title("CIFAR100 CAGrad Pareto Tuning vs Naive")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig_path = str(out_dir / "pareto_vs_naive.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"[summarize_cifar100_cagrad_boost] runs={len(rows_sorted)}")
    print(f"[summarize_cifar100_cagrad_boost] summary={summary_csv}")
    print(f"[summarize_cifar100_cagrad_boost] delta={delta_csv}")
    print(f"[summarize_cifar100_cagrad_boost] pareto={pareto_csv}")
    print(f"[summarize_cifar100_cagrad_boost] fig={fig_path}")


if __name__ == "__main__":
    main()
