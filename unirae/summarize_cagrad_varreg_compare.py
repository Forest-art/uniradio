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


def _write_csv(path: str, rows: List[Dict], keys: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _tag(exp_name: str, strategy: str, lambda_var: float) -> str:
    if strategy == "naive":
        return "naive_ref"
    if abs(lambda_var) < 1e-12:
        return "cagrad_base"
    return "cagrad_var"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", default="runs/cifar100_cagrad_varreg_cmp_*")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    rows: List[Dict] = []
    for rd in sorted(glob.glob(args.runs_glob)):
        if not os.path.isdir(rd):
            continue
        ts = _load_json(os.path.join(rd, "train_setup.json"))
        cs = _load_json(os.path.join(rd, "cos_summary.json"))
        u = _load_json(os.path.join(rd, "understanding.json"))
        g = _load_json(os.path.join(rd, "generation.json"))
        if not ts or not cs or not u or not g:
            continue
        strategy = str(ts.get("strategy", cs.get("strategy", "naive"))).lower()
        lambda_var = float(ts.get("lambda_var", cs.get("lambda_var", 0.0)))
        rows.append(
            {
                "exp_name": os.path.basename(rd),
                "run_dir": rd,
                "tag": _tag(os.path.basename(rd), strategy=strategy, lambda_var=lambda_var),
                "strategy": strategy,
                "acc": float(u.get("zero_shot_acc", u.get("acc_txt"))),
                "mse": float(g.get("mse", g.get("recon_mse"))),
                "cagrad_beta": ts.get("cagrad_beta", cs.get("cagrad_beta", "")),
                "lambda_txt": ts.get("lambda_txt", cs.get("lambda_txt", "")),
                "lambda_rec": ts.get("lambda_rec", cs.get("lambda_rec", "")),
                "lambda_var": lambda_var,
                "var_gamma": ts.get("var_gamma", cs.get("var_gamma", "")),
                "var_eps": ts.get("var_eps", cs.get("var_eps", "")),
                "grad_norm_mode": ts.get("grad_norm_mode", cs.get("grad_norm_mode", "")),
                "grad_norm_scope": ts.get("grad_norm_scope", cs.get("grad_norm_scope", "")),
                "grad_norm_layers": "|".join(ts.get("grad_norm_layers", cs.get("grad_norm_layers", [])))
                if isinstance(ts.get("grad_norm_layers", cs.get("grad_norm_layers", [])), list)
                else ts.get("grad_norm_layers", cs.get("grad_norm_layers", "")),
                "cos_mean": cs.get("cos_mean", ""),
                "cos_neg_ratio": cs.get("cos_neg_ratio", ""),
                "train_steps": cs.get("train_steps", ""),
            }
        )

    rows = sorted(rows, key=lambda x: (x["tag"], float(x["lambda_var"]), x["exp_name"]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        str(out_dir / "summary.csv"),
        rows,
        [
            "exp_name",
            "tag",
            "strategy",
            "acc",
            "mse",
            "cagrad_beta",
            "lambda_txt",
            "lambda_rec",
            "lambda_var",
            "var_gamma",
            "var_eps",
            "grad_norm_mode",
            "grad_norm_scope",
            "grad_norm_layers",
            "cos_mean",
            "cos_neg_ratio",
            "train_steps",
            "run_dir",
        ],
    )

    naive = next((r for r in rows if r["tag"] == "naive_ref"), None)
    base = next((r for r in rows if r["tag"] == "cagrad_base"), None)
    vars_rows = [r for r in rows if r["tag"] == "cagrad_var"]

    delta_rows: List[Dict] = []
    for cand in vars_rows:
        if base is not None:
            delta_rows.append(
                {
                    "compare_to": "cagrad_base",
                    "base_acc": base["acc"],
                    "base_mse": base["mse"],
                    "cand_exp_name": cand["exp_name"],
                    "cand_lambda_var": cand["lambda_var"],
                    "cand_acc": cand["acc"],
                    "cand_mse": cand["mse"],
                    "delta_acc": cand["acc"] - base["acc"],
                    "delta_mse": cand["mse"] - base["mse"],
                    "win_both": (cand["acc"] > base["acc"]) and (cand["mse"] < base["mse"]),
                    "cand_run_dir": cand["run_dir"],
                }
            )
        if naive is not None:
            delta_rows.append(
                {
                    "compare_to": "naive_ref",
                    "base_acc": naive["acc"],
                    "base_mse": naive["mse"],
                    "cand_exp_name": cand["exp_name"],
                    "cand_lambda_var": cand["lambda_var"],
                    "cand_acc": cand["acc"],
                    "cand_mse": cand["mse"],
                    "delta_acc": cand["acc"] - naive["acc"],
                    "delta_mse": cand["mse"] - naive["mse"],
                    "win_both": (cand["acc"] > naive["acc"]) and (cand["mse"] < naive["mse"]),
                    "cand_run_dir": cand["run_dir"],
                }
            )

    _write_csv(
        str(out_dir / "delta.csv"),
        delta_rows,
        [
            "compare_to",
            "base_acc",
            "base_mse",
            "cand_exp_name",
            "cand_lambda_var",
            "cand_acc",
            "cand_mse",
            "delta_acc",
            "delta_mse",
            "win_both",
            "cand_run_dir",
        ],
    )

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.6))
    color = {"naive_ref": "#111111", "cagrad_base": "#1f77b4", "cagrad_var": "#d62728"}
    marker = {"naive_ref": "*", "cagrad_base": "o", "cagrad_var": "s"}
    for r in rows:
        tag = r["tag"]
        ax.scatter(
            [r["mse"]],
            [r["acc"]],
            s=120 if tag == "naive_ref" else 72,
            c=color.get(tag, "#666666"),
            marker=marker.get(tag, "o"),
            label=tag,
        )
        txt = tag if tag != "cagrad_var" else f"var={float(r['lambda_var']):.3f}"
        ax.text(r["mse"] + 0.001, r["acc"] + 0.0002, txt, fontsize=8)

    if base is not None:
        for r in vars_rows:
            ax.annotate(
                "",
                xy=(r["mse"], r["acc"]),
                xytext=(base["mse"], base["acc"]),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "alpha": 0.55, "color": "#d62728"},
            )

    ax.set_xlabel("Generation MSE (lower is better)")
    ax.set_ylabel("Understanding Acc (higher is better)")
    ax.set_title("CAGrad + Feature Variance Regularization")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    hh, ll = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        hh.append(h)
        ll.append(l)
    ax.legend(hh, ll, loc="best")
    fig.tight_layout()
    fig_path = str(out_dir / "compare.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"[summarize_cagrad_varreg_compare] runs={len(rows)}")
    print(f"[summarize_cagrad_varreg_compare] summary={out_dir / 'summary.csv'}")
    print(f"[summarize_cagrad_varreg_compare] delta={out_dir / 'delta.csv'}")
    print(f"[summarize_cagrad_varreg_compare] fig={fig_path}")


if __name__ == "__main__":
    main()
