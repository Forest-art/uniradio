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


def _tag(exp_name: str, strategy: str, adaptive: bool) -> str:
    if strategy == "naive":
        return "naive_ref"
    if not adaptive:
        return "cagrad_fixed"
    return "cagrad_adaptive"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", default="runs/cifar100_cagrad_adaptivebeta_cmp_*")
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
        adaptive = bool(ts.get("cagrad_adaptive_beta", cs.get("cagrad_adaptive_beta", False)))
        rows.append(
            {
                "exp_name": os.path.basename(rd),
                "run_dir": rd,
                "tag": _tag(os.path.basename(rd), strategy=strategy, adaptive=adaptive),
                "strategy": strategy,
                "cagrad_adaptive_beta": adaptive,
                "acc": float(u.get("zero_shot_acc", u.get("acc_txt"))),
                "mse": float(g.get("mse", g.get("recon_mse"))),
                "cagrad_beta": ts.get("cagrad_beta", cs.get("cagrad_beta")),
                "lambda_txt": ts.get("lambda_txt", cs.get("lambda_txt")),
                "grad_norm_mode": ts.get("grad_norm_mode", cs.get("grad_norm_mode")),
                "grad_norm_scope": ts.get("grad_norm_scope", cs.get("grad_norm_scope")),
                "grad_norm_layers": "|".join(ts.get("grad_norm_layers", cs.get("grad_norm_layers", [])))
                if isinstance(ts.get("grad_norm_layers", cs.get("grad_norm_layers", [])), list)
                else ts.get("grad_norm_layers", cs.get("grad_norm_layers", "")),
                "cagrad_adaptive_scope": ts.get("cagrad_adaptive_scope", cs.get("cagrad_adaptive_scope", "")),
                "cagrad_adaptive_layers": "|".join(ts.get("cagrad_adaptive_layers", cs.get("cagrad_adaptive_layers", [])))
                if isinstance(ts.get("cagrad_adaptive_layers", cs.get("cagrad_adaptive_layers", [])), list)
                else ts.get("cagrad_adaptive_layers", cs.get("cagrad_adaptive_layers", "")),
                "cagrad_adaptive_nonconflict_merge": ts.get(
                    "cagrad_adaptive_nonconflict_merge",
                    cs.get("cagrad_adaptive_nonconflict_merge", ""),
                ),
                "cagrad_adaptive_conflict_threshold": ts.get(
                    "cagrad_adaptive_conflict_threshold",
                    cs.get("cagrad_adaptive_conflict_threshold", ""),
                ),
                "cagrad_adaptive_strength": ts.get(
                    "cagrad_adaptive_strength",
                    cs.get("cagrad_adaptive_strength", ""),
                ),
                "cagrad_adaptive_power": ts.get(
                    "cagrad_adaptive_power",
                    cs.get("cagrad_adaptive_power", ""),
                ),
                "cagrad_adaptive_beta_cap": ts.get(
                    "cagrad_adaptive_beta_cap",
                    cs.get("cagrad_adaptive_beta_cap", ""),
                ),
                "cagrad_adaptive_online_beta": ts.get(
                    "cagrad_adaptive_online_beta",
                    cs.get("cagrad_adaptive_online_beta", False),
                ),
                "cagrad_adaptive_online_lr": ts.get(
                    "cagrad_adaptive_online_lr",
                    cs.get("cagrad_adaptive_online_lr", ""),
                ),
                "cos_mean": cs.get("cos_mean", ""),
                "cos_neg_ratio": cs.get("cos_neg_ratio", ""),
                "train_steps": cs.get("train_steps", ""),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(rows, key=lambda x: x["tag"])
    _write_csv(
        str(out_dir / "summary.csv"),
        rows,
        [
            "exp_name",
            "tag",
            "strategy",
            "cagrad_adaptive_beta",
            "acc",
            "mse",
            "cagrad_beta",
            "lambda_txt",
            "grad_norm_mode",
            "grad_norm_scope",
            "grad_norm_layers",
            "cagrad_adaptive_scope",
            "cagrad_adaptive_layers",
            "cagrad_adaptive_nonconflict_merge",
            "cagrad_adaptive_conflict_threshold",
            "cagrad_adaptive_strength",
            "cagrad_adaptive_power",
            "cagrad_adaptive_beta_cap",
            "cagrad_adaptive_online_beta",
            "cagrad_adaptive_online_lr",
            "cos_mean",
            "cos_neg_ratio",
            "train_steps",
            "run_dir",
        ],
    )

    ref = next((r for r in rows if r["tag"] == "cagrad_fixed"), None)
    naive = next((r for r in rows if r["tag"] == "naive_ref"), None)
    ada_rows = [r for r in rows if r["tag"] == "cagrad_adaptive"]

    delta_rows: List[Dict] = []
    for ada in ada_rows:
        if ref:
            delta_rows.append(
                {
                    "compare_to": "cagrad_fixed",
                    "base_acc": ref["acc"],
                    "base_mse": ref["mse"],
                    "cand_exp_name": ada["exp_name"],
                    "cand_tag": ada["tag"],
                    "cand_acc": ada["acc"],
                    "cand_mse": ada["mse"],
                    "delta_acc": ada["acc"] - ref["acc"],
                    "delta_mse": ada["mse"] - ref["mse"],
                    "win_both": (ada["acc"] > ref["acc"]) and (ada["mse"] < ref["mse"]),
                    "cand_run_dir": ada["run_dir"],
                }
            )
        if naive:
            delta_rows.append(
                {
                    "compare_to": "naive_ref",
                    "base_acc": naive["acc"],
                    "base_mse": naive["mse"],
                    "cand_exp_name": ada["exp_name"],
                    "cand_tag": ada["tag"],
                    "cand_acc": ada["acc"],
                    "cand_mse": ada["mse"],
                    "delta_acc": ada["acc"] - naive["acc"],
                    "delta_mse": ada["mse"] - naive["mse"],
                    "win_both": (ada["acc"] > naive["acc"]) and (ada["mse"] < naive["mse"]),
                    "cand_run_dir": ada["run_dir"],
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
            "cand_tag",
            "cand_acc",
            "cand_mse",
            "delta_acc",
            "delta_mse",
            "win_both",
            "cand_run_dir",
        ],
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.6))
    color = {"naive_ref": "#111111", "cagrad_fixed": "#1f77b4", "cagrad_adaptive": "#d62728"}
    marker = {"naive_ref": "*", "cagrad_fixed": "o", "cagrad_adaptive": "s"}
    for r in rows:
        ax.scatter([r["mse"]], [r["acc"]], s=120 if r["tag"] == "naive_ref" else 64, c=color.get(r["tag"], "#666666"), marker=marker.get(r["tag"], "o"), label=r["tag"])
        label_text = r["tag"] if r["tag"] != "cagrad_adaptive" else r["exp_name"].replace("cifar100_cagrad_adaptivebeta_tune_", "").replace("cifar100_cagrad_adaptivebeta_cmp_", "")
        ax.text(r["mse"] + 0.001, r["acc"] + 0.0002, label_text, fontsize=8)

    if ref:
        for ada in ada_rows:
            ax.annotate(
                "",
                xy=(ada["mse"], ada["acc"]),
                xytext=(ref["mse"], ref["acc"]),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "alpha": 0.55, "color": "#d62728"},
            )

    ax.set_xlabel("Generation MSE (lower is better)")
    ax.set_ylabel("Understanding Acc (higher is better)")
    ax.set_title("Adaptive-Beta CAGrad vs Fixed-Beta")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="best")
    fig.tight_layout()
    fig_path = str(out_dir / "adaptivebeta_compare.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"[summarize_cagrad_adaptivebeta_compare] runs={len(rows)}")
    print(f"[summarize_cagrad_adaptivebeta_compare] summary={out_dir / 'summary.csv'}")
    print(f"[summarize_cagrad_adaptivebeta_compare] delta={out_dir / 'delta.csv'}")
    print(f"[summarize_cagrad_adaptivebeta_compare] fig={fig_path}")


if __name__ == "__main__":
    main()
