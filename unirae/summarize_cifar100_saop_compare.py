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


def _pareto_front(rows: List[Dict]) -> List[Dict]:
    out = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if (float(b["acc"]) >= float(a["acc"])) and (float(b["mse"]) <= float(a["mse"])):
                if (float(b["acc"]) > float(a["acc"])) or (float(b["mse"]) < float(a["mse"])):
                    dominated = True
                    break
        if not dominated:
            out.append(a)
    return sorted(out, key=lambda x: (float(x["mse"]), -float(x["acc"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", default="runs/cifar100_saop_cmp_*")
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
        rows.append(
            {
                "exp_name": os.path.basename(rd),
                "strategy": str(ts.get("strategy", cs.get("strategy", "naive"))).lower(),
                "acc": float(u.get("zero_shot_acc", u.get("acc_txt"))),
                "mse": float(g.get("mse", g.get("recon_mse"))),
                "cagrad_beta": ts.get("cagrad_beta", cs.get("cagrad_beta", "")),
                "saop_scope": ts.get("saop_scope", cs.get("saop_scope", "")),
                "saop_layers": "|".join(ts.get("saop_layers", cs.get("saop_layers", [])))
                if isinstance(ts.get("saop_layers", cs.get("saop_layers", [])), list)
                else ts.get("saop_layers", cs.get("saop_layers", "")),
                "saop_eps": ts.get("saop_eps", cs.get("saop_eps", "")),
                "grad_norm_mode": ts.get("grad_norm_mode", cs.get("grad_norm_mode", "")),
                "grad_norm_scope": ts.get("grad_norm_scope", cs.get("grad_norm_scope", "")),
                "grad_norm_layers": "|".join(ts.get("grad_norm_layers", cs.get("grad_norm_layers", [])))
                if isinstance(ts.get("grad_norm_layers", cs.get("grad_norm_layers", [])), list)
                else ts.get("grad_norm_layers", cs.get("grad_norm_layers", "")),
                "cos_mean": cs.get("cos_mean", ""),
                "cos_neg_ratio": cs.get("cos_neg_ratio", ""),
                "train_steps": cs.get("train_steps", ""),
                "run_dir": rd,
            }
        )

    order = {"naive": 0, "pcgrad": 1, "cagrad": 2, "saop": 3}
    rows = sorted(rows, key=lambda x: (order.get(x["strategy"], 99), x["exp_name"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        str(out_dir / "summary.csv"),
        rows,
        [
            "exp_name",
            "strategy",
            "acc",
            "mse",
            "cagrad_beta",
            "saop_scope",
            "saop_layers",
            "saop_eps",
            "grad_norm_mode",
            "grad_norm_scope",
            "grad_norm_layers",
            "cos_mean",
            "cos_neg_ratio",
            "train_steps",
            "run_dir",
        ],
    )

    by_strategy = {r["strategy"]: r for r in rows}
    naive = by_strategy.get("naive")
    delta_rows: List[Dict] = []
    if naive is not None:
        for r in rows:
            if r["strategy"] == "naive":
                continue
            delta_rows.append(
                {
                    "strategy": r["strategy"],
                    "naive_acc": float(naive["acc"]),
                    "naive_mse": float(naive["mse"]),
                    "acc": float(r["acc"]),
                    "mse": float(r["mse"]),
                    "delta_acc": float(r["acc"]) - float(naive["acc"]),
                    "delta_mse": float(r["mse"]) - float(naive["mse"]),
                    "win_both_vs_naive": (float(r["acc"]) > float(naive["acc"])) and (float(r["mse"]) < float(naive["mse"])),
                    "run_dir": r["run_dir"],
                }
            )
    _write_csv(
        str(out_dir / "delta_vs_naive.csv"),
        delta_rows,
        [
            "strategy",
            "naive_acc",
            "naive_mse",
            "acc",
            "mse",
            "delta_acc",
            "delta_mse",
            "win_both_vs_naive",
            "run_dir",
        ],
    )

    pareto = _pareto_front(rows)
    _write_csv(
        str(out_dir / "pareto_front.csv"),
        pareto,
        [
            "exp_name",
            "strategy",
            "acc",
            "mse",
            "run_dir",
        ],
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.4))
    color = {"naive": "#111111", "pcgrad": "#1f77b4", "cagrad": "#2ca02c", "saop": "#d62728"}
    marker = {"naive": "*", "pcgrad": "o", "cagrad": "s", "saop": "D"}
    for r in rows:
        st = r["strategy"]
        ax.scatter([r["mse"]], [r["acc"]], s=120 if st == "naive" else 80, c=color.get(st, "#666666"), marker=marker.get(st, "o"), label=st)
        ax.text(float(r["mse"]) + 0.001, float(r["acc"]) + 0.0002, st, fontsize=9)

    if len(pareto) >= 2:
        px = [float(r["mse"]) for r in pareto]
        py = [float(r["acc"]) for r in pareto]
        ax.plot(px, py, c="#ff7f0e", lw=1.5, marker="o", ms=4, label="pareto")

    ax.set_xlabel("Generation MSE (lower is better)")
    ax.set_ylabel("Understanding Acc (higher is better)")
    ax.set_title("CIFAR100 Strategy Compare: naive / pcgrad / cagrad / saop")
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
    fig_path = str(out_dir / "pareto_compare.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"[summarize_cifar100_saop_compare] runs={len(rows)}")
    print(f"[summarize_cifar100_saop_compare] summary={out_dir / 'summary.csv'}")
    print(f"[summarize_cifar100_saop_compare] delta={out_dir / 'delta_vs_naive.csv'}")
    print(f"[summarize_cifar100_saop_compare] pareto={out_dir / 'pareto_front.csv'}")
    print(f"[summarize_cifar100_saop_compare] fig={fig_path}")


if __name__ == "__main__":
    main()
