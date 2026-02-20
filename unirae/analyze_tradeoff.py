import argparse
import csv
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

from .utils import get_by_dotted_key, load_yaml, save_json


def _safe_load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_group(cfg: Dict, strategy: str, lambda_txt: float, lambda_rec: float) -> str:
    group = get_by_dotted_key(cfg, "experiment.group", "")
    if group:
        return group

    lora_enable = bool(get_by_dotted_key(cfg, "lora.enable", False))
    steps = int(get_by_dotted_key(cfg, "train.steps", 0))

    if not lora_enable or steps == 0:
        return "baseline"
    if lambda_rec == 0 and lambda_txt > 0:
        return "text_only"
    if lambda_txt == 0 and lambda_rec > 0:
        return "recon_only"
    if strategy == "conflict_aware":
        return "joint_conflict"
    return "joint_naive"


def _load_run_row(run_dir: str) -> Dict:
    cfg_path = os.path.join(run_dir, "run_config.yaml")
    if not os.path.exists(cfg_path):
        return {}

    cfg = load_yaml(cfg_path)
    understanding = _safe_load_json(os.path.join(run_dir, "understanding.json"))
    generation = _safe_load_json(os.path.join(run_dir, "generation.json"))
    cos_summary = _safe_load_json(os.path.join(run_dir, "cos_summary.json"))
    cos_curve = _safe_load_json(os.path.join(run_dir, "cos_curve.json")).get("curve", [])

    lambda_txt = float(get_by_dotted_key(cfg, "train.lambda_txt", 1.0))
    lambda_rec = float(get_by_dotted_key(cfg, "train.lambda_rec", 1.0))
    strategy_raw = str(get_by_dotted_key(cfg, "train.strategy", "naive"))
    strategy = "conflict" if strategy_raw.startswith("conflict") else "naive"

    recon_metric = generation.get("feature_l2", generation.get("mse", generation.get("loss", None)))

    row = {
        "exp_name": Path(run_dir).name,
        "seed": int(get_by_dotted_key(cfg, "seed", 42)),
        "lambda_txt": lambda_txt,
        "lambda_rec": lambda_rec,
        "strategy": strategy,
        "group": _infer_group(cfg, strategy_raw, lambda_txt, lambda_rec),
        "zero_shot_acc": understanding.get("zero_shot_acc", None),
        "recon_metric": recon_metric,
        "cos_mean": cos_summary.get("cos_mean", None),
        "cos_neg_ratio": cos_summary.get("cos_neg_ratio", None),
        "run_dir": run_dir,
        "recon_mode": generation.get("recon_mode", None),
        "cos_curve": cos_curve,
    }
    return row


def _pareto_front(rows: List[Dict]) -> List[Dict]:
    # Maximize zero_shot_acc, minimize recon_metric.
    valid = [r for r in rows if r.get("zero_shot_acc") is not None and r.get("recon_metric") is not None]
    front = []
    for r in valid:
        dominated = False
        for q in valid:
            if q is r:
                continue
            better_or_equal = (
                q["zero_shot_acc"] >= r["zero_shot_acc"] and q["recon_metric"] <= r["recon_metric"]
            )
            strictly_better = (
                q["zero_shot_acc"] > r["zero_shot_acc"] or q["recon_metric"] < r["recon_metric"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front


def _write_csv(path: str, rows: List[Dict], keys: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--out_dir", default="runs/analysis")
    args = parser.parse_args()

    run_dirs = [p for p in glob(os.path.join(args.runs_root, "*")) if os.path.isdir(p)]
    rows = []
    for rd in sorted(run_dirs):
        row = _load_run_row(rd)
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid runs found in {args.runs_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_keys = [
        "exp_name",
        "seed",
        "lambda_txt",
        "lambda_rec",
        "strategy",
        "group",
        "zero_shot_acc",
        "recon_metric",
        "cos_mean",
        "cos_neg_ratio",
        "recon_mode",
        "run_dir",
    ]
    results_csv = str(out_dir / "results.csv")
    _write_csv(results_csv, rows, csv_keys)

    pareto_rows = _pareto_front(rows)
    pareto_csv = str(out_dir / "pareto_points.csv")
    _write_csv(
        pareto_csv,
        pareto_rows,
        ["exp_name", "group", "zero_shot_acc", "recon_metric", "strategy", "seed", "run_dir"],
    )

    scatter_rows = [
        {
            "exp_name": r["exp_name"],
            "group": r["group"],
            "x_recon_metric": r["recon_metric"],
            "y_zero_shot_acc": r["zero_shot_acc"],
            "strategy": r["strategy"],
        }
        for r in rows
        if r.get("recon_metric") is not None and r.get("zero_shot_acc") is not None
    ]
    _write_csv(
        str(out_dir / "pareto_scatter.csv"),
        scatter_rows,
        ["exp_name", "group", "x_recon_metric", "y_zero_shot_acc", "strategy"],
    )
    save_json(scatter_rows, str(out_dir / "pareto_scatter.json"))

    cos_curves = {
        r["exp_name"]: {
            "group": r["group"],
            "strategy": r["strategy"],
            "curve": r.get("cos_curve", []),
        }
        for r in rows
    }
    save_json(cos_curves, str(out_dir / "cos_curves.json"))

    group_counts = {}
    for r in rows:
        group_counts[r["group"]] = group_counts.get(r["group"], 0) + 1

    coverage = {
        "baseline": group_counts.get("baseline", 0),
        "text_only": group_counts.get("text_only", 0),
        "recon_only": group_counts.get("recon_only", 0),
        "joint_naive": group_counts.get("joint_naive", 0),
        "joint_conflict": group_counts.get("joint_conflict", 0),
    }
    save_json({"group_counts": group_counts, "required_coverage": coverage}, str(out_dir / "coverage.json"))

    print(f"[analyze_tradeoff] runs={len(rows)}")
    print(f"[analyze_tradeoff] results_csv={results_csv}")
    print(f"[analyze_tradeoff] pareto_csv={pareto_csv}")


if __name__ == "__main__":
    main()
