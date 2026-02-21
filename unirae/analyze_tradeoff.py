import argparse
import csv
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List

from .utils import get_by_dotted_key, load_yaml, save_json


def _safe_load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_group(name: str) -> str:
    if not name:
        return ""
    norm = name.lower().replace("-", "_").replace(" ", "_")
    alias = {
        "text": "text_only",
        "recon": "recon_only",
        "joint_conflict_aware": "joint_conflict",
        "joint_conflictaware": "joint_conflict",
        "conflict": "joint_conflict",
    }
    return alias.get(norm, norm)


def _infer_group(cfg: Dict, strategy: str, lambda_txt: float, lambda_rec: float) -> str:
    group = _normalize_group(str(get_by_dotted_key(cfg, "experiment.group", "")))
    if group:
        return group

    mode = _normalize_group(str(get_by_dotted_key(cfg, "train.mode", "joint")))
    if mode == "baseline":
        return "baseline"
    if mode == "text_only":
        return "text_only"
    if mode == "recon_only":
        return "recon_only"

    lora_enable = bool(get_by_dotted_key(cfg, "lora.enable", False))
    steps = int(get_by_dotted_key(cfg, "train.steps", 0))

    if ("radio" in cfg and (not lora_enable or steps == 0)) or steps == 0:
        return "baseline"
    if lambda_rec == 0 and lambda_txt > 0:
        return "text_only"
    if lambda_txt == 0 and lambda_rec > 0:
        return "recon_only"
    s = strategy.lower()
    if s.startswith("conflict") or s == "pcgrad":
        return "joint_conflict"
    if "cagrad" in s:
        return "joint_cagrad"
    if "mgda" in s:
        return "joint_mgda"
    return "joint_naive"


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def _safe_neg_ratio(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for v in values if v < 0) / len(values))


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
    lambda_cons = float(get_by_dotted_key(cfg, "train.lambda_cons", get_by_dotted_key(cfg, "consistency.lambda_cons", 0.0)))
    consistency_enabled = bool(get_by_dotted_key(cfg, "consistency.enabled", False) or lambda_cons > 0.0)
    lambda_sup = float(get_by_dotted_key(cfg, "train.lambda_sup", get_by_dotted_key(cfg, "supcon.lambda", 0.0)))
    supcon_enabled = bool(get_by_dotted_key(cfg, "supcon.enabled", False) or lambda_sup > 0.0)
    supcon_tau = float(get_by_dotted_key(cfg, "supcon.tau", 0.1))
    strategy_raw = str(get_by_dotted_key(cfg, "train.strategy", "naive"))
    s = strategy_raw.lower()
    if s.startswith("conflict") or s == "pcgrad":
        strategy = "conflict"
    elif "cagrad" in s:
        strategy = "cagrad"
    elif "mgda" in s:
        strategy = "mgda"
    else:
        strategy = "naive"

    dataset = str(get_by_dotted_key(cfg, "data.dataset", "imagenet"))
    backbone = str(get_by_dotted_key(cfg, "model.backbone", "radio"))

    acc_txt = understanding.get("acc_txt", understanding.get("zero_shot_acc", None))
    recon_mse = generation.get("recon_mse", generation.get("mse", None))
    recon_metric = generation.get("feature_l2", recon_mse)

    cos_vals = [float(x["cos"]) for x in cos_curve if isinstance(x, dict) and "cos" in x]
    cos_mean = cos_summary.get("cos_mean", _safe_mean(cos_vals, default=0.0))
    cos_neg_ratio = cos_summary.get("cos_neg_ratio", _safe_neg_ratio(cos_vals))

    row = {
        "exp_name": Path(run_dir).name,
        "dataset": dataset,
        "backbone": backbone,
        "seed": int(get_by_dotted_key(cfg, "seed", 42)),
        "lambda_txt": lambda_txt,
        "lambda_rec": lambda_rec,
        "lambda_cons": lambda_cons,
        "consistency_enabled": consistency_enabled,
        "lambda_sup": lambda_sup,
        "supcon_enabled": supcon_enabled,
        "supcon_tau": supcon_tau,
        "strategy": strategy,
        "group": _infer_group(cfg, strategy_raw, lambda_txt, lambda_rec),
        "acc_txt": acc_txt,
        "zero_shot_acc": understanding.get("zero_shot_acc", acc_txt),
        "recon_mse": recon_mse,
        "recon_metric": recon_metric,
        "cos_mean": cos_mean,
        "cos_neg_ratio": cos_neg_ratio,
        "train_steps": int(cos_summary.get("train_steps", get_by_dotted_key(cfg, "train.steps", 0))),
        "walltime": cos_summary.get("walltime_sec", None),
        "run_dir": run_dir,
        "recon_mode": generation.get("recon_mode", None),
        "cos_curve": cos_curve,
    }
    return row


def _pareto_front(rows: List[Dict], y_key: str, x_key: str) -> List[Dict]:
    valid = [r for r in rows if r.get(y_key) is not None and r.get(x_key) is not None]
    front = []
    for r in valid:
        dominated = False
        for q in valid:
            if q is r:
                continue
            better_or_equal = q[y_key] >= r[y_key] and q[x_key] <= r[x_key]
            strictly_better = q[y_key] > r[y_key] or q[x_key] < r[x_key]
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


def _group_to_public_name(group: str) -> str:
    mapping = {
        "baseline": "baseline",
        "text_only": "text-only",
        "recon_only": "recon-only",
        "joint_naive": "joint-naive",
        "joint_conflict": "joint-conflict",
        "joint_cagrad": "joint-cagrad",
        "joint_mgda": "joint-mgda",
    }
    return mapping.get(group, group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--out_dir", default="runs/analysis")
    parser.add_argument("--results_dir", default="results")
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
        "dataset",
        "backbone",
        "seed",
        "lambda_txt",
        "lambda_rec",
        "lambda_cons",
        "consistency_enabled",
        "lambda_sup",
        "supcon_enabled",
        "supcon_tau",
        "strategy",
        "group",
        "acc_txt",
        "zero_shot_acc",
        "recon_mse",
        "recon_metric",
        "cos_mean",
        "cos_neg_ratio",
        "train_steps",
        "walltime",
        "recon_mode",
        "run_dir",
    ]
    results_csv = str(out_dir / "results.csv")
    _write_csv(results_csv, rows, csv_keys)

    pareto_rows = _pareto_front(rows, y_key="zero_shot_acc", x_key="recon_metric")
    pareto_csv = str(out_dir / "pareto_points.csv")
    _write_csv(
        pareto_csv,
        pareto_rows,
        ["exp_name", "dataset", "backbone", "group", "zero_shot_acc", "recon_metric", "strategy", "seed", "run_dir"],
    )

    scatter_rows = [
        {
            "exp_name": r["exp_name"],
            "dataset": r["dataset"],
            "backbone": r["backbone"],
            "group": r["group"],
            "x_recon_metric": r["recon_metric"],
            "y_zero_shot_acc": r["zero_shot_acc"],
            "strategy": r["strategy"],
            "lambda_cons": r["lambda_cons"],
            "consistency_enabled": r["consistency_enabled"],
            "lambda_sup": r["lambda_sup"],
            "supcon_enabled": r["supcon_enabled"],
            "supcon_tau": r["supcon_tau"],
        }
        for r in rows
        if r.get("recon_metric") is not None and r.get("zero_shot_acc") is not None
    ]
    _write_csv(
        str(out_dir / "pareto_scatter.csv"),
        scatter_rows,
        [
            "exp_name",
            "dataset",
            "backbone",
            "group",
            "x_recon_metric",
            "y_zero_shot_acc",
            "strategy",
            "lambda_cons",
            "consistency_enabled",
            "lambda_sup",
            "supcon_enabled",
            "supcon_tau",
        ],
    )
    save_json(scatter_rows, str(out_dir / "pareto_scatter.json"))

    cos_curves = {
        r["exp_name"]: {
            "dataset": r["dataset"],
            "backbone": r["backbone"],
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
    save_json({"group_counts": group_counts}, str(out_dir / "coverage.json"))

    # CIFAR-10 specific exports.
    cifar_rows = [r for r in rows if str(r.get("dataset", "")).lower() == "cifar10"]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cifar_csv = str(results_dir / "cifar10_results.csv")
    _write_csv(
        cifar_csv,
        cifar_rows,
        [
            "dataset",
            "backbone",
            "strategy",
            "seed",
            "lambda_txt",
            "lambda_rec",
            "lambda_cons",
            "consistency_enabled",
            "lambda_sup",
            "supcon_enabled",
            "supcon_tau",
            "acc_txt",
            "recon_mse",
            "cos_mean",
            "cos_neg_ratio",
            "train_steps",
            "walltime",
            "exp_name",
            "run_dir",
        ],
    )

    pareto_groups = {
        "baseline": [],
        "text-only": [],
        "recon-only": [],
        "joint-naive": [],
        "joint-conflict": [],
        "joint-cagrad": [],
        "joint-mgda": [],
    }
    for r in cifar_rows:
        if r.get("acc_txt") is None or r.get("recon_mse") is None:
            continue
        g = _group_to_public_name(str(r.get("group", "")))
        if g not in pareto_groups:
            continue
        pareto_groups[g].append(
            {
                "x": r["recon_mse"],
                "y": r["acc_txt"],
                "exp_name": r["exp_name"],
                "seed": r["seed"],
                "backbone": r["backbone"],
                "strategy": r["strategy"],
                "lambda_txt": r["lambda_txt"],
                "lambda_rec": r["lambda_rec"],
                "lambda_cons": r["lambda_cons"],
                "consistency_enabled": r["consistency_enabled"],
                "lambda_sup": r["lambda_sup"],
                "supcon_enabled": r["supcon_enabled"],
                "supcon_tau": r["supcon_tau"],
                "cos_mean": r["cos_mean"],
                "cos_neg_ratio": r["cos_neg_ratio"],
            }
        )

    cifar_pareto = {
        "dataset": "cifar10",
        "x": {"name": "recon_mse", "goal": "lower_better"},
        "y": {"name": "acc_txt", "goal": "higher_better"},
        "points": pareto_groups,
    }
    save_json(cifar_pareto, str(results_dir / "cifar10_pareto.json"))

    cifar_cos_curves = {
        r["exp_name"]: {
            "group": _group_to_public_name(r["group"]),
            "strategy": r["strategy"],
            "curve": r.get("cos_curve", []),
        }
        for r in cifar_rows
    }
    save_json(cifar_cos_curves, str(results_dir / "cifar10_cos_curve.json"))

    print(f"[analyze_tradeoff] runs={len(rows)}")
    print(f"[analyze_tradeoff] results_csv={results_csv}")
    print(f"[analyze_tradeoff] pareto_csv={pareto_csv}")
    print(f"[analyze_tradeoff] cifar10_results={cifar_csv}")


if __name__ == "__main__":
    main()
