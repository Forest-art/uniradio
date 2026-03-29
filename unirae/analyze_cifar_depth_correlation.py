import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _pearson_corr(xs: List[float], ys: List[float], eps: float = 1e-12) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= eps or vy <= eps:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / ((vx * vy) ** 0.5 + eps))


def _rank_values(values: List[float]) -> List[float]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return _pearson_corr(_rank_values(xs), _rank_values(ys))


def _load_jsonl_summary(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda x: int(x.get("step", 0)))
    return rows


def _load_step_csv(path: Path) -> Tuple[List[float], List[float]]:
    depths: List[float] = []
    cos: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            depths.append(float(row["depth"]))
            cos.append(float(row["cosine_similarity"]))
    return depths, cos


def _fallback_from_layer_probe_dir(layer_probe_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    step_files = []
    for p in layer_probe_dir.glob("step_*_layerwise.csv"):
        stem = p.stem
        # step_0000200_layerwise
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        if parts[1].isdigit():
            step_files.append((int(parts[1]), p))
    step_files.sort(key=lambda x: x[0])

    for step, p in step_files:
        depths, cos = _load_step_csv(p)
        if len(depths) < 2:
            continue
        rows.append(
            {
                "step": int(step),
                "depth_cos_pearson": _pearson_corr(depths, cos),
                "depth_cos_spearman": _spearman_corr(depths, cos),
            }
        )
    return rows


def _first_persistent_negative(
    steps: List[int],
    values: List[float],
    neg_threshold: float,
    persist_points: int,
) -> int:
    if len(steps) == 0 or len(steps) != len(values):
        return -1
    k = max(1, int(persist_points))
    for i in range(0, len(steps) - k + 1):
        ok = True
        for j in range(i, i + k):
            if not (values[j] <= neg_threshold):
                ok = False
                break
        if ok:
            return int(steps[i])
    return -1


def _mean(values: List[float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    v = sum((x - m) ** 2 for x in values) / len(values)
    return float(v**0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CIFAR layer-depth gradient-cos correlation over training.")
    parser.add_argument("--run_dirs", required=True, help="Comma-separated run directories.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--neg_threshold", type=float, default=-0.2)
    parser.add_argument("--persist_points", type=int, default=3)
    args = parser.parse_args()

    run_dirs = [Path(x.strip()) for x in str(args.run_dirs).split(",") if x.strip()]
    if len(run_dirs) == 0:
        raise ValueError("run_dirs is empty.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run: Dict[str, List[Dict]] = {}
    all_steps = set()
    for run_dir in run_dirs:
        summary_path = run_dir / "layer_probe_summary.jsonl"
        rows = _load_jsonl_summary(summary_path)
        if len(rows) == 0:
            rows = _fallback_from_layer_probe_dir(run_dir / "layer_probe")
        if len(rows) == 0:
            continue
        rows.sort(key=lambda x: int(x.get("step", 0)))
        per_run[str(run_dir)] = rows
        for r in rows:
            all_steps.add(int(r["step"]))

    if len(per_run) == 0:
        raise RuntimeError("No valid layer probe rows found in run_dirs.")

    ordered_steps = sorted(all_steps)
    curve_rows = []
    for step in ordered_steps:
        pearson_vals = []
        spearman_vals = []
        for rows in per_run.values():
            hit = None
            for r in rows:
                if int(r["step"]) == step:
                    hit = r
                    break
            if hit is None:
                continue
            pearson_vals.append(float(hit.get("depth_cos_pearson", 0.0)))
            spearman_vals.append(float(hit.get("depth_cos_spearman", 0.0)))
        if len(pearson_vals) == 0:
            continue
        curve_rows.append(
            {
                "step": int(step),
                "pearson_mean": _mean(pearson_vals),
                "pearson_std": _std(pearson_vals),
                "spearman_mean": _mean(spearman_vals),
                "spearman_std": _std(spearman_vals),
                "n_runs": len(pearson_vals),
            }
        )

    curve_csv = out_dir / "depth_corr_curve.csv"
    with open(curve_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "pearson_mean",
                "pearson_std",
                "spearman_mean",
                "spearman_std",
                "n_runs",
            ],
        )
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)

    run_findings: Dict[str, Dict] = {}
    for run_dir, rows in per_run.items():
        steps = [int(r["step"]) for r in rows]
        pearson = [float(r.get("depth_cos_pearson", 0.0)) for r in rows]
        spearman = [float(r.get("depth_cos_spearman", 0.0)) for r in rows]
        run_findings[run_dir] = {
            "first_persistent_pearson_step": _first_persistent_negative(
                steps, pearson, args.neg_threshold, args.persist_points
            ),
            "first_persistent_spearman_step": _first_persistent_negative(
                steps, spearman, args.neg_threshold, args.persist_points
            ),
            "final_pearson": float(pearson[-1]),
            "final_spearman": float(spearman[-1]),
            "num_points": len(rows),
        }

    agg_steps = [int(r["step"]) for r in curve_rows]
    agg_spearman = [float(r["spearman_mean"]) for r in curve_rows]
    agg_pearson = [float(r["pearson_mean"]) for r in curve_rows]
    agg = {
        "neg_threshold": float(args.neg_threshold),
        "persist_points": int(args.persist_points),
        "first_persistent_pearson_step_mean_curve": _first_persistent_negative(
            agg_steps, agg_pearson, args.neg_threshold, args.persist_points
        ),
        "first_persistent_spearman_step_mean_curve": _first_persistent_negative(
            agg_steps, agg_spearman, args.neg_threshold, args.persist_points
        ),
        "curve_csv": str(curve_csv),
    }

    out_json = out_dir / "depth_corr_findings.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_run": run_findings}, f, indent=2)

    # Publication-friendly quick plot.
    x = [int(r["step"]) for r in curve_rows]
    y_s = [float(r["spearman_mean"]) for r in curve_rows]
    e_s = [float(r["spearman_std"]) for r in curve_rows]
    y_p = [float(r["pearson_mean"]) for r in curve_rows]
    e_p = [float(r["pearson_std"]) for r in curve_rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=320)
    ax.plot(x, y_s, color="#1f77b4", linewidth=2.2, label="Spearman(depth, cos)")
    ax.fill_between(x, [a - b for a, b in zip(y_s, e_s)], [a + b for a, b in zip(y_s, e_s)], color="#1f77b4", alpha=0.15)
    ax.plot(x, y_p, color="#d62728", linewidth=2.0, label="Pearson(depth, cos)")
    ax.fill_between(x, [a - b for a, b in zip(y_p, e_p)], [a + b for a, b in zip(y_p, e_p)], color="#d62728", alpha=0.15)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(float(args.neg_threshold), color="#555", linestyle=":", linewidth=1.0, label=f"neg threshold={args.neg_threshold:.2f}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Correlation (depth vs cosine)")
    ax.set_title("CIFAR100 Layer-Depth Conflict Correlation")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "depth_corr_curve.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[done] curve_csv={curve_csv}")
    print(f"[done] findings_json={out_json}")
    print(f"[done] plot={out_dir / 'depth_corr_curve.png'}")


if __name__ == "__main__":
    main()
