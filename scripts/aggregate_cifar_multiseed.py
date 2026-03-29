#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CIFAR run manifests into raw + summary tables.")
    parser.add_argument("--manifest", required=True, help="TSV manifest emitted by a submit script.")
    parser.add_argument("--runs_root", default="runs", help="Root directory containing per-run folders.")
    parser.add_argument("--out_root", default="", help="Output directory. Defaults to manifest parent.")
    return parser.parse_args()


def _safe_mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _safe_std(values: List[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def _load_eval(eval_path: Path) -> Dict[str, float]:
    obj = json.loads(eval_path.read_text(encoding="utf-8"))
    u = obj.get("understanding", {})
    g = obj.get("generation", {})
    return {
        "acc_txt": float(u.get("acc_txt", float("nan"))),
        "zero_shot_loss": float(u.get("zero_shot_loss", float("nan"))),
        "recon_rmse": float(g.get("recon_rmse", float("nan"))),
        "recon_mse": float(g.get("recon_mse", g.get("mse", float("nan")))),
        "psnr": float(g.get("psnr", float("nan"))),
    }


def _is_completed_run(run_dir: Path) -> bool:
    # train_cifar10.py writes these only after the full training loop finishes.
    return (run_dir / "cos_summary.json").exists() and (run_dir / "checkpoints" / "latest.pt").exists()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    runs_root = Path(args.runs_root)
    out_root = Path(args.out_root) if args.out_root else manifest_path.parent
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_manifest(manifest_path)
    raw_rows: List[Dict[str, object]] = []
    grouped: "OrderedDict[str, List[Dict[str, object]]]" = OrderedDict()

    for row in manifest_rows:
        run_name = str(row["run_name"])
        label = str(row["label"])
        run_dir = runs_root / run_name
        eval_path = run_dir / "eval_last.json"
        completed = _is_completed_run(run_dir)
        if not eval_path.exists():
            raw = {
                **row,
                "run_dir": str(run_dir),
                "status": "missing_eval",
                "acc_txt": "",
                "zero_shot_loss": "",
                "recon_rmse": "",
                "recon_mse": "",
                "psnr": "",
            }
        elif not completed:
            raw = {
                **row,
                "run_dir": str(run_dir),
                "status": "incomplete",
                "acc_txt": "",
                "zero_shot_loss": "",
                "recon_rmse": "",
                "recon_mse": "",
                "psnr": "",
            }
        else:
            metrics = _load_eval(eval_path)
            raw = {
                **row,
                "run_dir": str(run_dir),
                "status": "ok",
                **metrics,
            }
            grouped.setdefault(label, []).append(raw)
        raw_rows.append(raw)

    raw_fields = list(raw_rows[0].keys()) if raw_rows else [
        "jobid",
        "run_name",
        "label",
        "seed",
        "status",
        "run_dir",
        "acc_txt",
        "zero_shot_loss",
        "recon_rmse",
        "recon_mse",
        "psnr",
    ]
    with (out_root / "raw_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows: List[Dict[str, object]] = []
    for label, rows in grouped.items():
        accs = [float(r["acc_txt"]) for r in rows]
        rmses = [float(r["recon_rmse"]) for r in rows]
        mses = [float(r["recon_mse"]) for r in rows]
        psnrs = [float(r["psnr"]) for r in rows]
        losses = [float(r["zero_shot_loss"]) for r in rows]
        summary_rows.append(
            {
                "label": label,
                "num_runs": len(rows),
                "acc_txt_mean": round(_safe_mean(accs), 6),
                "acc_txt_std": round(_safe_std(accs), 6),
                "recon_rmse_mean": round(_safe_mean(rmses), 6),
                "recon_rmse_std": round(_safe_std(rmses), 6),
                "recon_mse_mean": round(_safe_mean(mses), 6),
                "recon_mse_std": round(_safe_std(mses), 6),
                "psnr_mean": round(_safe_mean(psnrs), 6),
                "psnr_std": round(_safe_std(psnrs), 6),
                "zero_shot_loss_mean": round(_safe_mean(losses), 6),
                "zero_shot_loss_std": round(_safe_std(losses), 6),
            }
        )

    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "num_runs",
                "acc_txt_mean",
                "acc_txt_std",
                "recon_rmse_mean",
                "recon_rmse_std",
                "recon_mse_mean",
                "recon_mse_std",
                "psnr_mean",
                "psnr_std",
                "zero_shot_loss_mean",
                "zero_shot_loss_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=True)

    print(f"[done] wrote {out_root / 'raw_results.csv'}")
    print(f"[done] wrote {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
