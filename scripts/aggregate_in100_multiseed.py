#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate IN100 manifest runs into raw + summary tables.")
    parser.add_argument("--manifest", required=True, help="TSV manifest emitted by an IN100 launcher.")
    parser.add_argument("--runs_root", default="runs", help="Root directory containing per-run folders.")
    parser.add_argument("--out_root", default="", help="Output directory. Defaults to manifest parent.")
    return parser.parse_args()


def _safe_mean(values: List[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.fmean(vals)) if vals else float("nan")


def _safe_std(values: List[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def _load_eval(eval_path: Path) -> Dict[str, float]:
    obj = json.loads(eval_path.read_text(encoding="utf-8"))
    gen = obj.get("generation", {})
    return {
        "val_top1_acc": float(obj.get("val_top1_acc", obj.get("understanding", {}).get("acc_txt", float("nan")))),
        "val_mse": float(obj.get("val_mse", gen.get("recon_mse", float("nan")))),
        "val_rmse": float(obj.get("val_rmse", gen.get("recon_rmse", float("nan")))),
        "val_rfid": float(obj.get("val_rfid", gen.get("rfid", float("nan")))),
        "val_num_samples": int(obj.get("val_num_samples", 0)),
        "rfid_num_samples": int(obj.get("rfid_num_samples", 0)),
    }


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
        if not eval_path.exists():
            fallback = run_dir / "final_eval.json"
            eval_path = fallback if fallback.exists() else eval_path

        if not eval_path.exists():
            raw = {
                **row,
                "run_dir": str(run_dir),
                "status": "missing_eval",
                "val_top1_acc": "",
                "val_mse": "",
                "val_rmse": "",
                "val_rfid": "",
                "val_num_samples": "",
                "rfid_num_samples": "",
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
        "val_top1_acc",
        "val_mse",
        "val_rmse",
        "val_rfid",
        "val_num_samples",
        "rfid_num_samples",
    ]
    with (out_root / "raw_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows: List[Dict[str, object]] = []
    for label, rows in grouped.items():
        accs = [float(r["val_top1_acc"]) for r in rows]
        mses = [float(r["val_mse"]) for r in rows]
        rmses = [float(r["val_rmse"]) for r in rows]
        rfids = [float(r["val_rfid"]) for r in rows]
        summary_rows.append(
            {
                "label": label,
                "num_runs": len(rows),
                "val_top1_acc_mean": round(_safe_mean(accs), 6),
                "val_top1_acc_std": round(_safe_std(accs), 6),
                "val_mse_mean": round(_safe_mean(mses), 6),
                "val_mse_std": round(_safe_std(mses), 6),
                "val_rmse_mean": round(_safe_mean(rmses), 6),
                "val_rmse_std": round(_safe_std(rmses), 6),
                "val_rfid_mean": round(_safe_mean(rfids), 6),
                "val_rfid_std": round(_safe_std(rfids), 6),
            }
        )

    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "num_runs",
                "val_top1_acc_mean",
                "val_top1_acc_std",
                "val_mse_mean",
                "val_mse_std",
                "val_rmse_mean",
                "val_rmse_std",
                "val_rfid_mean",
                "val_rfid_std",
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
