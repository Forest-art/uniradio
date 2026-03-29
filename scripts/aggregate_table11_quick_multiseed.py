#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List

STRATEGIES = ["joint", "pcgrad", "cagrad", "dsga"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate quick multi-seed CIFAR100 results for Table 11.")
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--run_prefix", default="table11_cifar100_vit_small")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def run_name(prefix: str, strategy: str, seed: int) -> str:
    return f"{prefix}_{strategy}_seed{seed}"


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    table_rows: List[Dict[str, object]] = []
    for strategy in STRATEGIES:
        raw_rows = []
        accs: List[float] = []
        rmses: List[float] = []
        strat_dir = out_root / strategy
        strat_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            run_dir = runs_root / run_name(args.run_prefix, strategy, seed)
            eval_path = run_dir / "eval_last.json"
            if not eval_path.exists():
                raise FileNotFoundError(f"Missing eval file: {eval_path}")
            obj = json.loads(eval_path.read_text())
            row = {
                "seed": int(seed),
                "acc_top1": round(float(obj["understanding"]["acc_txt"]) * 100.0, 2),
                "rmse": round(float(obj["generation"]["recon_rmse"]), 3),
                "run_dir": str(run_dir),
            }
            raw_rows.append(row)
            accs.append(float(obj["understanding"]["acc_txt"]) * 100.0)
            rmses.append(float(obj["generation"]["recon_rmse"]))
            with open(strat_dir / f"seed_{seed}.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2, ensure_ascii=True)

        summary = {
            "strategy": strategy,
            "acc_top1_mean": round(statistics.fmean(accs), 2),
            "acc_top1_std": round(statistics.stdev(accs) if len(accs) > 1 else 0.0, 2),
            "rmse_mean": round(statistics.fmean(rmses), 3),
            "rmse_std": round(statistics.stdev(rmses) if len(rmses) > 1 else 0.0, 3),
        }
        with open(out_root / f"{strategy}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        table_rows.append(
            {
                "Method": strategy,
                "Acc_mean": summary["acc_top1_mean"],
                "Acc_std": summary["acc_top1_std"],
                "rMSE_mean": summary["rmse_mean"],
                "rMSE_std": summary["rmse_std"],
            }
        )

    with open(out_root / "table11_multiseed.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Acc_mean", "Acc_std", "rMSE_mean", "rMSE_std"])
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"[done] Table 11 aggregation written to {out_root}")


if __name__ == "__main__":
    main()
