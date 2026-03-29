#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize IN100 eval_last.json from final_eval.json or eval_metrics.jsonl.")
    parser.add_argument("--run-dir", required=True, help="Per-run output directory.")
    parser.add_argument(
        "--prefer-final",
        action="store_true",
        default=True,
        help="Prefer final_eval.json when present.",
    )
    parser.add_argument(
        "--no-prefer-final",
        dest="prefer_final",
        action="store_false",
        help="Use the last eval_metrics.jsonl row even if final_eval.json exists.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_last_jsonl(path: Path) -> Dict[str, Any]:
    last: Dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    if last is None:
        raise FileNotFoundError(f"No valid JSON rows found in {path}")
    return last


def _to_float(obj: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = obj.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_eval_last(payload: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    val_top1_acc = _to_float(payload, "val_top1_acc")
    val_mse = _to_float(payload, "val_mse")
    val_rmse = _to_float(payload, "val_rmse")
    val_rfid = _to_float(payload, "val_rfid")

    out: Dict[str, Any] = dict(payload)
    out["source_file"] = source_name
    out["understanding"] = {
        "acc_txt": val_top1_acc,
        "zero_shot_loss": float("nan"),
    }
    out["generation"] = {
        "recon_rmse": val_rmse,
        "recon_mse": val_mse,
        "psnr": float("nan"),
        "rfid": val_rfid,
    }
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir does not exist: {run_dir}")

    final_eval_path = run_dir / "final_eval.json"
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    eval_last_path = run_dir / "eval_last.json"

    if args.prefer_final and final_eval_path.exists():
        payload = _load_json(final_eval_path)
        source_name = final_eval_path.name
    elif eval_metrics_path.exists():
        payload = _load_last_jsonl(eval_metrics_path)
        source_name = eval_metrics_path.name
    elif final_eval_path.exists():
        payload = _load_json(final_eval_path)
        source_name = final_eval_path.name
    else:
        raise FileNotFoundError(
            f"Neither final_eval.json nor eval_metrics.jsonl exists under {run_dir}"
        )

    out = _build_eval_last(payload, source_name=source_name)
    eval_last_path.write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {eval_last_path}")


if __name__ == "__main__":
    main()
