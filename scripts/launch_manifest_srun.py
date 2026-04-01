#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, TextIO


@dataclass
class ActiveProcess:
    row: Dict[str, str]
    run_dir: Path
    stdout_handle: TextIO
    stderr_handle: TextIO
    stdout_path: Path
    stderr_path: Path
    proc: subprocess.Popen[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch manifest rows through reusable srun hold allocations.")
    parser.add_argument("--manifest", required=True, help="TSV manifest with run_name and launch_cmd columns.")
    parser.add_argument("--runs_root", default="runs", help="Root directory containing run folders.")
    parser.add_argument("--reuse_script", default="scripts/reuse_or_start_srun_train.sh", help="Path to hold-allocation launcher script.")
    parser.add_argument("--launcher_log_dir", default="", help="Directory for per-run launcher stdout/stderr.")
    parser.add_argument("--max_parallel", type=int, default=1, help="Maximum concurrent launcher processes.")
    parser.add_argument("--poll_seconds", type=float, default=20.0, help="Polling interval for active launchers.")
    parser.add_argument("--start_delay_seconds", type=float, default=5.0, help="Delay after starting each launcher.")
    parser.add_argument("--resume", action="store_true", help="Skip rows whose eval_last.json already exists.")
    parser.add_argument("--archive_stale_runs", action="store_true", default=True, help="Archive incomplete run dirs before relaunch.")
    parser.add_argument("--dry_run", action="store_true", help="Print launch plan without starting launchers.")
    return parser.parse_args()


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [dict(row) for row in reader]
    return rows


def _is_completed_run(run_dir: Path) -> bool:
    eval_last = run_dir / "eval_last.json"
    cifar_ckpt = run_dir / "checkpoints" / "latest.pt"
    in100_ckpt = run_dir / "latest.pt"
    cifar_done = (run_dir / "cos_summary.json").exists() and cifar_ckpt.exists()
    in100_done = eval_last.exists() and in100_ckpt.exists()
    return cifar_done or in100_done


def _write_event(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
        f.write("\n")


def _archive_stale_run_dir(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None
    if _is_completed_run(run_dir):
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = run_dir.with_name(f"{run_dir.name}_stale_{stamp}")
    suffix = 0
    while archived.exists():
        suffix += 1
        archived = run_dir.with_name(f"{run_dir.name}_stale_{stamp}_{suffix}")
    run_dir.rename(archived)
    return archived


def _status_for_run(run_dir: Path, return_code: int) -> str:
    if return_code == 0 and _is_completed_run(run_dir):
        return "ok"
    return "failed"


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest).resolve()
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (repo_root / runs_root).resolve()
    reuse_script = Path(args.reuse_script)
    if not reuse_script.is_absolute():
        reuse_script = (repo_root / reuse_script).resolve()
    log_dir = Path(args.launcher_log_dir) if args.launcher_log_dir else manifest_path.parent / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_log = log_dir / "launcher_state.jsonl"

    rows = _read_manifest(manifest_path)
    if not rows:
        print(f"[done] no rows in manifest: {manifest_path}")
        return 0
    required = {"run_name", "launch_cmd"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    pending: Deque[Dict[str, str]] = deque()
    skipped = 0
    for row in rows:
        run_name = str(row["run_name"]).strip()
        launch_cmd = str(row["launch_cmd"]).strip()
        if not run_name:
            raise ValueError("Manifest row has empty run_name.")
        if not launch_cmd:
            raise ValueError(f"Manifest row {run_name} has empty launch_cmd.")

        run_dir = runs_root / run_name
        if args.resume and _is_completed_run(run_dir):
            skipped += 1
            _write_event(
                state_log,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "status": "skipped_completed",
                    "run_dir": str(run_dir),
                },
            )
            continue
        pending.append(row)

    if skipped:
        print(f"[skip] completed rows already present: {skipped}")
    if not pending:
        print("[done] nothing to launch")
        return 0

    active: List[ActiveProcess] = []
    launched = 0
    finished = 0
    failed = 0

    while pending or active:
        while pending and len(active) < args.max_parallel:
            row = pending.popleft()
            run_name = str(row["run_name"]).strip()
            run_dir = runs_root / run_name
            archived_path = None
            if args.archive_stale_runs:
                archived_path = _archive_stale_run_dir(run_dir)

            if args.dry_run:
                print(f"[dry-run] {run_name}")
                _write_event(
                    state_log,
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "run_name": run_name,
                        "status": "dry_run",
                        "run_dir": str(run_dir),
                        "archived_stale_run_dir": str(archived_path) if archived_path else "",
                    },
                )
                launched += 1
                finished += 1
                continue

            stdout_path = log_dir / f"{run_name}.launcher.out"
            stderr_path = log_dir / f"{run_name}.launcher.err"
            stdout_handle = stdout_path.open("a", encoding="utf-8")
            stderr_handle = stderr_path.open("a", encoding="utf-8")
            proc = subprocess.Popen(
                ["bash", str(reuse_script), "--", row["launch_cmd"]],
                cwd=str(repo_root),
                env=os.environ.copy(),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            active.append(
                ActiveProcess(
                    row=row,
                    run_dir=run_dir,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    proc=proc,
                )
            )
            launched += 1
            print(f"[launch] {run_name} pid={proc.pid}")
            _write_event(
                state_log,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "status": "launched",
                    "run_dir": str(run_dir),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "archived_stale_run_dir": str(archived_path) if archived_path else "",
                    "pid": proc.pid,
                },
            )
            if args.start_delay_seconds > 0:
                time.sleep(args.start_delay_seconds)

        if not active:
            continue

        time.sleep(args.poll_seconds)
        next_active: List[ActiveProcess] = []
        for item in active:
            return_code = item.proc.poll()
            if return_code is None:
                next_active.append(item)
                continue

            item.stdout_handle.close()
            item.stderr_handle.close()
            finished += 1
            status = _status_for_run(item.run_dir, return_code)
            if status != "ok":
                failed += 1
            run_name = str(item.row["run_name"]).strip()
            print(f"[{status}] {run_name} rc={return_code}")
            _write_event(
                state_log,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "status": status,
                    "run_dir": str(item.run_dir),
                    "stdout_path": str(item.stdout_path),
                    "stderr_path": str(item.stderr_path),
                    "return_code": int(return_code),
                    "eval_last_exists": (item.run_dir / "eval_last.json").exists(),
                },
            )
        active = next_active

    print(
        f"[done] launched={launched} finished={finished} failed={failed} "
        f"manifest={manifest_path} log_dir={log_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
