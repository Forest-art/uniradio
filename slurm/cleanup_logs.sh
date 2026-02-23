#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARCHIVE_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
ARCHIVE_DIR="slurm/logs/archive_${ARCHIVE_TAG}"
mkdir -p "$ARCHIVE_DIR"

shopt -s nullglob
files=(slurm/logs/*.out slurm/logs/*.err)
if [ ${#files[@]} -eq 0 ]; then
  echo "[cleanup_logs] no logs to archive."
  exit 0
fi

mv "${files[@]}" "$ARCHIVE_DIR"/
echo "[cleanup_logs] moved ${#files[@]} files to $ARCHIVE_DIR"
