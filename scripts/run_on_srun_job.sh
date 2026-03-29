#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${JOB_ID:-}" ]]; then
  echo "[error] JOB_ID is not set." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: JOB_ID=<slurm_jobid> bash scripts/run_on_srun_job.sh -- \"<TRAIN_CMD>\"" >&2
  exit 1
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

TRAIN_CMD="$*"
if [[ -z "${TRAIN_CMD// }" ]]; then
  echo "[error] TRAIN_CMD is empty." >&2
  exit 1
fi

CPUS_PER_TASK=${CPUS_PER_TASK:-8}

module load slurm >/dev/null 2>&1 || true
srun --jobid "${JOB_ID}" --overlap \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  bash -lc "${TRAIN_CMD}"
