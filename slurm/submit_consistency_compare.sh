#!/usr/bin/env bash
set -euo pipefail

# One-click submit for consistency ablations:
# 1) baseline (cons=0, naive)
# 2) +consistency (naive)
# 3) +consistency (cagrad, merge into understanding)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-diffuser310}"
DATA_ROOT="${DATA_ROOT:-/path/to/cifar10}"
STEPS="${STEPS:-5000}"
SEEDS="${SEEDS:-42,43,44}"
RUN_PREFIX="${RUN_PREFIX:-study_cifar10_cons}"

IFS=',' read -r -a seed_arr <<< "$SEEDS"

submit_one() {
  local cfg="$1"
  local run_name="$2"
  local seed="$3"
  local extra="--set seed=${seed} --set data.root=${DATA_ROOT} --set train.steps=${STEPS} --set accelerate.mixed_precision=no"
  TRAIN_MODULE=unirae.train_cifar10 CONDA_ENV="$CONDA_ENV" EXTRA_ARGS="$extra" \
    sbatch slurm/sbatch_train.sh "$cfg" "$run_name"
}

for seed in "${seed_arr[@]}"; do
  submit_one "configs/cifar10_cons_baseline.yaml" "${RUN_PREFIX}_cons0_naive_s${seed}" "$seed"
  submit_one "configs/cifar10_cons_naive.yaml" "${RUN_PREFIX}_cons1_naive_s${seed}" "$seed"
  submit_one "configs/cifar10_cons_cagrad.yaml" "${RUN_PREFIX}_cons1_cagrad_s${seed}" "$seed"
done

echo "[submit_consistency_compare] submitted for seeds: ${SEEDS}"
