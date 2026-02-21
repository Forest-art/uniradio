#!/usr/bin/env bash
set -euo pipefail

# Submit CIFAR-10 SupCon comparisons:
# 1) baseline (lambda_sup=0)
# 2) naive + supcon
# 3) cagrad + supcon (merge_into_understanding=true)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-diffuser310}"
DATA_ROOT="${DATA_ROOT:-/path/to/cifar10}"
STEPS="${STEPS:-5000}"
SEEDS="${SEEDS:-42,43,44}"
LAMSUP="${LAMSUP:-0.1,0.5,1.0}"
RUN_PREFIX="${RUN_PREFIX:-study_cifar10_supcon}"

IFS=',' read -r -a seed_arr <<< "$SEEDS"
IFS=',' read -r -a lam_arr <<< "$LAMSUP"

submit_one() {
  local cfg="$1"
  local run_name="$2"
  local seed="$3"
  local lam_sup="$4"
  local extra="--set seed=${seed} --set data.root=${DATA_ROOT} --set train.steps=${STEPS} --set train.lambda_sup=${lam_sup} --set supcon.lambda=${lam_sup} --set accelerate.mixed_precision=no"
  TRAIN_MODULE=unirae.train_cifar10 CONDA_ENV="$CONDA_ENV" EXTRA_ARGS="$extra" \
    sbatch slurm/sbatch_train.sh "$cfg" "$run_name"
}

for seed in "${seed_arr[@]}"; do
  submit_one "configs/cifar10_supcon_baseline.yaml" "${RUN_PREFIX}_supcon0_cagrad0_lamsup0_s${seed}" "$seed" "0.0"
  for lam in "${lam_arr[@]}"; do
    lam_tag="${lam//./p}"
    submit_one "configs/cifar10_supcon_naive.yaml" "${RUN_PREFIX}_supcon1_cagrad0_lamsup${lam_tag}_s${seed}" "$seed" "$lam"
    submit_one "configs/cifar10_supcon_cagrad.yaml" "${RUN_PREFIX}_supcon1_cagrad1_lamsup${lam_tag}_s${seed}" "$seed" "$lam"
  done
done

echo "[submit_supcon_compare] submitted seeds=${SEEDS} lamsup=${LAMSUP}"
