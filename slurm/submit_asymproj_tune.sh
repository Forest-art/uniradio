#!/usr/bin/env bash
set -euo pipefail

# Grid search for layer-wise asymmetric projection tuning.
# Focus: deeper conflict groups + stricter threshold + higher lambda_rec.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-diffuser310}"
DATA_ROOT="${DATA_ROOT:-/path/to/cifar10}"
STEPS="${STEPS:-5000}"
SEEDS="${SEEDS:-43}"
RUN_PREFIX="${RUN_PREFIX:-study_asymproj_tune}"
BASE_CFG="${BASE_CFG:-configs/cifar10_resnet18_layerasymproj_deep.yaml}"

# Candidate sets.
GROUPS_CSV="${GROUPS_CSV:-layer4;layer3,layer4;layer2,layer3,layer4}"
THRESHOLDS_CSV="${THRESHOLDS_CSV:--0.02,-0.05,-0.08}"
LREC_CSV="${LREC_CSV:-1.5,2.0}"
LTXT="${LTXT:-0.8}"

IFS=',' read -r -a seed_arr <<< "$SEEDS"
IFS=';' read -r -a groups_arr <<< "$GROUPS_CSV"
IFS=',' read -r -a th_arr <<< "$THRESHOLDS_CSV"
IFS=',' read -r -a lrec_arr <<< "$LREC_CSV"

submit_one() {
  local seed="$1"
  local groups="$2"
  local th="$3"
  local lrec="$4"

  local groups_tag="${groups//,/}"
  local th_tag="${th//-/m}"
  th_tag="${th_tag//./p}"
  local lrec_tag="${lrec//./p}"
  local run_name="${RUN_PREFIX}_g${groups_tag}_th${th_tag}_lt${LTXT}_lr${lrec_tag}_s${seed}"
  local run_dir="runs/${run_name}"

  if [[ -f "${run_dir}/cos_summary.json" ]]; then
    echo "[skip] ${run_name} already completed"
    return 0
  fi
  if [[ -f "${run_dir}/metrics.jsonl" ]]; then
    echo "[skip] ${run_name} already exists (in-progress or partial)"
    return 0
  fi

  local extra="--set seed=${seed} --set data.root=${DATA_ROOT} --set train.steps=${STEPS} --set accelerate.mixed_precision=no"
  extra="${extra} --set train.lambda_txt=${LTXT} --set train.lambda_rec=${lrec}"
  extra="${extra} --set train.conflict_threshold=${th} --set train.layer_cagrad_conflict_groups=${groups}"

  TRAIN_MODULE=unirae.train_cifar10 CONDA_ENV="$CONDA_ENV" EXTRA_ARGS="$extra" \
    sbatch slurm/sbatch_train.sh "$BASE_CFG" "$run_name"
}

for seed in "${seed_arr[@]}"; do
  for groups in "${groups_arr[@]}"; do
    for th in "${th_arr[@]}"; do
      for lrec in "${lrec_arr[@]}"; do
        submit_one "$seed" "$groups" "$th" "$lrec"
      done
    done
  done
done

echo "[submit_asymproj_tune] submitted seeds=${SEEDS} groups=${GROUPS_CSV} thresholds=${THRESHOLDS_CSV} lrec=${LREC_CSV} ltxt=${LTXT}"
