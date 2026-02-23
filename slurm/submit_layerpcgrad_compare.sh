#!/usr/bin/env bash
set -euo pipefail

# Compare naive / cagrad / layer_pcgrad on CIFAR-10 ResNet-18.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-diffuser310}"
DATA_ROOT="${DATA_ROOT:-/path/to/cifar10}"
STEPS="${STEPS:-5000}"
SEEDS="${SEEDS:-42,43,44}"
RUN_PREFIX="${RUN_PREFIX:-study_layerpcgrad}"

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
  submit_one "configs/cifar10_resnet18_naive.yaml" "${RUN_PREFIX}_naive_s${seed}" "$seed"
  submit_one "configs/cifar10_resnet18_cagrad.yaml" "${RUN_PREFIX}_cagrad_s${seed}" "$seed"
  submit_one "configs/cifar10_resnet18_layerpcgrad.yaml" "${RUN_PREFIX}_layerpcgrad_s${seed}" "$seed"
done

echo "[submit_layerpcgrad_compare] submitted seeds=${SEEDS}"
