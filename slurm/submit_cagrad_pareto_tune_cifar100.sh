#!/usr/bin/env bash
# CAGrad Pareto tuning on CIFAR-100.
# Keep gradient decoupling setting fixed:
#   grad_norm_mode=mean, grad_norm_scope=conflict_deep, grad_norm_layers=layer3+layer4
# Sweep:
#   cagrad_beta, lambda_txt (lambda_rec fixed to 1.0)

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/unirae_radio/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_cagrad_pareto_tune}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
LAYERS=${LAYERS:-layer3+layer4}
NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_deep}
LAMBDA_REC=${LAMBDA_REC:-1.0}

# (beta, lambda_txt)
GRID=(
  "0.20 1.00"
  "0.35 1.00"
  "0.50 1.00"
  "0.65 1.00"
  "0.35 1.10"
  "0.50 1.10"
  "0.35 1.20"
  "0.50 1.20"
)

submit_one () {
  local beta="$1"
  local ltxt="$2"

  local beta_tag
  local ltxt_tag
  beta_tag=$(echo "${beta}" | tr '.' 'p')
  ltxt_tag=$(echo "${ltxt}" | tr '.' 'p')
  local run_name="${RUN_PREFIX}_b${beta_tag}_ltxt${ltxt_tag}_s${SEED}"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.mode=joint --set train.grad_strategy=cagrad --set train.cagrad_beta=${beta} --set train.lambda_txt=${ltxt} --set train.lambda_rec=${LAMBDA_REC} --set train.grad_norm_mode=${NORM_MODE} --set train.grad_norm_scope=${NORM_SCOPE} --set train.grad_norm_layers=${LAYERS}"
  sbatch \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "configs/cifar100_baseline_joint_cagrad.yaml" "${run_name}"
}

for item in "${GRID[@]}"; do
  # shellcheck disable=SC2086
  submit_one ${item}
done

echo "[submit_cagrad_pareto_tune_cifar100] submitted seed=${SEED} steps=${STEPS} grid=${#GRID[@]} layers=${LAYERS}"
