#!/usr/bin/env bash
# Fair comparison on CIFAR-100:
# strategies = naive | pcgrad | cagrad
# baseline: grad_norm_mode=none
# variant : grad_norm_mode=mean + grad_norm_scope=conflict_deep + grad_norm_layers=layer3+layer4

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/unirae_radio/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_gradnorm_conflictdeep_cmp}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
# Use '+' delimiter because sbatch --export uses comma as a separator.
LAYERS=${LAYERS:-layer3+layer4}

submit_one () {
  local cfg="$1"
  local run_name="$2"
  local norm_mode="$3"
  local norm_scope="$4"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.grad_norm_mode=${norm_mode} --set train.grad_norm_scope=${norm_scope} --set train.grad_norm_layers=${LAYERS}"
  sbatch \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "${cfg}" "${run_name}"
}

submit_one "configs/cifar100_baseline_joint_naive.yaml"  "${RUN_PREFIX}_joint_naive_normnone_s${SEED}"  "none" "all"
submit_one "configs/cifar100_baseline_joint_pcgrad.yaml" "${RUN_PREFIX}_joint_pcgrad_normnone_s${SEED}" "none" "all"
submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_joint_cagrad_normnone_s${SEED}" "none" "all"

submit_one "configs/cifar100_baseline_joint_naive.yaml"  "${RUN_PREFIX}_joint_naive_normmean_conflictdeep_s${SEED}"  "mean" "conflict_deep"
submit_one "configs/cifar100_baseline_joint_pcgrad.yaml" "${RUN_PREFIX}_joint_pcgrad_normmean_conflictdeep_s${SEED}" "mean" "conflict_deep"
submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_joint_cagrad_normmean_conflictdeep_s${SEED}" "mean" "conflict_deep"

echo "[submit_gradnorm_conflictdeep_compare_cifar100] submitted seed=${SEED} steps=${STEPS} layers=${LAYERS} prefix=${RUN_PREFIX}"
