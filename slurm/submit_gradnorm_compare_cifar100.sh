#!/usr/bin/env bash
# Fair comparison on CIFAR-100:
# strategies = naive | pcgrad | cagrad
# grad_norm_mode = none | mean
# All runs share the same seed / steps / data root.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-5000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_gradnorm_cmp}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}

submit_one () {
  local cfg="$1"
  local run_name="$2"
  local grad_norm_mode="$3"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.grad_norm_mode=${grad_norm_mode}"
  sbatch \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "${cfg}" "${run_name}"
}

submit_one "configs/cifar100_baseline_joint_naive.yaml"  "${RUN_PREFIX}_joint_naive_normnone_s${SEED}"  "none"
submit_one "configs/cifar100_baseline_joint_pcgrad.yaml" "${RUN_PREFIX}_joint_pcgrad_normnone_s${SEED}" "none"
submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_joint_cagrad_normnone_s${SEED}" "none"

submit_one "configs/cifar100_baseline_joint_naive.yaml"  "${RUN_PREFIX}_joint_naive_normmean_s${SEED}"  "mean"
submit_one "configs/cifar100_baseline_joint_pcgrad.yaml" "${RUN_PREFIX}_joint_pcgrad_normmean_s${SEED}" "mean"
submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_joint_cagrad_normmean_s${SEED}" "mean"

echo "[submit_gradnorm_compare_cifar100] submitted seed=${SEED} steps=${STEPS} prefix=${RUN_PREFIX}"
