#!/usr/bin/env bash
# Submit the 5 baseline groups on CIFAR-100:
# joint_naive, joint_pcgrad, joint_cagrad, text_only, recon_only.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/unirae_radio/data/cifar100}
STEPS=${STEPS:-5000}
SEEDS=${SEEDS:-42}
RUN_PREFIX=${RUN_PREFIX:-baseline5_cifar100}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}

IFS=',' read -r -a SEED_ARR <<<"${SEEDS}"

submit_one () {
  local cfg="$1"
  local run_name="$2"
  local seed="$3"
  local extra="--set seed=${seed} --set data.root=${DATA_ROOT} --set train.steps=${STEPS} --set accelerate.mixed_precision=no"
  sbatch \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "${cfg}" "${run_name}"
}

for seed in "${SEED_ARR[@]}"; do
  submit_one "configs/cifar100_baseline_joint_naive.yaml"  "${RUN_PREFIX}_joint_naive_s${seed}"  "${seed}"
  submit_one "configs/cifar100_baseline_joint_pcgrad.yaml" "${RUN_PREFIX}_joint_pcgrad_s${seed}" "${seed}"
  submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_joint_cagrad_s${seed}" "${seed}"
  submit_one "configs/cifar100_baseline_text_only.yaml"    "${RUN_PREFIX}_text_only_s${seed}"    "${seed}"
  submit_one "configs/cifar100_baseline_recon_only.yaml"   "${RUN_PREFIX}_recon_only_s${seed}"   "${seed}"
done

echo "[submit_baselines_cifar100] submitted seeds=${SEEDS} steps=${STEPS}"
