#!/usr/bin/env bash
# CIFAR100 real runs:
# 1) naive fixed lambda (reference)
# 2) naive + dynamic Lu/Lg grad-norm balancing
# Both with full-layer probe logging for depth-cos correlation.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-10000}
SEEDS=${SEEDS:-"42 43 44"}
RUN_PREFIX=${RUN_PREFIX:-cifar100_balance_fullprobe_10k}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/peilab/xlubl/unirae_runs}
LOG_DIR=${LOG_DIR:-/scratch/peilab/xlubl/unirae_slurm_logs}

PROBE_EVERY=${PROBE_EVERY:-200}
PROBE_UNTIL=${PROBE_UNTIL:-10000}

BALANCE_EVERY=${BALANCE_EVERY:-20}
BALANCE_EMA=${BALANCE_EMA:-0.9}
BALANCE_POWER=${BALANCE_POWER:-1.0}
BALANCE_MIN=${BALANCE_MIN:-0.1}
BALANCE_MAX=${BALANCE_MAX:-100.0}

submit_one() {
  local cfg="$1"
  local run_name="$2"
  local seed="$3"
  local extra="$4"

  local common="--set seed=${seed} --set data.root=${DATA_ROOT} --set output.root=${OUTPUT_ROOT} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.probe_every=${PROBE_EVERY} --set train.probe_until=${PROBE_UNTIL}"
  sbatch \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${common} ${extra}" \
    slurm/sbatch_train.sh "${cfg}" "${run_name}"
}

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  # Fixed lambda baseline.
  submit_one \
    "configs/cifar100_baseline_joint_naive.yaml" \
    "${RUN_PREFIX}_naive_fixed_s${seed}" \
    "${seed}" \
    "--set train.grad_norm_balance_every=0"

  # Dynamic balanced lambda.
  submit_one \
    "configs/cifar100_baseline_joint_naive.yaml" \
    "${RUN_PREFIX}_naive_balance_s${seed}" \
    "${seed}" \
    "--set train.grad_norm_balance_every=${BALANCE_EVERY} --set train.grad_norm_balance_ema=${BALANCE_EMA} --set train.grad_norm_balance_power=${BALANCE_POWER} --set train.grad_norm_balance_min_scale=${BALANCE_MIN} --set train.grad_norm_balance_max_scale=${BALANCE_MAX}"
done

echo "[submit_cifar100_balance_fullprobe] submitted"
echo "  RUN_PREFIX=${RUN_PREFIX}"
echo "  STEPS=${STEPS}"
echo "  SEEDS=${SEEDS}"
echo "  PROBE_EVERY=${PROBE_EVERY}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  LOG_DIR=${LOG_DIR}"
