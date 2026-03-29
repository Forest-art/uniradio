#!/usr/bin/env bash
# Compare fixed-beta CAGrad vs adaptive-beta CAGrad on CIFAR-100.
# Also run naive as context reference.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_cagrad_adaptivebeta_cmp}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
LAYERS=${LAYERS:-layer3+layer4}
NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_deep}
NUM_WORKERS=${NUM_WORKERS:-4}
BETA_BASE=${BETA_BASE:-0.35}
LAMBDA_TXT=${LAMBDA_TXT:-1.0}

submit_one () {
  local cfg="$1"
  local run_name="$2"
  local strategy="$3"
  local adaptive="$4"
  local dep_job="$5"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set data.num_workers=${NUM_WORKERS} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.mode=joint --set train.grad_strategy=${strategy} --set train.cagrad_beta=${BETA_BASE} --set train.lambda_txt=${LAMBDA_TXT} --set train.lambda_rec=1.0 --set train.grad_norm_mode=${NORM_MODE} --set train.grad_norm_scope=${NORM_SCOPE} --set train.grad_norm_layers=${LAYERS} --set train.cagrad_adaptive_beta=${adaptive} --set train.cagrad_adaptive_scope=deep --set train.cagrad_adaptive_layers=${LAYERS} --set train.cagrad_adaptive_nonconflict_merge=sum"

  local dep_args=()
  if [[ -n "${dep_job}" ]]; then
    dep_args=(--dependency="afterany:${dep_job}")
  fi

  local out
  out=$(sbatch \
    "${dep_args[@]}" \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "${cfg}" "${run_name}")
  local job_id
  job_id=$(echo "${out}" | awk '{print $4}')
  echo "${job_id}"
}

prev_job=""
jid=$(submit_one "configs/cifar100_baseline_joint_naive.yaml"  "${RUN_PREFIX}_naive_ref_s${SEED}"         "naive"  "false" "${prev_job}")
echo "[submit_cagrad_adaptivebeta_compare_cifar100] submitted ${jid} (naive_ref)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_cagrad_fixed_b${BETA_BASE}_s${SEED}" "cagrad" "false" "${prev_job}")
echo "[submit_cagrad_adaptivebeta_compare_cifar100] submitted ${jid} (cagrad_fixed)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_cagrad_adaptive_b${BETA_BASE}_s${SEED}" "cagrad" "true"  "${prev_job}")
echo "[submit_cagrad_adaptivebeta_compare_cifar100] submitted ${jid} (cagrad_adaptive)"
prev_job="${jid}"

echo "[submit_cagrad_adaptivebeta_compare_cifar100] done seed=${SEED} steps=${STEPS} beta_base=${BETA_BASE} lambda_txt=${LAMBDA_TXT}"
