#!/usr/bin/env bash
# CIFAR-100 fair comparison:
# naive vs pcgrad vs cagrad(best fixed) vs saop

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_saop_cmp}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
LAYERS=${LAYERS:-layer3+layer4}
NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_deep}
NUM_WORKERS=${NUM_WORKERS:-4}
LAMBDA_TXT=${LAMBDA_TXT:-1.0}
LAMBDA_REC=${LAMBDA_REC:-1.0}
CAGRAD_BETA=${CAGRAD_BETA:-0.35}
SAOP_EPS=${SAOP_EPS:-1e-8}

submit_one () {
  local cfg="$1"
  local run_name="$2"
  local strategy="$3"
  local dep_job="$4"
  local extra_more="${5:-}"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set data.num_workers=${NUM_WORKERS} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.mode=joint --set train.grad_strategy=${strategy} --set train.lambda_txt=${LAMBDA_TXT} --set train.lambda_rec=${LAMBDA_REC} --set train.grad_norm_mode=${NORM_MODE} --set train.grad_norm_scope=${NORM_SCOPE} --set train.grad_norm_layers=${LAYERS} ${extra_more}"

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
jid=$(submit_one "configs/cifar100_baseline_joint_naive.yaml" "${RUN_PREFIX}_naive_s${SEED}" "naive" "${prev_job}" "")
echo "[submit_saop_compare_cifar100] submitted ${jid} (naive)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_pcgrad.yaml" "${RUN_PREFIX}_pcgrad_s${SEED}" "pcgrad" "${prev_job}" "")
echo "[submit_saop_compare_cifar100] submitted ${jid} (pcgrad)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_cagrad_b${CAGRAD_BETA}_s${SEED}" "cagrad" "${prev_job}" "--set train.cagrad_beta=${CAGRAD_BETA}")
echo "[submit_saop_compare_cifar100] submitted ${jid} (cagrad)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_saop_s${SEED}" "saop" "${prev_job}" "--set train.saop_scope=deep --set train.saop_layers=${LAYERS} --set train.saop_eps=${SAOP_EPS} --set train.saop_log_norm_ratio=false")
echo "[submit_saop_compare_cifar100] submitted ${jid} (saop)"
prev_job="${jid}"

echo "[submit_saop_compare_cifar100] done seed=${SEED} steps=${STEPS} beta=${CAGRAD_BETA} layers=${LAYERS}"
