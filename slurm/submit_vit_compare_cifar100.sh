#!/usr/bin/env bash
# Transformer comparison on CIFAR-100 (default: patch4 backbone):
# naive / pcgrad / cagrad / cagrad+varreg

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_vit_cmp}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}

BACKBONE=${BACKBONE:-swin_tiny_patch4}
IMAGE_SIZE=${IMAGE_SIZE:-32}
PRETRAINED=${PRETRAINED:-false}

NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_all}
NORM_LAYERS=${NORM_LAYERS:-layer3+layer4}
LAMBDA_TXT=${LAMBDA_TXT:-1.0}
LAMBDA_REC=${LAMBDA_REC:-1.0}
CAGRAD_BETA=${CAGRAD_BETA:-0.35}

LAMBDA_VAR=${LAMBDA_VAR:-0.20}
VAR_GAMMA=${VAR_GAMMA:-1.0}
VAR_EPS=${VAR_EPS:-1e-4}

tag_float () {
  local x="$1"
  local neg=""
  if [[ "${x}" == -* ]]; then
    neg="m"
    x="${x#-}"
  fi
  echo "${neg}$(echo "${x}" | tr '.' 'p')"
}

submit_one () {
  local cfg="$1"
  local run_name="$2"
  local strategy="$3"
  local dep_job="$4"
  local extra_more="${5:-}"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set data.batch_size=${BATCH_SIZE} --set data.num_workers=${NUM_WORKERS} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set model.backbone=${BACKBONE} --set model.pretrained=${PRETRAINED} --set data.image_size=${IMAGE_SIZE} --set train.mode=joint --set train.grad_strategy=${strategy} --set train.lambda_txt=${LAMBDA_TXT} --set train.lambda_rec=${LAMBDA_REC} --set train.grad_norm_mode=${NORM_MODE} --set train.grad_norm_scope=${NORM_SCOPE} --set train.grad_norm_layers=${NORM_LAYERS} --set train.cagrad_beta=${CAGRAD_BETA} ${extra_more}"

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
jid=$(submit_one "configs/cifar100_baseline_joint_naive.yaml" "${RUN_PREFIX}_naive_s${SEED}" "naive" "${prev_job}")
echo "[submit_vit_compare_cifar100] submitted ${jid} (naive)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_pcgrad.yaml" "${RUN_PREFIX}_pcgrad_s${SEED}" "pcgrad" "${prev_job}")
echo "[submit_vit_compare_cifar100] submitted ${jid} (pcgrad)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_cagrad_b$(tag_float ${CAGRAD_BETA})_s${SEED}" "cagrad" "${prev_job}" "--set train.lambda_var=0.0")
echo "[submit_vit_compare_cifar100] submitted ${jid} (cagrad)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_cagrad_var_l$(tag_float ${LAMBDA_VAR})_s${SEED}" "cagrad" "${prev_job}" "--set train.lambda_var=${LAMBDA_VAR} --set train.var_gamma=${VAR_GAMMA} --set train.var_eps=${VAR_EPS}")
echo "[submit_vit_compare_cifar100] submitted ${jid} (cagrad_var)"
prev_job="${jid}"

echo "[submit_vit_compare_cifar100] done seed=${SEED} steps=${STEPS} backbone=${BACKBONE} batch_size=${BATCH_SIZE} lambda_var=${LAMBDA_VAR}"
