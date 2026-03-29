#!/usr/bin/env bash
# SUN397 + DINOv2 encoder compare:
# naive / pcgrad / cagrad (seed fixed for fair quick verification)

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/sun397}
HF_DATASET=${HF_DATASET:-dpdl-benchmark/sun397}
HF_CACHE_DIR=${HF_CACHE_DIR:-/project/peilab/luxiaocheng/projects/DSGA/data/sun397/hf_cache}
STEPS=${STEPS:-5000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-sun397_dino_cmp}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
NUM_WORKERS=${NUM_WORKERS:-8}

IMAGE_SIZE=${IMAGE_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-40}
RECON_SIZE=${RECON_SIZE:-64}
LR=${LR:-1e-4}

BACKBONE=${BACKBONE:-dinov2_vits14}
PRETRAINED=${PRETRAINED:-true}
CAGRAD_BETA=${CAGRAD_BETA:-0.35}
LAMBDA_TXT=${LAMBDA_TXT:-1.0}
LAMBDA_REC=${LAMBDA_REC:-1.0}
NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_all}
NORM_LAYERS=${NORM_LAYERS:-layer3+layer4}

SUN_MAX_TRAIN_SAMPLES=${SUN_MAX_TRAIN_SAMPLES:-0}
SUN_MAX_EVAL_SAMPLES=${SUN_MAX_EVAL_SAMPLES:-0}

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
  local run_name="$1"
  local strategy="$2"
  local dep_job="$3"
  local extra_more="${4:-}"

  local extra="--set seed=${SEED} \
    --set data.dataset=sun397 \
    --set data.root=${DATA_ROOT} \
    --set data.sun_source=hf \
    --set data.sun_hf_dataset=${HF_DATASET} \
    --set data.sun_hf_cache_dir=${HF_CACHE_DIR} \
    --set data.sun_hf_image_key=image \
    --set data.sun_hf_label_key=label \
    --set data.sun_max_train_samples=${SUN_MAX_TRAIN_SAMPLES} \
    --set data.sun_max_eval_samples=${SUN_MAX_EVAL_SAMPLES} \
    --set data.image_size=${IMAGE_SIZE} \
    --set data.batch_size=${BATCH_SIZE} \
    --set data.num_workers=${NUM_WORKERS} \
    --set model.backbone=${BACKBONE} \
    --set model.pretrained=${PRETRAINED} \
    --set model.recon_size=${RECON_SIZE} \
    --set train.steps=${STEPS} \
    --set train.mode=joint \
    --set train.grad_strategy=${strategy} \
    --set train.lambda_txt=${LAMBDA_TXT} \
    --set train.lambda_rec=${LAMBDA_REC} \
    --set train.cagrad_beta=${CAGRAD_BETA} \
    --set train.grad_norm_mode=${NORM_MODE} \
    --set train.grad_norm_scope=${NORM_SCOPE} \
    --set train.grad_norm_layers=${NORM_LAYERS} \
    --set optim.lr=${LR} \
    --set eval.batch_size=${EVAL_BATCH_SIZE} \
    --set eval.max_batches=${EVAL_MAX_BATCHES} \
    --set accelerate.mixed_precision=no \
    ${extra_more}"

  local dep_args=()
  if [[ -n "${dep_job}" ]]; then
    dep_args=(--dependency="afterany:${dep_job}")
  fi

  local out
  out=$(sbatch \
    "${dep_args[@]}" \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "configs/sun397_dino_joint_base.yaml" "${run_name}")
  local job_id
  job_id=$(echo "${out}" | awk '{print $4}')
  echo "${job_id}"
}

prev_job=""
jid=$(submit_one "${RUN_PREFIX}_naive_s${SEED}" "naive" "${prev_job}")
echo "[submit_dino_sun397_compare] submitted ${jid} (naive)"
prev_job="${jid}"

jid=$(submit_one "${RUN_PREFIX}_pcgrad_s${SEED}" "pcgrad" "${prev_job}")
echo "[submit_dino_sun397_compare] submitted ${jid} (pcgrad)"
prev_job="${jid}"

jid=$(submit_one "${RUN_PREFIX}_cagrad_b$(tag_float ${CAGRAD_BETA})_s${SEED}" "cagrad" "${prev_job}" "--set train.lambda_var=0.0")
echo "[submit_dino_sun397_compare] submitted ${jid} (cagrad)"
prev_job="${jid}"

echo "[submit_dino_sun397_compare] done seed=${SEED} steps=${STEPS} backbone=${BACKBONE} hf_dataset=${HF_DATASET}"
