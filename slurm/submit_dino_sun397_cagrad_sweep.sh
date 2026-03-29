#!/usr/bin/env bash
# SUN397 + DINOv2 full cagrad parameter sweep (seed fixed).

set -euo pipefail

DATE_TAG=$(date +%Y%m%d)

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/sun397}
HF_DATASET=${HF_DATASET:-dpdl-benchmark/sun397}
HF_CACHE_DIR=${HF_CACHE_DIR:-/project/peilab/luxiaocheng/projects/DSGA/data/sun397/hf_cache}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-sun397_dino_cagrad_sweep_${DATE_TAG}}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
NUM_WORKERS=${NUM_WORKERS:-8}

IMAGE_SIZE=${IMAGE_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-null}
RECON_SIZE=${RECON_SIZE:-64}
LR=${LR:-1e-4}

BACKBONE=${BACKBONE:-dinov2_vits14}
PRETRAINED=${PRETRAINED:-true}
LAMBDA_REC=${LAMBDA_REC:-1.0}
NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_all}
NORM_LAYERS=${NORM_LAYERS:-layer3+layer4}

BETA_LIST=${BETA_LIST:-"0.10 0.20 0.30 0.35 0.50"}
LAMBDA_TXT_LIST=${LAMBDA_TXT_LIST:-"1.0 1.1 1.2"}

SUN_MAX_TRAIN_SAMPLES=${SUN_MAX_TRAIN_SAMPLES:-0}
SUN_MAX_EVAL_SAMPLES=${SUN_MAX_EVAL_SAMPLES:-0}

RESULT_DIR="results/${RUN_PREFIX}"
mkdir -p "${RESULT_DIR}"
MANIFEST="${RESULT_DIR}/submitted_jobs.csv"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "job_id,run_name,strategy,beta,lambda_txt,lambda_rec,seed,steps" > "${MANIFEST}"
fi

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
  local beta="$3"
  local lambda_txt="$4"

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
    --set train.lambda_txt=${lambda_txt} \
    --set train.lambda_rec=${LAMBDA_REC} \
    --set train.lambda_var=0.0 \
    --set train.grad_norm_mode=${NORM_MODE} \
    --set train.grad_norm_scope=${NORM_SCOPE} \
    --set train.grad_norm_layers=${NORM_LAYERS} \
    --set optim.lr=${LR} \
    --set eval.batch_size=${EVAL_BATCH_SIZE} \
    --set eval.max_batches=${EVAL_MAX_BATCHES} \
    --set accelerate.mixed_precision=no"

  if [[ "${strategy}" == "cagrad" ]]; then
    extra="${extra} --set train.cagrad_beta=${beta}"
  fi

  local out
  if ! out=$(sbatch \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "configs/sun397_dino_joint_base.yaml" "${run_name}" 2>&1); then
    echo "${out}" >&2
    return 1
  fi
  local job_id
  job_id=$(echo "${out}" | awk '{print $4}')
  if [[ -z "${job_id}" ]]; then
    echo "[submit_dino_sun397_cagrad_sweep] failed to parse job id: ${out}" >&2
    return 1
  fi
  echo "${job_id}"
}

already_submitted () {
  local run_name="$1"
  awk -F',' -v rn="${run_name}" 'NR>1 && $2==rn && $1!="" {found=1} END {exit !found}' "${MANIFEST}"
}

try_submit () {
  local run_name="$1"
  local strategy="$2"
  local beta="$3"
  local lambda_txt="$4"

  if already_submitted "${run_name}"; then
    echo "[submit_dino_sun397_cagrad_sweep] skip existing ${run_name}"
    return 0
  fi

  local jid=""
  if jid=$(submit_one "${run_name}" "${strategy}" "${beta}" "${lambda_txt}"); then
    echo "[submit_dino_sun397_cagrad_sweep] submitted ${jid} ${run_name}"
    echo "${jid},${run_name},${strategy},${beta},${lambda_txt},${LAMBDA_REC},${SEED},${STEPS}" >> "${MANIFEST}"
    return 0
  fi

  echo "[submit_dino_sun397_cagrad_sweep] defer ${run_name} (submit failed, rerun script later)"
  return 1
}

naive_run="${RUN_PREFIX}_naive_s${SEED}"
try_submit "${naive_run}" "naive" "0.0" "1.0" || true

IFS=' ' read -r -a BETAS <<< "${BETA_LIST}"
IFS=' ' read -r -a LAMBDAS <<< "${LAMBDA_TXT_LIST}"

for beta in "${BETAS[@]}"; do
  for lambda_txt in "${LAMBDAS[@]}"; do
    run_name="${RUN_PREFIX}_cagrad_b$(tag_float "${beta}")_lt$(tag_float "${lambda_txt}")_lr$(tag_float "${LAMBDA_REC}")_s${SEED}"
    try_submit "${run_name}" "cagrad" "${beta}" "${lambda_txt}" || true
  done
done

echo "[submit_dino_sun397_cagrad_sweep] done"
echo "[submit_dino_sun397_cagrad_sweep] manifest=${MANIFEST}"
