#!/usr/bin/env bash
# Submit ImageNet baselines for unirae.train (Slurm / peilab / preempt via sbatch_train.sh).
#
# Baselines (per seed):
# 1) joint_naive      : strategy=naive,          lambda_txt=1.0, lambda_rec=1.0
# 2) joint_conflict   : strategy=conflict_aware, lambda_txt=1.0, lambda_rec=1.0
# 3) text_only_naive  : strategy=naive,          lambda_txt=1.0, lambda_rec=0.0
# 4) recon_only_naive : strategy=naive,          lambda_txt=0.0, lambda_rec=1.0

set -euo pipefail

CONFIG=${CONFIG:-configs/smoke.yaml}
DATA_ROOT=${DATA_ROOT:-/path/to/imagenet}
DATA_FORMAT=${DATA_FORMAT:-auto}            # auto | imagefolder | hf_disk
HF_LOAD_FROM_DISK=${HF_LOAD_FROM_DISK:-}    # optional; if set, force HF disk path
HF_SPLIT_TRAIN=${HF_SPLIT_TRAIN:-train}      # train / validation / ...
HF_SPLIT_VAL=${HF_SPLIT_VAL:-validation}
HF_IMAGE_KEY=${HF_IMAGE_KEY:-image}
HF_LABEL_KEY=${HF_LABEL_KEY:-label}
CLASS_NAMES_FILE=${CLASS_NAMES_FILE:-}

IMAGE_SIZE=${IMAGE_SIZE:-224}
NUM_WORKERS=${NUM_WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
STEPS=${STEPS:-20000}
SEEDS=${SEEDS:-42}

RUN_PREFIX=${RUN_PREFIX:-baseline4_imagenet}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train}
CONDA_ENV=${CONDA_ENV:-diffuser310}

submit_one () {
  local run_name="$1"
  local seed="$2"
  local strategy="$3"
  local lambda_txt="$4"
  local lambda_rec="$5"

  local extra="--set seed=${seed}"
  extra="${extra} --set data.data_root=${DATA_ROOT}"
  extra="${extra} --set data.data_format=${DATA_FORMAT}"
  extra="${extra} --set data.hf_split_train=${HF_SPLIT_TRAIN}"
  extra="${extra} --set data.hf_split_val=${HF_SPLIT_VAL}"
  extra="${extra} --set data.hf_image_key=${HF_IMAGE_KEY}"
  extra="${extra} --set data.hf_label_key=${HF_LABEL_KEY}"
  extra="${extra} --set data.image_size=${IMAGE_SIZE}"
  extra="${extra} --set data.num_workers=${NUM_WORKERS}"
  extra="${extra} --set train.batch_size=${BATCH_SIZE}"
  extra="${extra} --set eval.batch_size=${EVAL_BATCH_SIZE}"
  extra="${extra} --set train.steps=${STEPS}"
  extra="${extra} --set train.strategy=${strategy}"
  extra="${extra} --set train.lambda_txt=${lambda_txt}"
  extra="${extra} --set train.lambda_rec=${lambda_rec}"
  extra="${extra} --set accelerate.mixed_precision=no"

  if [[ -n "${HF_LOAD_FROM_DISK}" ]]; then
    extra="${extra} --set data.hf_load_from_disk=${HF_LOAD_FROM_DISK}"
  fi
  if [[ -n "${CLASS_NAMES_FILE}" ]]; then
    extra="${extra} --set data.class_names_file=${CLASS_NAMES_FILE}"
  fi

  sbatch \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "${CONFIG}" "${run_name}"
}

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
for seed in "${SEED_ARR[@]}"; do
  submit_one "${RUN_PREFIX}_joint_naive_s${seed}"      "${seed}" "naive"          "1.0" "1.0"
  submit_one "${RUN_PREFIX}_joint_conflict_s${seed}"   "${seed}" "conflict_aware" "1.0" "1.0"
  submit_one "${RUN_PREFIX}_text_only_naive_s${seed}"  "${seed}" "naive"          "1.0" "0.0"
  submit_one "${RUN_PREFIX}_recon_only_naive_s${seed}" "${seed}" "naive"          "0.0" "1.0"
done

echo "[submit_imagenet_baselines] submitted seeds=${SEEDS} steps=${STEPS} data_format=${DATA_FORMAT}"

