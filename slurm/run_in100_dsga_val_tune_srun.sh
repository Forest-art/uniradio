#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
CONDA_ENV=${CONDA_ENV:-diffuser}

RUNS_ROOT=${RUNS_ROOT:-${ROOT_DIR}/runs}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/results/in100_val_tune_srun_$(date +%Y%m%d_%H%M%S)}
MANIFEST=${MANIFEST:-${OUT_DIR}/jobs.tsv}
LAUNCH=${LAUNCH:-true}

MAX_PARALLEL=${MAX_PARALLEL:-2}
POLL_SECONDS=${POLL_SECONDS:-30}
START_DELAY_SECONDS=${START_DELAY_SECONDS:-20}
REUSE_SCRIPT=${REUSE_SCRIPT:-scripts/reuse_or_start_srun_train.sh}
LAUNCHER=${LAUNCHER:-scripts/launch_manifest_srun.py}

HOLD_JOB_NAME=${HOLD_JOB_NAME:-dsga-in100-hold}
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-64G}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE:-8}

DATASET_PATH=${DATASET_PATH:-}
HF_DATASET_ID=${HF_DATASET_ID:-clane9/imagenet-100}
CACHE_DIR=${CACHE_DIR:-}
IMAGE_KEY=${IMAGE_KEY:-image}
LABEL_KEY=${LABEL_KEY:-label}

ENCODER_INIT=${ENCODER_INIT:-dinov2}
ENCODER_CKPT=${ENCODER_CKPT:-}

TUNE_SEED=${TUNE_SEED:-3407}
MAX_STEPS=${MAX_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
LAMBDA_U=${LAMBDA_U:-1.0}
LAMBDA_G=${LAMBDA_G:-1.0}
RECON_LOSS_TYPE=${RECON_LOSS_TYPE:-mse}
RECON_RMSE_EPS=${RECON_RMSE_EPS:-1e-12}
AUTO_ALIGN_LAMBDA_G=${AUTO_ALIGN_LAMBDA_G:-true}

PROBE_EVERY=${PROBE_EVERY:-500}
PROBE_UNTIL=${PROBE_UNTIL:-10000}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-50}
EVAL_RFID_EVERY=${EVAL_RFID_EVERY:-0}
FINAL_EVAL_RFID=${FINAL_EVAL_RFID:-false}
EVAL_RFID_NUM_SAMPLES=${EVAL_RFID_NUM_SAMPLES:-512}
EVAL_RFID_BATCH_SIZE=${EVAL_RFID_BATCH_SIZE:-64}
EVAL_RFID_TMP_DIR=${EVAL_RFID_TMP_DIR:-${ROOT_DIR}/results/in100_rfid_tmp}
LOG_EVERY=${LOG_EVERY:-20}

CAGRAD_BETAS_STR=${CAGRAD_BETAS:-"0.35 0.5"}
DSGA_GLOBAL_GAMMAS_STR=${DSGA_GLOBAL_GAMMAS:-"0.5 0.8 1.0"}
DSGA_LAYERWISE_GAMMAS_STR=${DSGA_LAYERWISE_GAMMAS:-"0.5 0.8 1.0"}
DSGA_LAYERWISE_COARSE_GAMMAS_STR=${DSGA_LAYERWISE_COARSE_GAMMAS:-"0.5 0.8 1.0"}
DSGA_CONFLICT_TAUS_STR=${DSGA_CONFLICT_TAUS:-"0.0"}
DSGA_NORM_RESTORE=${DSGA_NORM_RESTORE:-false}

read -r -a CAGRAD_BETAS <<< "${CAGRAD_BETAS_STR}"
read -r -a DSGA_GLOBAL_GAMMAS <<< "${DSGA_GLOBAL_GAMMAS_STR}"
read -r -a DSGA_LAYERWISE_GAMMAS <<< "${DSGA_LAYERWISE_GAMMAS_STR}"
read -r -a DSGA_LAYERWISE_COARSE_GAMMAS <<< "${DSGA_LAYERWISE_COARSE_GAMMAS_STR}"
read -r -a DSGA_CONFLICT_TAUS <<< "${DSGA_CONFLICT_TAUS_STR}"

mkdir -p "${OUT_DIR}" "${RUNS_ROOT}" "${EVAL_RFID_TMP_DIR}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo -e "jobid\trun_name\tlabel\tseed\tmethod\tencoder_init\tcagrad_beta\tma_laga_mode\tma_laga_grouping\tma_laga_align_gamma\tma_laga_conflict_tau\tma_laga_norm_restore\trecon_loss_type\tlaunch_cmd" > "${MANIFEST}"
fi

tag_float() {
  local x="$1"
  local neg=""
  if [[ "${x}" == -* ]]; then
    neg="m"
    x="${x#-}"
  fi
  echo "${neg}$(echo "${x}" | tr '.' 'p')"
}

manifest_has_label() {
  local label="$1"
  awk -F'\t' -v target="${label}" 'NR>1 && $3==target {found=1} END {exit(found?0:1)}' "${MANIFEST}"
}

bool_flag() {
  local raw="${1:-false}"
  case "${raw,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

build_launch_cmd() {
  local run_name="$1"
  shift
  local args=("$@")
  local cmd=""
  local item

  cmd+="cd $(printf '%q' "${ROOT_DIR}")"
  cmd+=" && source /home/xlubl/anaconda3/etc/profile.d/conda.sh"
  cmd+=" && conda activate $(printf '%q' "${CONDA_ENV}")"
  cmd+=" && export OMP_NUM_THREADS=$(printf '%q' "${OMP_NUM_THREADS_VALUE}")"
  cmd+=" && export PYTHONUNBUFFERED=1"
  cmd+=" && python -m $(printf '%q' "unirae.train_imagenet100_methods")"
  for item in "${args[@]}"; do
    cmd+=" $(printf '%q' "${item}")"
  done
  cmd+=" && python $(printf '%q' "${ROOT_DIR}/scripts/materialize_in100_eval_last.py") --run-dir $(printf '%q' "${RUNS_ROOT}/${run_name}")"
  echo "${cmd}"
}

append_one() {
  local run_name="$1"
  local label="$2"
  local method="$3"
  local cagrad_beta="$4"
  local ma_laga_mode="$5"
  local ma_laga_grouping="$6"
  local ma_laga_align_gamma="$7"
  local ma_laga_conflict_tau="$8"
  local ma_laga_norm_restore="$9"

  if manifest_has_label "${label}"; then
    echo "[skip] ${label} already present in manifest"
    return 0
  fi

  local cmd_args=(
    "--method" "${method}"
    "--encoder_init" "${ENCODER_INIT}"
    "--batch_size" "${BATCH_SIZE}"
    "--num_workers" "${NUM_WORKERS}"
    "--max_steps" "${MAX_STEPS}"
    "--lr" "${LR}"
    "--weight_decay" "${WEIGHT_DECAY}"
    "--warmup_steps" "${WARMUP_STEPS}"
    "--lambda_u" "${LAMBDA_U}"
    "--lambda_g" "${LAMBDA_G}"
    "--recon_loss_type" "${RECON_LOSS_TYPE}"
    "--recon_rmse_eps" "${RECON_RMSE_EPS}"
    "--cagrad_beta" "${cagrad_beta}"
    "--ma_laga_mode" "${ma_laga_mode}"
    "--ma_laga_grouping" "${ma_laga_grouping}"
    "--ma_laga_align_gamma" "${ma_laga_align_gamma}"
    "--ma_laga_conflict_tau" "${ma_laga_conflict_tau}"
    "--probe_every" "${PROBE_EVERY}"
    "--probe_until" "${PROBE_UNTIL}"
    "--eval_every" "${EVAL_EVERY}"
    "--eval_max_batches" "${EVAL_MAX_BATCHES}"
    "--eval_rfid_every" "${EVAL_RFID_EVERY}"
    "--eval_rfid_num_samples" "${EVAL_RFID_NUM_SAMPLES}"
    "--eval_rfid_batch_size" "${EVAL_RFID_BATCH_SIZE}"
    "--eval_rfid_tmp_dir" "${EVAL_RFID_TMP_DIR}"
    "--run_name" "${run_name}"
    "--seed" "${TUNE_SEED}"
    "--output_root" "${RUNS_ROOT}"
    "--device" "auto"
    "--log_every" "${LOG_EVERY}"
    "--image_key" "${IMAGE_KEY}"
    "--label_key" "${LABEL_KEY}"
  )

  if [[ -n "${DATASET_PATH}" ]]; then
    cmd_args+=("--dataset_path" "${DATASET_PATH}")
  else
    cmd_args+=("--hf_dataset_id" "${HF_DATASET_ID}")
  fi
  if [[ -n "${CACHE_DIR}" ]]; then
    cmd_args+=("--cache_dir" "${CACHE_DIR}")
  fi
  if [[ -n "${ENCODER_CKPT}" ]]; then
    cmd_args+=("--encoder_ckpt" "${ENCODER_CKPT}")
  fi
  if bool_flag "${AUTO_ALIGN_LAMBDA_G}"; then
    cmd_args+=("--auto_align_lambda_g")
  else
    cmd_args+=("--disable_auto_align_lambda_g")
  fi
  if bool_flag "${ma_laga_norm_restore}"; then
    cmd_args+=("--ma_laga_norm_restore")
  fi
  if bool_flag "${FINAL_EVAL_RFID}"; then
    cmd_args+=("--final_eval_rfid")
  else
    cmd_args+=("--no_final_eval_rfid")
  fi

  local launch_cmd
  launch_cmd="$(build_launch_cmd "${run_name}" "${cmd_args[@]}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "" "${run_name}" "${label}" "${TUNE_SEED}" "${method}" "${ENCODER_INIT}" "${cagrad_beta}" "${ma_laga_mode}" "${ma_laga_grouping}" "${ma_laga_align_gamma}" \
    "${ma_laga_conflict_tau}" "${ma_laga_norm_restore}" "${RECON_LOSS_TYPE}" "${launch_cmd}" >> "${MANIFEST}"
  echo "[planned] ${run_name}"
}

append_one "in100_valtune_naive_seed${TUNE_SEED}" "naive" "joint" "0.0" "full" "global" "0.0" "0.0" "false"
append_one "in100_valtune_pcgrad_seed${TUNE_SEED}" "pcgrad" "pcgrad" "0.0" "full" "global" "0.0" "0.0" "false"

for beta in "${CAGRAD_BETAS[@]}"; do
  append_one "in100_valtune_cagrad_b$(tag_float "${beta}")_seed${TUNE_SEED}" \
    "cagrad_b$(tag_float "${beta}")" "cagrad" "${beta}" "full" "global" "0.0" "0.0" "false"
done

for gamma in "${DSGA_GLOBAL_GAMMAS[@]}"; do
  for tau in "${DSGA_CONFLICT_TAUS[@]}"; do
    append_one "in100_valtune_dsga_global_g$(tag_float "${gamma}")_t$(tag_float "${tau}")_seed${TUNE_SEED}" \
      "dsga_global_g$(tag_float "${gamma}")_t$(tag_float "${tau}")" "ma_laga" "0.0" "full" "global" "${gamma}" "${tau}" "${DSGA_NORM_RESTORE}"
  done
done

for gamma in "${DSGA_LAYERWISE_GAMMAS[@]}"; do
  for tau in "${DSGA_CONFLICT_TAUS[@]}"; do
    append_one "in100_valtune_dsga_layerwise_g$(tag_float "${gamma}")_t$(tag_float "${tau}")_seed${TUNE_SEED}" \
      "dsga_layerwise_g$(tag_float "${gamma}")_t$(tag_float "${tau}")" "ma_laga" "0.0" "full" "layerwise" "${gamma}" "${tau}" "${DSGA_NORM_RESTORE}"
  done
done

for gamma in "${DSGA_LAYERWISE_COARSE_GAMMAS[@]}"; do
  for tau in "${DSGA_CONFLICT_TAUS[@]}"; do
    append_one "in100_valtune_dsga_layerwisecoarse_g$(tag_float "${gamma}")_t$(tag_float "${tau}")_seed${TUNE_SEED}" \
      "dsga_layerwisecoarse_g$(tag_float "${gamma}")_t$(tag_float "${tau}")" "ma_laga" "0.0" "full" "layerwise_coarse" "${gamma}" "${tau}" "${DSGA_NORM_RESTORE}"
  done
done

echo "[done] manifest=${MANIFEST}"
echo "[hint] aggregate after completion:"
echo "python scripts/aggregate_in100_multiseed.py --manifest ${MANIFEST} --runs_root ${RUNS_ROOT} --out_root ${OUT_DIR}"

if [[ "${LAUNCH}" == "true" ]]; then
  export ACCOUNT PARTITION HOLD_JOB_NAME NODES GPUS_PER_NODE CPUS_PER_TASK MEM TIME_LIMIT
  python "${LAUNCHER}" \
    --manifest "${MANIFEST}" \
    --runs_root "${RUNS_ROOT}" \
    --reuse_script "${REUSE_SCRIPT}" \
    --max_parallel "${MAX_PARALLEL}" \
    --poll_seconds "${POLL_SECONDS}" \
    --start_delay_seconds "${START_DELAY_SECONDS}" \
    --resume
fi
