#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
CONDA_ENV=${CONDA_ENV:-diffuser}

RUNS_ROOT=${RUNS_ROOT:-${ROOT_DIR}/runs}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/results/in100_locked_multiseed_srun_$(date +%Y%m%d_%H%M%S)}
MANIFEST=${MANIFEST:-${OUT_DIR}/jobs.tsv}
LAUNCH=${LAUNCH:-true}

MAX_PARALLEL=${MAX_PARALLEL:-2}
POLL_SECONDS=${POLL_SECONDS:-30}
START_DELAY_SECONDS=${START_DELAY_SECONDS:-20}
REUSE_SCRIPT=${REUSE_SCRIPT:-scripts/reuse_or_start_srun_train.sh}
LAUNCHER=${LAUNCHER:-scripts/launch_manifest_srun.py}

HOLD_JOB_NAME=${HOLD_JOB_NAME:-dsga-in100-main-hold}
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
SEEDS=${SEEDS:-42,43,44}

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
EVAL_RFID_NUM_SAMPLES=${EVAL_RFID_NUM_SAMPLES:-1024}
EVAL_RFID_BATCH_SIZE=${EVAL_RFID_BATCH_SIZE:-64}
EVAL_RFID_TMP_DIR=${EVAL_RFID_TMP_DIR:-${ROOT_DIR}/results/in100_rfid_tmp}
LOG_EVERY=${LOG_EVERY:-20}

CAGRAD_BETA=${CAGRAD_BETA:-0.35}
DSGA_GLOBAL_GAMMA=${DSGA_GLOBAL_GAMMA:-0.8}
DSGA_GLOBAL_TAU=${DSGA_GLOBAL_TAU:-0.0}
DSGA_LAYERWISE_GROUPING=${DSGA_LAYERWISE_GROUPING:-layerwise_coarse}
DSGA_LAYERWISE_GAMMA=${DSGA_LAYERWISE_GAMMA:-0.8}
DSGA_LAYERWISE_TAU=${DSGA_LAYERWISE_TAU:-0.0}
DSGA_NORM_RESTORE=${DSGA_NORM_RESTORE:-false}
ENABLE_NAIVE=${ENABLE_NAIVE:-true}
ENABLE_PCGRAD=${ENABLE_PCGRAD:-true}
ENABLE_CAGRAD=${ENABLE_CAGRAD:-true}
ENABLE_DSGA_GLOBAL=${ENABLE_DSGA_GLOBAL:-true}
ENABLE_DSGA_LAYERWISE=${ENABLE_DSGA_LAYERWISE:-true}

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

mkdir -p "${OUT_DIR}" "${RUNS_ROOT}" "${EVAL_RFID_TMP_DIR}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo -e "jobid\trun_name\tlabel\tseed\tmethod\tencoder_init\tcagrad_beta\tma_laga_mode\tma_laga_grouping\tma_laga_align_gamma\tma_laga_conflict_tau\tma_laga_norm_restore\trecon_loss_type\tlaunch_cmd" > "${MANIFEST}"
fi

manifest_has_label_seed() {
  local label="$1"
  local seed="$2"
  awk -F'\t' -v target_label="${label}" -v target_seed="${seed}" 'NR>1 && $3==target_label && $4==target_seed {found=1} END {exit(found?0:1)}' "${MANIFEST}"
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
  local seed="$3"
  local method="$4"
  local cagrad_beta="$5"
  local ma_laga_mode="$6"
  local ma_laga_grouping="$7"
  local ma_laga_align_gamma="$8"
  local ma_laga_conflict_tau="$9"
  local ma_laga_norm_restore="${10}"

  if manifest_has_label_seed "${label}" "${seed}"; then
    echo "[skip] ${label} seed=${seed} already present in manifest"
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
    "--seed" "${seed}"
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
    "" "${run_name}" "${label}" "${seed}" "${method}" "${ENCODER_INIT}" "${cagrad_beta}" "${ma_laga_mode}" "${ma_laga_grouping}" "${ma_laga_align_gamma}" \
    "${ma_laga_conflict_tau}" "${ma_laga_norm_restore}" "${RECON_LOSS_TYPE}" "${launch_cmd}" >> "${MANIFEST}"
  echo "[planned] ${run_name}"
}

for seed in "${SEED_ARR[@]}"; do
  if bool_flag "${ENABLE_NAIVE}"; then
    append_one "in100_main_naive_seed${seed}" "naive" "${seed}" "joint" "0.0" "full" "global" "0.0" "0.0" "false"
  fi
  if bool_flag "${ENABLE_PCGRAD}"; then
    append_one "in100_main_pcgrad_seed${seed}" "pcgrad" "${seed}" "pcgrad" "0.0" "full" "global" "0.0" "0.0" "false"
  fi
  if bool_flag "${ENABLE_CAGRAD}"; then
    append_one "in100_main_cagrad_seed${seed}" "cagrad" "${seed}" "cagrad" "${CAGRAD_BETA}" "full" "global" "0.0" "0.0" "false"
  fi
  if bool_flag "${ENABLE_DSGA_GLOBAL}"; then
    append_one "in100_main_dsga_global_seed${seed}" "dsga_global" "${seed}" "ma_laga" "0.0" "full" "global" "${DSGA_GLOBAL_GAMMA}" "${DSGA_GLOBAL_TAU}" "${DSGA_NORM_RESTORE}"
  fi
  if bool_flag "${ENABLE_DSGA_LAYERWISE}"; then
    append_one "in100_main_dsga_layerwise_seed${seed}" "dsga_layerwise" "${seed}" "ma_laga" "0.0" "full" "${DSGA_LAYERWISE_GROUPING}" "${DSGA_LAYERWISE_GAMMA}" "${DSGA_LAYERWISE_TAU}" "${DSGA_NORM_RESTORE}"
  fi
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
