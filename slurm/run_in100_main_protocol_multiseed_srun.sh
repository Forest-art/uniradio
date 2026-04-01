#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
CONDA_ENV=${CONDA_ENV:-diffuser}

RUNS_ROOT=${RUNS_ROOT:-${ROOT_DIR}/runs}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/results/in100_main_protocol_multiseed_srun_$(date +%Y%m%d_%H%M%S)}
MANIFEST=${MANIFEST:-${OUT_DIR}/jobs.tsv}
LAUNCH=${LAUNCH:-true}

MAX_PARALLEL=${MAX_PARALLEL:-2}
POLL_SECONDS=${POLL_SECONDS:-30}
START_DELAY_SECONDS=${START_DELAY_SECONDS:-20}
REUSE_SCRIPT=${REUSE_SCRIPT:-scripts/reuse_or_start_srun_train.sh}
LAUNCHER=${LAUNCHER:-scripts/launch_manifest_srun.py}

HOLD_JOB_NAME=${HOLD_JOB_NAME:-dsga-in100-paper-hold}
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-96G}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE:-8}

HF_DATASET=${HF_DATASET:-clane9/imagenet-100}
HF_CONFIG=${HF_CONFIG:-}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-validation}
IMAGE_KEY=${IMAGE_KEY:-image}
LABEL_KEY=${LABEL_KEY:-label}

SEEDS=${SEEDS:-42,43,44}
STEPS=${STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-50}
LOG_EVERY=${LOG_EVERY:-20}

LAMBDA_U=${LAMBDA_U:-1.0}
LAMBDA_G=${LAMBDA_G:-1.0}
UNDERSTANDING_LOSS=${UNDERSTANDING_LOSS:-ce}
RECON_LOSS=${RECON_LOSS:-rmse}
RECON_RMSE_EPS=${RECON_RMSE_EPS:-1e-12}
LPIPS_WEIGHT=${LPIPS_WEIGHT:-1.0}
GAN_WEIGHT=${GAN_WEIGHT:-0.75}
LPIPS_START_STEP=${LPIPS_START_STEP:-0}
GAN_START_STEP=${GAN_START_STEP:-1000}
DISC_UPDATE_START_STEP=${DISC_UPDATE_START_STEP:-750}

ENCODER_UPDATE=${ENCODER_UPDATE:-full}
LR_ENCODER=${LR_ENCODER:-2e-5}
LR_DECODER=${LR_DECODER:-2e-5}
LR_UND=${LR_UND:-1e-4}
LR_DISC=${LR_DISC:-2e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
CLIP_GRAD=${CLIP_GRAD:-1.0}

CAGRAD_BETA=${CAGRAD_BETA:-0.35}
DSGA_GROUPING=${DSGA_GROUPING:-layerwise}
DSGA_ALIGN_GAMMA=${DSGA_ALIGN_GAMMA:-0.5}
DSGA_CONFLICT_TAU=${DSGA_CONFLICT_TAU:-0.0}
DSGA_MAGNITUDE_SCOPE=${DSGA_MAGNITUDE_SCOPE:-global}
DSGA_MODE=${DSGA_MODE:-full}
DSGA_NORM_RESTORE=${DSGA_NORM_RESTORE:-true}

RFID_NUM_SAMPLES=${RFID_NUM_SAMPLES:-5000}
RFID_BATCH_SIZE=${RFID_BATCH_SIZE:-64}
RFID_TMP_DIR=${RFID_TMP_DIR:-${ROOT_DIR}/results/in100_rfid_tmp}
FINAL_EVAL_RFID=${FINAL_EVAL_RFID:-true}

ENABLE_NAIVE=${ENABLE_NAIVE:-true}
ENABLE_PCGRAD=${ENABLE_PCGRAD:-true}
ENABLE_CAGRAD=${ENABLE_CAGRAD:-true}
ENABLE_DSGA=${ENABLE_DSGA:-true}

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

mkdir -p "${OUT_DIR}" "${RUNS_ROOT}" "${RFID_TMP_DIR}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo -e "jobid\trun_name\tlabel\tseed\tlaunch_cmd" > "${MANIFEST}"
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
  local label="$2"
  local seed="$3"
  local strategy="$4"
  local cmd=""

  cmd+="cd $(printf '%q' "${ROOT_DIR}")"
  cmd+=" && source /home/xlubl/anaconda3/etc/profile.d/conda.sh"
  cmd+=" && conda activate $(printf '%q' "${CONDA_ENV}")"
  cmd+=" && export OMP_NUM_THREADS=$(printf '%q' "${OMP_NUM_THREADS_VALUE}")"
  cmd+=" && export PYTHONUNBUFFERED=1"
  cmd+=" && export NUMEXPR_MAX_THREADS=\${NUMEXPR_MAX_THREADS:-64}"
  cmd+=" && python -m $(printf '%q' "unirae.train_dsga_rae_lora")"
  cmd+=" --out_dir $(printf '%q' "${RUNS_ROOT}")"
  cmd+=" --run_name $(printf '%q' "${run_name}")"
  cmd+=" --seed $(printf '%q' "${seed}")"
  cmd+=" --hf_dataset $(printf '%q' "${HF_DATASET}")"
  if [[ -n "${HF_CONFIG}" ]]; then
    cmd+=" --hf_config $(printf '%q' "${HF_CONFIG}")"
  fi
  cmd+=" --train_split $(printf '%q' "${TRAIN_SPLIT}")"
  cmd+=" --val_split $(printf '%q' "${VAL_SPLIT}")"
  cmd+=" --image_key $(printf '%q' "${IMAGE_KEY}")"
  cmd+=" --label_key $(printf '%q' "${LABEL_KEY}")"
  cmd+=" --image_size 224"
  cmd+=" --batch_size $(printf '%q' "${BATCH_SIZE}")"
  cmd+=" --grad_accum_steps $(printf '%q' "${GRAD_ACCUM_STEPS}")"
  cmd+=" --num_workers $(printf '%q' "${NUM_WORKERS}")"
  cmd+=" --steps $(printf '%q' "${STEPS}")"
  cmd+=" --eval_every $(printf '%q' "${EVAL_EVERY}")"
  cmd+=" --eval_max_batches $(printf '%q' "${EVAL_MAX_BATCHES}")"
  cmd+=" --log_every $(printf '%q' "${LOG_EVERY}")"
  cmd+=" --lambda_u $(printf '%q' "${LAMBDA_U}")"
  cmd+=" --lambda_g $(printf '%q' "${LAMBDA_G}")"
  cmd+=" --understanding_loss $(printf '%q' "${UNDERSTANDING_LOSS}")"
  cmd+=" --recon_loss $(printf '%q' "${RECON_LOSS}")"
  cmd+=" --recon_rmse_eps $(printf '%q' "${RECON_RMSE_EPS}")"
  cmd+=" --lpips_weight $(printf '%q' "${LPIPS_WEIGHT}")"
  cmd+=" --gan_weight $(printf '%q' "${GAN_WEIGHT}")"
  cmd+=" --lpips_start_step $(printf '%q' "${LPIPS_START_STEP}")"
  cmd+=" --gan_start_step $(printf '%q' "${GAN_START_STEP}")"
  cmd+=" --disc_update_start_step $(printf '%q' "${DISC_UPDATE_START_STEP}")"
  cmd+=" --shared_strategy $(printf '%q' "${strategy}")"
  cmd+=" --encoder_update $(printf '%q' "${ENCODER_UPDATE}")"
  cmd+=" --lr_encoder $(printf '%q' "${LR_ENCODER}")"
  cmd+=" --lr_decoder $(printf '%q' "${LR_DECODER}")"
  cmd+=" --lr_und $(printf '%q' "${LR_UND}")"
  cmd+=" --lr_disc $(printf '%q' "${LR_DISC}")"
  cmd+=" --weight_decay $(printf '%q' "${WEIGHT_DECAY}")"
  cmd+=" --clip_grad $(printf '%q' "${CLIP_GRAD}")"
  cmd+=" --rae_code_root $(printf '%q' "/project/peilab/luxiaocheng/projects/RAE")"
  cmd+=" --cagrad_beta $(printf '%q' "${CAGRAD_BETA}")"
  cmd+=" --dsga_grouping $(printf '%q' "${DSGA_GROUPING}")"
  cmd+=" --dsga_align_gamma $(printf '%q' "${DSGA_ALIGN_GAMMA}")"
  cmd+=" --dsga_conflict_tau $(printf '%q' "${DSGA_CONFLICT_TAU}")"
  cmd+=" --dsga_magnitude_scope $(printf '%q' "${DSGA_MAGNITUDE_SCOPE}")"
  cmd+=" --dsga_mode $(printf '%q' "${DSGA_MODE}")"
  if bool_flag "${DSGA_NORM_RESTORE}"; then
    cmd+=" --dsga_norm_restore"
  else
    cmd+=" --no_dsga_norm_restore"
  fi
  if bool_flag "${FINAL_EVAL_RFID}"; then
    cmd+=" --final_eval_rfid"
  else
    cmd+=" --no_final_eval_rfid"
  fi
  cmd+=" --rfid_num_samples $(printf '%q' "${RFID_NUM_SAMPLES}")"
  cmd+=" --rfid_batch_size $(printf '%q' "${RFID_BATCH_SIZE}")"
  cmd+=" --rfid_tmp_dir $(printf '%q' "${RFID_TMP_DIR}")"
  echo "${cmd}"
}

append_one() {
  local run_name="$1"
  local label="$2"
  local seed="$3"
  local strategy="$4"
  if manifest_has_label_seed "${label}" "${seed}"; then
    echo "[skip] ${label} seed=${seed} already present in manifest"
    return 0
  fi
  local launch_cmd
  launch_cmd="$(build_launch_cmd "${run_name}" "${label}" "${seed}" "${strategy}")"
  printf '%s\t%s\t%s\t%s\t%s\n' "" "${run_name}" "${label}" "${seed}" "${launch_cmd}" >> "${MANIFEST}"
  echo "[planned] ${run_name}"
}

for seed in "${SEED_ARR[@]}"; do
  if bool_flag "${ENABLE_NAIVE}"; then
    append_one "in100_mainprot_naive_seed${seed}" "naive" "${seed}" "naive"
  fi
  if bool_flag "${ENABLE_PCGRAD}"; then
    append_one "in100_mainprot_pcgrad_seed${seed}" "pcgrad" "${seed}" "pcgrad"
  fi
  if bool_flag "${ENABLE_CAGRAD}"; then
    append_one "in100_mainprot_cagrad_seed${seed}" "cagrad" "${seed}" "cagrad"
  fi
  if bool_flag "${ENABLE_DSGA}"; then
    append_one "in100_mainprot_dsga_seed${seed}" "dsga" "${seed}" "dsga"
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
