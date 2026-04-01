#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
CONDA_ENV=${CONDA_ENV:-diffuser310}

HF_DATASET=${HF_DATASET:-benjamin-paine/imagenet-1k-256x256}
HF_CONFIG=${HF_CONFIG:-}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-validation}
IMAGE_KEY=${IMAGE_KEY:-image}
LABEL_KEY=${LABEL_KEY:-label}
IMAGE_SIZE=${IMAGE_SIZE:-224}

BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
NUM_WORKERS=${NUM_WORKERS:-8}

SEED=${SEED:-42}
STEPS=${STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-20}
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

SHARED_STRATEGY=${SHARED_STRATEGY:-dsga}
CAGRAD_BETA=${CAGRAD_BETA:-0.35}
DSGA_GROUPING=${DSGA_GROUPING:-layerwise}
DSGA_ALIGN_GAMMA=${DSGA_ALIGN_GAMMA:-0.5}
DSGA_CONFLICT_TAU=${DSGA_CONFLICT_TAU:-0.0}
DSGA_MAGNITUDE_SCOPE=${DSGA_MAGNITUDE_SCOPE:-global}
DSGA_MODE=${DSGA_MODE:-full}
DSGA_NORM_RESTORE=${DSGA_NORM_RESTORE:-1}

ENCODER_UPDATE=${ENCODER_UPDATE:-full}
LR_ENCODER=${LR_ENCODER:-2e-5}
LR_DECODER=${LR_DECODER:-2e-5}
LR_UND=${LR_UND:-1e-4}
LR_DISC=${LR_DISC:-2e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
CLIP_GRAD=${CLIP_GRAD:-1.0}

RAE_CODE_ROOT=${RAE_CODE_ROOT:-/project/peilab/luxiaocheng/projects/RAE}
OUT_DIR=${OUT_DIR:-/project/peilab/luxiaocheng/projects/DSGA/runs}
RUN_NAME=${RUN_NAME:-in1k_rae_dsga_full_s${STEPS}_bs${BATCH_SIZE}x${GRAD_ACCUM_STEPS}_$(date +%Y%m%d_%H%M%S)}
RFID_TMP_DIR=${RFID_TMP_DIR:-/project/peilab/luxiaocheng/projects/DSGA/results/in1k_rfid_tmp}

CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-96G}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
HOLD_JOB_NAME=${HOLD_JOB_NAME:-dsga-in1k-rae-hold}
LOG_ROOT=${LOG_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/results/srun_reuse_logs}
OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE:-8}
NUMEXPR_MAX_THREADS_VALUE=${NUMEXPR_MAX_THREADS_VALUE:-64}
DATA_PARALLEL=${DATA_PARALLEL:-0}
SAVE_EVERY=${SAVE_EVERY:-500}
RESUME=${RESUME:-0}
RESUME_FROM=${RESUME_FROM:-}

HF_CONFIG_ARG=""
if [[ -n "${HF_CONFIG}" ]]; then
  HF_CONFIG_ARG="--hf_config ${HF_CONFIG}"
fi

DSGA_NORM_FLAG="--no_dsga_norm_restore"
if [[ "${DSGA_NORM_RESTORE}" == "1" ]]; then
  DSGA_NORM_FLAG="--dsga_norm_restore"
fi

TRAIN_CMD=$(cat <<EOC
cd ${ROOT_DIR}
source /home/xlubl/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS_VALUE}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS_VALUE}
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
python -u -m unirae.train_dsga_rae_lora \
  --out_dir ${OUT_DIR} \
  --run_name ${RUN_NAME} \
  --seed ${SEED} \
  --hf_dataset ${HF_DATASET} \
  ${HF_CONFIG_ARG} \
  --train_split ${TRAIN_SPLIT} \
  --val_split ${VAL_SPLIT} \
  --image_key ${IMAGE_KEY} \
  --label_key ${LABEL_KEY} \
  --image_size ${IMAGE_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --grad_accum_steps ${GRAD_ACCUM_STEPS} \
  --num_workers ${NUM_WORKERS} \
  --steps ${STEPS} \
  --eval_every ${EVAL_EVERY} \
  --eval_max_batches ${EVAL_MAX_BATCHES} \
  --log_every ${LOG_EVERY} \
  --lambda_u ${LAMBDA_U} \
  --lambda_g ${LAMBDA_G} \
  --understanding_loss ${UNDERSTANDING_LOSS} \
  --recon_loss ${RECON_LOSS} \
  --recon_rmse_eps ${RECON_RMSE_EPS} \
  --lpips_weight ${LPIPS_WEIGHT} \
  --gan_weight ${GAN_WEIGHT} \
  --lpips_start_step ${LPIPS_START_STEP} \
  --gan_start_step ${GAN_START_STEP} \
  --disc_update_start_step ${DISC_UPDATE_START_STEP} \
  --shared_strategy ${SHARED_STRATEGY} \
  --cagrad_beta ${CAGRAD_BETA} \
  --dsga_grouping ${DSGA_GROUPING} \
  --dsga_align_gamma ${DSGA_ALIGN_GAMMA} \
  --dsga_conflict_tau ${DSGA_CONFLICT_TAU} \
  --dsga_magnitude_scope ${DSGA_MAGNITUDE_SCOPE} \
  --dsga_mode ${DSGA_MODE} \
  ${DSGA_NORM_FLAG} \
  --encoder_update ${ENCODER_UPDATE} \
  --lr_encoder ${LR_ENCODER} \
  --lr_decoder ${LR_DECODER} \
  --lr_und ${LR_UND} \
  --lr_disc ${LR_DISC} \
  --weight_decay ${WEIGHT_DECAY} \
  --clip_grad ${CLIP_GRAD} \
  --rae_code_root ${RAE_CODE_ROOT} \
  --no_final_eval_rfid \
  --rfid_tmp_dir ${RFID_TMP_DIR} \
  --save_every ${SAVE_EVERY} \
  $(if [[ "${RESUME}" == "1" ]]; then echo "--resume"; fi) \
  $(if [[ -n "${RESUME_FROM}" ]]; then printf -- "--resume_from %q" "${RESUME_FROM}"; fi) \
  $(if [[ "${DATA_PARALLEL}" == "1" ]]; then echo "--data_parallel"; else echo "--no_data_parallel"; fi)
EOC
)

echo "[launch]"
echo "  run_name=${RUN_NAME}"
echo "  dataset=${HF_DATASET}"
echo "  out_dir=${OUT_DIR}"
echo "  batch_size=${BATCH_SIZE}"
echo "  grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "  effective_batch_size=$((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "  gpus_per_node=${GPUS_PER_NODE}"
echo "  steps=${STEPS}"
echo "  shared_strategy=${SHARED_STRATEGY}"
echo "  encoder_update=${ENCODER_UPDATE}"

ACCOUNT="${ACCOUNT}" \
PARTITION="${PARTITION}" \
HOLD_JOB_NAME="${HOLD_JOB_NAME}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
CPUS_PER_TASK="${CPUS_PER_TASK}" \
MEM="${MEM}" \
TIME_LIMIT="${TIME_LIMIT}" \
LOG_ROOT="${LOG_ROOT}" \
bash "${ROOT_DIR}/scripts/reuse_or_start_srun_train.sh" -- "${TRAIN_CMD}"
