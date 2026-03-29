#!/usr/bin/env bash
set -euo pipefail

# Joint RAE training (no LoRA): full encoder fine-tuning + CE understanding head
# with RAE-style reconstruction stack (L1 + LPIPS + GAN).

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
TIME_LIMIT=${TIME_LIMIT:-1-12:00:00}
CPUS=${CPUS:-8}
MEM=${MEM:-96G}
GPUS=${GPUS:-1}
CONDA_ENV=${CONDA_ENV:-diffuser310}

HF_DATASET=${HF_DATASET:-clane9/imagenet-100}
HF_CONFIG=${HF_CONFIG:-}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-validation}
IMAGE_SIZE=${IMAGE_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
TARGET_BATCH_SIZE=${TARGET_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}

SEED=${SEED:-42}
STEPS=${STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-20}
LOG_EVERY=${LOG_EVERY:-20}
PROBE_STEPS=${PROBE_STEPS:-0}

LAMBDA_U=${LAMBDA_U:-1.0}
LAMBDA_G=${LAMBDA_G:-1.0}
UNDERSTANDING_LOSS=${UNDERSTANDING_LOSS:-ce}
RECON_LOSS=${RECON_LOSS:-l1}
LPIPS_WEIGHT=${LPIPS_WEIGHT:-1.0}
GAN_WEIGHT=${GAN_WEIGHT:-0.75}
LPIPS_START_STEP=${LPIPS_START_STEP:-0}
GAN_START_STEP=${GAN_START_STEP:-0}
DISC_UPDATE_START_STEP=${DISC_UPDATE_START_STEP:-0}

SHARED_STRATEGY=${SHARED_STRATEGY:-naive}
ENCODER_UPDATE=${ENCODER_UPDATE:-full}

LR_ENCODER=${LR_ENCODER:-2e-5}
LR_DECODER=${LR_DECODER:-2e-5}
LR_UND=${LR_UND:-1e-4}
LR_DISC=${LR_DISC:-2e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}

RAE_CODE_ROOT=${RAE_CODE_ROOT:-/project/peilab/luxiaocheng/projects/RAE}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/peilab/xlubl/dsga_runs}
RUN_GROUP=${RUN_GROUP:-rae_joint_full_bs${TARGET_BATCH_SIZE}_s${STEPS}_$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-${RUN_GROUP}_naive_full}

mkdir -p /project/peilab/luxiaocheng/projects/DSGA/slurm/logs

if [[ $((BATCH_SIZE * GRAD_ACCUM_STEPS)) -ne ${TARGET_BATCH_SIZE} ]]; then
  echo "[warn] effective_batch_size=$((BATCH_SIZE * GRAD_ACCUM_STEPS)) != target_batch_size=${TARGET_BATCH_SIZE}" >&2
fi

HF_CONFIG_ARG=""
if [[ -n "${HF_CONFIG}" ]]; then
  HF_CONFIG_ARG="--hf_config ${HF_CONFIG}"
fi

cmd=$(cat <<EOC
set -euo pipefail
cd /project/peilab/luxiaocheng/projects/DSGA
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NUMEXPR_MAX_THREADS=\${NUMEXPR_MAX_THREADS:-128}
python -u -m unirae.train_dsga_rae_lora \
  --out_dir ${OUTPUT_ROOT} \
  --run_name ${RUN_NAME} \
  --seed ${SEED} \
  --hf_dataset ${HF_DATASET} \
  ${HF_CONFIG_ARG} \
  --train_split ${TRAIN_SPLIT} \
  --val_split ${VAL_SPLIT} \
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
  --lpips_weight ${LPIPS_WEIGHT} \
  --gan_weight ${GAN_WEIGHT} \
  --lpips_start_step ${LPIPS_START_STEP} \
  --gan_start_step ${GAN_START_STEP} \
  --disc_update_start_step ${DISC_UPDATE_START_STEP} \
  --shared_strategy ${SHARED_STRATEGY} \
  --encoder_update ${ENCODER_UPDATE} \
  --lr_encoder ${LR_ENCODER} \
  --lr_decoder ${LR_DECODER} \
  --lr_und ${LR_UND} \
  --lr_disc ${LR_DISC} \
  --weight_decay ${WEIGHT_DECAY} \
  --probe_steps ${PROBE_STEPS} \
  --rae_code_root ${RAE_CODE_ROOT}
EOC
)

jobid=$(sbatch \
  --parsable \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --gpus-per-node="${GPUS}" \
  --time="${TIME_LIMIT}" \
  --job-name="rae_jfull20k" \
  --output="/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/rae_jfull20k-%j.out" \
  --error="/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/rae_jfull20k-%j.err" \
  --wrap "$cmd")

cat <<EOM
[submitted]
  job_id=${jobid}
  run_name=${RUN_NAME}
  run_dir=${OUTPUT_ROOT}/${RUN_NAME}
  log_out=/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/rae_jfull20k-${jobid}.out
  log_err=/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/rae_jfull20k-${jobid}.err

[setting]
  shared_strategy=${SHARED_STRATEGY}
  encoder_update=${ENCODER_UPDATE}
  understanding_loss=${UNDERSTANDING_LOSS}
  recon_loss=${RECON_LOSS} + lpips(${LPIPS_WEIGHT}) + gan(${GAN_WEIGHT})
  batch_size=${BATCH_SIZE} (per_step micro-batch)
  grad_accum_steps=${GRAD_ACCUM_STEPS}
  effective_batch_size=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
  target_batch_size=${TARGET_BATCH_SIZE}
  steps=${STEPS}
  gpus=${GPUS}
EOM
