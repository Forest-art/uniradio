#!/usr/bin/env bash
set -euo pipefail

# Final ImageNet-100 fair 10k ablation:
# 1) Vanilla Joint
# 2) Global CAGrad (beta=0.5)
# 3) Pure LAGA (direction_only)
# 4) Pure MA (magnitude_only)
# 5) Full MA-LAGA (align + projection)

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
TIME_LIMIT=${TIME_LIMIT:-0-24:00:00}
CPUS=${CPUS:-8}
MEM=${MEM:-64G}
GPUS=${GPUS:-1}
CONDA_ENV=${CONDA_ENV:-diffuser}

ENCODER_INIT=${ENCODER_INIT:-scratch}
ENCODER_CKPT=${ENCODER_CKPT:-}
HF_DATASET_ID=${HF_DATASET_ID:-clane9/imagenet-100}
DATASET_PATH=${DATASET_PATH:-}

BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_STEPS=${MAX_STEPS:-10000}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-50}
EVAL_RFID_NUM_SAMPLES=${EVAL_RFID_NUM_SAMPLES:-1024}
EVAL_RFID_BATCH_SIZE=${EVAL_RFID_BATCH_SIZE:-64}
EVAL_RFID_TMP_DIR=${EVAL_RFID_TMP_DIR:-/tmp}

LR=${LR:-5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
LAMBDA_U=${LAMBDA_U:-1.0}
LAMBDA_G=${LAMBDA_G:-1.0}

MA_LAGA_ALIGN_GAMMA=${MA_LAGA_ALIGN_GAMMA:-1.0}
SEED=${SEED:-42}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/peilab/xlubl/unirae_runs/in100_ma_laga_ablation_10k}
RUN_PREFIX=${RUN_PREFIX:-in100_ma_laga_ablation10k}
LOG_EVERY=${LOG_EVERY:-20}

OUT_TSV=${OUT_TSV:-/scratch/peilab/xlubl/unirae_runs/in100_ma_laga_ablation10k_jobs_$(date +%Y%m%d_%H%M%S).tsv}
mkdir -p "$(dirname "${OUT_TSV}")"
mkdir -p slurm/logs
echo -e "jobid\trun_name\tmethod\tma_laga_mode\talign_gamma\tnorm_restore\tcagrad_beta\toutput_root" > "${OUT_TSV}"

submit_one() {
  local job_name="$1"
  local method="$2"
  local run_name="$3"
  local cagrad_beta="$4"
  local ma_laga_mode="$5"
  local align_gamma="$6"
  local norm_restore="$7"

  local maybe_ckpt=""
  if [[ "${ENCODER_INIT}" == "dinov2" && -n "${ENCODER_CKPT}" ]]; then
    maybe_ckpt="--encoder_ckpt ${ENCODER_CKPT}"
  fi

  local maybe_data_path=""
  if [[ -n "${DATASET_PATH}" ]]; then
    maybe_data_path="--dataset_path ${DATASET_PATH}"
  fi

  local maybe_norm_restore=""
  if [[ "${norm_restore}" == "true" ]]; then
    maybe_norm_restore="--ma_laga_norm_restore"
  fi

  local cmd
  cmd=$(cat <<EOC
set -euo pipefail
cd /project/peilab/luxiaocheng/projects/DSGA
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
python -m unirae.train_imagenet100_methods \
  --method ${method} \
  --encoder_init ${ENCODER_INIT} \
  --run_name ${run_name} \
  --hf_dataset_id ${HF_DATASET_ID} \
  ${maybe_data_path} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --max_steps ${MAX_STEPS} \
  --eval_every ${EVAL_EVERY} \
  --eval_max_batches ${EVAL_MAX_BATCHES} \
  --eval_rfid_num_samples ${EVAL_RFID_NUM_SAMPLES} \
  --eval_rfid_batch_size ${EVAL_RFID_BATCH_SIZE} \
  --eval_rfid_tmp_dir ${EVAL_RFID_TMP_DIR} \
  --lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_steps ${WARMUP_STEPS} \
  --lambda_u ${LAMBDA_U} \
  --lambda_g ${LAMBDA_G} \
  --cagrad_beta ${cagrad_beta} \
  --ma_laga_mode ${ma_laga_mode} \
  --ma_laga_align_gamma ${align_gamma} \
  ${maybe_norm_restore} \
  --seed ${SEED} \
  --output_root ${OUTPUT_ROOT} \
  --log_every ${LOG_EVERY} \
  --device auto \
  ${maybe_ckpt}
EOC
)

  local out
  out=$(sbatch \
    --parsable \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --gpus-per-node="${GPUS}" \
    --time="${TIME_LIMIT}" \
    --job-name="${job_name}" \
    --output="slurm/logs/${job_name}-%j.out" \
    --error="slurm/logs/${job_name}-%j.err" \
    --wrap "${cmd}")
  local job_id
  job_id=$(echo "${out}" | awk -F ';' '{print $1}')
  echo -e "${job_id}\t${run_name}\t${method}\t${ma_laga_mode}\t${align_gamma}\t${norm_restore}\t${cagrad_beta}\t${OUTPUT_ROOT}" >> "${OUT_TSV}"
  echo "${job_id}"
}

r="${RUN_PREFIX}_joint_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "in100_joint10k" "joint" "${r}" "0.5" "full" "${MA_LAGA_ALIGN_GAMMA}" "false")
echo "[submit_in100_ma_laga_ablation] submitted ${j} ${r}"

r="${RUN_PREFIX}_cagrad_b0p5_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "in100_cagrad10k" "cagrad" "${r}" "0.5" "full" "${MA_LAGA_ALIGN_GAMMA}" "false")
echo "[submit_in100_ma_laga_ablation] submitted ${j} ${r}"

r="${RUN_PREFIX}_pure_laga_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "in100_purelaga10k" "ma_laga" "${r}" "0.5" "direction_only" "${MA_LAGA_ALIGN_GAMMA}" "false")
echo "[submit_in100_ma_laga_ablation] submitted ${j} ${r}"

r="${RUN_PREFIX}_pure_ma_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "in100_purema10k" "ma_laga" "${r}" "0.5" "magnitude_only" "${MA_LAGA_ALIGN_GAMMA}" "false")
echo "[submit_in100_ma_laga_ablation] submitted ${j} ${r}"

r="${RUN_PREFIX}_full_malaga_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "in100_malaga10k" "ma_laga" "${r}" "0.5" "full" "${MA_LAGA_ALIGN_GAMMA}" "false")
echo "[submit_in100_ma_laga_ablation] submitted ${j} ${r}"

echo "[submit_in100_ma_laga_ablation] job table: ${OUT_TSV}"
