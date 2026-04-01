#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
CONDA_ENV=${CONDA_ENV:-diffuser310}

CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-96G}
TIME_LIMIT=${TIME_LIMIT:-06:00:00}
OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE:-8}
NUMEXPR_MAX_THREADS_VALUE=${NUMEXPR_MAX_THREADS_VALUE:-128}

RAE_CODE_ROOT=${RAE_CODE_ROOT:-/project/peilab/luxiaocheng/projects/RAE}
STAGE1_CONFIG=${STAGE1_CONFIG:-/project/peilab/luxiaocheng/projects/RAE/configs/stage1/pretrained/DINOv2-B.yaml}
HF_LOCAL_CACHE_DIR=${HF_LOCAL_CACHE_DIR:-/project/peilab/luxiaocheng/dataset/.cache/datasets/benjamin-paine___imagenet-1k-256x256/default/0.0.0/1bd0400450249a7fe90c0aece37d0d03e7ea956a}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-validation}

IMAGE_SIZE=${IMAGE_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
PROBE_STEPS=${PROBE_STEPS:-0}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-0}
RFID_NUM_SAMPLES=${RFID_NUM_SAMPLES:-5000}
RFID_BATCH_SIZE=${RFID_BATCH_SIZE:-64}
RFID_TMP_DIR=${RFID_TMP_DIR:-/project/peilab/luxiaocheng/projects/DSGA/results/in1k_rfid_tmp}

OUT_ROOT=${OUT_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/results}
RUN_NAME=${RUN_NAME:-in1k_rae_official_eval_$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
OUT_JSON="${RUN_DIR}/eval_summary.json"
LOG_FILE="${RUN_DIR}/eval.log"

mkdir -p "${RUN_DIR}" "${RFID_TMP_DIR}"

CMD=$(cat <<EOC
set -euo pipefail
cd ${ROOT_DIR}
source /home/xlubl/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export OMP_NUM_THREADS=${OMP_NUM_THREADS_VALUE}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS_VALUE}
export PYTHONUNBUFFERED=1
python -m unirae.eval_rae_table1_baselines \
  --rae_code_root ${RAE_CODE_ROOT} \
  --stage1_config ${STAGE1_CONFIG} \
  --hf_local_cache_dir ${HF_LOCAL_CACHE_DIR} \
  --train_split ${TRAIN_SPLIT} \
  --val_split ${VAL_SPLIT} \
  --image_size ${IMAGE_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --probe_steps ${PROBE_STEPS} \
  --max_eval_batches ${MAX_EVAL_BATCHES} \
  --rfid_num_samples ${RFID_NUM_SAMPLES} \
  --rfid_batch_size ${RFID_BATCH_SIZE} \
  --rfid_tmp_dir ${RFID_TMP_DIR} \
  --out_json ${OUT_JSON} \
  2>&1 | tee ${LOG_FILE}
EOC
)

echo "[launch] ${RUN_NAME}"
srun \
  --account "${ACCOUNT}" \
  --partition "${PARTITION}" \
  --nodes 1 \
  --ntasks 1 \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --gpus-per-node 1 \
  --mem "${MEM}" \
  --time "${TIME_LIMIT}" \
  --job-name "rae-in1k-eval" \
  bash -lc "${CMD}"
