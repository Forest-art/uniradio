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
REFERENCE_NPZ=${REFERENCE_NPZ:-/project/peilab/luxiaocheng/projects/DSGA/results/VIRTUAL_imagenet256_labeled.npz}
VAE_MODEL=${VAE_MODEL:-sd-vae-ft-mse}
POSTERIOR_MODE=${POSTERIOR_MODE:-sample}

BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
METRICS=${METRICS:-rfid}

OUT_ROOT=${OUT_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/results}
RUN_NAME=${RUN_NAME:-vavae_reference_npz_eval_$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
OUT_JSON="${RUN_DIR}/eval_summary.json"
RECON_NPZ="${RUN_DIR}/reconstructions.npz"
LOG_FILE="${RUN_DIR}/eval.log"

mkdir -p "${RUN_DIR}"

CMD=$(cat <<EOC
set -euo pipefail
cd ${ROOT_DIR}
source /home/xlubl/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export OMP_NUM_THREADS=${OMP_NUM_THREADS_VALUE}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS_VALUE}
export PYTHONUNBUFFERED=1
python -m unirae.eval_vavae_reference_npz \
  --rae_code_root ${RAE_CODE_ROOT} \
  --reference_npz ${REFERENCE_NPZ} \
  --vae_model ${VAE_MODEL} \
  --posterior_mode ${POSTERIOR_MODE} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --metrics ${METRICS} \
  --save_recon_npz ${RECON_NPZ} \
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
  --job-name "vavae-ref" \
  bash -lc "${CMD}"
