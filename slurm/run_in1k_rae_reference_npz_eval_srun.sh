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
REFERENCE_NPZ=${REFERENCE_NPZ:-/project/peilab/luxiaocheng/projects/DSGA/results/VIRTUAL_imagenet256_labeled.npz}
REFERENCE_URL=${REFERENCE_URL:-https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz}

BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
METRICS=${METRICS:-rfid}

OUT_ROOT=${OUT_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/results}
RUN_NAME=${RUN_NAME:-in1k_rae_reference_npz_eval_$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
OUT_JSON="${RUN_DIR}/eval_summary.json"
RECON_NPZ="${RUN_DIR}/reconstructions.npz"
LOG_FILE="${RUN_DIR}/eval.log"

mkdir -p "${RUN_DIR}" "$(dirname "${REFERENCE_NPZ}")"

python - <<'PY' "${REFERENCE_NPZ}" || rm -f "${REFERENCE_NPZ}"
import sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    data = np.load(path)
    arr = data["arr_0"]
    print(f"[reference] existing file valid: {path} shape={arr.shape} dtype={arr.dtype}")
except Exception as exc:  # noqa: BLE001
    print(f"[reference] invalid existing file, removing: {path} err={exc}")
    raise SystemExit(2)
PY

if [[ ! -f "${REFERENCE_NPZ}" ]]; then
  echo "[reference] downloading ${REFERENCE_URL}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=1M \
      --file-allocation=none \
      --dir "$(dirname "${REFERENCE_NPZ}")" \
      --out "$(basename "${REFERENCE_NPZ}")" \
      "${REFERENCE_URL}"
  else
    wget -c -O "${REFERENCE_NPZ}" "${REFERENCE_URL}"
  fi
fi

python - <<'PY' "${REFERENCE_NPZ}"
import sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
data = np.load(path)
arr = data["arr_0"]
print(f"[reference] ready: {path} shape={arr.shape} dtype={arr.dtype}")
PY

CMD=$(cat <<EOC
set -euo pipefail
cd ${ROOT_DIR}
source /home/xlubl/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export OMP_NUM_THREADS=${OMP_NUM_THREADS_VALUE}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS_VALUE}
export PYTHONUNBUFFERED=1
python -m unirae.eval_rae_reference_npz \
  --rae_code_root ${RAE_CODE_ROOT} \
  --stage1_config ${STAGE1_CONFIG} \
  --reference_npz ${REFERENCE_NPZ} \
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
  --job-name "rae-in1k-ref" \
  bash -lc "${CMD}"
