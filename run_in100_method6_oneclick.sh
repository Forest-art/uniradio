#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash run_in100_method6_oneclick.sh <dataset_path_or_hf_dataset_id>
  bash run_in100_method6_oneclick.sh hf:clane9/imagenet-100

Input supports:
  1) ImageFolder style: <root>/train, <root>/val (or validation)
  2) HF load_from_disk directory: contains dataset_dict.json / dataset_info.json
  3) HF dataset id for online load_dataset (e.g. clane9/imagenet-100)

Optional env vars:
  NPROC_PER_NODE=8            # number of GPUs to use on this node
  BATCH_SIZE_PER_GPU=32       # per-GPU batch size
  MAX_STEPS=10000
  LR=5e-4
  WARMUP_STEPS=1000
  NUM_WORKERS=8
  SEED=42
  OUTPUT_ROOT=results/in100_method6_runs
  RUN_GROUP=in100_method6_YYYYmmdd_HHMMSS
  METHODS="und_only gen_only joint pcgrad cagrad lacar"
  DATA_MODE=auto               # auto | path | hf
  PYTHON_BIN=python
  TORCHRUN_BIN=torchrun
  ENABLE_FINAL_RFID=1         # 1: compute final rFID, 0: disable
EOF
  exit 1
fi

DATASET_INPUT="$1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-32}
MAX_STEPS=${MAX_STEPS:-10000}
LR=${LR:-5e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/in100_method6_runs}
RUN_GROUP=${RUN_GROUP:-in100_method6_$(date +%Y%m%d_%H%M%S)}
METHODS=${METHODS:-"und_only gen_only joint pcgrad cagrad lacar"}
ENABLE_FINAL_RFID=${ENABLE_FINAL_RFID:-1}
DATA_MODE=${DATA_MODE:-auto}

if ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
  echo "[warn] $TORCHRUN_BIN not found, fallback to: $PYTHON_BIN -m torch.distributed.run"
  LAUNCH_BASE=("$PYTHON_BIN" "-m" "torch.distributed.run")
else
  LAUNCH_BASE=("$TORCHRUN_BIN")
fi

RUN_ROOT="${OUTPUT_ROOT}/${RUN_GROUP}"
mkdir -p "$RUN_ROOT"

# 统一数据输入：本地路径优先，否则按 HF dataset id 处理。
DATA_ARGS=()
if [[ "$DATASET_INPUT" == hf:* ]]; then
  HF_DATASET_ID="${DATASET_INPUT#hf:}"
  DATA_ARGS+=(--hf_dataset_id "$HF_DATASET_ID")
  DATA_MODE="hf_load_dataset"
elif [[ "$DATA_MODE" == "hf" ]]; then
  HF_DATASET_ID="$DATASET_INPUT"
  DATA_ARGS+=(--hf_dataset_id "$HF_DATASET_ID")
  DATA_MODE="hf_load_dataset"
elif [[ "$DATA_MODE" == "path" ]]; then
  if [[ ! -e "$DATASET_INPUT" ]]; then
    echo "[error] DATA_MODE=path but input path not found: $DATASET_INPUT" >&2
    exit 1
  fi
  DATASET_PATH="$DATASET_INPUT"
  DATA_ARGS+=(--dataset_path "$DATASET_PATH")
  DATA_MODE="local_path"
elif [[ -e "$DATASET_INPUT" ]]; then
  DATASET_PATH="$DATASET_INPUT"
  DATA_ARGS+=(--dataset_path "$DATASET_PATH")
  DATA_MODE="local_path"
else
  HF_DATASET_ID="$DATASET_INPUT"
  DATA_ARGS+=(--hf_dataset_id "$HF_DATASET_ID")
  DATA_MODE="hf_load_dataset"
fi

# 统一公共参数，保证公平对比。
COMMON_ARGS=(
  -m unirae.train_imagenet100_methods
  --encoder_init dinov2
  "${DATA_ARGS[@]}"
  --batch_size "$BATCH_SIZE_PER_GPU"
  --num_workers "$NUM_WORKERS"
  --max_steps "$MAX_STEPS"
  --lr "$LR"
  --warmup_steps "$WARMUP_STEPS"
  --probe_every 500
  --probe_until "$MAX_STEPS"
  --eval_every 1000
  --eval_max_batches 50
  --eval_rfid_every 0
  --eval_rfid_num_samples 2048
  --eval_rfid_batch_size 64
  --seed "$SEED"
  --output_root "$RUN_ROOT"
  --device auto
)

if [[ "$ENABLE_FINAL_RFID" == "1" ]]; then
  COMMON_ARGS+=(--final_eval_rfid)
else
  COMMON_ARGS+=(--no_final_eval_rfid)
fi

echo "[start] RUN_GROUP=${RUN_GROUP}"
echo "[start] DATA_MODE=${DATA_MODE}"
if [[ "$DATA_MODE" == "local_path" ]]; then
  echo "[start] DATASET_PATH=${DATASET_PATH}"
else
  echo "[start] HF_DATASET_ID=${HF_DATASET_ID}"
fi
echo "[start] NPROC_PER_NODE=${NPROC_PER_NODE}, BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU}, GLOBAL_BATCH=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU))"
echo "[start] OUTPUT_ROOT=${RUN_ROOT}"

echo "method,run_name,log_file" > "${RUN_ROOT}/run_manifest.csv"

for method in $METHODS; do
  run_name="${RUN_GROUP}_${method}_s${SEED}"
  log_file="${RUN_ROOT}/${run_name}.log"

  EXTRA_ARGS=()
  if [[ "$method" == "cagrad" ]]; then
    EXTRA_ARGS+=(--cagrad_beta 0.35)
  fi
  if [[ "$method" == "lacar" ]]; then
    EXTRA_ARGS+=(--lambda_var 0.20)
  fi

  echo "[run] method=${method} run_name=${run_name}"
  LAUNCH_CMD=(
    "${LAUNCH_BASE[@]}"
    --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE}"
    "${COMMON_ARGS[@]}"
    --method "${method}"
    --run_name "${run_name}"
  )
  # 兼容旧版 bash + set -u：空数组展开可能触发 unbound variable。
  LAUNCH_CMD+=( ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} )
  set -x
  "${LAUNCH_CMD[@]}" 2>&1 | tee "$log_file"
  set +x

  echo "${method},${run_name},${log_file}" >> "${RUN_ROOT}/run_manifest.csv"
  echo "[done] method=${method}"
done

echo "[all_done] results in: ${RUN_ROOT}"
