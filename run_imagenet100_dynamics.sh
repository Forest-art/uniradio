#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   bash run_imagenet100_dynamics.sh
# 可通过环境变量覆盖默认参数，例如:
#   NUM_GPUS=8 ENCODER_CKPT=/path/to/dino.pth HF_DATASET_ID=clane9/imagenet-100 BATCH_SIZE=32 MAX_STEPS=1200 bash run_imagenet100_dynamics.sh

HF_DATASET_ID="${HF_DATASET_ID:-clane9/imagenet-100}"
CACHE_DIR="${CACHE_DIR:-}"
ENCODER_CKPT="${ENCODER_CKPT:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_STEPS="${MAX_STEPS:-1200}"
PROBE_UNTIL="${PROBE_UNTIL:-1000}"
PROBE_EVERY="${PROBE_EVERY:-50}"
LR="${LR:-2e-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-in100_grad_dynamics}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
EVAL_RFID_NUM_SAMPLES="${EVAL_RFID_NUM_SAMPLES:-512}"
EVAL_RFID_BATCH_SIZE="${EVAL_RFID_BATCH_SIZE:-64}"
EVAL_RFID_TMP_DIR="${EVAL_RFID_TMP_DIR:-/tmp}"
SKIP_RFID="${SKIP_RFID:-0}"

COMMON_ARGS=(
  --hf_dataset_id "${HF_DATASET_ID}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_steps "${MAX_STEPS}"
  --probe_until "${PROBE_UNTIL}"
  --probe_every "${PROBE_EVERY}"
  --lr "${LR}"
  --output_root "${OUTPUT_ROOT}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --eval_every "${EVAL_EVERY}"
  --eval_max_batches "${EVAL_MAX_BATCHES}"
  --eval_rfid_num_samples "${EVAL_RFID_NUM_SAMPLES}"
  --eval_rfid_batch_size "${EVAL_RFID_BATCH_SIZE}"
  --eval_rfid_tmp_dir "${EVAL_RFID_TMP_DIR}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  COMMON_ARGS+=(--cache_dir "${CACHE_DIR}")
fi
if [[ -n "${ENCODER_CKPT}" ]]; then
  COMMON_ARGS+=(--encoder_ckpt "${ENCODER_CKPT}")
fi
if [[ "${SKIP_RFID}" == "1" ]]; then
  COMMON_ARGS+=(--skip_rfid)
fi

run_one() {
  local init_mode="$1"
  local run_name="${RUN_NAME_PREFIX}_${init_mode}_s${SEED}"

  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    echo "[run][ddp:${NUM_GPUS}gpus] ${init_mode}"
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
      -m unirae.train_imagenet100_dynamics \
      --encoder_init "${init_mode}" \
      --run_name "${run_name}" \
      "${COMMON_ARGS[@]}"
  else
    echo "[run][single] ${init_mode}"
    python -m unirae.train_imagenet100_dynamics \
      --encoder_init "${init_mode}" \
      --run_name "${run_name}" \
      "${COMMON_ARGS[@]}"
  fi
}

run_one scratch
run_one dinov2

echo "[done] outputs:"
echo "  ${OUTPUT_ROOT}/${RUN_NAME_PREFIX}_scratch_s${SEED}"
echo "  ${OUTPUT_ROOT}/${RUN_NAME_PREFIX}_dinov2_s${SEED}"
