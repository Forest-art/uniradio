#!/usr/bin/env bash
set -euo pipefail

# DSGA lambda_mag sensitivity sweep (CIFAR100, Swin-Tiny-P4, joint protocol)
# NOTE:
# - In this codebase there is no literal `lambda_mag` flag.
# - We map `lambda_mag` -> `train.dsga_m_align_gamma` (legacy alias: train.ma_laga_align_gamma).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG=${CONFIG:-configs/cifar100_baseline_joint_cagrad.yaml}
DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
SEED=${SEED:-42}
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}

# Keep protocol aligned with previous component ablation.
BACKBONE=${BACKBONE:-swin_tiny_patch4}
RUN_PREFIX=${RUN_PREFIX:-dsga_swin_cifar100_mag}
LOG_ROOT=${LOG_ROOT:-"${ROOT_DIR}/runs/${RUN_PREFIX}_logs_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${LOG_ROOT}"

MAG_VALUES=(0.1 0.2 0.3 0.4)
GPU_IDS=(0 1 2 3)

if [[ ${#MAG_VALUES[@]} -ne ${#GPU_IDS[@]} ]]; then
  echo "[error] MAG_VALUES and GPU_IDS length mismatch." >&2
  exit 1
fi

echo "[info] root=${ROOT_DIR}"
echo "[info] config=${CONFIG}"
echo "[info] data_root=${DATA_ROOT}"
echo "[info] logs=${LOG_ROOT}"

declare -a PIDS=()
declare -a RUNS=()
declare -a MAGS=()

for i in "${!MAG_VALUES[@]}"; do
  mag="${MAG_VALUES[$i]}"
  gpu="${GPU_IDS[$i]}"
  mag_tag="${mag/./p}"
  run_name="${RUN_PREFIX}_${mag_tag}_s${SEED}"
  log_file="${LOG_ROOT}/${run_name}.log"

  echo "[launch] GPU ${gpu} | lambda_mag=${mag} | run=${run_name}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONUNBUFFERED=1 \
  python -m accelerate.commands.launch --num_processes 1 -m unirae.train_cifar10 \
    --config "${CONFIG}" \
    --run_name "${run_name}" \
    --set "seed=${SEED}" \
    --set "data.dataset=cifar100" \
    --set "data.root=${DATA_ROOT}" \
    --set "data.image_size=32" \
    --set "data.batch_size=${BATCH_SIZE}" \
    --set "data.num_workers=${NUM_WORKERS}" \
    --set "model.backbone=${BACKBONE}" \
    --set "model.pretrained=false" \
    --set "train.mode=joint" \
    --set "train.steps=${STEPS}" \
    --set "train.grad_strategy=dsga" \
    --set "train.shared_params=backbone" \
    --set "train.lambda_txt=1.0" \
    --set "train.lambda_rec=1.0" \
    --set "train.grad_norm_mode=mean" \
    --set "train.grad_norm_scope=conflict_all" \
    --set "train.grad_norm_layers=layer3+layer4" \
    --set "train.laga_grouping=layerwise" \
    --set "train.dsga_d_mode=full" \
    --set "train.dsga_m_scope=global" \
    --set "train.dsga_m_norm_restore=false" \
    --set "train.dsga_m_align_gamma=${mag}" \
    --set "train.ma_laga_align_gamma=${mag}" \
    --set "train.dsga_m_eps=1e-8" \
    --set "train.lambda_var=0.0" \
    --set "train.lambda_gbvc=0.0" \
    --set "accelerate.mixed_precision=no" \
    > "${log_file}" 2>&1 &

  pid=$!
  PIDS+=("${pid}")
  RUNS+=("${run_name}")
  MAGS+=("${mag}")
done

fail=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  run_name="${RUNS[$i]}"
  mag="${MAGS[$i]}"
  if wait "${pid}"; then
    echo "[done] lambda_mag=${mag} | run=${run_name}"
  else
    echo "[fail] lambda_mag=${mag} | run=${run_name}" >&2
    fail=1
  fi
done

echo "[info] all processes finished. logs: ${LOG_ROOT}"
exit "${fail}"
