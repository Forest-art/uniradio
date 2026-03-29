#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG=${CONFIG:-configs/cifar100_baseline_joint_naive.yaml}
DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/unirae_radio/data/cifar100}
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-3407}
STEPS=${STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
RUN_PREFIX=${RUN_PREFIX:-item2_cifar100_vits_dsga_mag}
LOG_ROOT=${LOG_ROOT:-"${ROOT_DIR}/runs/${RUN_PREFIX}_logs_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${LOG_ROOT}"

MAG_VALUES_STR=${MAG_VALUES:-"0.1 0.2 0.4 0.6 0.8 1.0"}
read -r -a MAG_VALUES <<< "${MAG_VALUES_STR}"
GPU_IDS_STR=${GPU_IDS:-"0 1 2 3"}
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "[error] GPU_IDS is empty" >&2
  exit 1
fi

echo "[info] root=${ROOT_DIR}"
echo "[info] config=${CONFIG}"
echo "[info] data_root=${DATA_ROOT}"
echo "[info] seed=${SEED} steps=${STEPS} batch=${BATCH_SIZE}"
echo "[info] gamma_values=${MAG_VALUES[*]}"
echo "[info] gpu_ids=${GPU_IDS[*]}"
echo "[info] logs=${LOG_ROOT}"

fail=0
for ((offset=0; offset<${#MAG_VALUES[@]}; offset+=${#GPU_IDS[@]})); do
  declare -a PIDS=()
  declare -a RUNS=()
  declare -a MAGS=()

  for i in "${!GPU_IDS[@]}"; do
    idx=$((offset + i))
    if [[ ${idx} -ge ${#MAG_VALUES[@]} ]]; then
      break
    fi
    mag="${MAG_VALUES[$idx]}"
    gpu="${GPU_IDS[$i]}"
    mag_tag="${mag/./p}"
    run_name="${RUN_PREFIX}_${mag_tag}_s${SEED}"
    log_file="${LOG_ROOT}/${run_name}.log"

    echo "[launch] gpu=${gpu} gamma=${mag} run=${run_name}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -m accelerate.commands.launch --num_processes 1 -m unirae.train_cifar10 \
      --config "${CONFIG}" \
      --run_name "${run_name}" \
      --set "seed=${SEED}" \
      --set "data.dataset=cifar100" \
      --set "data.root=${DATA_ROOT}" \
      --set "data.image_size=32" \
      --set "data.batch_size=${BATCH_SIZE}" \
      --set "data.num_workers=${NUM_WORKERS}" \
      --set "model.backbone=vit_small" \
      --set "model.pretrained=false" \
      --set "train.mode=joint" \
      --set "train.steps=${STEPS}" \
      --set "train.grad_strategy=dsga" \
      --set "train.shared_params=backbone" \
      --set "train.lambda_txt=1.0" \
      --set "train.lambda_rec=1.0" \
      --set "train.grad_norm_mode=mean" \
      --set "train.grad_norm_scope=conflict_all" \
      --set "train.laga_grouping=layerwise" \
      --set "train.dsga_d_mode=full" \
      --set "train.dsga_d_conflict_threshold=0.0" \
      --set "train.dsga_d_conflict_only=false" \
      --set "train.dsga_m_scope=global" \
      --set "train.dsga_m_norm_restore=false" \
      --set "train.dsga_m_align_gamma=${mag}" \
      --set "train.probe_every=50" \
      --set "train.probe_until=${STEPS}" \
      --set "optim.lr=5e-4" \
      --set "optim.weight_decay=1e-4" \
      --set "log.every=50" \
      --set "log.cos_every=50" \
      --set "log.save_every=1000" \
      --set "log.eval_every=2000" \
      --set "eval.split=test" \
      --set "eval.batch_size=${EVAL_BATCH_SIZE}" \
      --set "eval.max_batches=40" \
      --set "eval.save_recon_samples=false" \
      --set "eval.compute_rfid=true" \
      --set "eval.rfid_num_samples=2048" \
      --set "eval.rfid_batch_size=${EVAL_BATCH_SIZE}" \
      --set "accelerate.mixed_precision=no" \
      > "${log_file}" 2>&1 &

    pid=$!
    PIDS+=("${pid}")
    RUNS+=("${run_name}")
    MAGS+=("${mag}")
  done

  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    run_name="${RUNS[$i]}"
    mag="${MAGS[$i]}"
    if wait "${pid}"; then
      echo "[done] gamma=${mag} run=${run_name}"
    else
      echo "[fail] gamma=${mag} run=${run_name}" >&2
      fail=1
    fi
  done

done

echo "[info] sweep finished. logs=${LOG_ROOT}"
exit "${fail}"
