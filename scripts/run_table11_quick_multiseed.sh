#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-configs/cifar100_baseline_joint_naive.yaml}
DATA_ROOT=${DATA_ROOT:-${ROOT_DIR}/data/cifar100}
RUN_PREFIX=${RUN_PREFIX:-table11_cifar100_vit_small}
RESULT_ROOT=${RESULT_ROOT:-${ROOT_DIR}/results/table11_quick_multiseed_$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-${RESULT_ROOT}/logs}
SEEDS_STR=${SEEDS:-"0 1 2"}
STRATEGIES_STR=${STRATEGIES:-"joint pcgrad cagrad dsga"}
read -r -a SEEDS <<< "${SEEDS_STR}"
read -r -a STRATEGIES <<< "${STRATEGIES_STR}"

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"

echo "[info] root=${ROOT_DIR}"
echo "[info] result_root=${RESULT_ROOT}"
echo "[info] seeds=${SEEDS[*]} strategies=${STRATEGIES[*]}"

for strategy in "${STRATEGIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="${RUN_PREFIX}_${strategy}_seed${seed}"
    log_file="${LOG_ROOT}/${run_name}.log"
    sets=(
      "seed=${seed}"
      "data.dataset=cifar100"
      "data.root=${DATA_ROOT}"
      "data.image_size=32"
      "data.batch_size=256"
      "data.num_workers=8"
      "data.download=true"
      "model.backbone=vit_small"
      "model.pretrained=false"
      "model.freeze_backbone=false"
      "train.mode=joint"
      "train.steps=10000"
      "train.lambda_txt=1.0"
      "train.lambda_rec=1.0"
      "train.shared_params=backbone"
      "train.recon_loss=rmse"
      "optim.lr=5e-4"
      "optim.weight_decay=1e-4"
      "optim.warmup_steps=1000"
      "log.every=100"
      "log.cos_every=100"
      "log.save_every=10000"
      "log.eval_every=10000"
      "eval.split=test"
      "eval.batch_size=256"
      "eval.max_batches=null"
      "eval.save_recon_samples=false"
      "eval.compute_rfid=false"
      "accelerate.mixed_precision=no"
    )

    case "${strategy}" in
      joint)
        sets+=("train.grad_strategy=naive")
        ;;
      pcgrad)
        sets+=("train.grad_strategy=pcgrad")
        ;;
      cagrad)
        sets+=("train.grad_strategy=cagrad" "train.cagrad_beta=0.35")
        ;;
      dsga)
        sets+=(
          "train.grad_strategy=dsga"
          "train.laga_grouping=layerwise"
          "train.dsga_d_mode=full"
          "train.dsga_d_conflict_threshold=0.0"
          "train.dsga_d_conflict_only=false"
          "train.dsga_m_scope=global"
          "train.dsga_m_norm_restore=false"
          "train.dsga_m_align_gamma=0.2"
        )
        ;;
      *)
        echo "[error] unsupported strategy=${strategy}" >&2
        exit 1
        ;;
    esac

    echo "[launch] strategy=${strategy} seed=${seed} run=${run_name}"
    cmd=("${PYTHON_BIN}" -m accelerate.commands.launch --num_processes 1 -m unirae.train_cifar10 --config "${CONFIG}" --run_name "${run_name}")
    for item in "${sets[@]}"; do
      cmd+=(--set "${item}")
    done
    PYTHONUNBUFFERED=1 "${cmd[@]}" > "${log_file}" 2>&1
    echo "[done] strategy=${strategy} seed=${seed} run=${run_name}"
  done
done

"${PYTHON_BIN}" scripts/aggregate_table11_quick_multiseed.py \
  --runs_root "${ROOT_DIR}/runs" \
  --run_prefix "${RUN_PREFIX}" \
  --out_root "${RESULT_ROOT}"

echo "[info] table11 done result_root=${RESULT_ROOT}"
