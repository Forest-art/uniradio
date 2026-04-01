#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

HOLD_JOB_ID="${HOLD_JOB_ID:-362171}"
CONDA_ENV="${CONDA_ENV:-diffuser310}"

SEED="${SEED:-42}"
STEPS="${STEPS:-50000}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/slurm/logs}"

LAGA_GROUPING="${LAGA_GROUPING:-layerwise}"
DSGA_M_SCOPE="${DSGA_M_SCOPE:-global}"
DSGA_M_ALIGN_GAMMA="${DSGA_M_ALIGN_GAMMA:-0.5}"
DSGA_M_NORM_RESTORE="${DSGA_M_NORM_RESTORE:-false}"
DSGA_LAYER_ADAPTIVE_BLEND="${DSGA_LAYER_ADAPTIVE_BLEND:-false}"
DSGA_LAYER_ADAPTIVE_STRENGTH="${DSGA_LAYER_ADAPTIVE_STRENGTH:-0.0}"
DSGA_LAYER_ADAPTIVE_POWER="${DSGA_LAYER_ADAPTIVE_POWER:-1.0}"

ENC_ANCHOR_REC_GATE_STRENGTH="${ENC_ANCHOR_REC_GATE_STRENGTH:-0.1}"
ENC_ANCHOR_REC_GATE_MIN="${ENC_ANCHOR_REC_GATE_MIN:-0.7}"

RUN_NAME="${RUN_NAME:-cifar100_main_dsga_enc_anchor_${LAGA_GROUPING}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

mkdir -p "${RUNS_ROOT}" "${LOG_DIR}"

read -r -d '' TRAIN_CMD <<EOF || true
cd "${ROOT_DIR}" && \
source /home/xlubl/anaconda3/etc/profile.d/conda.sh && \
conda activate "${CONDA_ENV}" && \
export OMP_NUM_THREADS=8 && \
export PYTHONUNBUFFERED=1 && \
MAIN_PROCESS_PORT=\$((10000 + RANDOM % 50000)) && \
python -m accelerate.commands.launch \
  --num_processes 1 \
  --num_machines 1 \
  --main_process_port "\$MAIN_PROCESS_PORT" \
  --mixed_precision no \
  -m unirae.train_cifar10 \
  --config configs/cifar100_baseline_joint_cagrad.yaml \
  --run_name "${RUN_NAME}" \
  --set output.root="${RUNS_ROOT}" \
  --set seed="${SEED}" \
  --set data.dataset=cifar100 \
  --set data.root="${ROOT_DIR}/data/cifar100" \
  --set data.batch_size=256 \
  --set data.num_workers=8 \
  --set data.image_size=32 \
  --set data.val_from_train=false \
  --set data.val_ratio=0.1 \
  --set model.backbone=resnet18 \
  --set model.pretrained=false \
  --set train.mode=joint \
  --set train.steps="${STEPS}" \
  --set train.lambda_txt=1.0 \
  --set train.lambda_rec=1.0 \
  --set train.grad_strategy=dsga \
  --set train.laga_grouping="${LAGA_GROUPING}" \
  --set train.dsga_m_scope="${DSGA_M_SCOPE}" \
  --set train.dsga_m_align_gamma="${DSGA_M_ALIGN_GAMMA}" \
  --set train.dsga_m_norm_restore="${DSGA_M_NORM_RESTORE}" \
  --set train.dsga_d_mode=full \
  --set train.dsga_d_conflict_threshold=0.0 \
  --set train.dsga_d_conflict_only=false \
  --set train.dsga_layer_adaptive_blend="${DSGA_LAYER_ADAPTIVE_BLEND}" \
  --set train.dsga_layer_adaptive_strength="${DSGA_LAYER_ADAPTIVE_STRENGTH}" \
  --set train.dsga_layer_adaptive_power="${DSGA_LAYER_ADAPTIVE_POWER}" \
  --set train.enc_anchor_rec_gate_strength="${ENC_ANCHOR_REC_GATE_STRENGTH}" \
  --set train.enc_anchor_rec_gate_min="${ENC_ANCHOR_REC_GATE_MIN}" \
  --set optim.lr=3e-4 \
  --set optim.weight_decay=1e-4 \
  --set optim.warmup_steps=0 \
  --set log.every=100 \
  --set log.cos_every=100 \
  --set log.save_every="${STEPS}" \
  --set log.eval_every=5000 \
  --set eval.split=test \
  --set eval.batch_size=256 \
  --set eval.max_batches=null \
  --set eval.save_recon_samples=false \
  --set eval.compute_rfid=false \
  --set accelerate.mixed_precision=no
EOF

echo "[launch] hold_job_id=${HOLD_JOB_ID}"
echo "[launch] run_name=${RUN_NAME}"
echo "[launch] run_dir=${RUN_DIR}"
echo "[launch] log_file=${LOG_FILE}"
echo "[launch] gate_strength=${ENC_ANCHOR_REC_GATE_STRENGTH} gate_min=${ENC_ANCHOR_REC_GATE_MIN}"

srun --jobid "${HOLD_JOB_ID}" --overlap bash -lc "${TRAIN_CMD}" 2>&1 | tee -a "${LOG_FILE}"
