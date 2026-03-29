#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
CONFIG=${CONFIG:-configs/cifar100_baseline_joint_cagrad.yaml}

RUNS_ROOT=${RUNS_ROOT:-${ROOT_DIR}/runs}
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/results/cifar100_50k_locked_multiseed_$(date +%Y%m%d_%H%M%S)}
MANIFEST=${MANIFEST:-${OUT_DIR}/jobs.tsv}
LAUNCH=${LAUNCH:-true}

MAX_PARALLEL=${MAX_PARALLEL:-4}
POLL_SECONDS=${POLL_SECONDS:-20}
START_DELAY_SECONDS=${START_DELAY_SECONDS:-5}
REUSE_SCRIPT=${REUSE_SCRIPT:-scripts/reuse_or_start_srun_train.sh}
LAUNCHER=${LAUNCHER:-scripts/launch_manifest_srun.py}

HOLD_JOB_NAME=${HOLD_JOB_NAME:-dsga-cifar100-hold}
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-32G}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE:-8}

DATA_ROOT=${DATA_ROOT:-${ROOT_DIR}/data/cifar100}
SEEDS=${SEEDS:-42,43,44}
STEPS=${STEPS:-50000}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
VAL_RATIO=${VAL_RATIO:-0.1}
BACKBONE=${BACKBONE:-resnet18}
IMAGE_SIZE=${IMAGE_SIZE:-32}
PRETRAINED=${PRETRAINED:-false}

LAMBDA_TXT=${LAMBDA_TXT:-1.0}
LAMBDA_REC=${LAMBDA_REC:-1.0}
LR=${LR:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-0}

LOG_EVERY=${LOG_EVERY:-100}
EVAL_EVERY=${EVAL_EVERY:-5000}

CAGRAD_BETA=${CAGRAD_BETA:-0.35}
DSGA_GLOBAL_GAMMA=${DSGA_GLOBAL_GAMMA:-0.5}
DSGA_LAYERWISE_GAMMA=${DSGA_LAYERWISE_GAMMA:-0.5}
DSGA_ADAPTIVE_GAMMA=${DSGA_ADAPTIVE_GAMMA:-0.5}
DSGA_ADAPTIVE_STRENGTH=${DSGA_ADAPTIVE_STRENGTH:-8.0}
DSGA_ADAPTIVE_POWER=${DSGA_ADAPTIVE_POWER:-0.5}
DSGA_NORM_RESTORE=${DSGA_NORM_RESTORE:-false}

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

mkdir -p "${OUT_DIR}" "${RUNS_ROOT}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo -e "jobid\trun_name\tlabel\tseed\tstrategy\tcagrad_beta\tlaga_grouping\tdsga_m_scope\tdsga_m_align_gamma\tdsga_layer_adaptive_blend\tdsga_layer_adaptive_strength\tdsga_layer_adaptive_power\tlaunch_cmd" > "${MANIFEST}"
fi

manifest_has_label_seed() {
  local label="$1"
  local seed="$2"
  awk -F'\t' -v target_label="${label}" -v target_seed="${seed}" 'NR>1 && $3==target_label && $4==target_seed {found=1} END {exit(found?0:1)}' "${MANIFEST}"
}

build_launch_cmd() {
  local run_name="$1"
  shift
  local overrides=("$@")
  local cmd=""
  local item

  cmd+="cd $(printf '%q' "${ROOT_DIR}")"
  cmd+=" && source /home/xlubl/anaconda3/etc/profile.d/conda.sh"
  cmd+=" && conda activate $(printf '%q' "${CONDA_ENV}")"
  cmd+=" && export OMP_NUM_THREADS=$(printf '%q' "${OMP_NUM_THREADS_VALUE}")"
  cmd+=" && export PYTHONUNBUFFERED=1"
  cmd+=" && MAIN_PROCESS_PORT=\$((10000 + RANDOM % 50000))"
  cmd+=" && python -m accelerate.commands.launch"
  cmd+=" --num_processes 1"
  cmd+=" --num_machines 1"
  cmd+=" --main_process_port \$MAIN_PROCESS_PORT"
  cmd+=" --mixed_precision no"
  cmd+=" -m $(printf '%q' "${TRAIN_MODULE}")"
  cmd+=" --config $(printf '%q' "${CONFIG}")"
  cmd+=" --run_name $(printf '%q' "${run_name}")"
  for item in "${overrides[@]}"; do
    cmd+=" --set $(printf '%q' "${item}")"
  done
  echo "${cmd}"
}

append_one() {
  local run_name="$1"
  local label="$2"
  local seed="$3"
  local strategy="$4"
  local cagrad_beta="$5"
  local laga_grouping="$6"
  local dsga_m_scope="$7"
  local dsga_m_align_gamma="$8"
  local dsga_layer_adaptive_blend="$9"
  local dsga_layer_adaptive_strength="${10}"
  local dsga_layer_adaptive_power="${11}"

  if manifest_has_label_seed "${label}" "${seed}"; then
    echo "[skip] ${label} seed=${seed} already present in manifest"
    return 0
  fi

  local overrides=(
    "seed=${seed}"
    "output.root=${RUNS_ROOT}"
    "data.dataset=cifar100"
    "data.root=${DATA_ROOT}"
    "data.batch_size=${BATCH_SIZE}"
    "data.num_workers=${NUM_WORKERS}"
    "data.image_size=${IMAGE_SIZE}"
    "data.val_from_train=false"
    "data.val_ratio=${VAL_RATIO}"
    "model.backbone=${BACKBONE}"
    "model.pretrained=${PRETRAINED}"
    "train.mode=joint"
    "train.steps=${STEPS}"
    "train.lambda_txt=${LAMBDA_TXT}"
    "train.lambda_rec=${LAMBDA_REC}"
    "train.grad_strategy=${strategy}"
    "train.cagrad_beta=${cagrad_beta}"
    "train.laga_grouping=${laga_grouping}"
    "train.dsga_m_scope=${dsga_m_scope}"
    "train.dsga_m_align_gamma=${dsga_m_align_gamma}"
    "train.dsga_m_norm_restore=${DSGA_NORM_RESTORE}"
    "train.dsga_d_mode=full"
    "train.dsga_d_conflict_threshold=0.0"
    "train.dsga_d_conflict_only=false"
    "train.dsga_layer_adaptive_blend=${dsga_layer_adaptive_blend}"
    "train.dsga_layer_adaptive_strength=${dsga_layer_adaptive_strength}"
    "train.dsga_layer_adaptive_power=${dsga_layer_adaptive_power}"
    "optim.lr=${LR}"
    "optim.weight_decay=${WEIGHT_DECAY}"
    "optim.warmup_steps=${WARMUP_STEPS}"
    "log.every=${LOG_EVERY}"
    "log.cos_every=${LOG_EVERY}"
    "log.save_every=${STEPS}"
    "log.eval_every=${EVAL_EVERY}"
    "eval.split=test"
    "eval.batch_size=${EVAL_BATCH_SIZE}"
    "eval.max_batches=null"
    "eval.save_recon_samples=false"
    "eval.compute_rfid=false"
    "accelerate.mixed_precision=no"
  )

  local launch_cmd
  launch_cmd="$(build_launch_cmd "${run_name}" "${overrides[@]}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "" "${run_name}" "${label}" "${seed}" "${strategy}" "${cagrad_beta}" "${laga_grouping}" "${dsga_m_scope}" "${dsga_m_align_gamma}" \
    "${dsga_layer_adaptive_blend}" "${dsga_layer_adaptive_strength}" "${dsga_layer_adaptive_power}" "${launch_cmd}" >> "${MANIFEST}"
  echo "[planned] ${run_name}"
}

for seed in "${SEED_ARR[@]}"; do
  append_one "cifar100_main_naive_seed${seed}" "naive" "${seed}" "naive" "0.0" "global" "global" "0.0" "false" "0.0" "1.0"
  append_one "cifar100_main_pcgrad_seed${seed}" "pcgrad" "${seed}" "pcgrad" "0.0" "global" "global" "0.0" "false" "0.0" "1.0"
  append_one "cifar100_main_cagrad_seed${seed}" "cagrad" "${seed}" "cagrad" "${CAGRAD_BETA}" "global" "global" "0.0" "false" "0.0" "1.0"
  append_one "cifar100_main_dsga_global_seed${seed}" "dsga_global" "${seed}" "dsga" "0.0" "global" "global" "${DSGA_GLOBAL_GAMMA}" "false" "0.0" "1.0"
  append_one "cifar100_main_dsga_layerwise_seed${seed}" "dsga_layerwise" "${seed}" "dsga" "0.0" "layerwise" "global" "${DSGA_LAYERWISE_GAMMA}" "false" "0.0" "1.0"
  append_one "cifar100_main_dsga_adaptive_seed${seed}" "dsga_adaptive" "${seed}" "dsga" "0.0" "layerwise" "global" "${DSGA_ADAPTIVE_GAMMA}" "true" "${DSGA_ADAPTIVE_STRENGTH}" "${DSGA_ADAPTIVE_POWER}"
done

echo "[done] manifest=${MANIFEST}"
echo "[hint] aggregate after completion:"
echo "python scripts/aggregate_cifar_multiseed.py --manifest ${MANIFEST} --runs_root ${RUNS_ROOT} --out_root ${OUT_DIR}"

if [[ "${LAUNCH}" == "true" ]]; then
  export ACCOUNT PARTITION HOLD_JOB_NAME NODES GPUS_PER_NODE CPUS_PER_TASK MEM TIME_LIMIT
  python "${LAUNCHER}" \
    --manifest "${MANIFEST}" \
    --runs_root "${RUNS_ROOT}" \
    --reuse_script "${REUSE_SCRIPT}" \
    --max_parallel "${MAX_PARALLEL}" \
    --poll_seconds "${POLL_SECONDS}" \
    --start_delay_seconds "${START_DELAY_SECONDS}" \
    --resume
fi
