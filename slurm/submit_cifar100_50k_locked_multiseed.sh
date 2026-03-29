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
mkdir -p "${OUT_DIR}"
echo -e "jobid\trun_name\tlabel\tseed\tstrategy\tcagrad_beta\tlaga_grouping\tdsga_m_scope\tdsga_m_align_gamma\tdsga_layer_adaptive_blend\tdsga_layer_adaptive_strength\tdsga_layer_adaptive_power" > "${MANIFEST}"

submit_one() {
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

  local extra=""
  extra+=" --set seed=${seed}"
  extra+=" --set output.root=${RUNS_ROOT}"
  extra+=" --set data.dataset=cifar100"
  extra+=" --set data.root=${DATA_ROOT}"
  extra+=" --set data.batch_size=${BATCH_SIZE}"
  extra+=" --set data.num_workers=${NUM_WORKERS}"
  extra+=" --set data.image_size=${IMAGE_SIZE}"
  extra+=" --set data.val_from_train=true"
  extra+=" --set data.val_ratio=${VAL_RATIO}"
  extra+=" --set model.backbone=${BACKBONE}"
  extra+=" --set model.pretrained=${PRETRAINED}"
  extra+=" --set train.mode=joint"
  extra+=" --set train.steps=${STEPS}"
  extra+=" --set train.lambda_txt=${LAMBDA_TXT}"
  extra+=" --set train.lambda_rec=${LAMBDA_REC}"
  extra+=" --set train.grad_strategy=${strategy}"
  extra+=" --set train.cagrad_beta=${cagrad_beta}"
  extra+=" --set train.laga_grouping=${laga_grouping}"
  extra+=" --set train.dsga_m_scope=${dsga_m_scope}"
  extra+=" --set train.dsga_m_align_gamma=${dsga_m_align_gamma}"
  extra+=" --set train.dsga_m_norm_restore=${DSGA_NORM_RESTORE}"
  extra+=" --set train.dsga_d_mode=full"
  extra+=" --set train.dsga_d_conflict_threshold=0.0"
  extra+=" --set train.dsga_d_conflict_only=false"
  extra+=" --set train.dsga_layer_adaptive_blend=${dsga_layer_adaptive_blend}"
  extra+=" --set train.dsga_layer_adaptive_strength=${dsga_layer_adaptive_strength}"
  extra+=" --set train.dsga_layer_adaptive_power=${dsga_layer_adaptive_power}"
  extra+=" --set optim.lr=${LR}"
  extra+=" --set optim.weight_decay=${WEIGHT_DECAY}"
  extra+=" --set optim.warmup_steps=${WARMUP_STEPS}"
  extra+=" --set log.every=${LOG_EVERY}"
  extra+=" --set log.cos_every=${LOG_EVERY}"
  extra+=" --set log.save_every=${STEPS}"
  extra+=" --set log.eval_every=${EVAL_EVERY}"
  extra+=" --set eval.split=test"
  extra+=" --set eval.batch_size=${EVAL_BATCH_SIZE}"
  extra+=" --set eval.max_batches=null"
  extra+=" --set eval.save_recon_samples=false"
  extra+=" --set eval.compute_rfid=false"
  extra+=" --set accelerate.mixed_precision=no"

  local job_id
  job_id=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "${CONFIG}" "${run_name}")
  echo -e "${job_id}\t${run_name}\t${label}\t${seed}\t${strategy}\t${cagrad_beta}\t${laga_grouping}\t${dsga_m_scope}\t${dsga_m_align_gamma}\t${dsga_layer_adaptive_blend}\t${dsga_layer_adaptive_strength}\t${dsga_layer_adaptive_power}" >> "${MANIFEST}"
  echo "[submitted] ${job_id} ${run_name}"
}

for seed in "${SEED_ARR[@]}"; do
  submit_one "cifar100_main_naive_seed${seed}_$(date +%Y%m%d_%H%M%S)" "naive" "${seed}" "naive" "0.0" "global" "global" "0.0" "false" "0.0" "0.0"
  submit_one "cifar100_main_pcgrad_seed${seed}_$(date +%Y%m%d_%H%M%S)" "pcgrad" "${seed}" "pcgrad" "0.0" "global" "global" "0.0" "false" "0.0" "0.0"
  submit_one "cifar100_main_cagrad_seed${seed}_$(date +%Y%m%d_%H%M%S)" "cagrad" "${seed}" "cagrad" "${CAGRAD_BETA}" "global" "global" "0.0" "false" "0.0" "0.0"
  submit_one "cifar100_main_dsga_global_seed${seed}_$(date +%Y%m%d_%H%M%S)" "dsga_global" "${seed}" "dsga" "0.0" "global" "global" "${DSGA_GLOBAL_GAMMA}" "false" "0.0" "0.0"
  submit_one "cifar100_main_dsga_layerwise_seed${seed}_$(date +%Y%m%d_%H%M%S)" "dsga_layerwise" "${seed}" "dsga" "0.0" "layerwise" "global" "${DSGA_LAYERWISE_GAMMA}" "false" "0.0" "0.0"
  submit_one "cifar100_main_dsga_adaptive_seed${seed}_$(date +%Y%m%d_%H%M%S)" "dsga_adaptive" "${seed}" "dsga" "0.0" "layerwise" "global" "${DSGA_ADAPTIVE_GAMMA}" "true" "${DSGA_ADAPTIVE_STRENGTH}" "${DSGA_ADAPTIVE_POWER}"
done

echo "[done] manifest=${MANIFEST}"
echo "[hint] aggregate after completion:"
echo "python scripts/aggregate_cifar_multiseed.py --manifest ${MANIFEST} --runs_root ${RUNS_ROOT} --out_root ${OUT_DIR}"
