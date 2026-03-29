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
OUT_DIR=${OUT_DIR:-${ROOT_DIR}/results/cifar100_50k_val_tune_$(date +%Y%m%d_%H%M%S)}
MANIFEST=${MANIFEST:-${OUT_DIR}/jobs.tsv}

DATA_ROOT=${DATA_ROOT:-${ROOT_DIR}/data/cifar100}
TUNE_SEED=${TUNE_SEED:-3407}
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

CAGRAD_BETAS_STR=${CAGRAD_BETAS:-"0.2 0.35 0.5"}
DSGA_GLOBAL_GAMMAS_STR=${DSGA_GLOBAL_GAMMAS:-"0.2 0.5 0.8"}
DSGA_LAYERWISE_GAMMAS_STR=${DSGA_LAYERWISE_GAMMAS:-"0.2 0.5 0.8"}
DSGA_ADAPTIVE_GAMMAS_STR=${DSGA_ADAPTIVE_GAMMAS:-"0.2 0.5 0.8"}
DSGA_ADAPTIVE_STRENGTHS_STR=${DSGA_ADAPTIVE_STRENGTHS:-"4 8 16"}
DSGA_ADAPTIVE_POWERS_STR=${DSGA_ADAPTIVE_POWERS:-"0.5 1.0"}
DSGA_NORM_RESTORE=${DSGA_NORM_RESTORE:-false}

read -r -a CAGRAD_BETAS <<< "${CAGRAD_BETAS_STR}"
read -r -a DSGA_GLOBAL_GAMMAS <<< "${DSGA_GLOBAL_GAMMAS_STR}"
read -r -a DSGA_LAYERWISE_GAMMAS <<< "${DSGA_LAYERWISE_GAMMAS_STR}"
read -r -a DSGA_ADAPTIVE_GAMMAS <<< "${DSGA_ADAPTIVE_GAMMAS_STR}"
read -r -a DSGA_ADAPTIVE_STRENGTHS <<< "${DSGA_ADAPTIVE_STRENGTHS_STR}"
read -r -a DSGA_ADAPTIVE_POWERS <<< "${DSGA_ADAPTIVE_POWERS_STR}"

mkdir -p "${OUT_DIR}"
echo -e "jobid\trun_name\tlabel\tseed\tstrategy\tcagrad_beta\tlaga_grouping\tdsga_m_scope\tdsga_m_align_gamma\tdsga_layer_adaptive_blend\tdsga_layer_adaptive_strength\tdsga_layer_adaptive_power" > "${MANIFEST}"

tag_float() {
  local x="$1"
  local neg=""
  if [[ "${x}" == -* ]]; then
    neg="m"
    x="${x#-}"
  fi
  echo "${neg}$(echo "${x}" | tr '.' 'p')"
}

submit_one() {
  local run_name="$1"
  local label="$2"
  local strategy="$3"
  local cagrad_beta="$4"
  local laga_grouping="$5"
  local dsga_m_scope="$6"
  local dsga_m_align_gamma="$7"
  local dsga_layer_adaptive_blend="$8"
  local dsga_layer_adaptive_strength="$9"
  local dsga_layer_adaptive_power="${10}"

  local extra=""
  extra+=" --set seed=${TUNE_SEED}"
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
  extra+=" --set eval.split=val"
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
  echo -e "${job_id}\t${run_name}\t${label}\t${TUNE_SEED}\t${strategy}\t${cagrad_beta}\t${laga_grouping}\t${dsga_m_scope}\t${dsga_m_align_gamma}\t${dsga_layer_adaptive_blend}\t${dsga_layer_adaptive_strength}\t${dsga_layer_adaptive_power}" >> "${MANIFEST}"
  echo "[submitted] ${job_id} ${run_name}"
}

submit_one "cifar100_valtune_naive_s${TUNE_SEED}_$(date +%Y%m%d_%H%M%S)" "naive" "naive" "0.0" "global" "global" "0.0" "false" "0.0" "0.0"
submit_one "cifar100_valtune_pcgrad_s${TUNE_SEED}_$(date +%Y%m%d_%H%M%S)" "pcgrad" "pcgrad" "0.0" "global" "global" "0.0" "false" "0.0" "0.0"

for beta in "${CAGRAD_BETAS[@]}"; do
  submit_one "cifar100_valtune_cagrad_b$(tag_float "${beta}")_s${TUNE_SEED}_$(date +%Y%m%d_%H%M%S)" \
    "cagrad_b$(tag_float "${beta}")" "cagrad" "${beta}" "global" "global" "0.0" "false" "0.0" "0.0"
done

for gamma in "${DSGA_GLOBAL_GAMMAS[@]}"; do
  submit_one "cifar100_valtune_dsga_global_g$(tag_float "${gamma}")_s${TUNE_SEED}_$(date +%Y%m%d_%H%M%S)" \
    "dsga_global_g$(tag_float "${gamma}")" "dsga" "0.0" "global" "global" "${gamma}" "false" "0.0" "0.0"
done

for gamma in "${DSGA_LAYERWISE_GAMMAS[@]}"; do
  submit_one "cifar100_valtune_dsga_layerwise_g$(tag_float "${gamma}")_s${TUNE_SEED}_$(date +%Y%m%d_%H%M%S)" \
    "dsga_layerwise_g$(tag_float "${gamma}")" "dsga" "0.0" "layerwise" "global" "${gamma}" "false" "0.0" "0.0"
done

for gamma in "${DSGA_ADAPTIVE_GAMMAS[@]}"; do
  for strength in "${DSGA_ADAPTIVE_STRENGTHS[@]}"; do
    for power in "${DSGA_ADAPTIVE_POWERS[@]}"; do
      submit_one "cifar100_valtune_dsga_adaptive_g$(tag_float "${gamma}")_s$(tag_float "${strength}")_p$(tag_float "${power}")_seed${TUNE_SEED}_$(date +%Y%m%d_%H%M%S)" \
        "dsga_adaptive_g$(tag_float "${gamma}")_s$(tag_float "${strength}")_p$(tag_float "${power}")" \
        "dsga" "0.0" "layerwise" "global" "${gamma}" "true" "${strength}" "${power}"
    done
  done
done

echo "[done] manifest=${MANIFEST}"
echo "[hint] aggregate after completion:"
echo "python scripts/aggregate_cifar_multiseed.py --manifest ${MANIFEST} --runs_root ${RUNS_ROOT} --out_root ${OUT_DIR}"
