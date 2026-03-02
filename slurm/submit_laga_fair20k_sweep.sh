#!/usr/bin/env bash
# Fair 20k comparison sweep:
# CIFAR100 + swin_tiny_patch4 + bs128 + seed42 + steps20000
# Includes one cagrad reference and several LAGA candidates.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/unirae_radio/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_patch4_fair20k}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-128}
BACKBONE=${BACKBONE:-swin_tiny_patch4}
IMAGE_SIZE=${IMAGE_SIZE:-32}
PRETRAINED=${PRETRAINED:-false}

NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_all}
NORM_LAYERS=${NORM_LAYERS:-layer3+layer4}
LAMBDA_TXT=${LAMBDA_TXT:-1.0}
LAMBDA_REC=${LAMBDA_REC:-1.0}

OUT_TSV=${OUT_TSV:-/scratch/peilab/xlubl/unirae_runs/laga_fair20k_jobs_$(date +%Y%m%d_%H%M%S).tsv}
mkdir -p "$(dirname "${OUT_TSV}")"
echo -e "jobid\trun_name\tstrategy\tbeta\tthresh\trestore\tlambda_gbvc\tgbvc_nu" > "${OUT_TSV}"

tag_float () {
  local x="$1"
  local neg=""
  if [[ "${x}" == -* ]]; then
    neg="m"
    x="${x#-}"
  fi
  echo "${neg}$(echo "${x}" | tr '.' 'p')"
}

submit_one () {
  local run_name="$1"
  local strategy="$2"
  local beta="$3"
  local threshold="$4"
  local restore="$5"
  local lambda_gbvc="$6"
  local gbvc_nu="$7"

  local extra=""
  extra+=" --set seed=${SEED}"
  extra+=" --set data.root=${DATA_ROOT}"
  extra+=" --set data.batch_size=${BATCH_SIZE}"
  extra+=" --set data.num_workers=${NUM_WORKERS}"
  extra+=" --set data.image_size=${IMAGE_SIZE}"
  extra+=" --set model.backbone=${BACKBONE}"
  extra+=" --set model.pretrained=${PRETRAINED}"
  extra+=" --set train.steps=${STEPS}"
  extra+=" --set accelerate.mixed_precision=no"
  extra+=" --set train.mode=joint"
  extra+=" --set train.grad_strategy=${strategy}"
  extra+=" --set train.lambda_txt=${LAMBDA_TXT}"
  extra+=" --set train.lambda_rec=${LAMBDA_REC}"
  extra+=" --set train.grad_norm_mode=${NORM_MODE}"
  extra+=" --set train.grad_norm_scope=${NORM_SCOPE}"
  extra+=" --set train.grad_norm_layers=${NORM_LAYERS}"
  extra+=" --set train.cagrad_beta=${beta}"
  extra+=" --set train.laga_conflict_threshold=${threshold}"
  extra+=" --set train.laga_restore_ratio=${restore}"
  extra+=" --set train.lambda_gbvc=${lambda_gbvc}"
  extra+=" --set train.gbvc_nu=${gbvc_nu}"
  extra+=" --set train.gbvc_eps=1e-8"

  local out
  out=$(sbatch \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh configs/cifar100_baseline_joint_cagrad.yaml "${run_name}")
  local job_id
  job_id=$(echo "${out}" | awk '{print $4}')
  echo -e "${job_id}\t${run_name}\t${strategy}\t${beta}\t${threshold}\t${restore}\t${lambda_gbvc}\t${gbvc_nu}" >> "${OUT_TSV}"
  echo "${job_id}"
}

# cagrad reference
r="${RUN_PREFIX}_cagrad_b0p5_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "cagrad" "0.5" "0.0" "0.0" "0.0" "1.0")
echo "[submit_laga_fair20k] submitted ${j} ${r}"

# LAGA candidates
r="${RUN_PREFIX}_laga_thm01_r0_gb0_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "laga" "0.5" "-0.1" "0.0" "0.0" "1.0")
echo "[submit_laga_fair20k] submitted ${j} ${r}"

r="${RUN_PREFIX}_laga_thm01_r025_gb0_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "laga" "0.5" "-0.1" "0.25" "0.0" "1.0")
echo "[submit_laga_fair20k] submitted ${j} ${r}"

r="${RUN_PREFIX}_laga_thm01_r05_gb0_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "laga" "0.5" "-0.1" "0.5" "0.0" "1.0")
echo "[submit_laga_fair20k] submitted ${j} ${r}"

r="${RUN_PREFIX}_laga_thm02_r05_gb0_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "laga" "0.5" "-0.2" "0.5" "0.0" "1.0")
echo "[submit_laga_fair20k] submitted ${j} ${r}"

r="${RUN_PREFIX}_laga_thm01_r025_gb001_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "laga" "0.5" "-0.1" "0.25" "0.01" "1.0")
echo "[submit_laga_fair20k] submitted ${j} ${r}"

echo "[submit_laga_fair20k] job table: ${OUT_TSV}"
