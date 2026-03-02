#!/usr/bin/env bash
# Fair 20k MA-LAGA sweep:
# CIFAR100 + swin_tiny_patch4 + bs128 + seed42 + steps20000.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/unirae_radio/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_patch4_ma_laga20k}
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

OUT_TSV=${OUT_TSV:-/scratch/peilab/xlubl/unirae_runs/ma_laga20k_jobs_$(date +%Y%m%d_%H%M%S).tsv}
mkdir -p "$(dirname "${OUT_TSV}")"
echo -e "jobid\trun_name\talign_gamma\tnorm_restore" > "${OUT_TSV}"

submit_one () {
  local run_name="$1"
  local align_gamma="$2"
  local norm_restore="$3"

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
  extra+=" --set train.grad_strategy=ma_laga"
  extra+=" --set train.lambda_txt=${LAMBDA_TXT}"
  extra+=" --set train.lambda_rec=${LAMBDA_REC}"
  extra+=" --set train.grad_norm_mode=${NORM_MODE}"
  extra+=" --set train.grad_norm_scope=${NORM_SCOPE}"
  extra+=" --set train.grad_norm_layers=${NORM_LAYERS}"
  extra+=" --set train.ma_laga_align_gamma=${align_gamma}"
  extra+=" --set train.ma_laga_norm_restore=${norm_restore}"
  extra+=" --set train.ma_laga_eps=1e-8"
  extra+=" --set train.lambda_var=0.0"
  extra+=" --set train.lambda_gbvc=0.0"

  local out
  out=$(sbatch \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh configs/cifar100_baseline_joint_cagrad.yaml "${run_name}")
  local job_id
  job_id=$(echo "${out}" | awk '{print $4}')
  echo -e "${job_id}\t${run_name}\t${align_gamma}\t${norm_restore}" >> "${OUT_TSV}"
  echo "${job_id}"
}

# MA-LAGA_v1: align_gamma=0.5, norm_restore=False
r="${RUN_PREFIX}_v1_g0p5_restore0_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "0.5" "false")
echo "[submit_ma_laga_fair20k] submitted ${j} ${r}"

# MA-LAGA_v2: align_gamma=1.0, norm_restore=False
r="${RUN_PREFIX}_v2_g1p0_restore0_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "1.0" "false")
echo "[submit_ma_laga_fair20k] submitted ${j} ${r}"

# MA-LAGA_v3: align_gamma=0.5, norm_restore=True
r="${RUN_PREFIX}_v3_g0p5_restore1_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "0.5" "true")
echo "[submit_ma_laga_fair20k] submitted ${j} ${r}"

# MA-LAGA_v4: align_gamma=1.0, norm_restore=True
r="${RUN_PREFIX}_v4_g1p0_restore1_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "1.0" "true")
echo "[submit_ma_laga_fair20k] submitted ${j} ${r}"

echo "[submit_ma_laga_fair20k] job table: ${OUT_TSV}"
