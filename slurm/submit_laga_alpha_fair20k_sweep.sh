#!/usr/bin/env bash
# Fair 20k LAGA alpha sweep:
# CIFAR100 + swin_tiny_patch4 + bs128 + seed42 + steps20000.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_patch4_laga_alpha20k}
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

OUT_TSV=${OUT_TSV:-/scratch/peilab/xlubl/unirae_runs/laga_alpha20k_jobs_$(date +%Y%m%d_%H%M%S).tsv}
mkdir -p "$(dirname "${OUT_TSV}")"
echo -e "jobid\trun_name\tthresh\trestore\talpha_mode\talpha_power\talpha_min\talpha_max\tlambda_gbvc" > "${OUT_TSV}"

submit_one () {
  local run_name="$1"
  local threshold="$2"
  local restore="$3"
  local alpha_mode="$4"
  local alpha_power="$5"
  local alpha_min="$6"
  local alpha_max="$7"
  local lambda_gbvc="$8"

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
  extra+=" --set train.grad_strategy=laga"
  extra+=" --set train.lambda_txt=${LAMBDA_TXT}"
  extra+=" --set train.lambda_rec=${LAMBDA_REC}"
  extra+=" --set train.grad_norm_mode=${NORM_MODE}"
  extra+=" --set train.grad_norm_scope=${NORM_SCOPE}"
  extra+=" --set train.grad_norm_layers=${NORM_LAYERS}"
  extra+=" --set train.laga_eps=1e-8"
  extra+=" --set train.laga_conflict_threshold=${threshold}"
  extra+=" --set train.laga_restore_ratio=${restore}"
  extra+=" --set train.laga_alpha_mode=${alpha_mode}"
  extra+=" --set train.laga_alpha_power=${alpha_power}"
  extra+=" --set train.laga_alpha_min=${alpha_min}"
  extra+=" --set train.laga_alpha_max=${alpha_max}"
  extra+=" --set train.lambda_gbvc=${lambda_gbvc}"
  extra+=" --set train.gbvc_nu=1.0"
  extra+=" --set train.gbvc_eps=1e-8"

  local out
  out=$(sbatch \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh configs/cifar100_baseline_joint_cagrad.yaml "${run_name}")
  local job_id
  job_id=$(echo "${out}" | awk '{print $4}')
  echo -e "${job_id}\t${run_name}\t${threshold}\t${restore}\t${alpha_mode}\t${alpha_power}\t${alpha_min}\t${alpha_max}\t${lambda_gbvc}" >> "${OUT_TSV}"
  echo "${job_id}"
}

# fixed baseline (should reproduce current best LAGA family baseline)
r="${RUN_PREFIX}_thm01_r0_afixed_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "-0.1" "0.0" "fixed" "1.0" "1.0" "1.0" "0.0")
echo "[submit_laga_alpha_fair20k] submitted ${j} ${r}"

# adaptive alpha candidates
r="${RUN_PREFIX}_thm01_r0_aratio_p1_m005_M1_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "-0.1" "0.0" "ratio" "1.0" "0.05" "1.0" "0.0")
echo "[submit_laga_alpha_fair20k] submitted ${j} ${r}"

r="${RUN_PREFIX}_thm01_r0_aratio_p1_m01_M06_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "-0.1" "0.0" "ratio" "1.0" "0.1" "0.6" "0.0")
echo "[submit_laga_alpha_fair20k] submitted ${j} ${r}"

r="${RUN_PREFIX}_thm01_r025_aratio_p1_m005_M1_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "-0.1" "0.25" "ratio" "1.0" "0.05" "1.0" "0.0")
echo "[submit_laga_alpha_fair20k] submitted ${j} ${r}"

r="${RUN_PREFIX}_thm005_r025_aratio_p1_m005_M1_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "-0.05" "0.25" "ratio" "1.0" "0.05" "1.0" "0.0")
echo "[submit_laga_alpha_fair20k] submitted ${j} ${r}"

r="${RUN_PREFIX}_thm01_r025_aratio_p05_m01_M1_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "-0.1" "0.25" "ratio" "0.5" "0.1" "1.0" "0.0")
echo "[submit_laga_alpha_fair20k] submitted ${j} ${r}"

echo "[submit_laga_alpha_fair20k] job table: ${OUT_TSV}"
