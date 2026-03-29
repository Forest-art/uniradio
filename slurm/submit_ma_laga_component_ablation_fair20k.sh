#!/usr/bin/env bash
# Strict MA-LAGA component ablation (fair 20k):
# 1) Vanilla Joint (naive)
# 2) Global CAGrad (beta=0.5)
# 3) Pure LAGA (direction_only)
# 4) Pure MA (magnitude_only)
# 5) Full MA-LAGA (full)

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_patch4_ma_laga_ablation20k}
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

OUT_TSV=${OUT_TSV:-/scratch/peilab/xlubl/unirae_runs/ma_laga_component_ablation20k_jobs_$(date +%Y%m%d_%H%M%S).tsv}
mkdir -p "$(dirname "${OUT_TSV}")"
echo -e "jobid\trun_name\tstrategy\tma_laga_mode\talign_gamma\tnorm_restore\tcagrad_beta" > "${OUT_TSV}"

submit_one () {
  local run_name="$1"
  local strategy="$2"
  local ma_laga_mode="$3"
  local align_gamma="$4"
  local norm_restore="$5"
  local cagrad_beta="$6"

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
  extra+=" --set train.cagrad_beta=${cagrad_beta}"
  extra+=" --set train.ma_laga_mode=${ma_laga_mode}"
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
  echo -e "${job_id}\t${run_name}\t${strategy}\t${ma_laga_mode}\t${align_gamma}\t${norm_restore}\t${cagrad_beta}" >> "${OUT_TSV}"
  echo "${job_id}"
}

# 1) Vanilla Joint
r="${RUN_PREFIX}_vanilla_naive_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "naive" "full" "1.0" "false" "0.5")
echo "[submit_ma_laga_component_ablation] submitted ${j} ${r}"

# 2) Global CAGrad
r="${RUN_PREFIX}_cagrad_b0p5_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "cagrad" "full" "1.0" "false" "0.5")
echo "[submit_ma_laga_component_ablation] submitted ${j} ${r}"

# 3) Pure LAGA (direction only)
r="${RUN_PREFIX}_pure_laga_dironly_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "ma_laga" "direction_only" "1.0" "false" "0.5")
echo "[submit_ma_laga_component_ablation] submitted ${j} ${r}"

# 4) Pure MA (magnitude only)
r="${RUN_PREFIX}_pure_ma_magonly_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "ma_laga" "magnitude_only" "1.0" "false" "0.5")
echo "[submit_ma_laga_component_ablation] submitted ${j} ${r}"

# 5) Full MA-LAGA
r="${RUN_PREFIX}_full_ma_laga_s${SEED}_$(date +%Y%m%d_%H%M%S)"
j=$(submit_one "${r}" "ma_laga" "full" "1.0" "false" "0.5")
echo "[submit_ma_laga_component_ablation] submitted ${j} ${r}"

echo "[submit_ma_laga_component_ablation] job table: ${OUT_TSV}"
