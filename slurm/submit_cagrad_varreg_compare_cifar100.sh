#!/usr/bin/env bash
# Compare feature variance regularization on top of strong fixed CAGrad line.
# Chain: naive_ref -> cagrad_base -> cagrad_varreg(lambda_var grid)

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_cagrad_varreg_cmp}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
LAYERS=${LAYERS:-layer3+layer4}
NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_deep}
NUM_WORKERS=${NUM_WORKERS:-4}
CAGRAD_BETA=${CAGRAD_BETA:-0.35}
LAMBDA_TXT=${LAMBDA_TXT:-1.0}
LAMBDA_REC=${LAMBDA_REC:-1.0}
VAR_GAMMA=${VAR_GAMMA:-1.0}
VAR_EPS=${VAR_EPS:-1e-4}
VAR_GRID=${VAR_GRID:-"0.05 0.10 0.20"}

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
  local cfg="$1"
  local run_name="$2"
  local strategy="$3"
  local lambda_var="$4"
  local dep_job="$5"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set data.num_workers=${NUM_WORKERS} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.mode=joint --set train.grad_strategy=${strategy} --set train.lambda_txt=${LAMBDA_TXT} --set train.lambda_rec=${LAMBDA_REC} --set train.grad_norm_mode=${NORM_MODE} --set train.grad_norm_scope=${NORM_SCOPE} --set train.grad_norm_layers=${LAYERS} --set train.cagrad_beta=${CAGRAD_BETA} --set train.lambda_var=${lambda_var} --set train.var_gamma=${VAR_GAMMA} --set train.var_eps=${VAR_EPS}"

  local dep_args=()
  if [[ -n "${dep_job}" ]]; then
    dep_args=(--dependency="afterany:${dep_job}")
  fi

  local out
  out=$(sbatch \
    "${dep_args[@]}" \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "${cfg}" "${run_name}")
  local job_id
  job_id=$(echo "${out}" | awk '{print $4}')
  echo "${job_id}"
}

prev_job=""
jid=$(submit_one "configs/cifar100_baseline_joint_naive.yaml" "${RUN_PREFIX}_naive_ref_s${SEED}" "naive" "0.0" "${prev_job}")
echo "[submit_cagrad_varreg_compare_cifar100] submitted ${jid} (naive_ref)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_cagrad_base_b$(tag_float ${CAGRAD_BETA})_s${SEED}" "cagrad" "0.0" "${prev_job}")
echo "[submit_cagrad_varreg_compare_cifar100] submitted ${jid} (cagrad_base)"
prev_job="${jid}"

for lam in ${VAR_GRID}; do
  lam_tag=$(tag_float "${lam}")
  run_name="${RUN_PREFIX}_cagrad_var_l${lam_tag}_s${SEED}"
  jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${run_name}" "cagrad" "${lam}" "${prev_job}")
  echo "[submit_cagrad_varreg_compare_cifar100] submitted ${jid} (${run_name})"
  prev_job="${jid}"
done

echo "[submit_cagrad_varreg_compare_cifar100] done seed=${SEED} steps=${STEPS} beta=${CAGRAD_BETA} var_grid='${VAR_GRID}'"
