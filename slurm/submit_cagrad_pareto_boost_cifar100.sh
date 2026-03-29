#!/usr/bin/env bash
# Sequential CAGrad tuning for CIFAR-100 Pareto improvement.
# Target: beat naive(conflict_deep) on both acc and mse.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_cagrad_pareto_boost}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
LAYERS=${LAYERS:-layer3+layer4}
NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_deep}
LAMBDA_REC=${LAMBDA_REC:-1.0}
NUM_WORKERS=${NUM_WORKERS:-4}

# beta lambda_txt conflict_only threshold nonconflict_merge
GRID=(
  "0.35 1.00 0 0.00 cagrad"
  "0.20 1.10 0 0.00 cagrad"
  "0.10 1.20 0 0.00 cagrad"
  "0.35 1.10 1 0.00 sum"
  "0.50 1.10 1 0.00 sum"
  "0.35 1.20 1 -0.05 sum"
)

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
  local beta="$1"
  local ltxt="$2"
  local conflict_only="$3"
  local threshold="$4"
  local nonconflict="$5"
  local dep_job="$6"

  local beta_tag ltxt_tag thr_tag
  beta_tag=$(tag_float "${beta}")
  ltxt_tag=$(tag_float "${ltxt}")
  thr_tag=$(tag_float "${threshold}")
  local run_name="${RUN_PREFIX}_b${beta_tag}_ltxt${ltxt_tag}_co${conflict_only}_thr${thr_tag}_ncm${nonconflict}_s${SEED}"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set data.num_workers=${NUM_WORKERS} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.mode=joint --set train.grad_strategy=cagrad --set train.cagrad_beta=${beta} --set train.cagrad_conflict_only=${conflict_only} --set train.cagrad_conflict_threshold=${threshold} --set train.cagrad_nonconflict_merge=${nonconflict} --set train.lambda_txt=${ltxt} --set train.lambda_rec=${LAMBDA_REC} --set train.grad_norm_mode=${NORM_MODE} --set train.grad_norm_scope=${NORM_SCOPE} --set train.grad_norm_layers=${LAYERS}"

  local dep_args=()
  if [[ -n "${dep_job}" ]]; then
    dep_args=(--dependency="afterany:${dep_job}")
  fi

  local out
  out=$(sbatch \
    "${dep_args[@]}" \
    --export=ALL,TRAIN_MODULE="${TRAIN_MODULE}",CONDA_ENV="${CONDA_ENV}",EXTRA_ARGS="${extra}" \
    slurm/sbatch_train.sh "configs/cifar100_baseline_joint_cagrad.yaml" "${run_name}")
  local job_id
  job_id=$(echo "${out}" | awk '{print $4}')
  echo "${job_id}"
}

prev_job=""
for item in "${GRID[@]}"; do
  # shellcheck disable=SC2086
  jid=$(submit_one ${item} "${prev_job}")
  echo "[submit_cagrad_pareto_boost_cifar100] submitted ${jid} (dep=${prev_job:-none})"
  prev_job="${jid}"
done

echo "[submit_cagrad_pareto_boost_cifar100] done seed=${SEED} steps=${STEPS} grid=${#GRID[@]} layers=${LAYERS} workers=${NUM_WORKERS}"
