#!/usr/bin/env bash
# Tune adaptive-beta CAGrad on CIFAR-100.
# Runs sequentially to reduce preempt/instability.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/DSGA/data/cifar100}
STEPS=${STEPS:-20000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_cagrad_adaptivebeta_tune}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
LAYERS=${LAYERS:-layer3+layer4}
NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_deep}
NUM_WORKERS=${NUM_WORKERS:-4}
BETA_BASE=${BETA_BASE:-0.35}
LAMBDA_TXT=${LAMBDA_TXT:-1.0}

# threshold strength power cap nonconflict_merge
GRID=(
  "0.00 1.00 1.00 1.00 sum"
  "-0.10 0.60 1.00 0.55 cagrad"
  "-0.10 0.80 1.00 0.65 cagrad"
  "-0.20 0.80 1.00 0.60 cagrad"
  "-0.10 0.60 2.00 0.60 cagrad"
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
  local cfg="$1"
  local run_name="$2"
  local strategy="$3"
  local adaptive="$4"
  local thr="$5"
  local stren="$6"
  local power="$7"
  local cap="$8"
  local nonconf="$9"
  local dep_job="${10}"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set data.num_workers=${NUM_WORKERS} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.mode=joint --set train.grad_strategy=${strategy} --set train.cagrad_beta=${BETA_BASE} --set train.lambda_txt=${LAMBDA_TXT} --set train.lambda_rec=1.0 --set train.grad_norm_mode=${NORM_MODE} --set train.grad_norm_scope=${NORM_SCOPE} --set train.grad_norm_layers=${LAYERS} --set train.cagrad_adaptive_beta=${adaptive} --set train.cagrad_adaptive_scope=deep --set train.cagrad_adaptive_layers=${LAYERS} --set train.cagrad_adaptive_nonconflict_merge=${nonconf} --set train.cagrad_adaptive_conflict_threshold=${thr} --set train.cagrad_adaptive_strength=${stren} --set train.cagrad_adaptive_power=${power} --set train.cagrad_adaptive_beta_cap=${cap}"

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
jid=$(submit_one "configs/cifar100_baseline_joint_naive.yaml" "${RUN_PREFIX}_naive_ref_s${SEED}" "naive" "false" "0.0" "0.0" "1.0" "1.0" "sum" "${prev_job}")
echo "[submit_cagrad_adaptivebeta_tune_cifar100] submitted ${jid} (naive_ref)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_cagrad_fixed_b$(tag_float ${BETA_BASE})_s${SEED}" "cagrad" "false" "0.0" "0.0" "1.0" "1.0" "sum" "${prev_job}")
echo "[submit_cagrad_adaptivebeta_tune_cifar100] submitted ${jid} (cagrad_fixed)"
prev_job="${jid}"

for item in "${GRID[@]}"; do
  # shellcheck disable=SC2086
  read -r thr stren power cap nonconf <<< "${item}"
  thr_tag=$(tag_float "${thr}")
  str_tag=$(tag_float "${stren}")
  pow_tag=$(tag_float "${power}")
  cap_tag=$(tag_float "${cap}")
  run_name="${RUN_PREFIX}_ada_t${thr_tag}_s${str_tag}_p${pow_tag}_c${cap_tag}_ncm${nonconf}_s${SEED}"
  jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${run_name}" "cagrad" "true" "${thr}" "${stren}" "${power}" "${cap}" "${nonconf}" "${prev_job}")
  echo "[submit_cagrad_adaptivebeta_tune_cifar100] submitted ${jid} (${run_name})"
  prev_job="${jid}"
done

echo "[submit_cagrad_adaptivebeta_tune_cifar100] done seed=${SEED} steps=${STEPS} beta_base=${BETA_BASE} lambda_txt=${LAMBDA_TXT} grid=${#GRID[@]}"
