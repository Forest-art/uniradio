#!/usr/bin/env bash
# Compare gradient intervention:
# - baseline joint training
# - text warmup then joint training
# Strategies: naive | pcgrad | cagrad (seed=42 by default)

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/project/peilab/luxiaocheng/projects/unirae_radio/data/cifar100}
STEPS=${STEPS:-20000}
WARMUP_STEPS=${WARMUP_STEPS:-4000}
SEED=${SEED:-42}
RUN_PREFIX=${RUN_PREFIX:-cifar100_textwarmup_cmp}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train_cifar10}
CONDA_ENV=${CONDA_ENV:-diffuser310}
LAYERS=${LAYERS:-layer3+layer4}
NORM_MODE=${NORM_MODE:-mean}
NORM_SCOPE=${NORM_SCOPE:-conflict_deep}
NUM_WORKERS=${NUM_WORKERS:-4}

submit_one () {
  local cfg="$1"
  local run_name="$2"
  local strategy="$3"
  local warmup="$4"
  local cagrad_beta="$5"
  local dep_job="$6"

  local extra="--set seed=${SEED} --set data.root=${DATA_ROOT} --set data.num_workers=${NUM_WORKERS} --set train.steps=${STEPS} --set accelerate.mixed_precision=no --set train.mode=joint --set train.grad_strategy=${strategy} --set train.cagrad_beta=${cagrad_beta} --set train.text_warmup_steps=${warmup} --set train.lambda_txt=1.0 --set train.lambda_rec=1.0 --set train.grad_norm_mode=${NORM_MODE} --set train.grad_norm_scope=${NORM_SCOPE} --set train.grad_norm_layers=${LAYERS}"

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

# baseline: no warmup
jid=$(submit_one "configs/cifar100_baseline_joint_naive.yaml"  "${RUN_PREFIX}_naive_joint_s${SEED}"  "naive"  "0"            "0.35" "${prev_job}")
echo "[submit_textwarmup_compare_cifar100] submitted ${jid} (naive joint)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_pcgrad.yaml" "${RUN_PREFIX}_pcgrad_joint_s${SEED}" "pcgrad" "0"            "0.35" "${prev_job}")
echo "[submit_textwarmup_compare_cifar100] submitted ${jid} (pcgrad joint)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_cagrad_joint_s${SEED}" "cagrad" "0"            "0.35" "${prev_job}")
echo "[submit_textwarmup_compare_cifar100] submitted ${jid} (cagrad joint)"
prev_job="${jid}"

# intervention: text warmup then joint
jid=$(submit_one "configs/cifar100_baseline_joint_naive.yaml"  "${RUN_PREFIX}_naive_warmup${WARMUP_STEPS}_s${SEED}"  "naive"  "${WARMUP_STEPS}" "0.35" "${prev_job}")
echo "[submit_textwarmup_compare_cifar100] submitted ${jid} (naive warmup)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_pcgrad.yaml" "${RUN_PREFIX}_pcgrad_warmup${WARMUP_STEPS}_s${SEED}" "pcgrad" "${WARMUP_STEPS}" "0.35" "${prev_job}")
echo "[submit_textwarmup_compare_cifar100] submitted ${jid} (pcgrad warmup)"
prev_job="${jid}"

jid=$(submit_one "configs/cifar100_baseline_joint_cagrad.yaml" "${RUN_PREFIX}_cagrad_warmup${WARMUP_STEPS}_s${SEED}" "cagrad" "${WARMUP_STEPS}" "0.35" "${prev_job}")
echo "[submit_textwarmup_compare_cifar100] submitted ${jid} (cagrad warmup)"
prev_job="${jid}"

echo "[submit_textwarmup_compare_cifar100] done seed=${SEED} steps=${STEPS} warmup=${WARMUP_STEPS} workers=${NUM_WORKERS}"
