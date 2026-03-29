#!/usr/bin/env bash
set -euo pipefail

# Fair comparison for gradient strategies on ImageNet-100 dynamics:
# naive vs pcgrad vs cagrad

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
TIME_LIMIT=${TIME_LIMIT:-0-24:00:00}
CPUS=${CPUS:-8}
MEM=${MEM:-64G}
GPUS=${GPUS:-1}
CONDA_ENV=${CONDA_ENV:-diffuser}

ENCODER_INIT=${ENCODER_INIT:-scratch}
HF_DATASET_ID=${HF_DATASET_ID:-clane9/imagenet-100}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_STEPS=${MAX_STEPS:-10000}
PROBE_UNTIL=${PROBE_UNTIL:-10000}
PROBE_EVERY=${PROBE_EVERY:-500}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-50}
EVAL_RFID_NUM_SAMPLES=${EVAL_RFID_NUM_SAMPLES:-512}
EVAL_RFID_BATCH_SIZE=${EVAL_RFID_BATCH_SIZE:-64}
EVAL_RFID_TMP_DIR=${EVAL_RFID_TMP_DIR:-/tmp}
LR=${LR:-2e-4}
LAMBDA_U=${LAMBDA_U:-1.0}
LAMBDA_G=${LAMBDA_G:-100.0}
CAGRAD_BETA=${CAGRAD_BETA:-0.5}
SEED=${SEED:-42}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/long_runs}
RUN_GROUP=${RUN_GROUP:-gradstrategy3_10k_eval1k_$(date +%Y%m%d_%H%M%S)}
ENCODER_CKPT=${ENCODER_CKPT:-}

mkdir -p slurm/logs

submit_one() {
  local job_name="$1"
  local strategy="$2"
  local run_name="$3"

  local maybe_ckpt=""
  if [[ "$ENCODER_INIT" == "dinov2" && -n "$ENCODER_CKPT" ]]; then
    maybe_ckpt="--encoder_ckpt $ENCODER_CKPT"
  fi

  local extra_cagrad=""
  if [[ "$strategy" == "cagrad" ]]; then
    extra_cagrad="--cagrad_beta ${CAGRAD_BETA}"
  fi

  local cmd
  cmd=$(cat <<EOC
set -euo pipefail
cd /project/peilab/luxiaocheng/projects/DSGA
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
python -m unirae.train_imagenet100_dynamics \
  --encoder_init ${ENCODER_INIT} \
  --grad_strategy ${strategy} \
  --run_name ${run_name} \
  --hf_dataset_id ${HF_DATASET_ID} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --max_steps ${MAX_STEPS} \
  --probe_until ${PROBE_UNTIL} \
  --probe_every ${PROBE_EVERY} \
  --eval_every ${EVAL_EVERY} \
  --eval_max_batches ${EVAL_MAX_BATCHES} \
  --eval_rfid_num_samples ${EVAL_RFID_NUM_SAMPLES} \
  --eval_rfid_batch_size ${EVAL_RFID_BATCH_SIZE} \
  --eval_rfid_tmp_dir ${EVAL_RFID_TMP_DIR} \
  --lr ${LR} \
  --lambda_u ${LAMBDA_U} \
  --lambda_g ${LAMBDA_G} \
  --seed ${SEED} \
  --output_root ${OUTPUT_ROOT} \
  --device auto \
  ${maybe_ckpt} \
  ${extra_cagrad}
EOC
)

  sbatch \
    --parsable \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --gpus-per-node="${GPUS}" \
    --time="${TIME_LIMIT}" \
    --job-name="${job_name}" \
    --output="slurm/logs/${job_name}-%j.out" \
    --error="slurm/logs/${job_name}-%j.err" \
    --wrap "$cmd"
}

run_naive="${RUN_GROUP}_${ENCODER_INIT}_naive_lu${LAMBDA_U}_lg${LAMBDA_G}_s${SEED}"
run_pcgrad="${RUN_GROUP}_${ENCODER_INIT}_pcgrad_lu${LAMBDA_U}_lg${LAMBDA_G}_s${SEED}"
run_cagrad="${RUN_GROUP}_${ENCODER_INIT}_cagrad_lu${LAMBDA_U}_lg${LAMBDA_G}_s${SEED}"

jid1=$(submit_one "in100_gn10k" "naive" "$run_naive")
jid2=$(submit_one "in100_gp10k" "pcgrad" "$run_pcgrad")
jid3=$(submit_one "in100_gc10k" "cagrad" "$run_cagrad")

cat <<EOM
[submitted]
  naive : job_id=${jid1}
    run_dir=/project/peilab/luxiaocheng/projects/DSGA/${OUTPUT_ROOT}/${run_naive}
    log=slurm/logs/in100_gn10k-${jid1}.out

  pcgrad : job_id=${jid2}
    run_dir=/project/peilab/luxiaocheng/projects/DSGA/${OUTPUT_ROOT}/${run_pcgrad}
    log=slurm/logs/in100_gp10k-${jid2}.out

  cagrad : job_id=${jid3}
    run_dir=/project/peilab/luxiaocheng/projects/DSGA/${OUTPUT_ROOT}/${run_cagrad}
    log=slurm/logs/in100_gc10k-${jid3}.out

[fairness]
  encoder_init=${ENCODER_INIT}, seed=${SEED}, steps=${MAX_STEPS}, eval_every=${EVAL_EVERY}
  batch_size=${BATCH_SIZE}, lr=${LR}, lambda_u=${LAMBDA_U}, lambda_g=${LAMBDA_G}
  cagrad_beta=${CAGRAD_BETA}
EOM
