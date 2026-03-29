#!/usr/bin/env bash
set -euo pipefail

# Submit 3 baselines for ImageNet-100 dynamics training:
# 1) scratch joint     : encoder_init=scratch, lambda_u=1, lambda_g=100
# 2) scratch gen-only  : encoder_init=scratch, lambda_u=0, lambda_g=100
# 3) dinov2 joint      : encoder_init=dinov2,  lambda_u=1, lambda_g=100

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
TIME_LIMIT=${TIME_LIMIT:-0-24:00:00}
CPUS=${CPUS:-8}
MEM=${MEM:-64G}
GPUS=${GPUS:-1}
CONDA_ENV=${CONDA_ENV:-diffuser}

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
SEED=${SEED:-42}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/long_runs}
RUN_GROUP=${RUN_GROUP:-baseline3_10k_eval1k_$(date +%Y%m%d_%H%M%S)}
ENCODER_CKPT=${ENCODER_CKPT:-}

mkdir -p slurm/logs

submit_one() {
  local job_name="$1"
  local encoder_init="$2"
  local lambda_u="$3"
  local lambda_g="$4"
  local run_name="$5"

  local maybe_ckpt=""
  if [[ "$encoder_init" == "dinov2" && -n "$ENCODER_CKPT" ]]; then
    maybe_ckpt="--encoder_ckpt $ENCODER_CKPT"
  fi

  local cmd
  cmd=$(cat <<EOC
set -euo pipefail
cd /project/peilab/luxiaocheng/projects/DSGA
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
python -m unirae.train_imagenet100_dynamics \
  --encoder_init ${encoder_init} \
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
  --lambda_u ${lambda_u} \
  --lambda_g ${lambda_g} \
  --seed ${SEED} \
  --output_root ${OUTPUT_ROOT} \
  --device auto \
  ${maybe_ckpt}
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

run_scratch_joint="${RUN_GROUP}_scratch_joint_lu1_lg100_s${SEED}"
run_scratch_gen="${RUN_GROUP}_scratch_genonly_lu0_lg100_s${SEED}"
run_dino_joint="${RUN_GROUP}_dinov2_joint_lu1_lg100_s${SEED}"

jid1=$(submit_one "in100_sj10k" "scratch" "1.0" "100.0" "$run_scratch_joint")
jid2=$(submit_one "in100_sg10k" "scratch" "0.0" "100.0" "$run_scratch_gen")
jid3=$(submit_one "in100_dj10k" "dinov2"  "1.0" "100.0" "$run_dino_joint")

cat <<EOM
[submitted]
  scratch_joint : job_id=${jid1}
    run_dir=/project/peilab/luxiaocheng/projects/DSGA/${OUTPUT_ROOT}/${run_scratch_joint}
    log=slurm/logs/in100_sj10k-${jid1}.out

  scratch_genonly : job_id=${jid2}
    run_dir=/project/peilab/luxiaocheng/projects/DSGA/${OUTPUT_ROOT}/${run_scratch_gen}
    log=slurm/logs/in100_sg10k-${jid2}.out

  dinov2_joint : job_id=${jid3}
    run_dir=/project/peilab/luxiaocheng/projects/DSGA/${OUTPUT_ROOT}/${run_dino_joint}
    log=slurm/logs/in100_dj10k-${jid3}.out

[monitor]
  squeue -u xlubl -A ${ACCOUNT}
  scontrol show job ${jid1}
  scontrol show job ${jid2}
  scontrol show job ${jid3}
EOM
