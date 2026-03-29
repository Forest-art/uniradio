#!/usr/bin/env bash
set -euo pipefail

# Table-1 style RAE pretrained baselines on ImageNet-100:
# - DINOv2-B
# - MAE-B
# - SigLIP2-B (used as the third released RAE pretrained baseline)
#
# Metrics per run:
# - linear probe top1 (probe_steps)
# - recon MSE / rMSE
# - rFID

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
TIME_LIMIT=${TIME_LIMIT:-0-24:00:00}
CPUS=${CPUS:-8}
MEM=${MEM:-96G}
GPUS=${GPUS:-1}
CONDA_ENV=${CONDA_ENV:-diffuser310}

RAE_CODE_ROOT=${RAE_CODE_ROOT:-/project/peilab/luxiaocheng/projects/RAE}
HF_DATASET=${HF_DATASET:-clane9/imagenet-100}
HF_CONFIG=${HF_CONFIG:-}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-validation}

BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
PROBE_STEPS=${PROBE_STEPS:-10000}
PROBE_LR=${PROBE_LR:-1e-3}
PROBE_WEIGHT_DECAY=${PROBE_WEIGHT_DECAY:-0.0}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-0}
RFID_NUM_SAMPLES=${RFID_NUM_SAMPLES:-5000}
RFID_BATCH_SIZE=${RFID_BATCH_SIZE:-64}
RFID_TMP_DIR=${RFID_TMP_DIR:-/scratch/peilab/xlubl/tmp_rfid}

OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/peilab/xlubl/dsga_runs}
RUN_GROUP=${RUN_GROUP:-in100_table1_rae_bs${BATCH_SIZE}_probe${PROBE_STEPS}_$(date +%Y%m%d_%H%M%S)}

mkdir -p slurm/logs

submit_one() {
  local method="$1"
  local stage1_cfg="$2"
  local run_name="$3"
  local out_json="${OUTPUT_ROOT}/${run_name}/eval_summary.json"

  local maybe_hf_cfg=""
  if [[ -n "${HF_CONFIG}" ]]; then
    maybe_hf_cfg="--hf_config ${HF_CONFIG}"
  fi

  local cmd
  cmd=$(cat <<EOC
set -euo pipefail
cd /project/peilab/luxiaocheng/projects/DSGA
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
python -m unirae.eval_rae_table1_baselines \
  --rae_code_root ${RAE_CODE_ROOT} \
  --stage1_config ${stage1_cfg} \
  --hf_dataset ${HF_DATASET} \
  --train_split ${TRAIN_SPLIT} \
  --val_split ${VAL_SPLIT} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --probe_steps ${PROBE_STEPS} \
  --probe_lr ${PROBE_LR} \
  --probe_weight_decay ${PROBE_WEIGHT_DECAY} \
  --max_eval_batches ${MAX_EVAL_BATCHES} \
  --rfid_num_samples ${RFID_NUM_SAMPLES} \
  --rfid_batch_size ${RFID_BATCH_SIZE} \
  --rfid_tmp_dir ${RFID_TMP_DIR} \
  --out_json ${out_json} \
  ${maybe_hf_cfg}
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
    --job-name="t1_${method}" \
    --output="slurm/logs/t1_${method}-%j.out" \
    --error="slurm/logs/t1_${method}-%j.err" \
    --wrap "$cmd"
}

CFG_DINO="${RAE_CODE_ROOT}/configs/stage1/pretrained/DINOv2-B.yaml"
CFG_MAE="${RAE_CODE_ROOT}/configs/stage1/pretrained/MAE.yaml"
CFG_SIGLIP2="${RAE_CODE_ROOT}/configs/stage1/pretrained/SigLIP2.yaml"

for p in "${CFG_DINO}" "${CFG_MAE}" "${CFG_SIGLIP2}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[error] missing config: ${p}" >&2
    exit 1
  fi
done

run_dino="${RUN_GROUP}_dinov2b"
run_mae="${RUN_GROUP}_maeb"
run_siglip2="${RUN_GROUP}_siglip2b"

jid1=$(submit_one "dino" "${CFG_DINO}" "${run_dino}")
jid2=$(submit_one "mae" "${CFG_MAE}" "${run_mae}")
jid3=$(submit_one "siglip2" "${CFG_SIGLIP2}" "${run_siglip2}")

cat <<EOM
[submitted]
  dino     job=${jid1}
    out=${OUTPUT_ROOT}/${run_dino}/eval_summary.json
    log=slurm/logs/t1_dino-${jid1}.out

  mae      job=${jid2}
    out=${OUTPUT_ROOT}/${run_mae}/eval_summary.json
    log=slurm/logs/t1_mae-${jid2}.out

  siglip2  job=${jid3}
    out=${OUTPUT_ROOT}/${run_siglip2}/eval_summary.json
    log=slurm/logs/t1_siglip2-${jid3}.out

[setting]
  hf_dataset=${HF_DATASET}
  batch_size=${BATCH_SIZE}
  probe_steps=${PROBE_STEPS}
  rfid_num_samples=${RFID_NUM_SAMPLES}
  run_group=${RUN_GROUP}
EOM
