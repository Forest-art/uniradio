#!/usr/bin/env bash
#SBATCH --job-name=unirae_sweep
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

CONFIG_PATH=${1:-configs/sweep_joint_conflict.yaml}
SWEEP_MODE=${2:-auto}            # text | recon | joint_naive | joint_pcgrad | auto
CONDA_ENV=${CONDA_ENV:-diffuser310}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train}
RUN_PREFIX=${RUN_PREFIX:-}

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

readarray -t META < <(python - "$CONFIG_PATH" "$SWEEP_MODE" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}
mode_in = sys.argv[2]
sweep_cfg = cfg.get('sweep', {})
mode = mode_in if mode_in != 'auto' else str(sweep_cfg.get('mode', 'joint_pcgrad'))

lambdas = sweep_cfg.get('lambda_values', [0.25, 0.5, 1.0, 2.0])
seeds = sweep_cfg.get('seeds', [42, 43])
dataset = str(cfg.get('data', {}).get('dataset', 'imagenet'))
backbone = str(cfg.get('model', {}).get('backbone', 'radio'))

print(mode)
print(dataset)
print(backbone)
print(','.join(str(x) for x in lambdas))
print(','.join(str(int(x)) for x in seeds))
PY
)

MODE=${META[0]}
DATASET=${META[1]}
BACKBONE=${META[2]}
IFS=',' read -r -a LAMBDAS <<<"${META[3]}"
IFS=',' read -r -a SEEDS <<<"${META[4]}"

TOTAL=$(( ${#LAMBDAS[@]} * ${#SEEDS[@]} ))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}, expected [0, $((TOTAL-1))]"
  exit 1
fi

LIDX=$(( TASK_ID / ${#SEEDS[@]} ))
SIDX=$(( TASK_ID % ${#SEEDS[@]} ))
LAMBDA=${LAMBDAS[$LIDX]}
SEED=${SEEDS[$SIDX]}

case "$MODE" in
  text)
    LTXT="$LAMBDA"; LREC="0"; STRATEGY="naive"; GROUP="text_only" ;;
  recon)
    LTXT="0"; LREC="$LAMBDA"; STRATEGY="naive"; GROUP="recon_only" ;;
  joint_naive)
    LTXT="$LAMBDA"; LREC="$LAMBDA"; STRATEGY="naive"; GROUP="joint_naive" ;;
  joint_conflict|joint_pcgrad)
    LTXT="$LAMBDA"; LREC="$LAMBDA"; STRATEGY="pcgrad"; GROUP="joint_pcgrad" ;;
  *)
    echo "Unknown mode=$MODE"
    exit 1
    ;;
esac

if [[ -n "$RUN_PREFIX" ]]; then
  RUN_NAME="${RUN_PREFIX}_${MODE}_l${LAMBDA}_s${SEED}"
else
  RUN_NAME="${DATASET}_${BACKBONE}_${MODE}_l${LAMBDA}_s${SEED}"
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}
MIXED_PRECISION=${MIXED_PRECISION:-no}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$((10000 + SLURM_JOB_ID % 50000))}

python -m accelerate.commands.launch \
  --num_processes "$NPROC_PER_NODE" \
  --num_machines 1 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision "$MIXED_PRECISION" \
  -m "$TRAIN_MODULE" \
  --config "$CONFIG_PATH" \
  --run_name "$RUN_NAME" \
  --set seed="$SEED" \
  --set train.lambda_txt="$LTXT" \
  --set train.lambda_rec="$LREC" \
  --set train.strategy="$STRATEGY" \
  --set experiment.group="$GROUP"
