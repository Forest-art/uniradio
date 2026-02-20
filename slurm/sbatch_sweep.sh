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
SWEEP_MODE=${2:-joint_conflict}   # text | recon | joint_naive | joint_conflict
CONDA_ENV=${CONDA_ENV:-diffuser310}

LAMBDAS=(0.25 0.5 1 2)
SEEDS=(42 43)

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

case "$SWEEP_MODE" in
  text)
    LTXT="$LAMBDA"; LREC="0"; STRATEGY="naive"; GROUP="text_only" ;;
  recon)
    LTXT="0"; LREC="$LAMBDA"; STRATEGY="naive"; GROUP="recon_only" ;;
  joint_naive)
    LTXT="$LAMBDA"; LREC="$LAMBDA"; STRATEGY="naive"; GROUP="joint_naive" ;;
  joint_conflict)
    LTXT="$LAMBDA"; LREC="$LAMBDA"; STRATEGY="conflict_aware"; GROUP="joint_conflict" ;;
  *)
    echo "Unknown SWEEP_MODE=$SWEEP_MODE"
    exit 1
    ;;
esac

RUN_NAME="${SWEEP_MODE}_l${LAMBDA}_s${SEED}"

cd "$(dirname "$0")/.."

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}
MIXED_PRECISION=${MIXED_PRECISION:-no}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$((10000 + SLURM_JOB_ID % 50000))}

python -m accelerate.commands.launch \
  --num_processes "$NPROC_PER_NODE" \
  --num_machines 1 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision "$MIXED_PRECISION" \
  -m unirae.train \
  --config "$CONFIG_PATH" \
  --run_name "$RUN_NAME" \
  --set seed="$SEED" \
  --set train.lambda_txt="$LTXT" \
  --set train.lambda_rec="$LREC" \
  --set train.strategy="$STRATEGY" \
  --set experiment.group="$GROUP"
