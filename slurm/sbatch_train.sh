#!/usr/bin/env bash
#SBATCH --job-name=unirae_train
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

CONFIG_PATH=${1:-configs/smoke.yaml}
RUN_NAME=${2:-smoke_$(date +%Y%m%d_%H%M%S)}
CONDA_ENV=${CONDA_ENV:-diffuser310}
TRAIN_MODULE=${TRAIN_MODULE:-unirae.train}
EXTRA_ARGS=${EXTRA_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-$PWD}"

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
  -m "$TRAIN_MODULE" \
  --config "$CONFIG_PATH" \
  --run_name "$RUN_NAME" \
  ${EXTRA_ARGS}
