#!/usr/bin/env bash
#SBATCH --job-name=unirae_train
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

CONFIG_PATH=${1:-configs/smoke.yaml}
RUN_NAME=${2:-smoke_$(date +%Y%m%d_%H%M%S)}
CONDA_ENV=${CONDA_ENV:-diffuser310}

cd "$(dirname "$0")/.."

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

python -u -m unirae.train \
  --config "$CONFIG_PATH" \
  --run_name "$RUN_NAME"
