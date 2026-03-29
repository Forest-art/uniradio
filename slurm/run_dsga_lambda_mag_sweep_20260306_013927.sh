#!/usr/bin/env bash
#SBATCH --job-name=dsga-mag4
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/%x-%j.out
#SBATCH --error=/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/%x-%j.err

set -euo pipefail
cd /project/peilab/luxiaocheng/projects/DSGA
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffuser310

export OMP_NUM_THREADS=8
export DATA_ROOT=/project/peilab/luxiaocheng/projects/DSGA/data/cifar100
export SEED=42
export STEPS=20000
export BATCH_SIZE=128
export NUM_WORKERS=4
export RUN_PREFIX=dsga_swin_cifar100_mag

bash scripts/run_dsga_lambda_mag_sweep.sh
