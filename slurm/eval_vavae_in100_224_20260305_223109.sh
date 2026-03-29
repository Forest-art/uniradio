#!/bin/bash
#SBATCH --job-name=vavae_in100_224
#SBATCH --partition=preempt
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/vavae_in100_224-%j.out
#SBATCH --error=/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/vavae_in100_224-%j.err

set -euo pipefail
source /home/xlubl/anaconda3/etc/profile.d/conda.sh
conda activate diffuser
export NUMEXPR_MAX_THREADS=256

RUN_DIR="${RUN_DIR}"
python /project/peilab/luxiaocheng/projects/eval_lightningdit_vavae_in100_rmse_rfid.py \
  --vae_model sd-vae-ft-mse \
  --posterior_mode sample \
  --batch_size 32 \
  --num_workers 8 \
  --rfid_num_samples 5000 \
  --rfid_batch_size 64 \
  --amp \
  --output_json "${RUN_DIR}/summary.json"
