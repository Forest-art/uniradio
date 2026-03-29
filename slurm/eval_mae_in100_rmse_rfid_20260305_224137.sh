#!/bin/bash
#SBATCH --job-name=mae_rmse_rfid
#SBATCH --partition=preempt
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/mae_rmse_rfid-%j.out
#SBATCH --error=/project/peilab/luxiaocheng/projects/DSGA/slurm/logs/mae_rmse_rfid-%j.err

set -euo pipefail
cd /project/peilab/luxiaocheng/projects/DSGA
source /home/xlubl/anaconda3/etc/profile.d/conda.sh
conda activate diffuser310
export NUMEXPR_MAX_THREADS=256

RUN_DIR="${RUN_DIR}"
python -m unirae.eval_rae_table1_baselines \
  --rae_code_root /project/peilab/luxiaocheng/projects/RAE \
  --stage1_config /project/peilab/luxiaocheng/projects/RAE/configs/stage1/pretrained/MAE.yaml \
  --hf_dataset clane9/imagenet-100 \
  --train_split train \
  --val_split validation \
  --batch_size 256 \
  --num_workers 8 \
  --probe_steps 0 \
  --max_eval_batches 0 \
  --rfid_num_samples 5000 \
  --rfid_batch_size 64 \
  --rfid_tmp_dir /scratch/peilab/xlubl/tmp_rfid \
  --out_json "${RUN_DIR}/eval_summary.json"
