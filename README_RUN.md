# UniRAE-RADIO Runbook (Baseline-Only)

## 1) Environment
```bash
cd /project/peilab/luxiaocheng/projects/unirae_radio
pip install -r requirements.txt
```

## 2) Active training entry
- CIFAR-10 baseline training only: `python -m unirae.train_cifar10`

`train_cifar10` is now intentionally simplified to only support:
- `train.mode`: `joint | text_only | recon_only`
- `train.grad_strategy` (joint mode only): `naive | pcgrad | cagrad`

No layer-wise, MGDA, consistency, or SupCon training paths are active in this baseline stage.

## 3) Canonical baseline configs
Use these 5 configs only:
- `configs/cifar10_baseline_joint_naive.yaml`
- `configs/cifar10_baseline_joint_pcgrad.yaml`
- `configs/cifar10_baseline_joint_cagrad.yaml`
- `configs/cifar10_baseline_text_only.yaml`
- `configs/cifar10_baseline_recon_only.yaml`

Set `data.root` before running.

## 4) Local runs
Example (joint + pcgrad):
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.train_cifar10 \
  --config configs/cifar10_baseline_joint_pcgrad.yaml \
  --run_name cifar10_joint_pcgrad_s42
```

Quick fake-data smoke:
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.train_cifar10 \
  --config configs/cifar10_baseline_joint_naive.yaml \
  --run_name cifar10_smoke_fake \
  --set data.use_fake_data=true \
  --set train.steps=100
```

## 5) Slurm single job
```bash
TRAIN_MODULE=unirae.train_cifar10 sbatch slurm/sbatch_train.sh \
  configs/cifar10_baseline_joint_cagrad.yaml \
  cifar10_joint_cagrad_s42
```

Submit all 5 baselines for multiple seeds:
```bash
DATA_ROOT=/path/to/cifar10 STEPS=5000 SEEDS=42,43,44 \
bash slurm/submit_baselines.sh
```

## 6) Outputs per run
Each run writes to `runs/<run_name>/`:
- `metrics.jsonl`
- `understanding.json`
- `generation.json`
- `cos_summary.json`
- `cos_curve.json`
- `train_setup.json`
- `checkpoints/latest.pt`

## 7) Notes
- Existing old configs/scripts are treated as legacy artifacts and are not part of the current baseline workflow.
- If you need to compare the five baselines, run with the same `seed`, `steps`, and data split settings.
