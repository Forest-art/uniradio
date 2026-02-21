# UniRAE-RADIO Runbook

## 1) Environment
```bash
cd /project/peilab/luxiaocheng/projects/unirae_radio
pip install -r requirements.txt
```

## 2) Two experiment tracks
- ImageNet + RADIO (existing): `python -m unirae.train`
- CIFAR-10 scale-up (new): `python -m unirae.train_cifar10`

Both write to `runs/<run_name>/` with shared artifacts:
- `metrics.jsonl` (`Lu/Lg/L_cons/L_supcon/total/cos/loss_txt/loss_g/loss_cons/loss_supcon/...`)
- `understanding.json`
- `generation.json`
- `cos_summary.json`
- `cos_curve.json`
- `checkpoints/latest.pt`

## 3) CIFAR-10 scale-up quick start

### 3.1 Config
Use `configs/cifar10_smoke.yaml` and set:
- `data.root`: CIFAR-10 cache/data root
- `model.backbone`: `resnet18` (default) or `vit_small`
- `train.mode`: `baseline | text_only | recon_only | joint`
- `train.strategy`: `naive | conflict_aware | cagrad | mgda_ub`
- `consistency.enabled/two_view/target/loss_type`: multi-view consistency switches
- `supcon.enabled/two_view/aug/tau/embed`: supervised contrastive switches

### 3.2 Local smoke (1000 steps)
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.train_cifar10 \
  --config configs/cifar10_smoke.yaml \
  --run_name cifar10_smoke_local
```

Dummy/fake-data smoke (不下载 CIFAR-10):
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.train_cifar10 \
  --config configs/cifar10_smoke.yaml \
  --run_name cifar10_smoke_fake \
  --set data.use_fake_data=true \
  --set train.steps=100
```

### 3.3 Sweep (text/recon/joint naive/joint conflict)
Configs:
- `configs/cifar10_text_sweep.yaml`
- `configs/cifar10_rec_sweep.yaml`
- `configs/cifar10_joint_naive.yaml`
- `configs/cifar10_joint_conflict.yaml`

Each config has:
- `sweep.lambda_values: [0.25, 0.5, 1.0, 2.0]`
- `sweep.seeds: [42, 43]`

Slurm array command example:
```bash
TRAIN_MODULE=unirae.train_cifar10 sbatch --array=0-7 slurm/sbatch_sweep.sh configs/cifar10_text_sweep.yaml text
```

## 4) Analyze trade-off
```bash
python -u -m unirae.analyze_tradeoff --runs_root runs --out_dir runs/analysis --results_dir results
```

Outputs:
- all runs: `runs/analysis/results.csv`
- CIFAR summary: `results/cifar10_results.csv` (includes `lambda_cons/lambda_sup`, `consistency_enabled/supcon_enabled`, `supcon_tau`)
- CIFAR Pareto data: `results/cifar10_pareto.json`
- CIFAR gradient conflict curves: `results/cifar10_cos_curve.json`

## 4.1 Consistency ablation (new)
Configs:
- `configs/cifar10_cons_baseline.yaml` (cons=0, naive)
- `configs/cifar10_cons_naive.yaml` (+consistency, naive sum)
- `configs/cifar10_cons_cagrad.yaml` (+consistency, cagrad, merge into understanding)

Local quick check:
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.train_cifar10 \
  --config configs/cifar10_cons_naive.yaml \
  --run_name cifar10_cons_naive_local \
  --set data.root=/path/to/cifar10
```

One-click Slurm submit (3 groups x multi-seed):
```bash
DATA_ROOT=/path/to/cifar10 STEPS=5000 SEEDS=42,43,44 \
bash slurm/submit_consistency_compare.sh
```

## 4.2 SupCon ablation (new)
Configs:
- `configs/cifar10_supcon_baseline.yaml` (no SupCon)
- `configs/cifar10_supcon_naive.yaml` (naive + SupCon)
- `configs/cifar10_supcon_cagrad.yaml` (CAGrad + SupCon, merged into understanding)

One-click Slurm submit (`lambda_sup` sweep + multi-seed):
```bash
DATA_ROOT=/path/to/cifar10 STEPS=5000 SEEDS=42,43,44 LAMSUP=0.1,0.5,1.0 \
bash slurm/submit_supcon_compare.sh
```

## 5) Slurm templates (HKUST SuperPOD)
Defaults in `slurm/sbatch_train.sh` and `slurm/sbatch_sweep.sh`:
- `--account=peilab`
- `--partition=preempt`
- `--gpus-per-node=1`
- `--cpus-per-task=8`
- `--mem=32G`
- `--time=24:00:00`

Single run:
```bash
TRAIN_MODULE=unirae.train_cifar10 sbatch slurm/sbatch_train.sh configs/cifar10_smoke.yaml cifar10_single
```

## 6) ImageNet + RADIO notes (existing)
`radio.code_root` means your local RADIO repo root (must contain `hubconf.py`).

The loader uses official local torch.hub style:
```python
torch.hub.load(radio_code_root, "radio_model", source="local", version=<model_version>, trust_repo=True)
```

## 7) Troubleshooting
1. `from .data_imagenet` import error:
   - run as module: `python -m unirae.xxx`.
2. CIFAR download blocked:
   - set `data.use_fake_data=true` for pipeline smoke.
3. OOM:
   - lower `data.batch_size`, switch to `resnet18`, or reduce `model.txt_dim/rec_dim`.
4. Sweep array index mismatch:
   - ensure `--array` matches `len(lambda_values) * len(seeds)`.
