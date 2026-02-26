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

## 8) ImageNet HF Loader (for `unirae.train.py`)
- ImageNet 数据入口在 `unirae/data_imagenet.py`。
- `data_format=auto` 时会自动选择：
  - `hf_disk`: 当设置 `hf_load_from_disk` 或 `data_root` 看起来像 HF `save_to_disk` 目录
  - `imagefolder`: 否则走 `torchvision.datasets.ImageFolder`
- 新增 `HFImageNetDataset` 的容错读取逻辑：
  - 读取 `item['image']` 与 `item['label']`
  - 非 RGB 图自动 `convert('RGB')`
  - 某个样本解码/处理失败时，会打印错误并自动尝试下一个样本（循环重试，最多一个 epoch 长度）
- 常用 HF 配置项（`unirae/train.py` 会透传）：
  - `data.hf_load_from_disk`
  - `data.hf_split_train` / `data.hf_split_val`
  - `data.hf_image_key` / `data.hf_label_key`

## 9) ImageNet 启动命令（`unirae.train`）
单卡本地快速启动（ImageFolder）：
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.train \
  --config configs/smoke.yaml \
  --run_name imagenet_joint_naive_s42 \
  --set seed=42 \
  --set data.data_root=/path/to/imagenet \
  --set data.data_format=imagefolder \
  --set train.steps=20000 \
  --set train.strategy=naive \
  --set train.lambda_txt=1.0 \
  --set train.lambda_rec=1.0
```

单卡本地快速启动（HF `load_from_disk`）：
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.train \
  --config configs/smoke.yaml \
  --run_name imagenet_joint_conflict_s42 \
  --set seed=42 \
  --set data.data_format=hf_disk \
  --set data.hf_load_from_disk=/path/to/imagenet_hf_saved \
  --set data.hf_split_train=train \
  --set data.hf_split_val=validation \
  --set train.steps=20000 \
  --set train.strategy=conflict_aware \
  --set train.lambda_txt=1.0 \
  --set train.lambda_rec=1.0
```

## 10) ImageNet Baseline 一键脚本（Slurm）
脚本：`slurm/submit_imagenet_baselines.sh`

默认会按每个 seed 提交 4 个 baseline：
- `joint_naive`
- `joint_conflict`
- `text_only_naive`
- `recon_only_naive`

ImageFolder 示例：
```bash
DATA_ROOT=/path/to/imagenet \
DATA_FORMAT=imagefolder \
STEPS=20000 \
SEEDS=42,43 \
bash slurm/submit_imagenet_baselines.sh
```

HF `load_from_disk` 示例：
```bash
DATA_FORMAT=hf_disk \
HF_LOAD_FROM_DISK=/path/to/imagenet_hf_saved \
HF_SPLIT_TRAIN=train \
HF_SPLIT_VAL=validation \
STEPS=20000 \
SEEDS=42 \
bash slurm/submit_imagenet_baselines.sh
```
