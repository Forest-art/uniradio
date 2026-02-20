# UniRAE-RADIO: Smoke / Sweep / Analyze

## 1) Environment
```bash
cd /project/peilab/luxiaocheng/projects/unirae_radio
pip install -r requirements.txt
```

## 2) 配置 ImageNet 与 RADIO
编辑 `configs/smoke.yaml`（以及 sweep 配置）中的以下字段：
- `data.data_root`: ImageNet 根目录（应至少包含 `train/` 和 `val/`）
- `data.data_format`: `auto` / `imagefolder` / `hf_disk`
- `data.hf_load_from_disk`: 若使用 HuggingFace `datasets.load_from_disk`，填本地数据路径
- `data.hf_split_train` / `data.hf_split_val`: HF split 名（通常 `train` / `validation`）
- `data.hf_image_key` / `data.hf_label_key`: HF 样本字段名（默认 `image` / `label`）
- `radio.code_root`: RADIO 代码目录（例如 `/project/peilab/luxiaocheng/projects/RADIO`）
- `radio.ckpt`: RADIO 权重路径（例如 `radio_v2.1_bf16.pth.tar`）
- 可选 `data.class_names_file`: 一行一个类名，用于替换默认文件夹名 prompt

## 3) 本地 smoke（1000 steps）
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.train --config configs/smoke.yaml --run_name smoke_local
```

多卡本地示例（4 卡）：
```bash
python -m accelerate.commands.launch --num_processes 4 -m unirae.train --config configs/smoke.yaml --run_name smoke_local_4gpu
```

训练过程中会在 `runs/<run_name>/` 生成：
- `metrics.jsonl`（包含 `Lu/Lg/total/cos`）
- `checkpoints/latest.pt`
- `understanding.json`
- `generation.json`
- `cos_summary.json`
- `cos_curve.json`

## 4) Slurm 单实验提交（HKUST SuperPOD）
```bash
cd /project/peilab/luxiaocheng/projects/unirae_radio
sbatch slurm/sbatch_train.sh configs/smoke.yaml smoke_slurm
```

默认资源：
- `--account=peilab`
- `--partition=preempt`
- `--gpus-per-node=1`（可在提交时覆盖成多卡）
- `--cpus-per-task=8`
- `--mem=32G`
- `--time=24:00:00`

脚本内部使用 `accelerate launch`；默认 `NPROC_PER_NODE=$SLURM_GPUS_ON_NODE`。
例如 4 卡运行：
```bash
sbatch --gpus-per-node=4 slurm/sbatch_train.sh configs/smoke.yaml smoke_slurm_4gpu
```

可通过环境变量切换 conda 环境：
```bash
CONDA_ENV=diffuser310 sbatch slurm/sbatch_train.sh configs/smoke.yaml smoke_slurm
```

## 5) Sweep（四组对照）
先跑 1 个 baseline（no finetune）：
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.train \
  --config configs/smoke.yaml \
  --run_name baseline_no_ft \
  --set lora.enable=false \
  --set train.steps=0 \
  --set experiment.group=baseline
```

四份配置：
- `configs/sweep_text.yaml`
- `configs/sweep_recon.yaml`
- `configs/sweep_joint_naive.yaml`
- `configs/sweep_joint_conflict.yaml`

每组默认扫描：
- 权重：`0.25 / 0.5 / 1 / 2`
- seed：`42 / 43`

### 5.1 text-only
```bash
sbatch --array=0-7 slurm/sbatch_sweep.sh configs/sweep_text.yaml text
```

### 5.2 recon-only
```bash
sbatch --array=0-7 slurm/sbatch_sweep.sh configs/sweep_recon.yaml recon
```

### 5.3 joint naive
```bash
sbatch --array=0-7 slurm/sbatch_sweep.sh configs/sweep_joint_naive.yaml joint_naive
```

### 5.4 joint conflict-aware
```bash
sbatch --array=0-7 slurm/sbatch_sweep.sh configs/sweep_joint_conflict.yaml joint_conflict
```

## 6) 分析 trade-off（自动汇总 + 帕累托数据）
```bash
python -u -m unirae.analyze_tradeoff --runs_root runs --out_dir runs/analysis
```

输出：
- `runs/analysis/results.csv`
- `runs/analysis/pareto_points.csv`
- `runs/analysis/pareto_scatter.csv`
- `runs/analysis/cos_curves.json`
- `runs/analysis/coverage.json`

`results.csv` 每行包含：
- `exp_name, seed, lambda_txt, lambda_rec, strategy, group, zero_shot_acc, recon_metric, cos_mean, cos_neg_ratio`

## 7) 默认 RADIO encoder 特征自检（kNN + linear probe）
用于你在另一个有 ImageNet 的集群上快速确认“特征是否正常”。

### 7.1 本地/交互式直接跑
```bash
cd /project/peilab/luxiaocheng/projects/unirae_radio
python -m accelerate.commands.launch --num_processes 1 -m unirae.eval_radio_repr \
  --radio_code_root /path/to/RADIO \
  --data_root /path/to/imagenet \
  --model_version c-radio_v3-b \
  --batch_size 128 \
  --workers 8 \
  --k 20 \
  --linear_steps 2000 \
  --output runs/radio_repr/c-radio_v3-b_repr.json
```

8 卡 linear probing（单机）：
```bash
python -m accelerate.commands.launch --num_processes 8 -m unirae.eval_radio_repr \
  --no_use_local_lib \
  --torchhub_repo NVlabs/RADIO \
  --data_root /path/to/imagenet \
  --model_version c-radio_v3-b \
  --batch_size 256 \
  --workers 8 \
  --linear_steps 2000 \
  --output runs/radio_repr/c-radio_v3-b_lp_8gpu.json
```

如果数据是 `datasets.save_to_disk` 格式（参考 RAE 的 load_from_disk）：
```bash
python -m accelerate.commands.launch --num_processes 8 -m unirae.eval_radio_repr \
  --no_use_local_lib \
  --torchhub_repo NVlabs/RADIO \
  --data_root /path/to/hf_imagenet_disk \
  --data_format hf_disk \
  --hf_load_from_disk /path/to/hf_imagenet_disk \
  --train_split train \
  --val_split validation \
  --hf_image_key image \
  --hf_label_key label \
  --model_version c-radio_v3-b \
  --batch_size 256 \
  --workers 8 \
  --linear_steps 2000 \
  --output runs/radio_repr/c-radio_v3-b_lp_8gpu_hf.json
```

加载逻辑（已对齐 RADIO 官方 `examples/common/model_loader.py` 的 RADIO 分支）：
- 默认：`--use_local_lib`（默认开启）时调用：
  - `torch.hub.load(<local_repo>, "radio_model", source="local", trust_repo=True, version=...)`
  - 其中 `<local_repo>` 优先取 `--radio_code_root`，否则取 `--torchhub_repo`（本地路径）
- 可选：`--use_huggingface` 走 HF 加载
- 可选：`--no_use_local_lib` 走远端 `torch.hub.load(...)`
- `--model_version` 同时支持：
  - 版本名（如 `c-radio_v3-b`）
  - 本地模型路径（如 `/path/to/radio_ckpt.pth.tar`）

按你习惯的本地加载写法可直接这样：
```bash
python -m accelerate.commands.launch --num_processes 1 -m unirae.eval_radio_repr \
  --use_local_lib \
  --torchhub_repo /path/to/RADIO \
  --data_root /path/to/imagenet \
  --model_version c-radio_v3-b
```

输入数据目录要求：
- `/path/to/imagenet/train/...`
- `/path/to/imagenet/val/...`

输出 JSON 包含：
- `feature_dim`
- `linear_probe_top1`
- `world_size`
- `knn_top1`（仅当加 `--run_knn`）

### 7.2 Slurm 一条命令（按你集群参数改 account/partition）
```bash
srun --account=<your_account> --partition=<your_partition> --nodes=1 --gpus-per-node=8 --cpus-per-task=32 --mem=128G --time=0-02:00:00 \
  bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate <your_env> && \
  cd /project/peilab/luxiaocheng/projects/unirae_radio && \
  python -m accelerate.commands.launch --num_processes 8 -m unirae.eval_radio_repr \
    --no_use_local_lib --torchhub_repo NVlabs/RADIO \
    --data_root /path/to/imagenet \
    --model_version c-radio_v3-b \
    --batch_size 256 --workers 8 --linear_steps 2000 \
    --output runs/radio_repr/c-radio_v3-b_repr.json'
```

## 8) 常见错误排查
1. `No Linear module matched lora.target_modules`：
   - 检查 `lora.target_modules` 是否匹配当前 RADIO trunk 的模块名。
2. `Cannot find ImageNet split=...`：
   - 检查 `data.data_root` 目录结构是否包含 `train/`、`val/`。
3. `RADIO hubconf not found` 或加载失败：
   - 检查 `radio.code_root` 是否指向 RADIO 仓库根目录。
   - 检查 `radio.ckpt` 路径是否可读。
4. OOM：
   - 降低 `train.batch_size` / `eval.batch_size`。
   - 可增大 `decoder.token_dropout` 或缩小 `decoder.hidden_dim`。
5. `zero-shot` 异常偏低：
   - 建议提供 `data.class_names_file`，避免直接使用 synset 文件夹名作为 prompt。
6. `ImportError: KeypointType`（来自 `RADIO/examples`）：
   - 这是 `albumentations` 版本兼容问题，不影响本项目 `unirae.eval_radio_repr` 脚本。
   - 若你只想验证 RADIO 表征，优先用这里的 `python -m unirae.eval_radio_repr`。
