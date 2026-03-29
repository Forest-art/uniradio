# UniRAE-RADIO Runbook (Baseline-Only)

## 1) Environment
```bash
cd /project/peilab/luxiaocheng/projects/DSGA
pip install -r requirements.txt
```

## 2) Active training entry
- CIFAR-10 baseline training only: `python -m unirae.train_cifar10`

`train_cifar10` is now intentionally simplified to only support:
- `train.mode`: `joint | text_only | recon_only`
- `train.grad_strategy` (joint mode only): `naive | pcgrad | cagrad | saop | laga | dsga`
  - legacy alias still accepted: `ma_laga` (mapped to DSGA path)

For `train.grad_strategy=dsga`, use the new naming:
- `train.dsga_m_align_gamma`: DSGA-M (magnitude alignment) gamma
- `train.dsga_m_scope`: DSGA-M scope: `global | layerwise` (default `global`)
- `train.dsga_m_norm_restore`: DSGA-M norm restore switch
- `train.dsga_m_eps`: DSGA-M epsilon
- `train.dsga_d_mode`: DSGA-D (direction decomposition) mode: `full | direction_only | magnitude_only`

Legacy aliases are still accepted for compatibility:
- `train.ma_laga_align_gamma`, `train.ma_laga_m_scope`, `train.ma_laga_norm_restore`, `train.ma_laga_eps`, `train.ma_laga_mode`

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

## 11) RAE 对齐参数（谢赛宁 RAE 风格）
- 当前 `unirae/decoder.py` 已支持 `decoder.noise_tau`，默认值 `0.8`：
  - 仅在 `train()` 时向 decoder 输入注入 RAE 风格噪声（`sigma~U(0,tau)` 后乘高斯）
  - `eval()` 时自动关闭噪声
- 推荐与当前 ImageNet baseline 一起显式设置：
  - `decoder.hidden_dim=1024`
  - `decoder.feature_target_dim=1024`
  - `decoder.token_dropout=0.0`
  - `decoder.noise_tau=0.8`

8 卡直跑（非 sbatch，HF `load_from_disk`，每 1000 step eval 并在日志打印结果）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m accelerate.commands.launch --num_processes 8 --num_machines 1 -m unirae.train \
  --config configs/smoke.yaml \
  --run_name imagenet8x_conflict_rae_tau08_s42 \
  --set seed=42 \
  --set data.data_format=hf_disk \
  --set data.hf_load_from_disk=/path/to/imagenet_hf_saved \
  --set data.hf_split_train=train \
  --set data.hf_split_val=validation \
  --set train.steps=20000 \
  --set train.strategy=conflict_aware \
  --set train.lambda_txt=1.0 \
  --set train.lambda_rec=1.0 \
  --set train.batch_size=128 \
  --set eval.batch_size=128 \
  --set log.eval_every=1000 \
  --set decoder.hidden_dim=1024 \
  --set decoder.feature_target_dim=1024 \
  --set decoder.token_dropout=0.0 \
  --set decoder.noise_tau=0.8
```

## 12) DINO RAE Stage-1（只训练重建 Decoder）
- 新入口：`unirae/train_dino_decoder_only.py`
- 逻辑对齐 RAE Stage-1 的核心训练范式：
  - 冻结 DINO encoder
  - 不注入 LoRA（`lora_last_n_blocks=0`）
  - 理解头冻结（参数不更新）
  - 只优化 Transformer decoder
  - 训练噪声 `tau=0.8`（`sigma~U(0,tau)` 的 RAE 风格噪声）
  - 数据预处理对齐：`Resize -> (Random/Center)Crop -> ToTensor`，encoder 侧做 DINO mean/std 归一化
  - 默认 `L1` 重建损失、`EMA(decay=0.9978)`、`cosine + warmup` 学习率

配置文件：
- `configs/imagenet_dino_rae_decoder_only.yaml`

8 卡直跑（HF `load_from_disk`）：
```bash
cd /project/peilab/luxiaocheng/projects/DSGA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m accelerate.commands.launch --num_processes 8 --num_machines 1 -m unirae.train_dino_decoder_only \
  --config configs/imagenet_dino_rae_decoder_only.yaml \
  --run_name imagenet_dino_rae_dec_only_s42 \
  --set data.data_format=hf_disk \
  --set data.hf_load_from_disk=/path/to/imagenet_hf_saved \
  --set data.hf_split_train=train \
  --set data.hf_split_val=validation \
  --set train.steps=20000 \
  --set log.eval_every=1000 \
  --set eval.max_batches=50
```

日志中会直接打印：
- 训练：`loss/mse/psnr`
- 评估：`[eval][step=...] {recon_loss, mse, psnr, num_samples}`

## 13) DINO RAE Stage-1 Full（DSGA 内集成训练+评测）
- 入口：`unirae/train_dino_rae_stage1.py`
- 目标：在 `unirae_radio` 内复用 RAE 全量 Stage-1 逻辑（不是单独跑 RAE 仓库）
  - 训练：`L1 + LPIPS + GAN + adaptive d_weight + DiffAug + EMA + cosine`
  - 评测：可选调用官方 `RAE eval` 计算 `PSNR/SSIM/rFID`

配置文件：
- `configs/imagenet_dino_rae_stage1_full.yaml`

使用前准备：
1. 准备 RAE 依赖代码（默认路径）：`/project/peilab/luxiaocheng/projects/RAE`
2. 准备判别器权重：`gan.disc.arch.dino_ckpt_path`
3. 若启用 `rFID`，准备 `eval.reference_npz_path`

8 卡直跑（HF `load_from_disk`）：
```bash
cd /project/peilab/luxiaocheng/projects/DSGA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m accelerate.commands.launch --num_processes 8 --num_machines 1 -m unirae.train_dino_rae_stage1 \
  --config configs/imagenet_dino_rae_stage1_full.yaml \
  --run_name dsga_rae_stage1_full_s42 \
  --set data.data_format=hf_disk \
  --set data.hf_load_from_disk=/path/to/imagenet_hf_saved \
  --set data.hf_split_train=train \
  --set data.hf_split_val=validation \
  --set gan.disc.arch.dino_ckpt_path=/path/to/RAE/models/discs/dino_vit_small_patch8_224.pth \
  --set eval.reference_npz_path=/path/to/imagenet_val_224_or_256.npz \
  --set train.steps=20000 \
  --set log.eval_every=1000
```

日志会打印：
- 训练：`total_loss/recon_l1/lpips/gan_g/disc_loss/disc_acc/mse/psnr`
- 评估：`[eval][step=...]`（官方 eval 开启时输出 `psnr/ssim/rfid`）

## 14) 官方 RAE 预训练 Decoder 初始化校验（看初始 rMSE）
- 新脚本：`unirae/eval_rae_official_init.py`
- 目的：确认官方 decoder 权重加载正确，并用同一数据对比
  - `official_pretrained`（加载官方 ckpt）
  - `random_decoder`（不加载 decoder ckpt）
- 输出：`mse / rmse / psnr`

先下载官方权重到本地 RAE 目录（只需一次）：
```bash
cd /project/peilab/luxiaocheng/projects/RAE
hf download nyu-visionx/RAE-collections \
  decoders/dinov2/wReg_base/ViTXL_n08/model.pt \
  stats/dinov2/wReg_base/imagenet1k/stat.pt \
  discs/dino_vit_small_patch8_224.pth \
  --repo-type model \
  --local-dir models
```

HF `load_from_disk` 完整校验命令（推荐）：
```bash
cd /project/peilab/luxiaocheng/projects/DSGA

python -m unirae.eval_rae_official_init \
  --dataset_mode imagenet \
  --data_root /path/to/imagenet_hf_saved \
  --data_format hf_disk \
  --hf_load_from_disk /path/to/imagenet_hf_saved \
  --hf_split_override validation \
  --image_size 256 \
  --batch_size 32 \
  --num_workers 8 \
  --max_batches 50 \
  --device cuda \
  --compare_random_decoder \
  --out_json results/rae_official_init_cmp.json
```

如果加载正确，通常会看到：
- `official_pretrained.rmse` 显著低于 `random_decoder.rmse`
- `official_pretrained.psnr` 显著高于 `random_decoder.psnr`

## 15) ImageNet-100 Early Gradient Dynamics（Bridge Experiment）
- 新脚本：`unirae/train_imagenet100_dynamics.py`
- 目标：在前 1000 step 内高频记录 encoder 各层 `g_u`(CE) 与 `g_g`(MSE) 冲突
  - `step<=1000` 且 `step%50==0` 触发探针
  - 每次输出：`results/in100_grad_dynamics_{init_mode}/step_XXXX.csv`
- 初始化对照：
  - `--encoder_init scratch`：`vit_small_patch14_dinov2` 架构随机初始化
  - `--encoder_init dinov2`：同架构加载 DINOv2 预训练权重

单组启动示例：
```bash
cd /project/peilab/luxiaocheng/projects/DSGA

python -m unirae.train_imagenet100_dynamics \
  --encoder_init scratch \
  --hf_dataset_id clane9/imagenet-100 \
  --batch_size 128 \
  --num_workers 8 \
  --max_steps 1200 \
  --probe_until 1000 \
  --probe_every 50 \
  --output_root results
```

一键顺序跑 `scratch -> dinov2`：
```bash
cd /project/peilab/luxiaocheng/projects/DSGA
bash run_imagenet100_dynamics.sh
```
