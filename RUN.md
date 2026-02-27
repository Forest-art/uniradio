# UniRAE ImageNet-100 Baseline Compare Runbook

## 1. Goal
对比两个 baseline（同架构 ViT-S，唯一变量是 encoder 初始化）：

- `scratch`: `vit_small_patch14_dinov2` 随机初始化
- `dinov2`: `vit_small_patch14_dinov2` 预训练初始化

训练入口：

- `unirae/train_imagenet100_dynamics.py`

绘图入口：

- `unirae/plot_imagenet100_dynamics_pub.py`

一键脚本：

- `run_imagenet100_dynamics.sh`

---

## 2. New Machine Setup

### 2.1 Create venv
```bash
cd /project/peilab/luxiaocheng/projects/unirae_radio
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 2.2 Install PyTorch (pick your CUDA build first)
示例（CUDA 12.1）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 Install remaining deps
```bash
pip install -r requirements.txt
```

### 2.4 Optional sanity check
```bash
python - <<'PY'
import torch, timm
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("timm:", timm.__version__)
PY
```

---

## 3. Data Source
默认直接用 Hugging Face:

- `clane9/imagenet-100`

脚本会自动做：

- `Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet mean/std)`

---

## 4. Run Two Baselines (Single Seed)

默认 seed=42，按顺序跑 `scratch -> dinov2`：
```bash
cd /project/peilab/luxiaocheng/projects/unirae_radio
source .venv/bin/activate

bash run_imagenet100_dynamics.sh
```

默认输出目录：

- `results/in100_grad_dynamics_scratch_s42`
- `results/in100_grad_dynamics_dinov2_s42`

默认会在训练中边训边评：

- 每 `1000` step 做一次验证评测（`eval_every=1000`）
- 输出指标：`Top-1 Acc`（理解）、`rMSE`（重建）、`rFID`（重建）
- 评测日志写入：`<run_dir>/eval_metrics.jsonl`

可覆盖参数（示例）：
```bash
HF_DATASET_ID=clane9/imagenet-100 \
BATCH_SIZE=32 \
NUM_WORKERS=8 \
MAX_STEPS=1000 \
PROBE_UNTIL=1000 \
PROBE_EVERY=50 \
EVAL_EVERY=1000 \
EVAL_MAX_BATCHES=50 \
EVAL_RFID_NUM_SAMPLES=512 \
EVAL_RFID_BATCH_SIZE=64 \
EVAL_RFID_TMP_DIR=/tmp \
SEED=42 \
RUN_NAME_PREFIX=in100_grad_dynamics \
OUTPUT_ROOT=results/bridge_in100_20260227_1k \
bash run_imagenet100_dynamics.sh
```

---

## 5. Multi-Seed Compare (Recommended for Report)

建议 3 seeds: `42,43,44`
```bash
cd /project/peilab/luxiaocheng/projects/unirae_radio
source .venv/bin/activate

for seed in 42 43 44; do
  HF_DATASET_ID=clane9/imagenet-100 \
  BATCH_SIZE=32 \
  NUM_WORKERS=8 \
  MAX_STEPS=1000 \
  PROBE_UNTIL=1000 \
  PROBE_EVERY=50 \
  EVAL_EVERY=1000 \
  EVAL_MAX_BATCHES=50 \
  EVAL_RFID_NUM_SAMPLES=512 \
  EVAL_RFID_BATCH_SIZE=64 \
  EVAL_RFID_TMP_DIR=/tmp \
  SEED=${seed} \
  RUN_NAME_PREFIX=in100_grad_dynamics \
  OUTPUT_ROOT=results/bridge_in100_20260227_1k \
  bash run_imagenet100_dynamics.sh
done
```

---

## 6. Plot Publication-Style Figures

```bash
cd /project/peilab/luxiaocheng/projects/unirae_radio
source .venv/bin/activate

python -m unirae.plot_imagenet100_dynamics_pub \
  --root results/bridge_in100_20260227_1k \
  --seeds 42,43,44 \
  --modes scratch,dinov2 \
  --run_name_template 'in100_grad_dynamics_{mode}_s{seed}'
```

输出目录：

- `results/bridge_in100_20260227_1k/analysis_pub`

关键文件：

- `loss_curves_understanding_generation.png`
- `conflict_dynamics_mean_cos_neg_ratio.png`
- `blockwise_mean_cosine_compare.png`
- `per_run_summary.csv`
- `mode_aggregate_summary.csv`
- `key_step_loss_summary.csv`
- 每个 run 下 `eval_metrics.jsonl`（包含 `val_top1_acc/val_rmse/val_rfid`）

---

## 7. Quick Read of Loss Curves

如果只想快速看 loss 数字（不看图）：
```bash
python - <<'PY'
import pandas as pd
p='results/bridge_in100_20260227_1k/analysis_pub/key_step_loss_summary.csv'
df=pd.read_csv(p)
print(df.to_string(index=False))
PY
```

---

## 8. Slurm Example (HKUST SuperPOD)

推荐先拿 1 卡交互资源：
```bash
srun --partition preempt --account=peilab --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --time 0-6:00:00 --pty bash
```

进入后再执行第 4/5 节命令。

---

## 9. Troubleshooting

### 9.1 HuggingFace cache path / disk pressure
若 `/project` 空间紧张，可把 HF cache 临时切到 `/tmp`：
```bash
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_cache
```

`rFID` 评测也建议落到 `/tmp` 临时目录：
```bash
export EVAL_RFID_TMP_DIR=/tmp
```

### 9.2 Output overwrite
请确保 `SEED` 或 `RUN_NAME_PREFIX` 不同，否则可能覆盖旧 run。

### 9.3 GPU OOM
先把 `BATCH_SIZE` 从 `32` 降到 `16` 或 `8`。

### 9.4 rFID 太慢
可以减小 `EVAL_RFID_NUM_SAMPLES`（例如 `256`），或临时关闭：
```bash
SKIP_RFID=1 bash run_imagenet100_dynamics.sh
```
