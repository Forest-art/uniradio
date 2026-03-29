# Full-Update（No LoRA）+ CE 分类对比（ImageNet-100）

Date: 2026-03-05  
Group: `dsga_ce_in100_fullft_cmp_bs32_20260305_112543`

## 1) 设置

- 训练脚本：`unirae/train_dsga_rae_lora.py`
- 关键参数：
  - `--encoder_update full`（全量 encoder 共享更新）
  - `--understanding_loss ce`（分类 CE）
  - `--recon_loss rmse`
  - `steps=3000`, `batch_size=32`, `seed=42`
- 方法：`naive / cagrad / dsga`

每个 run 参数规模：
- `num_shared_params = 86,582,016`
- `num_lora_params = 0`
- `num_decoder_params = 415,352,704`
- `num_und_params = 76,900`

## 2) 结果（quick eval, step=3000, 640 samples）

| method | acc | rmse | mse |
|---|---:|---:|---:|
| naive | 0.953125 | 0.072959 | 0.005323 |
| cagrad | 0.948438 | 0.063726 | 0.004061 |
| dsga | **0.962500** | **0.056950** | **0.003243** |

## 3) 训练后段（last 1000 step 均值）

| method | lu | lg_obj | rmse | acc | grad_cos_mean | grad_neg_ratio |
|---|---:|---:|---:|---:|---:|---:|
| naive | **0.243737** | 0.071425 | 0.071425 | **0.933844** | 0.049341 | 0.3960 |
| cagrad | 0.247700 | 0.062880 | 0.062880 | 0.932125 | 0.046607 | 0.3740 |
| dsga | 0.259146 | **0.056134** | **0.056134** | 0.928312 | 0.038188 | 0.3920 |

## 4) 对照：同协议 CLIP-cos full-update

对照组：`dsga_clipcos_in100_fullft_cmp_bs32_20260305_110055`

主要变化：CE 版理解侧更强监督（分类），三方法在理解指标上的差距相对缩小，而生成侧排序更明显向 `dsga > cagrad > naive`。

## 5) 作业信息

- naive: `351029`
- cagrad: `351030`
- dsga: `351031`
- manifest: `/scratch/peilab/xlubl/dsga_runs/dsga_ce_in100_fullft_cmp_bs32_20260305_112543_jobs.tsv`
