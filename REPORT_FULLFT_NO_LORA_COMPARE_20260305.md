# No-LoRA Full-Update 对比（ImageNet-100, CLIP-cos + rMSE）

Date: 2026-03-05  
Group: `dsga_clipcos_in100_fullft_cmp_bs32_20260305_110055`

## 1) 设置

- 训练脚本：`unirae/train_dsga_rae_lora.py`
- 关键变更：`--encoder_update full`（不注入 LoRA，encoder 全量共享参数更新）
- 数据：`clane9/imagenet-100`
- batch: `32`, steps: `3000`, seed: `42`
- 理解损失：`clip_cosine`
- 生成损失：`rmse`
- 方法：`naive / cagrad / dsga`

每个 run 的可训练参数（来自 `run_setup.json`）：
- `num_shared_params = 86,582,016`
- `num_lora_params = 0`
- `num_decoder_params = 415,352,704`
- `num_und_params = 590,592`

## 2) 结果

### 2.1 quick eval（step=3000, 640 samples）

| method | u_cosine | rmse | mse |
|---|---:|---:|---:|
| naive | **0.871640** | 0.040995 | 0.001681 |
| cagrad | 0.864217 | **0.034980** | **0.001224** |
| dsga | 0.867891 | 0.037415 | 0.001400 |

### 2.2 训练末段（last 1000 step 均值）

| method | lu | lg_obj | rmse | u_cosine | grad_cos_mean | grad_neg_ratio |
|---|---:|---:|---:|---:|---:|---:|
| naive | **0.151241** | 0.040571 | 0.040570 | **0.848759** | 0.013964 | 0.4130 |
| cagrad | 0.160236 | **0.035003** | **0.035003** | 0.839764 | 0.004161 | 0.4860 |
| dsga | 0.155798 | 0.037289 | 0.037289 | 0.844202 | 0.007389 | 0.4430 |

## 3) 与 LoRA 版同协议对照（step=3000）

对照组：`dsga_clipcos_in100_rmse_cmp_bs32_20260305_025937`

| method | Δlu (full-lora) | Δlg_obj | Δrmse | Δu_cosine |
|---|---:|---:|---:|---:|
| naive | +0.005769 | -0.039492 | -0.039492 | -0.005769 |
| cagrad | +0.018144 | -0.044928 | -0.044928 | -0.018144 |
| dsga | +0.011630 | -0.042941 | -0.042941 | -0.011630 |

结论：去掉 LoRA 后，三种算法差异显著放大，出现清晰 trade-off：
- `cagrad` 明显更偏生成（`rmse/mse` 最优）
- `naive` 明显更偏理解（`u_cosine` 最优）
- `dsga` 居中

## 4) 作业信息

- naive: `351017`
- cagrad: `351018`
- dsga: `351019`
- manifest: `/scratch/peilab/xlubl/dsga_runs/dsga_clipcos_in100_fullft_cmp_bs32_20260305_110055_jobs.tsv`
