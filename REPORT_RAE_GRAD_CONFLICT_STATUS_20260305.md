# RAE 训练梯度冲突状态快报（2026-03-05）

## 结论

基于当前 RAE 版本训练，梯度冲突问题**仍然存在**，但强度属于“中等偏轻”，未出现完全消失。

## 证据 A：最新 RAE-LoRA 联训（ImageNet-100, steps=3000）

实验组：`dsga_clipcos_in100_rmse_cmp_bs32_20260305_025937`  
任务：`clip_cosine`（理解）+ `rmse`（重建）  
共享参数冲突指标：`grad_cos_lora_u_g`（LoRA 共享参数上 `cos(g_u, g_g)`）

来源目录：
- `/scratch/peilab/xlubl/dsga_runs/dsga_clipcos_in100_rmse_cmp_bs32_20260305_025937_naive_s42/metrics.jsonl`
- `/scratch/peilab/xlubl/dsga_runs/dsga_clipcos_in100_rmse_cmp_bs32_20260305_025937_pcgrad_s42/metrics.jsonl`
- `/scratch/peilab/xlubl/dsga_runs/dsga_clipcos_in100_rmse_cmp_bs32_20260305_025937_cagrad_s42/metrics.jsonl`
- `/scratch/peilab/xlubl/dsga_runs/dsga_clipcos_in100_rmse_cmp_bs32_20260305_025937_dsga_s42/metrics.jsonl`

### 全程统计（1~3000 step）

| method | mean cos | neg ratio (`cos<0`) | strong neg (`cos<-0.05`) |
|---|---:|---:|---:|
| naive | -0.013586 | 0.5980 | 0.2470 |
| pcgrad | -0.013097 | 0.5957 | 0.2433 |
| cagrad | -0.009717 | 0.5757 | 0.2217 |
| dsga | -0.010751 | 0.5833 | 0.2290 |

### 分阶段统计（neg ratio）

| method | step 1-1000 | step 1001-2000 | step 2001-3000 |
|---|---:|---:|---:|
| naive | 0.5350 | 0.6290 | 0.6300 |
| pcgrad | 0.5350 | 0.6260 | 0.6260 |
| cagrad | 0.5250 | 0.5990 | 0.6030 |
| dsga | 0.5260 | 0.6120 | 0.6120 |

观察：
- 四种策略下，`cos<0` 比例均显著高于 0.5，冲突持续存在。
- 后 2k step 冲突比例更高，说明冲突不是仅在 warmup 早期出现。
- `cagrad/dsga` 能降低冲突强度，但不能消除冲突。

## 证据 B：分层冲突 probe（RAE 官方预训练解码器）

来源：
- `results/grad_conflict_rae_pretrained_clsce_cpu_20260227/conflict_stats.json`
- `results/grad_conflict_rae_pretrained_dinoreg_cpu_20260227/conflict_stats.json`

关键值：
- `global_neg_ratio ≈ 0.482`
- 分层上深层与浅层均出现负余弦（非局部单层偶发）。

## 备注

1. 上述 A 证据基于训练在线记录的 LoRA 共享参数冲突（最直接反映联训时共享梯度关系）。
2. 若要进一步确认“全 encoder（非仅 LoRA）”在当前 checkpoint 上的冲突热图，可追加一次 layerwise probe（固定同一 batch 协议）。
