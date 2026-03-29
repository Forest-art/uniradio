# REPORT: 实验结果整理与当前结论（整理于 2026-03-23）

## 1) 范围与证据分级

本报告整理当前仓库内可直接追溯的实验材料，覆盖截至 2026-03-13 的最新结果。

为避免过度 claim，本文将证据分成两档：

- 主结果：`full eval`、`rFID` 或 `50k` 完整训练结果，可作为当前主叙事依据。
- 补充证据：`quick eval`、`2k proxy`、`offline probe`、运行时基准、多种子 quick 结果，只能作为趋势或 caveat。

说明：

- ImageNet-100 的若干底层 `summary.json` 位于 `/scratch/...`，不在当前工作区 allowlist 内；因此本报告对这些实验采用仓库内已提交报告中的数字。
- 本仓库里最新的综合总报告是 `REPORT_EXPERIMENT_SUMMARY_20260306.md`，但 `results/` 中还有 2026-03-13 的新结果，本报告已将其补入。

## 2) 主结果

### 2.1 ImageNet-100：预训练重建器横评

来源：`REPORT_EXPERIMENT_SUMMARY_20260306.md`

| 方法 | 分辨率 | rMSE | rFID | 结论 |
|---|---:|---:|---:|---|
| VQGAN | 224 | 0.1041 | 22.3989 | 明显落后 |
| SD-VAE | 224 | 0.0442 | 4.0457 | 次优 |
| MAE | 256 | **0.0301** | **3.0855** | 当前最好 |
| RAE (DINOv2-B) | 224 | 0.1123 | 7.5470 | 重建侧不占优 |

结论：

- 当前已测集合里，`MAE` 的重建指标最好，`SD-VAE` 次之。
- `RAE(DINOv2-B)` 在当前评测协议下没有表现出重建优势。
- 该表混合了 `224` 与 `256` 分辨率，只适合趋势讨论，不适合做严格公平主表。

### 2.2 ImageNet-100：CLIP-cos + rMSE（LoRA, 3k, full eval）

来源：`REPORT_CLIPCOS_IN100_RMSE_20260305.md`

| 方法 | eval_u_cosine | eval_rMSE | rFID |
|---|---:|---:|---:|
| naive | 0.805826 | 0.093790 | 46.0110 |
| pcgrad | **0.806110** | 0.093769 | 46.0252 |
| cagrad | 0.805128 | **0.093451** | **45.8176** |
| dsga | 0.805485 | 0.093504 | 45.8436 |

结论：

- 理解侧几乎重合，差异不到 `0.001`。
- 生成侧排序较清晰：`cagrad ≈ dsga > pcgrad > naive`。
- 在这个较温和的冲突设定里，`dsga` 改善了生成，但没有明显超过 `cagrad`。

### 2.3 ImageNet-100：CE + rMSE（full update, no LoRA, full eval）

来源：`REPORT_EXPERIMENT_SUMMARY_20260306.md`、`REPORT_FULLFT_CE_COMPARE_20260305.md`

| 方法 | eval_acc | eval_rMSE | rFID |
|---|---:|---:|---:|
| naive | 0.9336 | 0.070468 | 41.0046 |
| cagrad | **0.9380** | 0.061622 | 36.5226 |
| dsga | 0.9320 | **0.054835** | **33.2027** |

结论：

- 去掉 LoRA、改成 full update 后，方法差异显著放大。
- `dsga` 在生成侧最强，`cagrad` 在理解侧最强。
- 这是当前支持“DSGA 更偏生成侧收益”的主证据之一。

### 2.4 ImageNet-100：CE + rMSE + LPIPS+GAN（full update, full eval）

来源：`REPORT_FULLFT_CE_RMSE_LPIPSGAN_COMPARE_20260305.md`

| 方法 | eval_acc | eval_rMSE | rFID |
|---|---:|---:|---:|
| naive | **0.9296** | 0.079134 | 4.5354 |
| cagrad | 0.9286 | 0.062108 | 3.9498 |
| dsga | 0.9216 | **0.058830** | **3.7969** |

结论：

- 加入 `LPIPS+GAN` 后，所有方法的 `rFID` 都大幅改善，这是比梯度策略更强的一阶因素。
- 在该设定下，`dsga` 继续拿到最好生成指标，但理解侧最低。
- 如果论文要讲“生成质量提升”，必须把损失协议写清楚，否则容易把损失改动和策略改动混为一谈。

### 2.5 CIFAR100：bs=256, 50k, joint 主结果

来源：`REPORT_CIFAR100_ALGCMP_BS256_50K_20260305.md`

| 方法 | acc_txt | recon_rmse | psnr |
|---|---:|---:|---:|
| naive | 0.5001 | 0.462828 | 6.6916 |
| pcgrad | 0.5104 | 0.453860 | 6.8616 |
| cagrad | 0.5079 | 0.398520 | 7.9910 |
| dsga | **0.5112** | **0.362968** | **8.8026** |

结论：

- 在当前 CIFAR100 主协议下，`dsga` 同时取得最好理解与生成，是最干净的 Pareto-best 结果。
- 这是当前最适合当作 CIFAR100 主表结论的结果。

## 3) 调参与诊断

### 3.1 Layer-wise / deep DSGA 调参

来源：`REPORT_DSGA_MD_IMPLEMENTATION_AND_CIFAR100_TUNING_20260305.md`

关键结论：

- 没有任何一个 completed `50k` 的 `layer-wise` 或 `deep` 变体能同时优于 `global DSGA` 的 `acc_txt` 和 `recon_rmse`。
- 最强生成候选是 `g08_nr1_lrec1`：`acc_txt=0.4965`, `recon_rmse=0.317159`，但理解掉点明显。
- 最强理解候选是 `deep34_g08_lrec095`：`acc_txt=0.5114`，但 `recon_rmse=0.427274`，生成明显退化。

当前结论：

- `layer-wise` 更像是在移动准确率-重建质量 tradeoff 曲线，而不是稳定优于 `global DSGA`。
- 目前不能强 claim “layer-wise DSGA 普遍优于 global DSGA”。

### 3.2 DSGA-M (`lambda_mag`) fresh sweep

来源：`results/extended_diagnostics_item2_fresh_20260312_212304/README.md`、`REPORT_EXPERIMENT_SUMMARY_20260306.md`

关键观察：

- `lambda_mag` 从 `0.1` 增大到 `1.0` 时，`mean_norm_ratio` 从 `1.016` 升到 `1.549`，说明幅值恢复确实生效。
- 最佳理解侧出现在 `lambda_mag=0.4`：`Acc=16.99%`。
- 最佳生成侧出现在 `lambda_mag=1.0`：`rMSE=0.568`, `rFID=245.753`。
- `mean_abs_alpha_post` 长期维持在 `1e-8` 量级，说明方向分解后的残差几乎为零。

当前结论：

- 更强的 magnitude alignment 更有利于生成侧。
- 理解侧最优并不随 `lambda_mag` 单调提升。
- 该组是 `2k` proxy，不应直接替代主结果。

### 3.3 扩展诊断：冲突结构、初始化、架构异质性

来源：`results/extended_diagnostics_20260312_195642/README.md`、`REPORT_EXPERIMENT_SUMMARY_20260306.md`

当前可保留的结论：

- 空间/时间异质性成立：IN100 10k probe 中，DSGA 最深四层约有 `0.54` 的采样时间处在 `cos<0` 区域。
- 初始化锚定存在：
  - `scratch joint`: `37.38% / rFID 24.181`
  - `dino joint`: `38.81% / rFID 30.536`
  - `dsga scratch`: `39.62% / rFID 17.921`
- 架构异质性存在：
  - `resnet18`: mean `rho^- = 0.20`
  - `vit_small`: mean `rho^- = 0.31`
  - `swin_tiny_patch4`: mean `rho^- = 0.24`

更谨慎的表述：

- 冲突不是只出现在浅层或训练早期。
- `DINO` 初始化会把理解侧往上锚定，但会损伤生成侧。
- `ViT` 上冲突占比更高，但这不自动推出“DSGA 在 ViT 上一定最好”。

## 4) 2026-03-13 的新补充结果

### 4.1 Table 10：运行时与显存开销

来源：`results/table10_runtime_memory_20260313_015304/table10_runtime_memory_all.csv`

| 架构 | 方法 | Time ms/iter | Peak GPU GB | 相对 Joint 开销 |
|---|---|---:|---:|---:|
| vit_small_patch16 | joint | 14.858 | 0.778 | 0.00% |
| vit_small_patch16 | pcgrad | 23.486 | 1.466 | 58.07% |
| vit_small_patch16 | cagrad | 30.228 | 1.548 | 103.45% |
| vit_small_patch16 | dsga | 24.911 | 1.366 | 67.66% |
| swin_tiny_patch4 | joint | 18.955 | 1.208 | 0.00% |
| swin_tiny_patch4 | pcgrad | 32.346 | 2.055 | 70.65% |
| swin_tiny_patch4 | cagrad | 38.679 | 2.177 | 104.06% |
| swin_tiny_patch4 | dsga | 35.641 | 2.302 | 88.03% |

结论：

- 在 `vit_small_patch16` 上，`dsga` 明显便宜于 `cagrad`，且显存更低。
- 在 `swin_tiny_patch4` 上，`dsga` 仍比 `cagrad` 更快，但峰值显存更高。
- 因此“DSGA 一定比 CAGrad 更省资源”不能无条件强 claim，更准确的说法是：时间开销通常更低，但显存优势依赖架构。

### 4.2 Table 11：quick multi-seed（3 seeds）

来源：`results/table11_quick_multiseed_20260313_015304/table11_multiseed.csv`

| 方法 | Acc mean | Acc std | rMSE mean | rMSE std |
|---|---:|---:|---:|---:|
| joint | 40.56 | 0.14 | 0.380 | 0.000 |
| pcgrad | 40.58 | 0.25 | 0.378 | 0.001 |
| cagrad | **41.19** | 0.22 | **0.341** | 0.003 |
| dsga | 40.75 | 0.26 | 0.376 | 0.003 |

结论：

- 在这个 `quick multiseed` 协议里，`cagrad` 同时拿到最好均值 `Acc` 和 `rMSE`。
- `dsga` 只比 `joint/pcgrad` 略好，没有复现 CIFAR100 `50k` 主协议里的统治性优势。
- 这个结果应当作为 caveat：`dsga` 的优势更依赖具体协议和训练预算，而不是在所有轻量设置下都稳定领先。

## 5) 当前最稳的结论

1. `DSGA` 当前最强的证据集中在生成侧，尤其是 ImageNet-100 的 full-update 设定和 CIFAR100 的 50k 主协议。
2. `CAGrad` 在若干 quick/proxy 或弱冲突协议上仍然非常强，尤其是最新 `table11` quick multiseed。
3. `LPIPS+GAN` 对 `rFID` 的影响远大于梯度策略差异，本身就是一阶变量。
4. `global DSGA` 目前比 `layer-wise` 更稳，后者只有局部正信号，没有系统性胜出证据。
5. 冲突的 spatial + temporal heterogeneity 有足够证据支撑，可作为方法动机。

## 6) 当前不该强 claim 的点

1. 不能说 `layer-wise DSGA` 已经稳定优于 `global DSGA`。
2. 不能说 `DSGA` 在所有协议、所有训练预算、所有架构上都优于 `CAGrad`。
3. 不能把不同分辨率或不同损失协议下的结果直接并成一个公平主表。
4. 不能把 `table11` 这类 quick multiseed 结果直接外推成 full-budget 最终结论。

## 7) 汇报口径建议

如果现在要对外或对组内汇报，建议主线这样讲：

- CIFAR100 主结果：`global DSGA` 在 `bs=256, 50k` 协议下是当前最好的平衡点。
- IN100 主结果：`DSGA` 在 full-update 设定下显著强化生成侧；若加 `LPIPS+GAN`，该优势继续保留。
- 方法动机：冲突确实具有 spatial + temporal heterogeneity，且存在初始化锚定与架构异质性。
- 诚实 caveat：`CAGrad` 在一些 quick/proxy 协议上更强；`layer-wise DSGA` 尚未证明比 `global DSGA` 更稳。

## 8) 关键本地材料

- `REPORT_EXPERIMENT_SUMMARY_20260306.md`
- `REPORT_CIFAR100_ALGCMP_BS256_50K_20260305.md`
- `REPORT_DSGA_MD_IMPLEMENTATION_AND_CIFAR100_TUNING_20260305.md`
- `REPORT_CLIPCOS_IN100_RMSE_20260305.md`
- `REPORT_FULLFT_CE_COMPARE_20260305.md`
- `REPORT_FULLFT_CE_RMSE_LPIPSGAN_COMPARE_20260305.md`
- `results/extended_diagnostics_20260312_195642/README.md`
- `results/extended_diagnostics_item2_fresh_20260312_212304/README.md`
- `results/table10_runtime_memory_20260313_015304/table10_runtime_memory_all.csv`
- `results/table11_quick_multiseed_20260313_015304/table11_multiseed.csv`
