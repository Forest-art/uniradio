# REPORT: 实验总览更新（截至 2026-03-12）

## 1) 汇总范围

本报告汇总目前已完成并有结果文件支撑的实验，覆盖：

- ImageNet-100 上官方/公开预训练重建模型评测（`rMSE`, `rFID`）
- ImageNet-100 上多任务梯度策略对比（`naive/pcgrad/cagrad/dsga`）
- CIFAR100 上 joint 训练策略对比
- DSGA 当前实现语义（方向 layer-wise，幅度 global）状态
- 扩展诊断（冲突空间/时间异质性、初始化锚定、架构异质性、`lambda_mag` sweep）

时间范围：2026-03-03 至 2026-03-12（含本次更新）。

## 2) 实验设计总览

### 2.1 数据与指标

- 数据集：`clane9/imagenet-100`（`train` / `validation`）
- 验证集大小：`5000`
- 生成指标：`rMSE`（越低越好）、`rFID`（越低越好）
- 理解指标（按任务不同）：`u_cosine` 或 `acc`

### 2.2 主要实验轴

- A. 官方预训练重建模型横向评测（冻结权重，直接重建）
- B. IN100 冲突优化（CLIP-cos + rMSE，LoRA）
- C. IN100 全量 encoder 更新（No-LoRA）：
  - `CE + rMSE`
  - `CE + rMSE + LPIPS + GAN`
- D. CIFAR100 50k 步 joint 方法对比
- E. 扩展诊断：
  - 冲突深度-时间热图
  - `DSGA-M` 幅值恢复 sweep
  - granularity / initialization / architecture probes

## 3) 核心结果

### 3.1 论文表可直接引用：官方模型在 IN100 的 `rMSE/rFID`

| 方法 | 评测分辨率 | rMSE | rFID | 备注 |
|---|---:|---:|---:|---|
| VQGAN (`vqgan_imagenet_f16_1024`) | 224 | 0.1041386 | 22.3989 | 官方权重 |
| VA-VAE / SD-VAE (`stabilityai/sd-vae-ft-mse`) | 224 | 0.0441609 | 4.0457 | lightning-dit 路径 |
| MAE (RAE `MAE.yaml`) | 256 | 0.0300767 | 3.0855 | 官方预训练配置 |
| RAE (DINOv2-B, `DINOv2-B.yaml`) | 224 | 0.1123207 | 7.5470 | 官方预训练配置 |

结论（该表）：

- 当前已测集合里，`MAE` 的 `rMSE/rFID` 最好，`SD-VAE` 次之。
- `VQGAN` 与 `RAE(DINOv2-B)` 的重建侧指标明显更高。
- 注意公平性：本表含 `224` 与 `256` 混合分辨率，论文主表若要求严格公平，建议统一分辨率后复跑一次。

### 3.2 IN100：CLIP-cos + rMSE（LoRA, 3k steps, full eval）

| 方法 | eval_u_cosine | eval_rMSE | rFID |
|---|---:|---:|---:|
| naive | 0.805826 | 0.093790 | 46.0110 |
| pcgrad | **0.806110** | 0.093769 | 46.0252 |
| cagrad | 0.805128 | **0.093451** | **45.8176** |
| dsga | 0.805485 | 0.093504 | 45.8436 |

结论：

- 理解侧几乎持平（差异 < 0.001）。
- 生成侧排序为 `cagrad ≈ dsga > pcgrad > naive`。

### 3.3 IN100：CE + rMSE（Full-update, No-LoRA, full eval）

| 方法 | eval_acc | eval_rMSE | rFID |
|---|---:|---:|---:|
| naive | 0.9336 | 0.070468 | 41.0046 |
| cagrad | **0.9380** | 0.061622 | 36.5226 |
| dsga | 0.9320 | **0.054835** | **33.2027** |

结论：

- `dsga` 在生成侧（`rMSE/rFID`）最好。
- `cagrad` 在理解侧（`acc`）最好。

### 3.4 IN100：CE + rMSE + LPIPS+GAN（Full-update, full eval）

| 方法 | eval_acc | eval_rMSE | rFID |
|---|---:|---:|---:|
| naive | **0.9296** | 0.079134 | 4.5354 |
| cagrad | 0.9286 | 0.062108 | 3.9498 |
| dsga | 0.9216 | **0.058830** | **3.7969** |

结论：

- 加入 `LPIPS+GAN` 后，三方法 `rFID` 大幅下降（质量显著改善）。
- 该设定下 `dsga` 继续在生成侧最优，但理解侧分数最低。

### 3.5 CIFAR100：bs=256, 50k, joint

| 方法 | acc_txt | recon_rmse | psnr |
|---|---:|---:|---:|
| naive | 0.5001 | 0.462828 | 6.6916 |
| pcgrad | 0.5104 | 0.453860 | 6.8616 |
| cagrad | 0.5079 | 0.398520 | 7.9910 |
| dsga | **0.5112** | **0.362968** | **8.8026** |

结论：

- 在该 CIFAR100 协议下，`dsga` 同时拿到最好理解与生成，Pareto 最优。

### 3.6 CIFAR100：ViT-S, 2k steps, fresh `DSGA-M` (`lambda_mag`) sweep

协议：

- 数据集：`CIFAR100`
- 骨干：`vit_small`
- steps：`2000`
- batch size：`64`
- 优化：`AdamW lr=5e-4`
- 策略：`train.grad_strategy=dsga`
- `DSGA-D=layer-wise`, `DSGA-M=global`
- sweep：`lambda_mag ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}`

| lambda_mag | Acc (%) | rMSE | rFID | mean_norm_ratio | mean_m_t | mean_abs_alpha_post |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 16.48 | 0.605 | 250.548 | 1.0163 | 1.1743 | 1.25e-08 |
| 0.2 | 16.70 | 0.605 | 252.098 | 1.0309 | 1.3915 | 1.15e-08 |
| 0.4 | **16.99** | 0.595 | 249.753 | 1.0748 | 1.9792 | 1.83e-08 |
| 0.6 | 15.66 | 0.611 | 248.729 | 1.1538 | 2.8002 | 3.06e-08 |
| 0.8 | 16.15 | 0.589 | 249.671 | 1.2997 | 4.4538 | 5.68e-08 |
| 1.0 | 16.88 | **0.568** | **245.753** | 1.5488 | 6.7645 | 3.85e-08 |

结论：

- `lambda_mag` 增大后，`DSGA-M` 的全局增益 `m_t` 明显抬升，`mean_norm_ratio` 也单调增大，说明幅值恢复确实生效。
- `mean_abs_alpha_post` 始终在 `1e-8` 量级，说明 `DSGA-D` 的投影后残差基本为零，方向分解是干净的。
- 该 2k-step proxy 上，理解侧最佳点在 `lambda_mag=0.4`，而生成侧最佳点在 `lambda_mag=1.0`；即更强的 magnitude alignment 更有利于生成，但理解侧最优并不随 `lambda_mag` 单调上升。

图：

![Fresh lambda_mag sensitivity](results/extended_diagnostics_item2_fresh_20260312_212304/figs/sensitivity_lambda_mag.png)

### 3.7 扩展诊断（03-12 evidence pack）

来源：`results/extended_diagnostics_20260312_195642` 与 `results/extended_diagnostics_item2_fresh_20260312_212304`

#### 3.7.1 冲突的空间/时间异质性仍然成立

- IN100 10k probe 显示，四种方法（`joint/pcgrad/cagrad/dsga`）都存在持续的 antagonistic patches。
- 在 DSGA 的 IN100 ViT-S 运行中，最深四层平均有 `0.54` 的采样时间处在 `cos<0` 区域。
- 这说明冲突不是“只在浅层或只在训练早期出现”的现象。

#### 3.7.2 初始化锚定效应明确存在

| 方法 | Acc (%) | rMSE | rFID |
|---|---:|---:|---:|
| scratch joint | 37.38 | 0.021 | 24.181 |
| dino joint | 38.81 | 0.027 | 30.536 |
| dsga (scratch) | **39.62** | 0.028 | **17.921** |

结论：

- `DINO` 初始化确实把理解侧往上锚定，但会明显伤害生成侧（`rFID` 从 `24.181` 恶化到 `30.536`）。
- `DSGA scratch` 把 Pareto 前沿推到了更好的区域：理解高于 `scratch joint` 和 `dino joint`，同时 `rFID` 最低。

#### 3.7.3 架构异质性成立，不是单一架构特例

- `resnet18`: mean `rho^- = 0.20`
- `vit_small`: mean `rho^- = 0.31`
- `swin_tiny_patch4`: mean `rho^- = 0.24`

结论：

- 空间异质性在 `ResNet / ViT / Swin` 三类架构上都存在，只是模式强弱不同；`ViT` 冲突占比最高。

#### 3.7.4 granularity 证据是“局部支持，整体未定”

单个 offline probe 对比（CIFAR100 Swin-T final checkpoint）显示：

| granularity | Acc (%) | rMSE |
|---|---:|---:|
| global | 52.87 | 0.241 |
| layer-wise | **54.93** | **0.237** |

但结合 03-05 的 50k tuning 报告：

- 没有一个 completed 50k `layer-wise` 变体能同时优于 `global DSGA` 的 `acc_txt` 与 `recon_rmse`。
- 因此当前**还不能**把“layer-wise routing 普遍优于 global routing”写成稳定结论。
- 更准确的表述是：`layer-wise` 在部分离线 probe/局部协议上有正面信号，但在系统化 tuning 和主结果协议上证据仍不足。

## 4) 当前总结论（可写入论文讨论）

1. 在 IN100 上，方法排序明显依赖训练目标：
   - `CLIP-cos + rMSE`：`cagrad` 与 `dsga` 接近；
   - `CE + rMSE`：`dsga` 生成更强，`cagrad` 理解更强；
   - `CE + rMSE + LPIPS+GAN`：`dsga` 生成优势进一步扩大。
2. 感知/对抗损失（LPIPS+GAN）对 `rFID` 的影响远大于梯度策略本身，属于一阶因素。
3. 对“预训练重建器本体能力”的横向比较（VQGAN/SD-VAE/MAE/RAE）显示：`MAE` 与 `SD-VAE` 当前更优。
4. 扩展诊断表明，梯度冲突具有明确的空间/时间异质性，并非某一层或某一阶段的偶发现象。
5. `DSGA-M` 的核心作用更接近“幅值恢复/生成侧补偿”：在 fresh `lambda_mag` sweep 中，较大的 `lambda_mag` 明显改善 `rMSE/rFID`，而理解侧最优点出现在中等 `lambda_mag`（`0.4`）附近。
6. `DSGA-D` 的投影残差在 fresh probe 中稳定处于 `1e-8` 量级，可认为方向分解实现正确。
7. 关于 granularity，当前证据只支持“layer-wise 可能有局部收益”，**不支持**“layer-wise 已稳定优于 global”的强 claim。
8. 论文主表必须同协议比较（分辨率、损失定义、`rfid_num_samples`），跨协议只可做趋势讨论。

## 5) 当前不能强 claim 的点

1. 目前还没有足够强的 full-budget 证据支持“layer-wise DSGA 普遍优于 global DSGA”。
2. `GBVC / feature-variance` 方向的证据还不够稳：旧版 pooled-feature probe 不支持强结论；新版 intermediate-feature 路径已经补到脚本，但还应以 full-budget fresh rerun 为准。
3. `local Pareto micro-step` 的代码路径已经接入 `scripts/extended_diagnostics_pack.py`，但论文若要把它作为主证据，建议再跑一版更充分的采样预算。

## 6) DSGA 实现状态（本次确认）

当前代码已按要求落地为：

- 方向分解：`layer-wise`
- 幅度对齐：`global`

对应实现点：

- `unirae/grad_conflict.py`: `apply_ma_laga(..., magnitude_scope=...)`
- `unirae/train_cifar10.py`: `dsga` 默认 `dsga_m_scope=global`
- `unirae/train_dsga_rae_lora.py`: `dsga` 路径已切到全局幅度 + 分组方向

## 7) 关键结果文件（可追溯）

- VQGAN: `/scratch/peilab/xlubl/dsga_runs/vqgan_official_in100_eval224_1024only_20260305_221823/summary.json`
- SD-VAE: `/scratch/peilab/xlubl/dsga_runs/lightningdit_vavae_in100_eval224_sdvaeftmse_20260305_223109/summary.json`
- MAE: `/scratch/peilab/xlubl/dsga_runs/mae_official_in100_rmse_rfid_256full_5000rfid_20260305_224137/eval_summary.json`
- RAE(DINOv2-B): `/scratch/peilab/xlubl/dsga_runs/rae_dinov2b_in100_rmse_rfid_256full_5000rfid_20260305_225533/eval_summary.json`
- CLIP-cos + rMSE full eval: `/scratch/peilab/xlubl/dsga_runs/dsga_clipcos_in100_rmse_cmp_bs32_20260305_025937_full_eval_summary.json`
- CE + rMSE full eval: `/scratch/peilab/xlubl/dsga_runs/dsga_ce_in100_fullft_cmp_bs32_20260305_112543_full_eval_rmse_rfid_summary.json`
- CE + rMSE + LPIPS+GAN full eval: `/scratch/peilab/xlubl/dsga_runs/dsga_ce_in100_fullft_lpipsgan_rmse_cmp_bs32_20260305_132551_full_eval_rmse_rfid_summary.json`
- CIFAR100 50k: `/project/peilab/luxiaocheng/projects/DSGA/REPORT_CIFAR100_ALGCMP_BS256_50K_20260305.md`
- Extended diagnostics (historical pack): `/project/peilab/luxiaocheng/projects/DSGA/results/extended_diagnostics_20260312_195642/README.md`
- Extended diagnostics (fresh item2 pack): `/project/peilab/luxiaocheng/projects/DSGA/results/extended_diagnostics_item2_fresh_20260312_212304/README.md`
- Fresh item2 figures:
  - `/project/peilab/luxiaocheng/projects/DSGA/results/extended_diagnostics_item2_fresh_20260312_212304/figs/sensitivity_lambda_mag.png`
  - `/project/peilab/luxiaocheng/projects/DSGA/results/extended_diagnostics_item2_fresh_20260312_212304/figs/sensitivity_lambda_mag.pdf`
  - `/project/peilab/luxiaocheng/projects/DSGA/results/extended_diagnostics_item2_fresh_20260312_212304/figs/diag_mt_time_cifar_vits.pdf`

## 8) 这次更新的最短版本

- 能说的：
  - 冲突是 spatial + temporal heterogeneous 的。
  - `DSGA-D` 投影是干净的。
  - `DSGA-M` 增大后，生成侧会明显受益。
  - 初始化锚定和架构异质性都是真实存在的。
- 不能强说的：
  - `layer-wise` 现在还没有被稳定证明优于 `global`。
  - `GBVC` 和 `local Pareto` 还需要更强的 full-budget supporting evidence。
