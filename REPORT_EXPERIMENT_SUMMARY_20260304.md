# DSGA 实验总报告（更新至 2026-03-04）

## 0. 报告范围与判定口径

### 数据来源
- 项目日志与旧报告：
  - `EXPERIMENT_LOG.md`
  - `REPORT_EXPERIMENT_SUMMARY_20260303.md`
  - `REPORT_LAGA_VS_CAGRAD_20260302.md`
- 项目内结果表：`results/*/summary.csv`
- Scratch 汇总与最终评测：`/scratch/peilab/xlubl/unirae_runs/*.csv` 与各 run 的 `final_eval.json`

### 判定口径（用于“有效/无效”）
- 有效：在**同协议**（数据、步数、batch、seed、损失定义一致）下，相对基线出现可复现优势；优先看是否 `win_both`（ACC 上升且 MSE/rMSE 下降）。
- 条件有效：能改善某一侧指标，但存在明显 trade-off，或跨批次不稳定。
- 无效：当前证据下不能带来目标收益（双降/双升失败，或被基线支配）。

---

## 1. 实验设计总览

### 1.1 任务与统一目标
- 双任务联合训练：
  - Understanding：分类 Top-1 Acc（越高越好）
  - Generation：重建误差（MSE/rMSE，越低越好）与 rFID（越低越好）
- 核心问题：多目标梯度冲突如何处理，才能同时提升理解与生成。

### 1.2 数据与主干设置
- CIFAR100：ResNet18 / ViT-small / Swin-Tiny-Patch4
- SUN397：DINOv2 ViT-S14
- ImageNet-100：ViT-S（scratch init），对比 `joint/pcgrad/cagrad/la_cagrad/gma_laga/ma_laga/...`

### 1.3 方法线
- 基线：`naive`, `pcgrad`, `cagrad`
- 扩展：adaptive-beta（static/online）、SAOP、Feature Variance Reg、LAGA/MA-LAGA/GBVC、LA-CAGrad、EGD 等。

---

## 2. 关键结果与结论

## 2.1 CIFAR100（ResNet18，20k）

### A) GradNorm 范围实验（`mean + conflict_deep`）
来源：`results/cifar100_gradnorm_conflictdeep_cmp_20260223_20k_v2/summary.csv`

- `naive`: acc `0.5297 -> 0.5322`, mse `0.2961 -> 0.2635`
- `pcgrad`: acc `0.5277 -> 0.5315`, mse `0.2664 -> 0.2473`
- `cagrad`: acc `0.5257 -> 0.5299`, mse `0.2281 -> 0.2056`

结论：`mean + conflict_deep(layer3|layer4)` 在该线稳定有效。

### B) 固定 CAGrad 公平调参
来源：`results/cifar100_cagrad_pareto_boost_worker4fair_20260223/summary.csv`

- 最佳：`beta=0.35, lambda_txt=1.0`
- acc `0.5384`, mse `0.2210`
- 相对同批 naive（日志记录）：`delta_acc=+0.0003`, `delta_mse=-0.0387`, `win_both=True`

结论：固定 CAGrad 在该公平设定可实现 Pareto 改善。

### C) Text warmup（先 4k text-only，再 joint）
来源：`results/cifar100_textwarmup_cmp_20260223_20k/summary.csv`

- naive: acc `0.5306 -> 0.5246`, mse `0.2623 -> 0.2832`
- pcgrad: acc `0.5322 -> 0.5257`, mse `0.2497 -> 0.2681`
- cagrad: acc `0.5350 -> 0.5258`, mse `0.2239 -> 0.2410`

结论：该 warmup 策略明确无效（双指标同时变差）。

### D) Adaptive beta / Online beta
来源：
- `results/cifar100_cagrad_adaptivebeta_cmp_20260223_20k/summary.csv`
- `results/cifar100_cagrad_adaptivebeta_tune_20260223_20k/summary.csv`
- `results/cifar100_cagrad_onlinebeta_cmp_20260223_20k/summary.csv`
- `results/cifar100_cagrad_boost_overlay_adaptive_20260223_20k/summary.csv`

核心观察：
- 早期 compare：adaptive 比 naive 好，但不如 fixed cagrad。
- online compare 一批次里，online/static 都优于 fixed cagrad（小幅双赢）。
- overlay 到强 fixed 基线后，adaptive 又明显回退。

结论：adaptive-beta 有潜力但不稳定，当前属于“条件有效”。

### E) Feature Variance Regularization（`lambda_var`）
来源：`results/cifar100_cagrad_varreg_cmp_20260223_20k/summary.csv`

- cagrad_base: acc `0.5256`, mse `0.2247`
- cagrad_var(0.10): acc `0.5314`, mse `0.2191`
- cagrad_var(0.20): acc `0.5370`, mse `0.2219`

结论：在该批次有效；`0.10` 偏重重建，`0.20` 更平衡。

### F) SAOP
来源：`results/cifar100_saop_cmp_20260223_20k/summary.csv`

- saop: acc `0.5343`, mse `0.2591`
- naive: acc `0.5326`, mse `0.2653`

结论：SAOP 相对 naive 有效，但不占据该批次 Pareto 前沿（条件有效）。

---

## 2.2 CIFAR100（Transformer 线）

### A) ViT-small（20k）
来源：`results/cifar100_vit_cmp_20260224_20k/summary.csv`

- naive: acc `0.3903`, mse `0.0874`
- pcgrad: acc `0.3884`, mse `0.0888`（被 naive 支配）
- cagrad: acc `0.4118`, mse `0.0781`（双赢）
- cagrad+var: acc `0.3965`, mse `0.0744`（偏生成）

结论：CAGrad 有效，PCGrad 无效。

### B) Swin-Tiny-Patch4（20k）
来源：`results/cifar100_patch4_cmp_20260224_20k/summary.csv`

- naive: acc `0.5251`, mse `0.0973`
- cagrad: acc `0.5302`, mse `0.0809`
- cagrad+var: acc `0.5310`, mse `0.0812`

结论：cagrad 与 cagrad+var 均明显有效（一个偏 MSE，一个偏 ACC）。

### C) 纯 LAGA vs CAGrad（Patch4）
来源：`/scratch/peilab/xlubl/unirae_runs/cifar100_patch4_cmp_20260302_laga_vs_baselines.csv`

- cagrad: acc `0.5302`, mse `0.08087`
- LAGA: acc `0.5296`, mse `0.09986`
- LAGA+GBVC: acc `0.5339`, mse `0.09977`

结论：纯 LAGA 系列 ACC 接近/略高，但 MSE 显著劣于 CAGrad（无效于“同时优化”目标）。

### D) MA-LAGA 组件与 GBVC（Patch4, fair20k）
来源：
- `/scratch/peilab/xlubl/unirae_runs/ma_laga_component_ablation20k_final_20260303.csv`
- `/scratch/peilab/xlubl/unirae_runs/ma_laga_gbvc20k_final_compare_20260302.csv`

关键数字：
- Global CAGrad: acc `0.5272`, mse `0.07695`
- Pure MA: acc `0.5457`, mse `0.05540`
- MA-LAGA(full): acc `0.5493`, mse `0.05629`
- MA-LAGA + GBVC (`lambda_gbvc=0.05~0.2`) 未超越无 GBVC 版本

结论：
- MA（幅值调控）是主要增益来源；MA-LAGA(full)整体显著优于 CAGrad（本批次最强结果之一）。
- GBVC 在此设置无增益。

---

## 2.3 SUN397 + DINOv2（20k）
来源：
- `results/sun397_dino_cmp_quick_20260224/summary.csv`（1k quick）
- `results/sun397_dino_cmp_full_20260224/summary.csv`（20k full）

full 20k：
- naive: acc `0.4668`, mse `0.5801`
- pcgrad: acc `0.4594`, mse `0.5954`（双差）
- cagrad: acc `0.4530`, mse `0.5140`（重建好、分类差）

结论：
- 该任务下无方法实现相对 naive 的双赢。
- cagrad 仅在生成侧有效，pcgrad 无效。

---

## 2.4 ImageNet-100（10k）

### A) MSE-loss 训练目标（bs64）
来源：`/scratch/peilab/xlubl/unirae_runs/in100_ma_laga_ablation10k_bs64_final_20260303.csv`

- joint: acc `0.4081`, rmse `0.02876`, rfid `27.47`
- cagrad: acc `0.4213`, rmse `0.02586`, rfid `24.43`
- pure LAGA: acc `0.4203`, rmse `0.02952`, rfid `29.32`
- pure MA: acc `0.4013`, rmse `0.02646`, rfid `24.06`
- MA-LAGA(full): acc `0.3959`, rmse `0.02859`, rfid `25.37`

结论：
- cagrad 在该协议下最稳健。
- pure LAGA 与 MA-LAGA(full) 无效；pure MA 仅改善生成、伤害分类（条件有效）。

### B) RMSE-loss 训练目标（公平重跑）
来源：`/scratch/peilab/xlubl/unirae_runs/in100_baselines_recon_rmse_loss_bs64/summary_rmse_loss_methods_20260303.csv`

- joint: acc `0.4125`, rmse `0.019425`, rfid `15.21`
- pcgrad: acc `0.4163`, rmse `0.019439`, rfid `16.31`
- cagrad: acc `0.4150`, rmse `0.015732`, rfid `9.51`
- la_cagrad: acc `0.4069`, rmse `0.016786`, rfid `12.89`
- gma_laga: acc `0.4022`, rmse `0.016633`, rfid `14.81`

结论：
- cagrad 明确主导重建侧（rmse/rfid 最优）。
- pcgrad 仅在 acc 略高，整体不占优。
- la_cagrad / gma_laga 在本批次不如 cagrad。

### C) 变体补跑（2026-03-03 晚）
来源：
- `/scratch/peilab/xlubl/unirae_runs/in100_laga_fix_bs64_vs_cagrad_20260303.csv`
- `/scratch/peilab/xlubl/unirae_runs/in100_egd_recon_rmse_loss_bs64/.../final_eval.json`

- capped MA-LAGA: acc `0.4072`, rmse `0.02830`, rfid `28.99`
- NR-LAGA: acc `0.4209`, rmse `0.02899`, rfid `28.42`
- EGD: acc `0.4028`, rmse `0.01789`, rfid `13.07`

结论：相对 cagrad 基线，这些补跑当前都不成立为更优解。

### D) 今日新增（2026-03-04）：MA-LAGA grouping（global vs layerwise）
来源：`/scratch/peilab/xlubl/unirae_runs/in100_ma_laga_grouping_rmse_10k/final_compare_layerwise_vs_global.csv`

- global, gamma=0.5: acc `0.4206`, rmse `0.01648`, rfid `7.46`
- global, gamma=1.0: acc `0.4078`, rmse `0.01510`, rfid `6.36`
- layerwise, gamma=0.5: acc `0.4088`, rmse `0.01755`, rfid `9.69`
- layerwise, gamma=1.0: acc `0.3916`, rmse `0.01607`, rfid `7.60`

结论（初步）：
- 在该组实验内，`global` 明显优于 `layerwise`。
- `global,gamma=1.0` 给出当前最低 rmse/rfid，但有 ACC 损失；`global,gamma=0.5` 给更好 ACC。
- 注意：本组 `eval_rfid_num_samples=2048`，而 03-03 基线常用 1024；跨批次比较 rFID 时需谨慎。

## 2.5 ImageNet-100（DINO 初始化 DSGA 简化 Pipeline）

### A) Quick 验证（1200 step）
来源：`/scratch/peilab/xlubl/unirae_runs/uniradio_quick_20260304_165726/quick_in100_20260304_165726/summary.json`

- frozen LP: top1 `0.9104`
- naive: top1 `0.1780`, rmse `0.07592`, rfid `221.17`
- cagrad: top1 `0.1942`, rmse `0.07131`, rfid `187.16`
- ma_laga_global: top1 `0.2052`, rmse `0.06421`, rfid `162.29`

结论：在短程 quick setting 下，分解法显著优于 naive；其中 MA-LAGA 同时给出更高 top1 与更低重建误差。

### B) Fair 20k（本轮最新）
来源：
- `/scratch/peilab/xlubl/unirae_runs/uniradio_20k_20260304_183016/unir20k_naive_retry_20260304_183016/summary.json`
- `/scratch/peilab/xlubl/unirae_runs/uniradio_20k_20260304_183016/unir20k_cagrad_20260304_183016/summary.json`
- `/scratch/peilab/xlubl/unirae_runs/uniradio_20k_20260304_183016/unir20k_malaga_20260304_183016/summary.json`

统一结果（phase4 eval）：
- frozen LP: top1 `0.9232`
- naive(20k): top1 `0.4572`, rmse `0.02578`, rfid `33.58`
- cagrad(20k): top1 `0.4696`, rmse `0.02179`, rfid `24.13`
- ma_laga_global(20k): top1 `0.4534`, rmse `0.01861`, rfid `17.88`

结论：
- `cagrad` 在理解指标（linear probe top1）最佳。
- `ma_laga_global` 在生成指标（rmse/rfid）最佳。
- 当前仍无“同时超过 cagrad 的双赢解”。

### C) 关键诊断（为什么 linear probe 仍明显下降）
- 即使 20k 后最优 top1 约 `0.47`，相对 frozen LP `0.9232` 仍有大幅语义损失。
- 该结果支持“联合重建会破坏预训练语义”的核心假设；现有分解仅能部分缓解。
- 本轮评估为快速协议（`probe_steps=1200`, `eval_max_batches=40`, `rfid_num_samples=512`），用于快速比较趋势；用于论文最终表格前建议再跑 full-eval 版本复核绝对值。

---

## 3. 有效 / 无效清单（给论文写作直接引用）

## 3.1 有效（当前证据支持）
- `mean + conflict_deep` 梯度归一化（CIFAR100-ResNet）。
- fixed CAGrad（尤其 `beta=0.35`）在 CIFAR100 多批次优于 naive。
- CAGrad / CAGrad+Var 在 Transformer-CIFAR100 上显著优于 naive。
- MA-LAGA（含 MA 成分）在 CIFAR100-Patch4 fair20k 显著优于 CAGrad。
- ImageNet-100（ViT-s scratch, RMSE-loss, 10k）里，CAGrad 对重建指标（rmse/rfid）稳定最强。
- DINO 初始化 DSGA（IN100, 20k）里，CAGrad 相对 naive 可同时提升 top1 与重建。
- 03-04 新增：MA-LAGA global grouping 相对 layerwise 有明显优势（初步）。

## 3.2 条件有效（看目标或批次）
- adaptive beta（含 online）：部分批次可双赢 fixed cagrad，但复现性不足。
- SAOP：相对 naive 有提升，但不稳定占 Pareto 前沿。
- CAGrad 在 SUN397：只改善生成，不改善分类。
- pure MA（ImageNet-100 MSE-loss）：生成提升但分类下降。
- DINO 初始化 DSGA（IN100, 20k）里，MA-LAGA global 可显著改善重建（rmse/rfid），但 top1 低于 CAGrad。

## 3.3 无效（当前证据不支持）
- text warmup 4k -> joint（CIFAR100）
- PCGrad 在 ViT-small CIFAR100 与 SUN397 full 的效果
- 纯 LAGA（CIFAR100 patch4 与 IN100 MSE-loss）
- MA-LAGA(full) 在 IN100 MSE-loss 版本
- LA-CAGrad / capped MA-LAGA / NR-LAGA / EGD（相对 IN100 cagrad 基线）

---

## 4. 总结性结论

1. 当前最稳的主线仍是 CAGrad：在多条线中都能稳定抬升理解或给出更平衡解。
2. MA-LAGA 不是“普适替代 CAGrad”，而是更偏生成侧：在 DINO 初始化 IN100 20k 中，重建最好但 top1 未超 CAGrad。
3. 主要未解问题是“语义保真”：DINO frozen LP 约 `0.92`，联合训练后仅约 `0.45~0.47`，说明语义漂移仍重。
4. 训练目标定义与评估协议会显著改变排序（`mse` vs `rmse`、rfid 采样数、probe 步数），跨实验比较必须严格同协议。

---

## 5. 给 Gemini 的论文/画图输入建议（可直接用）

优先图表（含数据源）：
1. CIFAR100-ResNet：`naive/pcgrad/cagrad` 在 `normnone` vs `mean+conflict_deep`
   - 源：`results/cifar100_gradnorm_conflictdeep_cmp_20260223_20k_v2/summary.csv`
2. CIFAR100-Patch4：`cagrad vs LAGA vs MA-LAGA` 主对比
   - 源：
     - `/scratch/peilab/xlubl/unirae_runs/cifar100_patch4_cmp_20260302_laga_vs_baselines.csv`
     - `/scratch/peilab/xlubl/unirae_runs/ma_laga_component_ablation20k_final_20260303.csv`
3. ImageNet-100：MSE-loss 与 RMSE-loss 两个 regime 的方法排序变化
   - 源：
     - `/scratch/peilab/xlubl/unirae_runs/in100_ma_laga_ablation10k_bs64_final_20260303.csv`
     - `/scratch/peilab/xlubl/unirae_runs/in100_baselines_recon_rmse_loss_bs64/summary_rmse_loss_methods_20260303.csv`
4. SUN397：`naive/pcgrad/cagrad` 的 ACC-MSE tradeoff
   - 源：`results/sun397_dino_cmp_full_20260224/summary.csv`
5. 03-04 新增 MA-LAGA grouping：global vs layerwise
   - 源：`/scratch/peilab/xlubl/unirae_runs/in100_ma_laga_grouping_rmse_10k/final_compare_layerwise_vs_global.csv`
6. 03-04 新增 DINO 初始化 DSGA 简化 pipeline（1200 quick + 20k）
   - 源：
     - `/scratch/peilab/xlubl/unirae_runs/uniradio_quick_20260304_165726/quick_in100_20260304_165726/summary.json`
     - `/scratch/peilab/xlubl/unirae_runs/uniradio_20k_20260304_183016/unir20k_naive_retry_20260304_183016/summary.json`
     - `/scratch/peilab/xlubl/unirae_runs/uniradio_20k_20260304_183016/unir20k_cagrad_20260304_183016/summary.json`
     - `/scratch/peilab/xlubl/unirae_runs/uniradio_20k_20260304_183016/unir20k_malaga_20260304_183016/summary.json`

图形建议：
- 主图：ACC-MSE（或 ACC-rMSE）Pareto 散点。
- 辅图：不同 regime（MSE-loss / RMSE-loss）的排名变化条形图。
- 消融图：MA-LAGA component（Vanilla/CAGrad/Pure-LAGA/Pure-MA/Full）。
- 新增诊断图：`Frozen-LP -> Joint-Naive -> CAGrad -> MA-LAGA` 的 linear-probe 保真曲线（显示语义漂移幅度）。
