# UniRadio Experiment Summary (up to 2026-03-03)

## Scope
- This report summarizes the latest reproducible comparison tables discussed in this round.
- Focus: CIFAR100 (20k steps) and ImageNet-100 (10k steps), seed=42.
- Key code update in this round: `unirae/train_imagenet100_methods.py` supports `--recon_loss_type rmse` for training objective.

## A. CIFAR100 Patch4: Baselines vs LAGA
- Source: `/scratch/peilab/xlubl/unirae_runs/cifar100_patch4_cmp_20260302_laga_vs_baselines.csv`
- Setting: `swin_tiny_patch4`, `steps=20000`, `seed=42`.

| Method | Top-1 Acc | MSE | PSNR |
|---|---:|---:|---:|
| naive | 0.5251 | 0.097286 | 10.1195 |
| pcgrad | 0.5219 | 0.093520 | 10.2909 |
| cagrad | 0.5302 | 0.080870 | 10.9221 |
| LAGA | 0.5296 | 0.099859 | 10.0061 |
| LAGA+GBVC | 0.5339 | 0.099769 | 10.0100 |

## B. CIFAR100 Patch4: MA-LAGA + GBVC Sweep
- Source: `/scratch/peilab/xlubl/unirae_runs/ma_laga_gbvc20k_final_compare_20260302.csv`
- Setting: fair 20k protocol.

| Group | lambda_gbvc | align_gamma | norm_restore | Acc | MSE | PSNR |
|---|---:|---:|---:|---:|---:|---:|
| CAGrad_ref | 0.0 | - | - | 0.5272 | 0.076948 | 11.1380 |
| MA-LAGA_v2_no_GBVC | 0.0 | 1.0 | False | 0.5493 | 0.056287 | 12.4959 |
| MA-LAGA_v4_no_GBVC | 0.0 | 1.0 | True | 0.5512 | 0.056791 | 12.4572 |
| MA-LAGA_v2_GBVC | 0.05 | 1.0 | False | 0.5489 | 0.057119 | 12.4322 |
| MA-LAGA_v2_GBVC | 0.1 | 1.0 | False | 0.5444 | 0.057438 | 12.4080 |
| MA-LAGA_v2_GBVC | 0.2 | 1.0 | False | 0.5432 | 0.058155 | 12.3541 |
| MA-LAGA_v4_GBVC | 0.05 | 1.0 | True | 0.5455 | 0.057188 | 12.4269 |
| MA-LAGA_v4_GBVC | 0.1 | 1.0 | True | 0.5429 | 0.059847 | 12.2296 |
| MA-LAGA_v4_GBVC | 0.2 | 1.0 | True | 0.5437 | 0.056860 | 12.4519 |

## C. CIFAR100 Patch4: Strict Component Ablation
- Source runs: `runs/cifar100_patch4_ma_laga_ablation20k_*`
- Setting (from run config): `bs=128`, `steps=20000`, `swin_tiny_patch4`, `seed=42`.

| Method | Top-1 Acc | MSE | PSNR |
|---|---:|---:|---:|
| Vanilla Joint | 0.5251 | 0.097286 | 10.1195 |
| Global CAGrad (beta=0.5) | 0.5272 | 0.076948 | 11.1380 |
| Pure LAGA (direction only) | 0.5296 | 0.099859 | 10.0061 |
| Pure MA (magnitude only) | 0.5457 | 0.055403 | 12.5646 |
| MA-LAGA (full) | 0.5493 | 0.056287 | 12.4959 |

## D. ImageNet-100 (MSE-loss regime): 5-way Main Ablation (bs=64, steps=10k)
- Source dir: `/scratch/peilab/xlubl/unirae_runs/in100_ma_laga_ablation_10k_bs64/`

| Method | Top-1 Acc | MSE | rMSE(eval) | rFID |
|---|---:|---:|---:|---:|
| Vanilla Joint | 0.4081 | 0.000827303 | 0.028763 | 27.4748 |
| Global CAGrad (beta=0.5) | 0.4213 | 0.000668685 | 0.025859 | 24.4254 |
| Pure LAGA | 0.4203 | 0.000871613 | 0.029523 | 29.3186 |
| Pure MA | 0.4012 | 0.000700248 | 0.026462 | 24.0565 |
| MA-LAGA (full) | 0.3959 | 0.000817597 | 0.028594 | 25.3692 |

## E. ImageNet-100 (MSE-loss regime): Follow-up Variants (bs=64, steps=10k)
| Method | Top-1 Acc | MSE | rMSE(eval) | rFID |
|---|---:|---:|---:|---:|
| LA-CAGrad (beta=0.5) | 0.4134 | 0.000715920 | 0.026757 | 26.3467 |
| GMA-LAGA | 0.4100 | 0.000243291 | 0.015598 | 11.7325 |
| Capped MA-LAGA (scale<=3) | 0.4072 | 0.000801013 | 0.028302 | 28.9850 |
| NR-LAGA | 0.4209 | 0.000840443 | 0.028990 | 28.4238 |

## F. ImageNet-100 (rMSE-loss as training objective): Fair Baseline Rerun (bs=64, steps=10k)
- Source: `/scratch/peilab/xlubl/unirae_runs/in100_baselines_recon_rmse_loss_bs64/summary_rmse_loss_methods_20260303.csv`

| Method | Top-1 Acc | MSE | rMSE | rFID |
|---|---:|---:|---:|---:|
| joint | 0.4125 | 0.000377326 | 0.019425 | 15.2059 |
| pcgrad | 0.4163 | 0.000377871 | 0.019439 | 16.3055 |
| cagrad | 0.4150 | 0.000247509 | 0.015732 | 9.5128 |
| la_cagrad | 0.4069 | 0.000281772 | 0.016786 | 12.8863 |
| gma_laga | 0.4022 | 0.000276653 | 0.016633 | 14.8148 |

## Consolidated Conclusions
1. CIFAR100 (20k): MA-LAGA family clearly improves over CAGrad on both classification and reconstruction in this setup.
2. CIFAR100 GBVC sweep: adding GBVC on top of MA-LAGA did not show additional gains under tested weights (0.05/0.1/0.2).
3. ImageNet-100 (10k, MSE-loss regime): Global CAGrad remains a strong and stable baseline; several LAGA variants did not consistently surpass it on both Top-1 and rFID simultaneously.
4. ImageNet-100 (10k, rMSE-loss training): rerun confirms method ranking changes under objective switch; CAGrad currently has best reconstruction side (rMSE/rFID), while PCGrad has slightly higher Top-1 than CAGrad.
5. For fair cross-method claims, compare only within the same loss regime (`mse` vs `rmse`) and same protocol (`bs`, `steps`, `seed`, `lr`, warmup).

## Notes
- Some early runs were preempted and resumed with new run names; this report uses completed run artifacts only.
- Numbers are copied from `final_eval.json`, `eval_last.json`, or generated summary CSVs listed above.
