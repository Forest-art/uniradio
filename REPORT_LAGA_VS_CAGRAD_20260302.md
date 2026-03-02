# LAGA vs CAGrad (CIFAR100, Swin-Tiny-P4) Report

Date: 2026-03-02  
Author: Codex (for Gemini handoff)

## 1. Goal

Test whether our LAGA family can **surpass CAGrad** under strictly fair settings on CIFAR100 joint understanding+generation training.

## 2. Fair Protocol (strictly matched)

- Dataset: `CIFAR100`
- Model: `swin_tiny_patch4`
- Batch size: `128`
- Total steps: `20000`
- Seed: `42`
- Joint objective: text/classification + reconstruction
- Shared params: backbone
- Gradient norm setup: `train.grad_norm_mode=mean`, `train.grad_norm_scope=conflict_all`
- Same train/eval pipeline (`unirae.train_cifar10`)

Only runs with `eval_last.step=20000` are used in conclusions.

## 3. Methods Compared

### 3.1 Reference baseline

- CAGrad: `beta=0.5`
- Run: `cifar100_patch4_fair20k_cagrad_b0p5_s42_20260302_154425`

### 3.2 LAGA fair20k sweep

- `th=-0.1, restore=0.0`
- `th=-0.1, restore=0.25`
- `th=-0.1, restore=0.5`
- `th=-0.2, restore=0.5`
- `th=-0.1, restore=0.25, lambda_gbvc=0.01`

### 3.3 LAGA-alpha (new implementation)

Code changes:
- `unirae/grad_conflict.py`: added LAGA adaptive soft-projection knobs  
  `alpha_mode`, `alpha_power`, `alpha_min`, `alpha_max`
- `unirae/train_cifar10.py`: parsed and logged `train.laga_alpha_*` and passed to `apply_laga_objective`

Sweep:
- fixed alpha baseline (`alpha_mode=fixed`, alpha=1)
- adaptive ratio alpha (`alpha_mode=ratio`) with several `power/min/max` settings

## 4. Final Results (all at step=20000)

Primary metrics:
- Understanding: Top-1 Acc
- Generation: MSE (lower is better), PSNR (higher is better)

| Method | Key setting | Acc | MSE | PSNR |
|---|---|---:|---:|---:|
| CAGrad | `beta=0.5` | 0.5272 | **0.07695** | **11.1380** |
| LAGA | `th=-0.1, r=0.0` | 0.5266 | 0.09826 | 10.0763 |
| LAGA | `th=-0.1, r=0.25` | 0.5272 | **0.09780** | 10.0966 |
| LAGA | `th=-0.1, r=0.5` | **0.5340** | 0.09786 | 10.0941 |
| LAGA | `th=-0.2, r=0.5` | 0.5294 | 0.09836 | 10.0718 |
| LAGA+GBVC | `th=-0.1, r=0.25, gbvc=0.01` | 0.5266 | 0.09961 | 10.0172 |
| LAGA-alpha | `th=-0.1, r=0.25, ratio p1 min0.05 max1` | 0.5324 | **0.09793** | 10.0908 |
| LAGA-alpha | other tested configs | 0.5249~0.5314 | 0.09831~0.09987 | 10.0058~10.0740 |

## 5. Key Findings

1. LAGA family consistently has **higher or comparable Acc** than CAGrad in best settings.
2. LAGA family consistently has **much worse MSE** than CAGrad:
   - Best LAGA MSE: `0.09780`
   - CAGrad MSE: `0.07695`
   - Gap: `+0.02085` absolute (about `+27%` relative).
3. Adding `restore_ratio` and adaptive `alpha_k` only gives **small local gains** inside LAGA family; it does not close the gap to CAGrad.
4. Light GBVC (`lambda_gbvc=0.01`) did not help in this setting.

## 6. Why LAGA did not beat CAGrad (evidence-backed hypotheses)

1. **Asymmetric projection removes generation-critical parallel component too aggressively**.  
   Even with softening (`restore_ratio`, adaptive alpha), generation branch still underperforms in MSE.
2. **LAGA optimization bias favors understanding objective** under conflict-heavy regions.  
   Observed pattern: Acc rises, MSE stalls high.
3. **Current alpha scheduling is not objective-aware**.  
   `alpha_k` based on norm ratio is geometric, but not directly tied to reconstruction loss sensitivity.
4. **Single-point merge rule may be too rigid** vs CAGrad's global compromise direction.

## 7. Suggested Next Experiments (for Gemini to analyze/propose)

1. Add per-layer diagnostic logging for conflict surgery:
   - removed parallel energy ratio  
   - effective merge angle vs original `g_g`  
   - per-layer contribution to reconstruction loss decrease
2. Replace static geometric alpha with **loss-aware alpha**:
   - choose alpha to bound per-step increase in reconstruction loss proxy
3. Try staged strategy:
   - early training use CAGrad-like blend, late training gradually increase asymmetry
4. Add reconstruction-protected constraint:
   - keep a minimum retained parallel component budget for generation on high-sensitivity layers

## 8. Repro Artifacts

- Report table CSV:  
  `/scratch/peilab/xlubl/unirae_runs/laga_vs_cagrad_report_table_20260302.csv`
- LAGA fair20k jobs:  
  `/scratch/peilab/xlubl/unirae_runs/laga_fair20k_jobs_20260302_154425.tsv`
- LAGA-alpha jobs:  
  `/scratch/peilab/xlubl/unirae_runs/laga_alpha20k_jobs_20260302_161927.tsv`

