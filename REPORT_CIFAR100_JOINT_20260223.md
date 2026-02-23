# UniRAE-RADIO CIFAR100 Joint Training Report (2026-02-23)

## 1. Scope and Code Status
- Current code has been rolled back to the **pre-intervention joint-training version**:
  - Revert commit: `7690d93`
  - Reverted commit: `b6f4d86` (text-warmup intervention schedule)
- Main objective in this phase:
  - Compare `naive`, `pcgrad`, `cagrad` under unified CIFAR100 settings.
  - Find a `cagrad` setup that is Pareto-better on both:
    - Understanding: `acc` (higher better)
    - Generation: `mse` (lower better)

## 2. Experiment Design

### 2.1 Common setup
- Dataset: `CIFAR100`
- Model: `ResNet18` backbone + text prototype head + reconstruction decoder
- Seed: `42`
- Joint objective:
  - Understanding loss (`Lu`) + Generation loss (`Lg`)
  - `lambda_txt=1.0`, `lambda_rec=1.0` unless noted
- Train steps: mainly `20,000` (overridden from base config)
- Eval split: `test`
- Key metrics:
  - `acc_txt` / `zero_shot_acc` (understanding)
  - `mse` / `recon_mse` (generation)

### 2.2 Methods compared
- Gradient merge strategy:
  - `naive`
  - `pcgrad`
  - `cagrad`
- Gradient normalization/decoupling setting:
  - `none`
  - `mean + conflict_deep + layer3|layer4`
- CAGrad tuning:
  - `beta ∈ {0.10, 0.20, 0.35}`
  - `lambda_txt ∈ {1.0, 1.1, 1.2}`

## 3. Results

### 3.1 Baseline strategy comparison (20k, seed=42)
Source: `results/cifar100_gradnorm_conflictdeep_cmp_20260223_20k_v2/summary.csv`

| strategy | setting | acc | mse |
|---|---|---:|---:|
| naive | normnone | 0.5297 | 0.2961 |
| naive | normmean+conflictdeep | 0.5322 | 0.2635 |
| pcgrad | normnone | 0.5277 | 0.2664 |
| pcgrad | normmean+conflictdeep | 0.5315 | 0.2473 |
| cagrad | normnone | 0.5257 | 0.2281 |
| cagrad | normmean+conflictdeep | 0.5299 | 0.2056 |

Observation:
- `mean + conflict_deep` improves both metrics vs `normnone` for all three strategies.
- But in this batch, `cagrad` still trails `naive` slightly on `acc`.

### 3.2 Fair CAGrad tuning vs naive reference
Source: `results/cifar100_cagrad_pareto_boost_worker4fair_20260223/delta_vs_naive.csv`

Reference naive (same worker setting):  
- `acc=0.5381`, `mse=0.2597`

Best CAGrad candidate:
- `beta=0.35`, `lambda_txt=1.0`
- `acc=0.5384`, `mse=0.2210`
- Delta vs naive:
  - `delta_acc=+0.0003`
  - `delta_mse=-0.0387`
  - `win_both=True`

Interpretation:
- Under this fair setup, **CAGrad achieved Pareto improvement over naive**.

### 3.3 Intervention ablation (text warmup then joint)
Source: `results/cifar100_textwarmup_cmp_20260223_20k/delta_warmup_minus_joint.csv`

Warmup policy tested:
- First `4000` steps text-only update, then joint updates.

Delta (warmup - direct joint):
- naive: `delta_acc=-0.0060`, `delta_mse=+0.0209`
- pcgrad: `delta_acc=-0.0065`, `delta_mse=+0.0184`
- cagrad: `delta_acc=-0.0092`, `delta_mse=+0.0171`

Conclusion for this intervention:
- Warmup policy consistently hurt both tasks in this setting.
- Therefore this branch was reverted from training code.

## 4. Final Conclusions (Current Stage)
- Current recommended training line is back to **direct joint training** (no warmup intervention).
- Useful setting confirmed:
  - `grad_norm_mode=mean`
  - `grad_norm_scope=conflict_deep`
  - `grad_norm_layers=layer3+layer4`
- Best observed CAGrad candidate in fair comparison:
  - `cagrad_beta=0.35`, `lambda_txt=1.0`
  - Achieves slight `acc` gain and clear `mse` gain over naive.

## 5. Suggested Questions for Gemini (Next Optimization Round)
- Why does `text-warmup -> joint` degrade both tasks on CIFAR100 here?
- How to design a smoother curriculum (instead of hard warmup switch), e.g.:
  - linear ramp of `lambda_rec` from 0 to 1
  - cosine/step schedule for task weights
- How to improve CAGrad stability on `acc` while preserving `mse` gains?
  - finer `beta` search around `0.30~0.45`
  - adaptive `beta` based on gradient cosine/conflict rate
- Should normalization target only conflict batches or selected layers with adaptive scaling?
- Would multi-objective early stopping / checkpoint selection by Pareto score improve final pick?

## 6. Key Artifacts
- Strategy/normalization comparison:
  - `results/cifar100_gradnorm_conflictdeep_cmp_20260223_20k_v2/summary.csv`
- Fair CAGrad tuning vs naive:
  - `results/cifar100_cagrad_pareto_boost_worker4fair_20260223/delta_vs_naive.csv`
  - `results/cifar100_cagrad_pareto_boost_worker4fair_20260223/pareto_vs_naive.png`
- Warmup intervention ablation:
  - `results/cifar100_textwarmup_cmp_20260223_20k/delta_warmup_minus_joint.csv`
  - `results/cifar100_textwarmup_cmp_20260223_20k/pareto_warmup_compare.png`
