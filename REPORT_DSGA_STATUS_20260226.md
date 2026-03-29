# DSGA Current Code & Experiment Consolidated Report (2026-02-26)

## 0. Scope and Snapshot

This report consolidates the **current code state** and **experiment outcomes** in:

- Project root: `/project/peilab/luxiaocheng/projects/DSGA`
- Focus period: 2026-02-23 to 2026-02-24 (latest logged experiment window)
- Primary metrics:
  - Understanding: `acc` / `zero_shot_acc` (higher is better)
  - Generation: `mse` / `recon_mse` (lower is better)

Current branch/worktree status:

- Branch: `main` (tracking `origin/main`)
- Worktree is intentionally in-progress:
  - modified: core training modules and report markdowns
  - untracked: `EXPERIMENT_LOG.md`, `results/`, `runs/`, new `slurm/*` and `unirae/summarize_*` scripts

---

## 1. Codebase Organization (整理后的代码视图)

## 1.1 Training Entry and Core Pipeline

Main entry:

- `unirae/train_cifar10.py`

Current capability summary in this entry:

- Unified train modes:
  - `joint`
  - `text_only`
  - `recon_only`
- Joint gradient strategies:
  - `naive`
  - `pcgrad`
  - `cagrad`
  - `saop`
- Gradient normalization controls:
  - `grad_norm_mode`: `none|mean|geom|unit`
  - `grad_norm_scope`: `all|conflict_all|conflict_deep`
  - optional layer filtering for deep conflict normalization
- CAGrad extensions:
  - fixed beta
  - adaptive beta (conflict-aware)
  - online adaptive beta (EMA-style beta state update)
- Additional regularization:
  - `FeatureVarianceLoss` via `lambda_var`
- Standard outputs per run:
  - `train_setup.json`, `metrics.jsonl`, `cos_summary.json`, `understanding.json`, `generation.json`, checkpoints

## 1.2 Data Layer

Main module:

- `unirae/data_cifar10.py`

Now supports datasets:

- `cifar10`
- `cifar100`
- `sun397`

SUN397 support details:

- source modes:
  - HuggingFace (`sun_source=hf`, default dataset `dpdl-benchmark/sun397`)
  - torchvision (`sun_source=torchvision`)
- normalization defaults to ImageNet stats for SUN397
- optional train/eval sample caps (`sun_max_train_samples`, `sun_max_eval_samples`) for quick-turn runs

## 1.3 Backbone Layer

Main module:

- `unirae/models/backbone.py`

Active backbone options:

- `resnet18`
- `vit_small`
- `swin_tiny_patch4` (aliases: `patch4`, `vit_patch4`)
- `dinov2_vits14` (aliases: `dino_vits14`, `vit_small_dinov2`)

This enables three model families in one pipeline:

- CNN baseline line (ResNet)
- lightweight transformer line (ViT/Swin)
- pretrained DINO line (SUN397 migration)

## 1.4 Gradient Merge/Conflict Layer

Main module:

- `unirae/grad_conflict.py`

Implemented strategies:

- `apply_naive`
- `apply_conflict_aware` (PCGrad-style)
- `apply_cagrad` (with fixed/adaptive/online variants)
- `apply_saop` and layerwise SAOP merge helper

## 1.5 Loss Layer

Main module:

- `unirae/losses.py`

Newly used component:

- `FeatureVarianceLoss` for feature collapse mitigation (`lambda_var`, VICReg-style variance hinge)

## 1.6 Experiment Automation and Summarization

Submission scripts (Slurm):

- CIFAR100 lines:
  - `slurm/submit_cagrad_adaptivebeta_compare_cifar100.sh`
  - `slurm/submit_cagrad_adaptivebeta_tune_cifar100.sh`
  - `slurm/submit_cagrad_onlinebeta_compare_cifar100.sh`
  - `slurm/submit_cagrad_varreg_compare_cifar100.sh`
  - `slurm/submit_saop_compare_cifar100.sh`
  - `slurm/submit_vit_compare_cifar100.sh`
- SUN397 lines:
  - `slurm/submit_dino_sun397_compare.sh`
  - `slurm/submit_dino_sun397_cagrad_sweep.sh`

Summarizers:

- `unirae/summarize_cagrad_adaptivebeta_compare.py`
- `unirae/summarize_cagrad_varreg_compare.py`
- `unirae/summarize_cifar100_saop_compare.py`
- `unirae/summarize_vit_compare_cifar100.py`
- `unirae/summarize_sun397_dino_compare.py`
- `unirae/summarize_sun397_cagrad_sweep.py`

---

## 2. Experiment Design and Conclusions (详细复盘)

## 2.1 CIFAR100 (ResNet18 line): strategy and normalization baseline

Design:

- Compare `naive`, `pcgrad`, `cagrad`
- Compare `normnone` vs `mean + conflict_deep(layer3|layer4)`
- seed=42, steps=20000

Evidence:

- `results/cifar100_gradnorm_conflictdeep_cmp_20260223_20k_v2/summary.csv`

Key findings:

- `mean+conflict_deep` improves both ACC and MSE for all three strategies versus `normnone`
- In this batch:
  - best MSE: cagrad+conflict_deep (`mse=0.2056`)
  - best ACC: naive+conflict_deep (`acc=0.5322`)

Conclusion:

- **Effective**: deep-conflict gradient normalization is robustly useful on CIFAR100 ResNet line.

---

## 2.2 CIFAR100 fixed CAGrad fair Pareto boost

Design:

- Tune fixed CAGrad with beta/lambda_txt grid under fair worker setting
- Compare directly against matched naive reference

Evidence:

- `results/cifar100_cagrad_pareto_boost_worker4fair_20260223/delta_vs_naive.csv`

Best candidate:

- `beta=0.35`, `lambda_txt=1.0`
- `delta_acc=+0.0003`, `delta_mse=-0.0387`, `win_both=True`

Conclusion:

- **Effective**: fixed CAGrad can be Pareto-better than naive in fair setting.

---

## 2.3 Text warmup intervention (4k text-only -> joint)

Design:

- For naive/pcgrad/cagrad:
  - first 4000 steps text-only
  - then switch to joint
- Compare against direct joint training

Evidence:

- `results/cifar100_textwarmup_cmp_20260223_20k/delta_warmup_minus_joint.csv`

Result summary:

- all three strategies show `delta_acc<0` and `delta_mse>0`
- `win_both=False` for all

Conclusion:

- **Ineffective**: hard warmup switch degrades both understanding and generation under this setup.

---

## 2.4 Adaptive beta (simple/static/online) on CIFAR100

### 2.4.1 Early adaptive compare

Evidence:

- `results/cifar100_cagrad_adaptivebeta_cmp_20260223_20k/summary.csv`

Finding:

- adaptive beats naive but is worse than fixed CAGrad in this early compare batch

### 2.4.2 Adaptive tune grid

Evidence:

- `results/cifar100_cagrad_adaptivebeta_tune_20260223_20k/summary.csv`

Finding:

- tuned adaptive can improve ACC over fixed baseline in some settings
- but no consistent win on both ACC and MSE versus fixed CAGrad

### 2.4.3 Online beta compare

Evidence:

- `results/cifar100_cagrad_onlinebeta_cmp_20260223_20k/delta.csv`

Finding:

- vs fixed CAGrad:
  - static adaptive: `delta_acc=+0.0018`, `delta_mse=-0.0021`, `win_both=True`
  - online adaptive: `delta_acc=+0.0027`, `delta_mse=-0.0038`, `win_both=True`
- vs naive:
  - both adaptive variants still trail naive ACC in this batch

### 2.4.4 Overlay on previously strong fixed-CAGrad line

Evidence:

- `results/cifar100_cagrad_boost_overlay_adaptive_20260223_20k/delta.csv`

Finding:

- both adaptive variants lose to strong fixed CAGrad (`win_both=False` vs fixed)
- but both still beat naive (`win_both=True` vs naive)

Conclusion:

- **Partially effective / unstable**:
  - adaptive beta has upside in some batches
  - cannot yet replace fixed CAGrad as stable default

---

## 2.5 SAOP strategy compare (CIFAR100)

Design:

- Compare `naive`, `pcgrad`, `cagrad`, `saop`
- same deep conflict settings

Evidence:

- `results/cifar100_saop_cmp_20260223_20k/summary.csv`
- `results/cifar100_saop_cmp_20260223_20k/delta_vs_naive.csv`

Key result:

- SAOP vs naive: `delta_acc=+0.0017`, `delta_mse=-0.0062`, `win_both=True`
- But in same run:
  - pcgrad has stronger ACC
  - cagrad has stronger MSE

Conclusion:

- **Effective but not dominant**:
  - SAOP is better than naive
  - still not global Pareto-front winner in this batch

---

## 2.6 Feature variance regularization (CIFAR100 strong CAGrad line)

Design:

- add `L_var` with `lambda_var in {0.05,0.10,0.20}`
- compare against `naive_ref` and `cagrad_base`

Evidence:

- `results/cifar100_cagrad_varreg_cmp_20260223_20k/summary.csv`
- `results/cifar100_cagrad_varreg_cmp_20260223_20k/delta.csv`

Key conclusions by lambda:

- `lambda_var=0.10`: best MSE among var-reg settings; beats cagrad_base on both, but not naive on ACC
- `lambda_var=0.20`: beats naive on both and beats cagrad_base on both
- `lambda_var=0.05`: beats naive on both, but not cagrad_base on MSE

Conclusion:

- **Effective**:
  - feature variance regularization is a reliable positive direction
  - `lambda_var=0.20` is best-balanced in this batch

---

## 2.7 Transformer migration on CIFAR100

### 2.7.1 ViT-small compare

Evidence:

- `results/cifar100_vit_cmp_20260224_20k/summary.csv`
- `results/cifar100_vit_cmp_20260224_20k/delta_vs_naive.csv`

Key results:

- cagrad vs naive: `delta_acc=+0.0215`, `delta_mse=-0.0092`, `win_both=True`
- cagrad+var vs naive: `delta_acc=+0.0062`, `delta_mse=-0.0129`, `win_both=True`
- pcgrad fails (`win_both=False`)

### 2.7.2 Patch4 (Swin-Tiny) compare

Evidence:

- `results/cifar100_patch4_cmp_20260224_20k/summary.csv`
- `results/cifar100_patch4_cmp_20260224_20k/delta_vs_naive.csv`

Key results:

- cagrad vs naive: `delta_acc=+0.0051`, `delta_mse=-0.0164`, `win_both=True`
- cagrad+var vs naive: `delta_acc=+0.0059`, `delta_mse=-0.0161`, `win_both=True`
- pcgrad still fails on ACC (`win_both=False`)

Cross-backbone observation:

- patch4 significantly recovers understanding compared with vit_small baseline

Conclusion:

- **Effective**:
  - cagrad remains strong on transformer backbones
  - cagrad+var gives extra MSE/ACC tradeoff point
- **Ineffective**:
  - pcgrad remains weak in transformer runs

---

## 2.8 SUN397 + DINOv2 migration

### 2.8.1 Quick verify (1000 steps)

Evidence:

- `results/sun397_dino_cmp_quick_20260224/summary.csv`
- `results/sun397_dino_cmp_quick_20260224/delta_vs_naive.csv`

Result:

- cagrad slightly improves ACC, worsens MSE
- `win_both_vs_naive=False`

### 2.8.2 Full verify (20000 steps)

Evidence:

- `results/sun397_dino_cmp_full_20260224/summary.csv`
- `results/sun397_dino_cmp_full_20260224/delta_vs_naive.csv`

Result:

- pcgrad is worse than naive on both ACC and MSE
- cagrad significantly improves MSE but drops ACC
- neither is `win_both` vs naive

Conclusion:

- **Not yet solved** on SUN397+DINO:
  - no strategy currently Pareto-dominates naive
  - cagrad introduces clear ACC↔MSE tradeoff

---

## 2.9 SUN397 CAGrad sweep status

Planned design:

- `beta in {0.10,0.20,0.30,0.35,0.50}`
- `lambda_txt in {1.0,1.1,1.2}`
- plus naive anchor

Artifacts:

- `results/sun397_dino_cagrad_sweep_20260224/submitted_jobs.csv`
- `results/sun397_dino_cagrad_sweep_20260224/pending_jobs_plan.csv`

Current execution state (checked via sacct):

- submitted jobs `347646` to `347655`: all `CANCELLED`
- two jobs (`347654`,`347655`) were cancelled before node allocation
- no currently running jobs (`squeue -u xlubl` empty)
- no complete sweep summary generated yet (`summary.csv` missing)

Conclusion:

- **Pending / blocked by scheduling stability**:
  - sweep data is currently incomplete and unusable for final SUN397 tuning decision

---

## 3. Effective vs Ineffective Summary (结论总表)

| Item | Verdict | Evidence |
|---|---|---|
| Deep conflict normalization (`mean+conflict_deep`) on CIFAR100 ResNet | Effective | `results/cifar100_gradnorm_conflictdeep_cmp_20260223_20k_v2/summary.csv` |
| Fixed CAGrad (`beta=0.35`) fair tuning line | Effective | `results/cifar100_cagrad_pareto_boost_worker4fair_20260223/delta_vs_naive.csv` |
| Text warmup (4k text-only then joint) | Ineffective | `results/cifar100_textwarmup_cmp_20260223_20k/delta_warmup_minus_joint.csv` |
| Adaptive beta (static/online) | Conditionally effective, unstable | `results/cifar100_cagrad_onlinebeta_cmp_20260223_20k/delta.csv`, `results/cifar100_cagrad_boost_overlay_adaptive_20260223_20k/delta.csv` |
| SAOP | Effective vs naive, not frontier-best | `results/cifar100_saop_cmp_20260223_20k/delta_vs_naive.csv` |
| Feature variance regularization (`lambda_var`) | Effective (especially `0.20`) | `results/cifar100_cagrad_varreg_cmp_20260223_20k/delta.csv` |
| ViT-small + CAGrad | Effective | `results/cifar100_vit_cmp_20260224_20k/delta_vs_naive.csv` |
| Patch4(Swin) + CAGrad/(+var) | Effective | `results/cifar100_patch4_cmp_20260224_20k/delta_vs_naive.csv` |
| PCGrad on transformer/SUN397 lines | Mostly ineffective | `results/cifar100_vit_cmp_20260224_20k/delta_vs_naive.csv`, `results/cifar100_patch4_cmp_20260224_20k/delta_vs_naive.csv`, `results/sun397_dino_cmp_full_20260224/delta_vs_naive.csv` |
| SUN397+DINO current cagrad setup | Tradeoff only (not win-both) | `results/sun397_dino_cmp_full_20260224/delta_vs_naive.csv` |

---

## 4. Code/Workflow Gaps Identified

1. Runbook mismatch with actual capabilities

- `README_RUN.md` still describes a baseline-only view, while code now includes SAOP, adaptive/online beta, var-reg, SUN397, transformer backbones.
- This mismatch increases onboarding friction and misuse risk.

2. Sweep manifest vs actual completion

- `submitted_jobs.csv` only records submission attempts, not completion success.
- In current SUN397 sweep, all submitted jobs are cancelled but manifest still looks "filled".

3. Result robustness on new domain (SUN397)

- CIFAR100 improvements do not transfer cleanly to SUN397+DINO.
- More robust scheduling/retry + full grid completion is required before claiming migration success.

---

## 5. Recommended Next Actions (按优先级)

1. Finish and harden SUN397 sweep execution

- re-submit cancelled combinations with retry logic and state tracking (`submitted`, `running`, `completed`, `failed/cancelled`)
- only summarize from completed runs

2. Update runbook to match current training surface

- refresh `README_RUN.md` to reflect real supported datasets/backbones/strategies and active scripts

3. Lock default "stable CIFAR100 production line"

- Recommended stable candidate to freeze for now:
  - `grad_norm_mode=mean`
  - `grad_norm_scope=conflict_deep`
  - `grad_norm_layers=layer3|layer4`
  - `cagrad_beta=0.35`
  - optional `lambda_var=0.20` when balanced Pareto gain is desired

4. For SUN397, separate two optimization objectives explicitly

- If understanding-first: tune for ACC recovery with controlled MSE loss
- If generation-first: keep cagrad-like settings and tune ACC penalty bounds

---

## 6. Final Assessment

As of now, the project has transitioned from a CIFAR-only baseline pipeline to a multi-dataset, multi-backbone, multi-strategy framework with stronger experiment automation. On CIFAR100, several strategies are validated and reproducible. On SUN397+DINO, the migration is in-progress and still unresolved due both method tradeoff and interrupted sweep execution.

