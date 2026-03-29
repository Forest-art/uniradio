# REPORT: DSGA-M / DSGA-D Implementation + CIFAR100 Layerwise Tuning (2026-03-05)

## 1) Naming Update (Applied)

- `DSGA-M` = Magnitude Alignment
- `DSGA-D` = Direction Decomposition

Code-level naming policy (current):
- New keys (recommended):
  - `train.dsga_m_align_gamma`
  - `train.dsga_m_norm_restore`
  - `train.dsga_m_eps`
  - `train.dsga_d_mode`
- Legacy aliases (still accepted):
  - `train.ma_laga_align_gamma`
  - `train.ma_laga_norm_restore`
  - `train.ma_laga_eps`
  - `train.ma_laga_mode`

Current training metadata (`train_setup.json` / `train_summary`) writes both new keys and legacy aliases for compatibility.

## 2) Current DSGA Implementation (train_cifar10)

Main call path:
- `train.grad_strategy=dsga` -> `apply_ma_laga_objective(...)`
- Shared-parameter merge is done per group (`global` / `layerwise` / `deep`).

### 2.1 Objective weighting before merge
For shared params, two task grads are first weighted:

- `g_u = lambda_txt * grad(L_u)`
- `g_g = lambda_rec * grad(L_g)`

(then optional grad-norm normalization if enabled)

### 2.2 DSGA-M (magnitude alignment)
Per group, define:
- `m_u = ||g_u||`
- `m_g = ||g_g||`

Scale factor:
- `s = (m_u / (m_g + eps)) ^ gamma`

Aligned generation grad:
- `g_g_aligned = s * g_g`

where:
- `gamma = train.dsga_m_align_gamma`
- `eps = train.dsga_m_eps`

### 2.3 DSGA-D (direction decomposition)
Mode is controlled by `train.dsga_d_mode`:
- `full`: if conflict (`cos<0`), project aligned `g_g` orthogonal to `g_u`; otherwise sum.
- `direction_only`: no magnitude alignment, only direction projection on conflict.
- `magnitude_only`: do DSGA-M only, no projection.

In `full` mode with conflict and `train.dsga_m_norm_restore=true`, projected branch is restored to the aligned norm.

### 2.4 Grouping modes now available
`train.laga_grouping`:
- `global`: one group over all shared params
- `layerwise`: fixed 5 groups (`stem, layer1, layer2, layer3, layer4`)
- `deep`: selected layers only via `train.laga_layers` (newly enabled in this round)

## 3) Experiment Protocol (This Round)

Target:
- Make `layer-wise DSGA` outperform `global DSGA` on CIFAR100.

Fixed setup:
- Dataset: CIFAR100
- Seed: 42
- Batch size: 256
- Steps: 50000
- Train module: `unirae.train_cifar10`
- Base config: `configs/cifar100_baseline_joint_cagrad.yaml`
- Grad strategy: `dsga`

Reference baseline (global DSGA):
- Run: `cifar100_algcmp_bs256_50k_20260305_142142_dsga_s42`
- Job: `351106`
- Final: `acc_txt=0.5112`, `recon_rmse=0.362968`, `psnr=8.8026`

## 4) Final Results (Completed 50k runs only)

### 4.1 Layerwise default

| variant | job | elapsed | acc_txt | recon_rmse | psnr | Δacc vs global | Δrmse vs global |
|---|---:|---:|---:|---:|---:|---:|---:|
| layerwise_5groups_default | 351193 | 00:13:37 | 0.5094 | 0.368132 | 8.6799 | -0.0018 | +0.0052 |

### 4.2 Layerwise tuning set A (`cifar100_dsga_layerwise_tune2_bs256_50k_20260305_183009_jobs.tsv`)

| variant | job | elapsed | acc_txt | recon_rmse | psnr | Δacc | Δrmse |
|---|---:|---:|---:|---:|---:|---:|---:|
| g03_nr1_lrec1 | 351202 | 00:13:44 | 0.5041 | 0.410897 | 7.7253 | -0.0071 | +0.0479 |
| g08_nr1_lrec1 | 351203 | 00:13:29 | 0.4965 | **0.317159** | **9.9745** | -0.0147 | **-0.0458** |
| g05_nr0_lrec1 | 351204 | 00:13:35 | 0.5024 | 0.371647 | 8.5974 | -0.0088 | +0.0087 |
| dironly_lrec1 | 351205 | 00:13:27 | 0.5042 | 0.461965 | 6.7078 | -0.0070 | +0.0990 |
| g05_nr1_lrec09 | 351206 | 00:13:54 | 0.5108 | 0.374404 | 8.5332 | -0.0004 | +0.0114 |
| magonly_lrec1 | 351207 | 00:13:41 | 0.5102 | 0.369105 | 8.6570 | -0.0010 | +0.0061 |

### 4.3 Layerwise tuning set B (`cifar100_dsga_layerwise_tune4_bs256_50k_20260305_184615_jobs.tsv`)

| variant | job | elapsed | acc_txt | recon_rmse | psnr | Δacc | Δrmse |
|---|---:|---:|---:|---:|---:|---:|---:|
| g05_nr1_lrec085 | 351214 | 00:13:39 | 0.5086 | 0.373029 | 8.5651 | -0.0026 | +0.0101 |
| g05_nr1_lrec08 | 351215 | 00:13:37 | 0.5060 | 0.375538 | 8.5069 | -0.0052 | +0.0126 |
| g045_nr1_lrec09 | 351216 | 00:13:32 | 0.5061 | 0.385880 | 8.2710 | -0.0051 | +0.0229 |

### 4.4 Deep grouping tuning (`cifar100_dsga_layerwise_deep_tune_bs256_50k_20260305_190407_jobs.tsv`)

| variant | job | elapsed | acc_txt | recon_rmse | psnr | Δacc | Δrmse |
|---|---:|---:|---:|---:|---:|---:|---:|
| deep34_g08_lrec1 | 351222 | 00:13:04 | 0.5111 | 0.425700 | 7.4179 | -0.0001 | +0.0627 |
| deep34_g07_lrec1 | 351223 | 00:13:01 | 0.5086 | 0.423677 | 7.4593 | -0.0026 | +0.0607 |
| deep34_g08_lrec095 | 351224 | 00:13:07 | **0.5114** | 0.427274 | 7.3859 | **+0.0002** | +0.0643 |

## 5) Key Findings

- No completed 50k `layer-wise` variant achieved simultaneous improvement over `global DSGA` on both `acc_txt` and `recon_rmse`.
- Best understanding (smallest positive gain over global):
  - `deep34_g08_lrec095`: `acc_txt=0.5114` (only +0.0002), but reconstruction degraded strongly.
- Best generation:
  - `g08_nr1_lrec1`: `recon_rmse=0.3172` and `psnr=9.9745` (large gain), but `acc_txt` dropped to `0.4965`.
- Practical behavior in this setup:
  - DSGA-M/DSGA-D hyperparameters shift the solution along an accuracy-reconstruction tradeoff curve.
  - On CIFAR100 bs=256 50k, the current `global DSGA` remains the best balanced point among tested candidates.

## 6) Run Artifacts

Manifests:
- `/scratch/peilab/xlubl/dsga_runs/cifar100_algcmp_bs256_50k_20260305_142142_jobs.tsv`
- `/scratch/peilab/xlubl/dsga_runs/cifar100_dsga_layerwise_tune2_bs256_50k_20260305_183009_jobs.tsv`
- `/scratch/peilab/xlubl/dsga_runs/cifar100_dsga_layerwise_tune4_bs256_50k_20260305_184615_jobs.tsv`
- `/scratch/peilab/xlubl/dsga_runs/cifar100_dsga_layerwise_deep_tune_bs256_50k_20260305_190407_jobs.tsv`

Representative eval files:
- `/scratch/peilab/xlubl/dsga_runs/cifar100_algcmp_bs256_50k_20260305_142142_dsga_s42/eval_last.json`
- `/scratch/peilab/xlubl/dsga_runs/cifar100_dsga_layerwise_bs256_50k_s42_20260305_181025/eval_last.json`
- `/scratch/peilab/xlubl/dsga_runs/cifar100_dsga_layerwise_tune2_bs256_50k_20260305_183009_g08_nr1_lrec1_s42/eval_last.json`
- `/scratch/peilab/xlubl/dsga_runs/cifar100_dsga_layerwise_deep_tune_bs256_50k_20260305_190407_deep34_g08_lrec095_s42/eval_last.json`
