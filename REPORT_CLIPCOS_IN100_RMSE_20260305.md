# CLIP-Cosine vs Reconstruction Conflict Benchmark (ImageNet-100)

Date: 2026-03-05  
Group: `dsga_clipcos_in100_rmse_cmp_bs32_20260305_025937`

## 1. Experiment Plan

Goal: Replace classification/linear-probing objective with CLIP visual feature alignment to amplify understanding-generation conflict.

- Understanding loss:
  - `L_u = 1 - cos( normalize(h(z_enc)), normalize(f_clip(x)) )`
  - `f_clip`: frozen `openai/clip-vit-base-patch16` vision encoder
  - `h`: trainable understanding projection head
- Generation loss:
  - `L_g = rMSE(recon, x)`
- Shared parameter conflict solver (compared methods):
  - `naive`, `pcgrad`, `cagrad`, `dsga`
- Fair protocol:
  - Dataset: `clane9/imagenet-100`
  - Seed: `42`
  - Batch size: `32`
  - Steps: `3000`
  - LoRA: last 4 blocks, rank 8, alpha 16
  - Learning rates: `lr_lora=2e-5`, `lr_decoder=2e-5`, `lr_und=1e-4`
  - LPIPS/GAN: disabled (`0.0`)
  - Init checkpoint: `results/train_conflict_bottleneck_rae_cpu_20260227_v2/latest.pt`

## 2. Job Manifest

- naive: `350972`
- pcgrad: `350973`
- cagrad: `350974`
- dsga: `350975`
- full eval: `350982`

Manifest file:  
`/scratch/peilab/xlubl/dsga_runs/dsga_clipcos_in100_rmse_cmp_bs32_20260305_025937_jobs.tsv`

## 3. Results

### 3.1 Quick Eval (subset, step=3000)

| Method | u_cosine | rMSE | MSE |
|---|---:|---:|---:|
| naive | 0.874322 | 0.083436 | 0.006962 |
| pcgrad | 0.874369 | 0.083443 | 0.006963 |
| cagrad | 0.874929 | 0.083642 | 0.006996 |
| dsga | 0.874764 | 0.083572 | 0.006984 |

### 3.2 Full Validation + rFID (5000 samples)

| Method | eval_u_cosine | eval_rMSE | eval_MSE | rFID |
|---|---:|---:|---:|---:|
| naive | 0.805826 | 0.093790 | 0.008797 | 46.0110 |
| pcgrad | **0.806110** | 0.093769 | 0.008793 | 46.0252 |
| cagrad | 0.805128 | **0.093451** | **0.008733** | **45.8176** |
| dsga | 0.805485 | 0.093504 | 0.008743 | 45.8436 |

Full summary file:  
`/scratch/peilab/xlubl/dsga_runs/full_eval_rmse_rfid_summary.json`

## 4. Conclusions

1. Under CLIP-cosine understanding + rMSE reconstruction, all four methods are very close on understanding (`eval_u_cosine` spread < 0.001).
2. Generation quality ranking is clearer: `cagrad ≈ dsga > pcgrad > naive` by `rMSE/rFID`.
3. In this setup, DSGA improves generation over naive/pcgrad but is slightly behind cagrad on final full-val `rMSE/rFID`.
4. This objective did not produce a large trade-off gap on understanding; conflict is present but mild at this training horizon (3k steps).

