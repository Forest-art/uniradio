# REPORT: CE FullFT + RMSE + LPIPS+GAN Compare (2026-03-05)

Group: `dsga_ce_in100_fullft_lpipsgan_rmse_cmp_bs32_20260305_132551`

## 1) Setup

- Dataset: `clane9/imagenet-100` (`train` / `validation`)
- Model update: `encoder_update=full` (no LoRA)
- Understanding loss: `ce`
- Reconstruction loss: `rmse`
- Perceptual/adversarial: `lpips_weight=1.0`, `gan_weight=0.75`
- GAN schedule: `disc_update_start_step=750`, `gan_start_step=1000`
- Steps: `3000`, batch size: `32`, seed: `42`
- Shared strategies: `naive`, `cagrad`, `dsga`

Slurm train jobs:
- naive: `351075`
- cagrad: `351076`
- dsga: `351077`

Manifest:
- `/scratch/peilab/xlubl/dsga_runs/dsga_ce_in100_fullft_lpipsgan_rmse_cmp_bs32_20260305_132551_jobs.tsv`

## 2) Quick Eval at Step 3000 (`eval_last.json`, 640 samples)

| method | acc | rmse | mse |
|---|---:|---:|---:|
| naive | 0.959375 | 0.081871 | 0.006703 |
| cagrad | 0.951563 | 0.063790 | 0.004069 |
| dsga | 0.945313 | 0.060545 | 0.003666 |

## 3) Full Eval + rFID (val=5000, rFID samples=5000)

Slurm eval job: `351090`

Summary files:
- `/scratch/peilab/xlubl/dsga_runs/dsga_ce_in100_fullft_lpipsgan_rmse_cmp_bs32_20260305_132551_full_eval_rmse_rfid_summary.json`
- `/scratch/peilab/xlubl/dsga_runs/full_eval_rmse_rfid_summary.json`

| method | eval_acc | eval_rmse | eval_mse | val_rfid |
|---|---:|---:|---:|---:|
| naive | 0.9296 | 0.079134 | 0.006262 | 4.5354 |
| cagrad | 0.9286 | 0.062108 | 0.003857 | 3.9498 |
| dsga | 0.9216 | 0.058830 | 0.003461 | 3.7969 |

## 4) Compare to previous CE FullFT RMSE-only group

Reference group: `dsga_ce_in100_fullft_cmp_bs32_20260305_112543` (LPIPS/GAN disabled)

Delta = (new LPIPS+GAN run) - (old RMSE-only run)

| method | delta_acc | delta_rmse | delta_rfid |
|---|---:|---:|---:|
| naive | -0.0040 | +0.008665 | -36.4692 |
| cagrad | -0.0094 | +0.000486 | -32.5728 |
| dsga | -0.0104 | +0.003996 | -29.4057 |

Takeaway:
- Enabling `LPIPS+GAN` drastically improves rFID for all methods.
- RMSE and ACC are slightly worse than RMSE-only setting.
- Under this setup, `dsga` gives best reconstruction (`rmse/mse`) and best rFID; `naive` gives best ACC.
