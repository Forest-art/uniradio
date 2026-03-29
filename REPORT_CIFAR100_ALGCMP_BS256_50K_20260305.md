# REPORT: CIFAR100 Algorithm Compare (bs=256, steps=50000)

Naming update (2026-03-05):
- `DSGA-M`: magnitude alignment
- `DSGA-D`: directional decomposition
- Legacy `ma_laga_*` parameter names are compatibility aliases.
- Detailed implementation + layerwise tuning report:
  `REPORT_DSGA_MD_IMPLEMENTATION_AND_CIFAR100_TUNING_20260305.md`

Group: `cifar100_algcmp_bs256_50k_20260305_142142`
Manifest: `/scratch/peilab/xlubl/dsga_runs/cifar100_algcmp_bs256_50k_20260305_142142_jobs.tsv`

## Setup

- Dataset: CIFAR100
- Seed: 42
- Batch size: 256
- Steps: 50000
- Train module: `unirae.train_cifar10`
- Compared methods: `naive`, `pcgrad`, `cagrad`, `dsga`

## Slurm Jobs

- naive: `351097`
- pcgrad: `351098`
- cagrad: `351099`
- dsga: `351106`

All completed successfully.

## Final Eval (`eval_last.json` @ step=50000)

| method | acc_txt | zero_shot_loss | recon_mse | recon_rmse | psnr |
|---|---:|---:|---:|---:|---:|
| naive | 0.5001 | 2.8462 | 0.214210 | 0.462828 | 6.6916 |
| pcgrad | 0.5104 | 2.7850 | 0.205989 | 0.453860 | 6.8616 |
| cagrad | 0.5079 | 2.7347 | 0.158818 | 0.398520 | 7.9910 |
| dsga | 0.5112 | 2.6908 | 0.131746 | 0.362968 | 8.8026 |

## Takeaway

- Understanding (`acc_txt`): `dsga` best (slightly above `pcgrad`).
- Generation (`recon_mse/rmse`, `psnr`): `dsga` best with clear margin.
- Overall tradeoff on this 50k/bs256 setting: `dsga` is Pareto-best among the four.
