# DSGA Handoff TODO

Date: 2026-04-01
Branch: `xlubl/enc-anchor-dsga-prototype-20260328`

## What This Push Contains

- `scripts/launch_manifest_srun.py`
  - completed-run detection now covers both CIFAR-style checkpoints and IN100 top-level `latest.pt`
- `slurm/run_in100_locked_multiseed_srun.sh`
  - can now disable methods via `ENABLE_*` env flags, so the minimal paper set can be launched without editing the script
- `unirae/train_dsga_rae_lora.py`
  - non-empty full-update RAE/DSGA training entrypoint for ImageNet-style runs
- `unirae/eval_rae_table1_baselines.py`
  - supports local Arrow cache, optional checkpoint loading, probe skipping, and repo-local rFID tmp dir
- `unirae/eval_rae_reference_npz.py`
  - evaluates official/pretrained RAE against the official ImageNet-256 reference NPZ
- `unirae/eval_vavae_reference_npz.py`
  - evaluates SD-VAE / VA-VAE baselines against the same reference NPZ
- `unirae/train_cifar10.py` + `unirae/grad_conflict.py`
  - `enc_anchor_rec_gate_strength/min` are wired through for CIFAR100 ablations

## Priority Order

1. Finish the minimal IN100 locked multiseed main-table run.
2. Aggregate and read the 3-seed summary before opening any new sweep.
3. Only if the result is still promising, spend budget on full-update RAE or appendix-only variants.

## Cluster Commands

### 1. IN100 locked multiseed, minimal paper set

Run only `naive`, `cagrad(beta=0.5)`, and `dsga_global(gamma=0.5, tau=0.0)`.

```bash
cd /project/peilab/luxiaocheng/projects/DSGA
ENABLE_PCGRAD=false \
ENABLE_DSGA_LAYERWISE=false \
CAGRAD_BETA=0.5 \
DSGA_GLOBAL_GAMMA=0.5 \
DSGA_GLOBAL_TAU=0.0 \
SEEDS=42,43,44 \
LAUNCH=true \
bash slurm/run_in100_locked_multiseed_srun.sh
```

### 2. Aggregate the locked multiseed result

Replace `OUT_DIR` with the actual run directory printed by the launcher.

```bash
cd /project/peilab/luxiaocheng/projects/DSGA
python scripts/aggregate_in100_multiseed.py \
  --manifest "${OUT_DIR}/jobs.tsv" \
  --runs_root runs \
  --out_root "${OUT_DIR}"
```

Read these fields first:

- `val_top1_acc_mean`
- `val_rmse_mean`
- `val_rfid_mean`

### 3. Full-update RAE pilot only if needed

This is not the main-table blocker. Use it only after the locked multiseed result is in hand.

```bash
cd /project/peilab/luxiaocheng/projects/DSGA
SHARED_STRATEGY=dsga \
DSGA_ALIGN_GAMMA=0.5 \
DSGA_GROUPING=layerwise \
STEPS=10000 \
EVAL_EVERY=1000 \
EVAL_MAX_BATCHES=50 \
bash slurm/run_in1k_dsga_rae_full_srun.sh
```

### 4. Official reference-NPZ eval

If a full-update checkpoint or official pretrained baseline needs a paper-facing rFID number:

```bash
cd /project/peilab/luxiaocheng/projects/DSGA
bash slurm/run_in1k_rae_reference_npz_eval_srun.sh
bash slurm/run_vavae_reference_npz_eval_srun.sh
```

## Decision Rules

- If the IN100 locked 3-seed result does not beat `cagrad(beta=0.5)` cleanly on the tradeoff, stop expanding DSGA on IN100.
- Do not spend more budget on `enc-anchor` as a main-line method. Current evidence is appendix-level at best.
- Do not reopen `dsga_layerwise` as a paper main result unless the new 3-seed evidence clearly beats `dsga_global`.

## Notes

- Prefer the manifest/srun reuse path instead of ad-hoc `sbatch`.
- Keep outputs under repo-local `results/` or `runs/`.
- `unirae/eval_rae_table1_baselines.py` now defaults its rFID tmp dir to `results/in1k_rfid_tmp`, so it stays inside the allowed workspace.
