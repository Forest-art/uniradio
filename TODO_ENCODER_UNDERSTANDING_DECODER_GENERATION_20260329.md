# Encoder-Understands / Decoder-Generates TODO

## Goal

Validate the asymmetric training idea:

- `encoder` should mainly preserve understanding gradients
- `decoder` should remain purely generation-driven
- gradient decomposition / conflict handling should happen only on the shared `encoder`

This is the concrete follow-up to the current `enc_anchor_dsga` prototype.

## Current Code Status

The current codebase already matches the intended structural split in
`unirae/train_imagenet100_methods.py`:

- `cls_head` is updated only by understanding loss `L_u`
- `decoder` is updated only by reconstruction loss `L_g`
- `encoder` is the only shared part where gradient arbitration happens

Relevant entry:

- `python -m unirae.train_imagenet100_methods --method enc_anchor_dsga`

Relevant method behavior:

- `enc_anchor_dsga` = `ma_laga`-style gradient decomposition on encoder params
- adds a depth-aware gate to suppress reconstruction gradients more strongly in deeper encoder groups
- keeps the semantic branch dominant in deeper encoder layers

## Main Question

Does this asymmetric split improve the understanding/generation tradeoff over the current
encoder-shared baselines?

Primary comparison:

1. `ma_laga`
2. `enc_anchor_dsga`

Same seed, same data split, same optimizer, same batch size, same eval cadence.

## Immediate Run Plan

### Phase 1: quick GPU sanity compare

Purpose:

- check whether the encoder-anchored routing is directionally useful before spending a larger budget

Protocol:

- dataset: `ImageNet-100`
- encoder init: `dinov2`
- steps: `500`
- eval every: `250`
- eval batches: `8`
- seed: `42`

Baseline:

```bash
python -m unirae.train_imagenet100_methods \
  --method ma_laga \
  --encoder_init dinov2 \
  --device cuda \
  --hf_dataset_id clane9/imagenet-100 \
  --batch_size 64 \
  --num_workers 8 \
  --max_steps 500 \
  --eval_every 250 \
  --eval_max_batches 8 \
  --log_every 20 \
  --warmup_steps 50 \
  --lambda_u 1.0 \
  --lambda_g 1.0 \
  --recon_loss_type rmse \
  --ma_laga_mode full \
  --ma_laga_grouping layerwise \
  --ma_laga_align_gamma 1.0 \
  --output_root results/in100_encdec_gradsplit_quick \
  --run_name ma_laga_quick_s42
```

Proposed method:

```bash
python -m unirae.train_imagenet100_methods \
  --method enc_anchor_dsga \
  --encoder_init dinov2 \
  --device cuda \
  --hf_dataset_id clane9/imagenet-100 \
  --batch_size 64 \
  --num_workers 8 \
  --max_steps 500 \
  --eval_every 250 \
  --eval_max_batches 8 \
  --log_every 20 \
  --warmup_steps 50 \
  --lambda_u 1.0 \
  --lambda_g 1.0 \
  --recon_loss_type rmse \
  --ma_laga_mode full \
  --ma_laga_grouping layerwise \
  --ma_laga_align_gamma 1.0 \
  --enc_anchor_rec_gate_strength 0.75 \
  --enc_anchor_rec_gate_min 0.20 \
  --output_root results/in100_encdec_gradsplit_quick \
  --run_name enc_anchor_dsga_quick_s42
```

### Phase 2: 3-seed confirmation

Run only if Phase 1 is promising.

Seeds:

- `42`
- `43`
- `44`

Lock all hyperparameters from Phase 1 before expanding.

Suggested output root:

- `results/in100_encdec_gradsplit_multiseed`

### Phase 3: gate sweep

Run only if `enc_anchor_dsga` shows a non-trivial tradeoff gain.

Sweep:

- `--enc_anchor_rec_gate_strength` in `{0.25, 0.50, 0.75, 1.00}`
- `--enc_anchor_rec_gate_min` in `{0.10, 0.20, 0.35}`
- `--ma_laga_grouping` in `{layerwise, layerwise_coarse}`

## What To Read First

Look at these metrics first:

- `val_top1_acc`
- `val_rmse`
- `val_rfid`
- `train_grad_cosine`

Interpretation rule:

- if `enc_anchor_dsga` improves `val_top1_acc` without a large `val_rmse` collapse, the encoder anchor is helping semantics
- if it improves `val_rmse` while keeping `val_top1_acc` close, the split is helping tradeoff quality
- if both sides move little relative to `ma_laga`, the extra asymmetry is probably not worth expanding

## Decision Rule

- If the 500-step compare is not clearly better than `ma_laga`, stop.
- If it is better on at least one side without a severe collapse on the other, run 3 seeds.
- Only after 3-seed confirmation should we treat this as a real method branch rather than a prototype.

## Notes For Running

- This experiment is about encoder-only arbitration; do not add new decoder-side routing in this round.
- Keep `cls_head <- L_u` and `decoder <- L_g` unchanged so the ablation isolates encoder conflict handling.
- Prefer `srun` reuse workflow if running on HKUST SuperPOD.

