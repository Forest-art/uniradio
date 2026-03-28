# Encoder-Anchor DSGA TODO

## Scope

This note documents the new `enc_anchor_dsga` prototype added to
`unirae/train_imagenet100_methods.py` and the next experiments needed to
validate it on a real GPU cluster.

## What Changed

- New method: `enc_anchor_dsga`
- Entry point: `python -m unirae.train_imagenet100_methods`
- Rationale:
  - Keep the encoder conflict solver asymmetric in favor of semantic gradients.
  - Suppress reconstruction gradients more strongly in deeper encoder groups.
  - Leave `cls_head` fully driven by understanding loss and `decoder` fully driven by reconstruction loss.

## New Flags

- `--method enc_anchor_dsga`
- `--ma_laga_grouping {layerwise,layerwise_coarse,global}`
- `--enc_anchor_rec_gate_strength`
- `--enc_anchor_rec_gate_min`
- `--ma_laga_conflict_tau`
- `--ma_laga_conflict_tau_end`

## Smoke Validation Already Done

- CPU smoke run completed successfully:
  - `results/in100_enc_anchor_smoke/enc_anchor_dsga_cpu_smoke_20260328_v2`
- Baseline smoke run completed successfully:
  - `results/in100_enc_anchor_smoke/ma_laga_cpu_smoke_20260328`
- Interpretation:
  - The new branch trains, evaluates, writes checkpoints, and respects the same auto-loss-alignment path as `ma_laga`.
  - The smoke run is not evidence of quality improvement. It only validates the code path.

## Priority Experiments

1. Quick GPU compare on ImageNet-100:
   - `ma_laga`
   - `enc_anchor_dsga`
   - Same seed, same steps, same batch size, same eval cadence
2. If the quick compare is promising, run 3 seeds.
3. If `enc_anchor_dsga` wins mainly on understanding or mainly on reconstruction, sweep:
   - `--enc_anchor_rec_gate_strength` in `{0.25, 0.50, 0.75, 1.00}`
   - `--enc_anchor_rec_gate_min` in `{0.10, 0.20, 0.35}`
   - `--ma_laga_grouping` in `{layerwise, layerwise_coarse}`

## Recommended Quick Commands

Use the same output root for both methods.

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
  --output_root results/in100_enc_anchor_quick \
  --run_name ma_laga_quick_s42
```

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
  --output_root results/in100_enc_anchor_quick \
  --run_name enc_anchor_dsga_quick_s42
```

## Comparison Targets

Compare these fields first:

- `val_top1_acc`
- `val_rmse`
- `val_rfid`
- `train_grad_cosine`

Primary question:

- Does encoder-anchored gating improve the understanding/generation tradeoff over plain `ma_laga` under the same protocol?

Secondary question:

- Does the gain mainly come from protecting understanding, or from improving overall Pareto tradeoff?

## Decision Rule

- If `enc_anchor_dsga` is not better than `ma_laga` in the 500-step quick compare, do not expand to a big sweep.
- If it is clearly better on at least one side without a major collapse on the other side, continue to 3-seed confirmation.
