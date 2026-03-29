# Appendix Code

This folder is a clean, self-contained reference implementation for the core gradient strategies used in the paper:

- `joint`: vanilla loss summation
- `pcgrad`: symmetric conflict projection
- `cagrad`: the same CAGrad-like average/MGDA interpolation used in this repository
- `dsga`: DSGA-D (direction decomposition) + DSGA-M (magnitude alignment)

## Files

- `gradient_strategies.py`: core gradient merge logic
- `toy_multitask_model.py`: a tiny shared-encoder model with classification and reconstruction heads
- `demo_train.py`: runnable synthetic demo with no dataset dependency

## Design choices

- The code is intentionally independent from the main training framework.
- `DSGA-D` is applied group-wise. In the demo, groups are layer-wise (`input` + one group per block).
- `DSGA-M` supports `global` and `layerwise` magnitude alignment.
- The default setting in `demo_train.py` matches the current main experiments: layer-wise direction routing with global magnitude alignment.

## Run

```bash
python -m appendix_code.demo_train --strategy dsga --steps 10 --device cpu
python -m appendix_code.demo_train --strategy cagrad --steps 10 --device cpu
```

## Plug into your own model

1. Split parameters into `shared_params` and `aux_params`.
2. Build `dsga_groups` from the shared encoder blocks.
3. Compute the two task losses.
4. Call `apply_two_task_strategy(...)` before `optimizer.step()`.

Minimal usage:

```python
stats = apply_two_task_strategy(
    loss_understanding=loss_u,
    loss_generation=loss_g,
    shared_params=shared_params,
    aux_params=aux_params,
    strategy="dsga",
    dsga_groups=layer_groups,
    dsga_lambda_mag=0.2,
    dsga_magnitude_scope="global",
)
```

## Notes for the appendix

- This is the clean reference code meant for paper release.
- It is not a drop-in replacement for the full experiment pipeline in `unirae/`.
- The goal here is clarity and traceability of the method, not maximal training throughput.
