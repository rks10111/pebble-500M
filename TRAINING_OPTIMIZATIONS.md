# Training Optimizations

Date: 2026-06-26

This file tracks the optimization work done so far for the Pebble 500M training run. It is meant to be updated whenever we change the training path, benchmark a new setting, or reject an optimization after testing.

## Current Recommended Runtime Settings

The current budgeted target is 35B training tokens. The tokenized dataset remains the restored 50B dataset; the extra tokens are just available data.

For phase 1 this month, stop at 24B tokens while keeping the LR schedule planned for 35B:

```sh
pebble-train \
  --config configs/pebble_500m_35b.yaml \
  --data-dir /opt/dlami/nvme/pebble-data-50b \
  --out-dir /opt/dlami/nvme/pebble-runs/pebble-500m-35b \
  --max-tokens 24000000000 \
  --micro-batch-size 32 \
  --compile \
  --compile-mode max-autotune-no-cudagraphs \
  --s3-sync-uri s3://statement-llm-training/pebble-500m/runs/pebble-500m-35b
```

For phase 2 next month, restore the run artifacts from S3 and resume to the config default of 35B tokens.

Why:

- Keeps `global_batch_tokens=524288` unchanged.
- Uses the fastest tested valid micro-batch size.
- Keeps gradient accumulation compatible with `torch.compile`.
- Leaves about 43.8 GB memory headroom on the current 97.9 GB GPU.
- Keeps the LR schedule aligned to the real final 35B-token target instead of an abandoned 50B target.

## Implemented Code Optimizations

### Default compile mode without CUDA graphs

Commit:

```text
3bd2b30 fix(train): default compile without cudagraphs
```

Change:

- Changed the default compile mode to `max-autotune-no-cudagraphs`.
- Added logic to remap CUDA-graph compile modes to the no-CUDAGraph mode when CUDA gradient accumulation is active.
- Kept Inductor fusion and max-autotune behavior while avoiding CUDA graph replay buffers.

Reason:

- `max-autotune` enables CUDA graphs.
- CUDA graphs reuse static buffers across graph replays.
- Gradient accumulation replays the compiled forward/backward multiple times before one optimizer step.
- That caused overwritten graph outputs/gradients and made CUDA graph modes incompatible with this accumulation path.

Result:

- Compile works with gradient accumulation.
- The best benchmarked compiled setting reached 85,218 post-warmup train tok/s at `micro_batch_size=32`.
- This is about 55% faster than the earlier best no-compile baseline.

### Avoid returning unused logits during train and validation

Commit:

```text
47275e2 refactor(train): avoid returning unused logits
```

Change:

- Added `Transformer.forward(..., return_logits=False)`.
- Training now calls `forward_model(x, y, return_logits=False)`.
- Validation now calls `model(x, y, return_logits=False)`.
- Generation and default inference keep returning logits.
- Added a test that verifies the loss-only path returns no logits and preserves the same loss value.

Reason:

- Training and validation were discarding returned logits with `_, loss = ...`.
- Returning logits from the compiled forward makes logits a graph output.
- At `micro_batch_size=32`, `context_length=1024`, `vocab_size=50304`, the returned bf16 logits output is about 3.3 GB.

Expected effect:

- Mostly a memory-headroom improvement.
- It does not remove the full vocab projection or the cross-entropy backward state.
- Tok/s may improve modestly from lower output traffic or allocator pressure, but this needs remote benchmarking.

Validation:

- Local test suite passed: `uv run pytest`, 19 tests.
- Remote benchmark completed with `micro_batch_size=32`:
  - Output: `/opt/dlami/nvme/pebble-runs/mbs32-returnlogits-false-20260626-015642`
  - Result: 86,057 post-warmup train tok/s
  - Peak allocated memory: 46.1 GB
  - Peak reserved memory: 47.8 GB

Measured difference versus the previous compiled `micro_batch_size=32` run:

- Previous: 85,218 tok/s, 52.7 GB peak allocated, 54.1 GB peak reserved
- New: 86,057 tok/s, 46.1 GB peak allocated, 47.8 GB peak reserved
- Throughput delta: +839 tok/s, about +1.0%
- Peak allocated memory delta: -6.6 GB, about -12.5%
- Peak reserved memory delta: -6.3 GB, about -11.7%

## Benchmarked Runtime Optimizations

### Micro-batch size search with compiled no-CUDAGraph mode

Benchmark output:

```text
/opt/dlami/nvme/pebble-runs/mbs-bench-compile-nocg-20260625-183311
```

Benchmark command shape:

```sh
pebble-train \
  --config configs/pebble_500m_35b.yaml \
  --data-dir /opt/dlami/nvme/pebble-data-50b \
  --out-dir "$out" \
  --max-tokens 15728640 \
  --micro-batch-size "$mbs" \
  --warmup-steps 10 \
  --compile \
  --compile-mode max-autotune-no-cudagraphs \
  --no-wandb \
  --no-s3-sync
```

Results:

- `micro_batch_size=8`: 79,049 tok/s, `accum_steps=64`, 20.9 GB peak reserved
- `micro_batch_size=16`: 82,729 tok/s, `accum_steps=32`, 32.4 GB peak reserved
- `micro_batch_size=32`: 85,218 tok/s, `accum_steps=16`, 54.1 GB peak reserved

Conclusion:

- `micro_batch_size=32` is the fastest tested valid setting.
- It is about 3% faster than `16`.
- It is about 8% faster than `8`.
- It is memory-safe on the current GPU.

### Maximum micro-batch probe

Probe output:

```text
/opt/dlami/nvme/pebble-runs/mbs64-probe-20260625-185415
```

Result:

- `micro_batch_size=64` reached Inductor compile/autotune.
- GPU memory reached about 80.6 GB during compile/autotune.
- It failed before writing training metrics.

Failure:

```text
torch._inductor.exc.InductorError: AcceleratorError: CUDA error: an illegal memory access was encountered
```

Second probe after `return_logits=False`:

```text
/opt/dlami/nvme/pebble-runs/mbs64-returnlogits-false-20260626-020917
```

Result:

- `micro_batch_size=64` still reached Inductor compile/autotune.
- It still failed before writing training metrics.
- Failure was the same class: Inductor backward compilation hit `CUDA error: an illegal memory access was encountered`.
- Exit code: `RC=1`.

Conclusion:

- Treat `micro_batch_size=64` as not usable with the current compiled path.
- The `return_logits=False` memory reduction did not make `64` viable.
- Do not try `128+` under the same model/config/compiler setup unless the compiler path changes.

## Baselines and Superseded Attempts

### No-compile baseline

Earlier no-compile results:

- `micro_batch_size=8`: about 54,955 train tok/s, about 45,717 wall tok/s, about 27.7 GB reserved
- `micro_batch_size=16`: about 51,199 train tok/s, about 42,417 wall tok/s
- `micro_batch_size=32`: about 47,392 tok/s at step 20, about 88.1 GB reserved

Conclusion:

- Before the compile-mode fix, the best practical setting was `micro_batch_size=8 --no-compile`.
- After the compile-mode fix, `micro_batch_size=32 --compile` is clearly better.

### Superseded CUDA graph fixes

Commits in this line of investigation:

```text
38eeba4 fix(train): clone compiled loss for accumulation metrics
db7e956 fix(train): mark compiled accumulation micro steps
ce3da53 fix(train): disable compile cudagraphs during accumulation
3526f8f fix(train): configure inductor cudagraphs for accumulation
5b21b56 fix(train): fall back from compile for cuda accumulation
3bd2b30 fix(train): default compile without cudagraphs
```

What we learned:

- Cloning detached loss fixed only the metrics path.
- Marking CUDA graph step boundaries was not enough for accumulated backward.
- Setting `inductor_config.triton.cudagraphs = False` while still using `mode="max-autotune"` was contradictory, because the mode preset can re-enable CUDA graphs.
- Fully disabling compile was correct as a temporary fallback but left performance on the table.
- The durable fix is to use `max-autotune-no-cudagraphs`.

## Constraints We Are Preserving

These are intentionally unchanged unless we decide to run a different experiment:

- `global_batch_tokens=524288`
- `context_length=1024`
- optimizer settings
- LR schedule
- model architecture
- tokenizer/data
- target 35B-token run setup using the restored 50B tokenized dataset

Notes:

- `micro_batch_size=48` is not valid with the current global batch because `524288 / 1024 = 512` sequences per step and `512 / 48` is not an integer.
- Changing global batch to make `48` work would change the experiment.

## Candidate Future Optimizations

These are not implemented yet:

- Benchmark the `return_logits=False` commit remotely at `micro_batch_size=32`.
- Test whether the reduced logits output headroom allows a compiler-stable larger valid batch in a future PyTorch/Triton version.
- Add fused or chunked linear cross entropy to avoid materializing the full `[B*T, V]` tensor in the loss path. This is the real larger logits-memory optimization.
- Benchmark pinned-memory or prefetched data loading if GPU utilization shows input stalls.
- Tune operational overhead separately: W&B mode, validation cadence, checkpoint cadence, and logging interval. These mostly affect wall-clock throughput, not core post-warmup train tok/s.
