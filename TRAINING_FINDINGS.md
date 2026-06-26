# Training Findings

Date: 2026-06-26

These notes summarize the compile, gradient accumulation, and micro-batch sizing findings for the Pebble 500M run. The tokenized dataset contains about 50B train tokens; the current budgeted training target is 35B tokens.

## Current Run Context

- Remote host: `statement-llm`
- Remote repo: `~/pebble-500M`
- Local training data: `/opt/dlami/nvme/pebble-data-50b`
- Main long-run output target: `/opt/dlami/nvme/pebble-runs/pebble-500m-35b`
- Benchmark output: `/opt/dlami/nvme/pebble-runs/mbs-bench-compile-nocg-20260625-183311`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- GPU memory: about 97.9 GB

The restored dataset manifest was verified after the instance/NVMe restore:

- Train split: 50,000,000,208 tokens, 501 shards, no missing or wrong-sized shards
- Validation split: 50,000,592 tokens, 1 shard, no missing or wrong-sized shards

Budget update:

- Final planned training target: 35B tokens
- First budget phase stop: 24B tokens via `--max-tokens 24000000000`
- Phase 2 resumes from the synced S3 checkpoint and trains to the 35B config target

## Batch Setup

The current training config uses:

- `global_batch_tokens = 524,288`
- `context_length = 1,024`
- sequences per optimizer step: `524,288 / 1,024 = 512`

Gradient accumulation should preserve the global batch. Reducing `global_batch_tokens` just to fit a micro-batch changes the experiment because it changes the optimizer batch size, noise scale, and learning-rate schedule interpretation. The normal approach is to keep the global batch fixed and choose the largest stable micro-batch, then set:

```text
accum_steps = global_batch_tokens / (micro_batch_size * context_length)
```

For this config:

- `micro_batch_size=8` gives `accum_steps=64`
- `micro_batch_size=16` gives `accum_steps=32`
- `micro_batch_size=32` gives `accum_steps=16`
- `micro_batch_size=64` gives `accum_steps=8`

## Torch Compile Finding

`torch.compile(mode="max-autotune")` is not compatible with this gradient accumulation path on CUDA because the mode enables CUDA graphs. CUDA graphs reuse static memory buffers across replays. With accumulation, the compiled forward/backward is replayed multiple times before an optimizer step, so later micro-steps can overwrite graph outputs or gradient buffers from earlier micro-steps.

Symptoms seen:

- Loss outputs can be overwritten by a later graph replay.
- Cloning the detached loss fixes the metrics path only.
- The real blocker is backward/gradient accumulation: graph-managed gradient tensors can be replayed/reused across micro-steps.
- The failure mode included PyTorch CUDAGraph overwrite errors.

Setting `inductor_config.triton.cudagraphs = False` while still using `mode="max-autotune"` was not enough, because the mode preset can turn CUDA graphs back on.

The usable compile mode is:

```text
max-autotune-no-cudagraphs
```

That keeps Inductor fusion and autotuning, but removes CUDA graph replay buffers. The code default was changed to this mode in commit:

```text
3bd2b30 fix(train): default compile without cudagraphs
```

Avoid these modes with gradient accumulation on CUDA:

- `max-autotune`
- `reduce-overhead`

## No-Compile Baseline

Earlier no-compile benchmark results:

- `micro_batch_size=8`: about 54,955 train tok/s, about 45,717 wall tok/s, about 27.7 GB reserved
- `micro_batch_size=16`: about 51,199 train tok/s, about 42,417 wall tok/s
- `micro_batch_size=32`: about 47,392 tok/s at step 20, about 88.1 GB reserved

Before the compile fix, the best practical setting appeared to be `micro_batch_size=8 --no-compile`.

## Compile No-CUDAGraph Benchmark

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

Post-warmup train throughput:

- `micro_batch_size=8`: 79,049 tok/s, `accum_steps=64`, 20.9 GB peak reserved
- `micro_batch_size=16`: 82,729 tok/s, `accum_steps=32`, 32.4 GB peak reserved
- `micro_batch_size=32`: 85,218 tok/s, `accum_steps=16`, 54.1 GB peak reserved

`micro_batch_size=32` was the fastest tested working configuration:

- About 3% faster than `micro_batch_size=16`
- About 8% faster than `micro_batch_size=8`
- About 55% faster than the earlier best no-compile baseline
- About 43.8 GB memory headroom on the current GPU

The wall-throughput numbers in these short benchmarks include compile/autotune and final validation overhead, so use post-warmup train throughput for micro-batch comparisons.

## Maximum Micro-Batch Probe

`micro_batch_size=64` was tested separately:

- Probe output: `/opt/dlami/nvme/pebble-runs/mbs64-probe-20260625-185415`
- It reached Inductor compile/autotune.
- GPU memory reached about 80.6 GB during compile/autotune.
- It failed before writing training metrics.
- Failure:

```text
torch._inductor.exc.InductorError: AcceleratorError: CUDA error: an illegal memory access was encountered
```

Because `64` fails on the compiled path, `128+` is not worth trying under the current model/config/compiler setup.

## Recommendation

Use:

```sh
pebble-train \
  --config configs/pebble_500m_35b.yaml \
  --data-dir /opt/dlami/nvme/pebble-data-50b \
  --out-dir /opt/dlami/nvme/pebble-runs/pebble-500m-35b \
  --micro-batch-size 32 \
  --compile \
  --compile-mode max-autotune-no-cudagraphs \
  --s3-sync-uri s3://statement-llm-training/pebble-500m/runs/pebble-500m-35b
```

With the current committed code, `max-autotune-no-cudagraphs` is already the default compile mode. Keeping it explicit in remote run commands is still useful for auditability.

Treat `micro_batch_size=32` as the maximum safe and practical micro-batch size for the current setup. Retest if any of these change:

- model size
- context length
- global batch tokens
- compile mode
- PyTorch/Triton version
- GPU type or available memory
- activation checkpointing or memory-affecting model changes
