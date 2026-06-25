# Pebble 500M

Pebble 500M is a from-scratch pretraining project for a roughly `500M` parameter decoder-only transformer.

The codebase focuses on reproducible data preparation, throughput benchmarking, training, checkpointing, and resuming on a single GPU instance.

## Target Hardware

- AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04)
- Instance: `g7e.2xlarge`
- GPU: `1x NVIDIA RTX PRO Server 6000`, 96GB VRAM
- Local SSD: about 1900GB

## First Boot Check

```bash
nvidia-smi
source /opt/pytorch/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("BF16 matmul OK:", y.shape)
PY
```

## Install

From the instance:

```bash
git clone <this-repo-url> pebble-500M
cd pebble-500M
python -m pip install -e ".[dev]"
```

## Prepare Data

The data pipeline uses a deterministic document-level stream from `HuggingFaceFW/fineweb-edu`, reserves fixed validation documents first, then writes GPT-2-tokenized `uint16` mmap shards. Training never tokenizes raw text live.

For a 100M-token benchmark dataset:

```bash
pebble-prepare-data \
  --config configs/pebble_500m.yaml \
  --out-dir /local_nvme/pebble-data \
  --train-tokens 120000000 \
  --val-tokens 50000000 \
  --shard-tokens 50000000
```

For the long run, prepare only the target you plan to train:

```bash
pebble-prepare-data \
  --config configs/pebble_500m.yaml \
  --out-dir /local_nvme/pebble-data \
  --train-tokens 15000000000 \
  --val-tokens 50000000 \
  --shard-tokens 100000000
```

The output includes:

- `manifest.json`: shard paths, token counts, seed, tokenizer, and split metadata
- `documents.jsonl.gz`: one deterministic record per included source document
- `train/*.bin` and `val/*.bin`: contiguous `uint16` token shards

## Benchmark Before Full Training

For a quick synthetic smoke benchmark that does not download FineWeb-Edu:

```bash
pebble-make-synthetic-data \
  --out-dir /local_nvme/pebble-smoke-data \
  --train-tokens 20000000 \
  --val-tokens 1000000 \
  --shard-tokens 1000000 \
  --vocab-size 50304

pebble-train \
  --config configs/pebble_500m.yaml \
  --data-dir /local_nvme/pebble-smoke-data \
  --out-dir /local_nvme/pebble-runs/smoke-mbs16 \
  --max-tokens 200000000 \
  --micro-batch-size 16
```

Use the final `average_train_tok/s` value for comparing micro-batch sizes. This metric measures synchronized training-loop throughput after the configured warmup steps and excludes validation/checkpoint time. `average_wall_tok/s` includes the full run overhead. By default, `pebble-train` excludes the first log interval from train-throughput averages; override with `--warmup-steps`.

Run a 100M-token benchmark before the long run:

```bash
pebble-train \
  --config configs/pebble_500m.yaml \
  --data-dir /local_nvme/pebble-data \
  --out-dir /local_nvme/pebble-runs/bench-100m \
  --max-tokens 100000000 \
  --micro-batch-size 16
```

Decision rule:

- `<45k tok/s`: optimize before the long run
- `45k-55k tok/s`: target `10B-15B`
- `55k-70k tok/s`: target `15B-20B`
- `80k+ tok/s`: `20B` is comfortable
- `100k+ tok/s`: `30B` becomes plausible

## Long Run

The default config plans the LR schedule for `20B` tokens so training can continue past `10B` or `15B` without the LR decaying too early.

```bash
pebble-train \
  --config configs/pebble_500m.yaml \
  --data-dir /local_nvme/pebble-data \
  --out-dir /local_nvme/pebble-runs/pebble-500m-20b \
  --max-tokens 15000000000 \
  --micro-batch-size 16
```

Try micro-batch sizes `8, 16, 24, 32, 40, 48, 64` and use the largest stable value that improves throughput.

## Checkpoints

The trainer writes:

- rolling operational checkpoints in `checkpoints/latest-*`
- milestone checkpoints at `2B`, `5B`, `10B`, `15B`, and `20B`
- `metrics.jsonl`
- fixed prompt samples at milestones

Local SSD is temporary. Before stopping or terminating the instance:

```bash
scripts/sync_checkpoints_to_s3.sh /local_nvme/pebble-runs/pebble-500m-20b s3://your-bucket/pebble-500m/pebble-500m-20b
```

## Resume

```bash
pebble-train \
  --config configs/pebble_500m.yaml \
  --data-dir /local_nvme/pebble-data \
  --out-dir /local_nvme/pebble-runs/pebble-500m-20b \
  --resume /local_nvme/pebble-runs/pebble-500m-20b/checkpoints/latest-000000000123.pt \
  --max-tokens 20000000000
```

The checkpoint stores model, optimizer, scheduler counters, `tokens_seen`, RNG state, config, and dataloader state.

## Local Smoke Tests

```bash
pytest
```

The tests use tiny CPU configs and synthetic token shards. They do not download FineWeb-Edu.
