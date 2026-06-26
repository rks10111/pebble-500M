# Pebble 500M

Pebble 500M is a from-scratch pretraining project for a roughly `500M` parameter decoder-only transformer.

The codebase focuses on reproducible data preparation, throughput benchmarking, training, checkpointing, and resuming on a single GPU instance.

## Target Hardware

- AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04)
- Instance: `g7e.2xlarge`
- GPU: `1x NVIDIA RTX PRO Server 6000`, 96GB VRAM
- Root EBS: about 300GB
- Local NVMe: about 1700GB mounted at `/opt/dlami/nvme`
- S3 backup prefix: `s3://statement-llm-training/pebble-500m/` in `eu-west-2`

Use the root EBS volume for the OS, repo, Python environment, scripts, and small logs. Use
`/opt/dlami/nvme` for the active tokenized dataset and training output. Use S3 as the durable
copy for tokenized data, checkpoints, metrics, samples, and final weights.

The training scripts read from local files. Do not train directly from S3.

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

## Storage Smoke Test

Verify the IAM role, root EBS, NVMe mount, and S3 read/write access:

```bash
AWS_REGION=eu-west-2 scripts/s3_smoke_test.sh
```

Expected layout:

- `/`: about `300GB` root EBS, used for system files and repo state
- `/opt/dlami/nvme`: about `1.7TB`, used for active data and training runs
- `s3://statement-llm-training/pebble-500m/`: durable backup prefix

## W&B Tracking

`pebble-train` logs scalar metrics to W&B by default. Before a long run, either log in:

```bash
wandb login
```

or disable W&B explicitly:

```bash
--no-wandb
```

For disconnected runs, use:

```bash
--wandb-mode offline
```

Parameter and gradient histogram logging is off by default because it can add overhead on long
runs. Enable it only for debugging:

```bash
--wandb-watch
```

The 50B tokenization wrapper also logs W&B progress by default. It records rows/documents seen,
train/validation tokens written, total-token progress, shard counts, elapsed time, and tokenization
throughput. Local W&B files are written under `/opt/dlami/nvme/logs/wandb`, not inside the
tokenized dataset directory.

Disable tokenization W&B, or force offline mode, with:

```bash
PEBBLE_WANDB=0 scripts/start_data_prep_tmux.sh
WANDB_MODE=offline scripts/start_data_prep_tmux.sh
```

## Prepare Data

The data pipeline uses a deterministic document-level stream from `HuggingFaceFW/fineweb-edu`, reserves fixed validation documents first, then writes GPT-2-tokenized `uint16` mmap shards. Training never tokenizes raw text live.

For a 100M-token benchmark dataset:

```bash
pebble-prepare-data \
  --config configs/pebble_500m.yaml \
  --out-dir /opt/dlami/nvme/pebble-data-100m \
  --train-tokens 120000000 \
  --val-tokens 50000000 \
  --shard-tokens 50000000
```

For the 50B-token dataset used by the two-budget-window run:

```bash
pebble-prepare-data \
  --config configs/pebble_500m_35b.yaml \
  --out-dir /opt/dlami/nvme/pebble-data-50b \
  --train-tokens 50000000000 \
  --val-tokens 50000000 \
  --shard-tokens 100000000
```

Or start the full tokenization-and-S3-backup workflow in tmux:

```bash
cd ~/pebble-500M
scripts/start_data_prep_tmux.sh
```

This runs `scripts/prepare_50b_data_and_sync.sh`, which verifies disk/S3 access, tokenizes the
50B dataset, writes logs under `/opt/dlami/nvme/logs`, logs W&B data-prep progress by default,
and syncs the finished dataset to S3. It skips tokenization if `manifest.json` already exists.
It refuses to run on a non-empty partial data directory because tokenization is not resumable.

After a successful run, the script asks AWS to stop the EC2 instance to avoid idle GPU spend.
It discovers the instance id from `PEBBLE_INSTANCE_ID` or the local EC2 DMI asset tag; no public
IP address is needed in the scripts.
Disable that behavior with:

```bash
PEBBLE_STOP_INSTANCE_ON_DONE=0 scripts/start_data_prep_tmux.sh
```

The output includes:

- `manifest.json`: shard paths, token counts, seed, tokenizer, and split metadata
- `documents.jsonl.gz`: one deterministic record per included source document
- `train/*.bin` and `val/*.bin`: contiguous `uint16` token shards

Back up the prepared 50B tokenized dataset to S3 after preparation:

```bash
AWS_REGION=eu-west-2 scripts/sync_data_to_s3.sh \
  /opt/dlami/nvme/pebble-data-50b \
  s3://statement-llm-training/pebble-500m/data/fineweb-edu-gpt2-50b
```

Restore it onto a fresh instance before resuming:

```bash
AWS_REGION=eu-west-2 scripts/restore_nvme_from_s3.sh
```

This restores:

- `s3://statement-llm-training/pebble-500m/data/fineweb-edu-gpt2-50b` to `/opt/dlami/nvme/pebble-data-50b`
- `s3://statement-llm-training/pebble-500m/runs/pebble-500m-35b` to `/opt/dlami/nvme/pebble-runs/pebble-500m-35b`

The script refuses to run if `/opt/dlami/nvme` is not mounted, so a large restore does not
accidentally fill the 300GB root EBS volume.

## Benchmark Before Full Training

For a quick synthetic smoke benchmark that does not download FineWeb-Edu:

```bash
pebble-make-synthetic-data \
  --out-dir /opt/dlami/nvme/pebble-smoke-data \
  --train-tokens 20000000 \
  --val-tokens 1000000 \
  --shard-tokens 1000000 \
  --vocab-size 50304

pebble-train \
  --config configs/pebble_500m.yaml \
  --data-dir /opt/dlami/nvme/pebble-smoke-data \
  --out-dir /opt/dlami/nvme/pebble-runs/smoke-mbs16 \
  --max-tokens 200000000 \
  --micro-batch-size 16
```

Use the final `average_train_tok/s` value for comparing micro-batch sizes. This metric measures synchronized training-loop throughput after the configured warmup steps and excludes validation/checkpoint time. `average_wall_tok/s` includes the full run overhead. By default, `pebble-train` excludes the first log interval from train-throughput averages; override with `--warmup-steps`.

Run a 100M-token benchmark before the long run:

```bash
pebble-train \
  --config configs/pebble_500m.yaml \
  --data-dir /opt/dlami/nvme/pebble-data-100m \
  --out-dir /opt/dlami/nvme/pebble-runs/bench-100m \
  --max-tokens 100000000 \
  --micro-batch-size 16
```

Decision rule:

- `<45k tok/s`: optimize before the long run
- `45k-55k tok/s`: target `10B-15B`
- `55k-70k tok/s`: target `15B-20B`
- `80k+ tok/s`: `20B` is comfortable
- `100k+ tok/s`: `30B` becomes plausible
- `85k+ tok/s` sustained across runs: target `24B` in the first `$500` window and `35B` total

## Long Run

The `configs/pebble_500m_35b.yaml` config uses the restored 50B tokenized dataset, but schedules
training for a 35B-token budget. With two `$500` budget windows, run to about `24B` in the first
window, sync to S3, then resume to `35B` in the next window.

First budget window:

```bash
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

Use `micro_batch_size=32`; it is the fastest stable tested value for this run.

## Checkpoints

The trainer writes:

- rolling operational checkpoints in `checkpoints/latest-*`
- milestone checkpoints every `1B` tokens through the 35B target
- `metrics.jsonl`
- fixed prompt samples at milestones

Sync run artifacts to S3 periodically and before stopping or terminating the instance:

```bash
AWS_REGION=eu-west-2 scripts/sync_checkpoints_to_s3.sh \
  /opt/dlami/nvme/pebble-runs/pebble-500m-35b \
  s3://statement-llm-training/pebble-500m/runs/pebble-500m-35b
```

This script uploads checkpoints, `metrics.jsonl`, and `samples.jsonl`. Training remains local and
does not block on S3 uploads.

For long runs, prefer automatic sync from the trainer:

```bash
pebble-train \
  --config configs/pebble_500m_35b.yaml \
  --data-dir /opt/dlami/nvme/pebble-data-50b \
  --out-dir /opt/dlami/nvme/pebble-runs/pebble-500m-35b \
  --resume /opt/dlami/nvme/pebble-runs/pebble-500m-35b/checkpoints/latest-000000000123.pt \
  --max-tokens 35000000000 \
  --s3-sync-uri s3://statement-llm-training/pebble-500m/runs/pebble-500m-35b
```

For the first budget phase, use `--max-tokens 24000000000` to stop at about 24B tokens while keeping
the config's 35B learning-rate schedule for the later resume.

When `--s3-sync-uri` is set, `pebble-train` syncs checkpoints, `metrics.jsonl`, and `samples.jsonl`
after every milestone checkpoint, rolling latest checkpoint, and final checkpoint. You can also set
`PEBBLE_S3_RUN_URI` in the environment instead of passing the flag. Use `--no-s3-sync` to disable
automatic sync for a run. Checkpoints are written atomically via a temporary file and rename before
sync, so S3 should not receive a partially written `.pt` file.

## Resume

Restore the NVMe working state first if this is a fresh instance or the local NVMe was reset:

```bash
AWS_REGION=eu-west-2 scripts/restore_nvme_from_s3.sh
```

The restore script syncs the tokenized dataset and run artifacts, verifies the dataset manifest
and shard sizes, then prints the latest available resume command if a `latest-*.pt` checkpoint
exists locally after restore.

Resume from the latest local checkpoint:

```bash
pebble-train \
  --config configs/pebble_500m_35b.yaml \
  --data-dir /opt/dlami/nvme/pebble-data-50b \
  --out-dir /opt/dlami/nvme/pebble-runs/pebble-500m-35b \
  --resume /opt/dlami/nvme/pebble-runs/pebble-500m-35b/checkpoints/latest-000000000123.pt \
  --max-tokens 35000000000 \
  --s3-sync-uri s3://statement-llm-training/pebble-500m/runs/pebble-500m-35b
```

The checkpoint stores model, optimizer, scheduler counters, `tokens_seen`, RNG state, config, and dataloader state.

## Local Smoke Tests

```bash
pytest
```

The tests use tiny CPU configs and synthetic token shards. They do not download FineWeb-Edu.
