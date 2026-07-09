#!/usr/bin/env bash
set -euo pipefail

data_dir="${DATA_DIR:-/opt/dlami/nvme/pebble-sft-data/pebble-identity-sft-continuation}"
checkpoint="${CHECKPOINT:-/opt/dlami/nvme/pebble-sft-runs/pebble-500m-chat-sft-cleaned/checkpoints/latest-sft-00003500.pt}"
out_dir="${OUT_DIR:-/opt/dlami/nvme/pebble-sft-runs/pebble-500m-chat-sft-identity-from-3500}"
wandb_run_name="${WANDB_RUN_NAME:-pebble-500m-chat-sft-identity-from-3500}"

epochs="${EPOCHS:-1}"
micro_batch_size="${MICRO_BATCH_SIZE:-8}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-8}"
lr="${LR:-1e-5}"
min_lr="${MIN_LR:-1e-6}"
warmup_steps="${WARMUP_STEPS:-5}"
eval_interval_steps="${EVAL_INTERVAL_STEPS:-25}"
save_interval_steps="${SAVE_INTERVAL_STEPS:-25}"
eval_batches="${EVAL_BATCHES:-20}"
precision="${PRECISION:-bf16}"
max_steps="${MAX_STEPS:-}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<EOF
usage: $0

Launch a low-LR SFT-on-SFT identity continuation run from checkpoint 3500.
Run this on statement-llm from /home/ubuntu/pebble-500M.

Environment overrides:
  DATA_DIR              Prepared masked SFT dataset.
                        Default: ${data_dir}
  CHECKPOINT            Starting SFT checkpoint.
                        Default: ${checkpoint}
  OUT_DIR               Output run directory.
                        Default: ${out_dir}
  WANDB_RUN_NAME        W&B run name.
                        Default: ${wandb_run_name}
  EPOCHS                Default: ${epochs}
  MAX_STEPS             Optional hard optimizer-step cap.
  LR                    Default: ${lr}
  MIN_LR                Default: ${min_lr}
  WARMUP_STEPS          Default: ${warmup_steps}
  EVAL_INTERVAL_STEPS   Default: ${eval_interval_steps}
  SAVE_INTERVAL_STEPS   Default: ${save_interval_steps}
EOF
  exit 0
fi

if [[ ! -f "${data_dir}/manifest.json" ]]; then
  echo "error: data dir is missing manifest.json: ${data_dir}" >&2
  exit 1
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "error: checkpoint not found: ${checkpoint}" >&2
  exit 1
fi

cmd=(
  /opt/pytorch/bin/python train_sft.py
  --config configs/pebble_500m_30b.yaml
  --data-dir "${data_dir}"
  --checkpoint "${checkpoint}"
  --out-dir "${out_dir}"
  --epochs "${epochs}"
  --micro-batch-size "${micro_batch_size}"
  --grad-accum-steps "${grad_accum_steps}"
  --lr "${lr}"
  --min-lr "${min_lr}"
  --warmup-steps "${warmup_steps}"
  --precision "${precision}"
  --eval-interval-steps "${eval_interval_steps}"
  --eval-batches "${eval_batches}"
  --save-interval-steps "${save_interval_steps}"
  --no-ground-chat-tokens
  --wandb
  --wandb-run-name "${wandb_run_name}"
)

if [[ -n "${max_steps}" ]]; then
  cmd+=(--max-steps "${max_steps}")
fi

exec "${cmd[@]}"
