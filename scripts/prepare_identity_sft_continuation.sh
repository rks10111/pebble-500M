#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_dir="${RAW_IDENTITY_SFT_DIR:-/Users/rakeshbs/Projects/Statement/pebble identity sft}"
out_dir="${OUT_DIR:-${repo_root}/data/pebble-identity-sft-continuation}"
shard_tokens="${SHARD_TOKENS:-1000000}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<EOF
usage: $0

Prepare the Pebble identity-control dataset for assistant-only SFT continuation.

Environment overrides:
  RAW_IDENTITY_SFT_DIR  Raw messages JSONL directory.
                         Default: ${raw_dir}
  OUT_DIR               Prepared masked SFT output directory.
                         Default: ${out_dir}
  SHARD_TOKENS          Tokenized shard size.
                         Default: ${shard_tokens}
EOF
  exit 0
fi

train_jsonl="${raw_dir}/train.jsonl.gz"
val_jsonl="${raw_dir}/val.jsonl.gz"

if [[ ! -f "${train_jsonl}" ]]; then
  echo "error: missing train jsonl: ${train_jsonl}" >&2
  exit 1
fi
if [[ ! -f "${val_jsonl}" ]]; then
  echo "error: missing validation jsonl: ${val_jsonl}" >&2
  exit 1
fi
if [[ -e "${out_dir}" ]] && [[ -n "$(find "${out_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "error: OUT_DIR must be empty or absent: ${out_dir}" >&2
  exit 1
fi

cd "${repo_root}"
uv run pebble-prepare-sft-data \
  --no-hf-dataset \
  --extra-train-jsonl "${train_jsonl}" \
  --extra-val-jsonl "${val_jsonl}" \
  --out-dir "${out_dir}" \
  --shard-tokens "${shard_tokens}" \
  --log-interval-rows 1000

echo "prepared identity SFT continuation dataset at ${out_dir}"
