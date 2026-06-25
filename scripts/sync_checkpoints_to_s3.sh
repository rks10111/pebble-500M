#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <run-dir> <s3-uri>" >&2
  exit 2
fi

run_dir="$1"
s3_uri="$2"

aws s3 sync "${run_dir}/checkpoints" "${s3_uri}/checkpoints" --only-show-errors
aws s3 cp "${run_dir}/metrics.jsonl" "${s3_uri}/metrics.jsonl" --only-show-errors || true
aws s3 cp "${run_dir}/samples.jsonl" "${s3_uri}/samples.jsonl" --only-show-errors || true
