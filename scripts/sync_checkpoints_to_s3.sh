#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <run-dir> <s3-uri>" >&2
  exit 2
fi

run_dir="$1"
s3_uri="${2%/}"
region="${AWS_REGION:-eu-west-2}"

if [[ ! -d "${run_dir}/checkpoints" ]]; then
  echo "error: run dir does not contain checkpoints/: ${run_dir}" >&2
  exit 1
fi

aws s3 sync "${run_dir}/checkpoints" "${s3_uri}/checkpoints" --region "${region}" --only-show-errors --exclude "*.tmp"
aws s3 cp "${run_dir}/metrics.jsonl" "${s3_uri}/metrics.jsonl" --region "${region}" --only-show-errors || true
aws s3 cp "${run_dir}/samples.jsonl" "${s3_uri}/samples.jsonl" --region "${region}" --only-show-errors || true

echo "synced run artifacts from ${run_dir} to ${s3_uri}"
