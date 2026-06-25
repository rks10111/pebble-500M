#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [s3-prefix] [nvme-path]" >&2
  exit 2
fi

s3_prefix="${1:-s3://statement-llm-training/pebble-500m}"
nvme_path="${2:-/opt/dlami/nvme}"
region="${AWS_REGION:-eu-west-2}"
s3_prefix="${s3_prefix%/}"

stamp="$(date +%Y%m%d-%H%M%S)"
local_file="/tmp/codex-s3-test-${stamp}.txt"
restored_file="/tmp/codex-s3-test-restored-${stamp}.txt"
test_uri="${s3_prefix}/test/codex-s3-test-${stamp}.txt"

printf "codex s3 smoke test %s\n" "${stamp}" > "${local_file}"

echo "== identity =="
aws sts get-caller-identity --region "${region}"

echo "== disk =="
df -h / "${nvme_path}"

echo "== prefix list =="
aws s3 ls "${s3_prefix}/" --region "${region}"

echo "== upload =="
aws s3 cp "${local_file}" "${test_uri}" --region "${region}" --only-show-errors

echo "== download =="
aws s3 cp "${test_uri}" "${restored_file}" --region "${region}" --only-show-errors

echo "== compare =="
cmp "${local_file}" "${restored_file}"

echo "S3 smoke test passed: ${test_uri}"
