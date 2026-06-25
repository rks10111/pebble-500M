#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <data-dir> <s3-uri>" >&2
  exit 2
fi

data_dir="$1"
s3_uri="${2%/}"
region="${AWS_REGION:-eu-west-2}"

if [[ ! -f "${data_dir}/manifest.json" ]]; then
  echo "error: data dir does not contain manifest.json: ${data_dir}" >&2
  exit 1
fi

aws s3 sync "${data_dir}" "${s3_uri}" --region "${region}" --only-show-errors
echo "synced tokenized data from ${data_dir} to ${s3_uri}"
