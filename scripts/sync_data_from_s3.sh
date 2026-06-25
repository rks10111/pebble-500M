#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <s3-uri> <data-dir>" >&2
  exit 2
fi

s3_uri="${1%/}"
data_dir="$2"
region="${AWS_REGION:-eu-west-2}"

mkdir -p "${data_dir}"
aws s3 sync "${s3_uri}" "${data_dir}" --region "${region}" --only-show-errors

if [[ ! -f "${data_dir}/manifest.json" ]]; then
  echo "error: restored data dir does not contain manifest.json: ${data_dir}" >&2
  exit 1
fi

echo "restored tokenized data from ${s3_uri} to ${data_dir}"
