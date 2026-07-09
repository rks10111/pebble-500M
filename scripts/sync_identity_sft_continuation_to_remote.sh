#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_dir="${LOCAL_DIR:-${repo_root}/data/pebble-identity-sft-continuation}"
remote_host="${STATEMENT_LLM_HOST:-statement-llm}"
remote_dir="${REMOTE_DIR:-/opt/dlami/nvme/pebble-sft-data/pebble-identity-sft-continuation}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<EOF
usage: $0

Sync the prepared Pebble identity SFT continuation dataset to the remote trainer.

Environment overrides:
  LOCAL_DIR          Local prepared masked SFT directory.
                     Default: ${local_dir}
  STATEMENT_LLM_HOST SSH host.
                     Default: ${remote_host}
  REMOTE_DIR         Remote prepared masked SFT directory.
                     Default: ${remote_dir}
EOF
  exit 0
fi

if [[ ! -f "${local_dir}/manifest.json" ]]; then
  echo "error: local prepared dataset is missing manifest.json: ${local_dir}" >&2
  exit 1
fi

remote_dir_quoted="$(printf "%q" "${remote_dir}")"
ssh "${remote_host}" "mkdir -p -- ${remote_dir_quoted}"
rsync -avP --delete "${local_dir}/" "${remote_host}:${remote_dir}/"

echo "synced identity SFT continuation dataset to ${remote_host}:${remote_dir}"
