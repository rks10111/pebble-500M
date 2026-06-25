#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

declare -A caller_env=()
for key in \
  AWS_REGION \
  PEBBLE_NVME_ROOT \
  PEBBLE_DATA_DIR \
  PEBBLE_RUN_DIR \
  PEBBLE_S3_DATA_URI \
  PEBBLE_S3_RUN_URI \
  PEBBLE_RESTORE_DATA \
  PEBBLE_RESTORE_RUN \
  PEBBLE_ALLOW_UNMOUNTED_NVME \
  PEBBLE_VERIFY_TARGET_TOKENS \
  PEBBLE_EXPECT_TRAIN_TOKENS \
  PEBBLE_EXPECT_VAL_TOKENS; do
  if [[ -v "${key}" ]]; then
    caller_env["${key}"]="${!key}"
  fi
done

usage() {
  cat >&2 <<'EOF'
usage: scripts/restore_nvme_from_s3.sh [data-s3-uri] [run-s3-uri]

Restores the known Pebble 50B S3 prefixes back onto the local NVMe volume.

Environment overrides:
  AWS_REGION                 default: eu-west-2
  PEBBLE_NVME_ROOT           default: /opt/dlami/nvme
  PEBBLE_DATA_DIR            default: /opt/dlami/nvme/pebble-data-50b
  PEBBLE_RUN_DIR             default: /opt/dlami/nvme/pebble-runs/pebble-500m-50b
  PEBBLE_S3_DATA_URI         default: s3://statement-llm-training/pebble-500m/data/fineweb-edu-gpt2-50b
  PEBBLE_S3_RUN_URI          default: s3://statement-llm-training/pebble-500m/runs/pebble-500m-50b
  PEBBLE_RESTORE_DATA        default: 1
  PEBBLE_RESTORE_RUN         default: 1
  PEBBLE_ALLOW_UNMOUNTED_NVME default: 0
  PEBBLE_VERIFY_TARGET_TOKENS default: 1
  PEBBLE_EXPECT_TRAIN_TOKENS default: 50000000000
  PEBBLE_EXPECT_VAL_TOKENS   default: 50000000
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  usage
  exit 2
fi

if [[ -f "${HOME}/.pebble-training-env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.pebble-training-env"
fi

for key in "${!caller_env[@]}"; do
  export "${key}=${caller_env[${key}]}"
done

if [[ -f /opt/pytorch/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /opt/pytorch/bin/activate
fi

region="${AWS_REGION:-eu-west-2}"
nvme_root="${PEBBLE_NVME_ROOT:-/opt/dlami/nvme}"
data_dir="${PEBBLE_DATA_DIR:-/opt/dlami/nvme/pebble-data-50b}"
run_dir="${PEBBLE_RUN_DIR:-/opt/dlami/nvme/pebble-runs/pebble-500m-50b}"
s3_data_uri="${1:-${PEBBLE_S3_DATA_URI:-s3://statement-llm-training/pebble-500m/data/fineweb-edu-gpt2-50b}}"
s3_run_uri="${2:-${PEBBLE_S3_RUN_URI:-s3://statement-llm-training/pebble-500m/runs/pebble-500m-50b}}"
restore_data="${PEBBLE_RESTORE_DATA:-1}"
restore_run="${PEBBLE_RESTORE_RUN:-1}"
allow_unmounted_nvme="${PEBBLE_ALLOW_UNMOUNTED_NVME:-0}"
verify_target_tokens="${PEBBLE_VERIFY_TARGET_TOKENS:-1}"
expect_train_tokens="${PEBBLE_EXPECT_TRAIN_TOKENS:-50000000000}"
expect_val_tokens="${PEBBLE_EXPECT_VAL_TOKENS:-50000000}"

command -v aws >/dev/null

require_nvme_mount() {
  if [[ "${allow_unmounted_nvme}" == "1" ]]; then
    mkdir -p "${nvme_root}"
    return
  fi

  if [[ ! -d "${nvme_root}" ]]; then
    echo "error: NVMe root does not exist: ${nvme_root}" >&2
    echo "Mount the NVMe volume first, or set PEBBLE_ALLOW_UNMOUNTED_NVME=1 for a deliberate override." >&2
    exit 1
  fi

  if command -v mountpoint >/dev/null && mountpoint -q "${nvme_root}"; then
    return
  fi

  root_device="$(df -P / | awk 'NR == 2 {print $1}')"
  nvme_device="$(df -P "${nvme_root}" | awk 'NR == 2 {print $1}')"
  if [[ -z "${nvme_device}" || "${nvme_device}" == "${root_device}" ]]; then
    echo "error: ${nvme_root} does not appear to be a separate mounted volume" >&2
    echo "Refusing to restore large artifacts onto root EBS. Mount NVMe first." >&2
    exit 1
  fi
}

verify_data_manifest() {
  local manifest_path="${data_dir}/manifest.json"
  if [[ ! -f "${manifest_path}" ]]; then
    echo "error: restored data dir does not contain manifest.json: ${data_dir}" >&2
    exit 1
  fi

  DATA_DIR="${data_dir}" \
    VERIFY_TARGET_TOKENS="${verify_target_tokens}" \
    EXPECT_TRAIN_TOKENS="${expect_train_tokens}" \
    EXPECT_VAL_TOKENS="${expect_val_tokens}" \
    python - <<'PY'
import json
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
manifest = json.loads((data_dir / "manifest.json").read_text())
verify_target_tokens = os.environ["VERIFY_TARGET_TOKENS"] == "1"
expected_tokens = {
    "train": int(os.environ["EXPECT_TRAIN_TOKENS"]),
    "val": int(os.environ["EXPECT_VAL_TOKENS"]),
}

for split in ("train", "val"):
    info = manifest["splits"].get(split)
    if not info:
        raise SystemExit(f"missing split in manifest: {split}")
    tokens = int(info["tokens"])
    if verify_target_tokens and tokens < expected_tokens[split]:
        raise SystemExit(
            f"restored {split} split has only {tokens:,} tokens; "
            f"expected at least {expected_tokens[split]:,}"
        )
    missing = []
    wrong_size = []
    for shard in info["shards"]:
        path = data_dir / shard["path"]
        if not path.is_file():
            missing.append(str(path))
            continue
        expected_bytes = shard.get("bytes")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            wrong_size.append(f"{path}: expected {expected_bytes}, got {path.stat().st_size}")
    if missing:
        raise SystemExit("missing shard files:\n" + "\n".join(missing[:20]))
    if wrong_size:
        raise SystemExit("wrong shard sizes:\n" + "\n".join(wrong_size[:20]))
    print(f"{split}: tokens={tokens:,} shards={len(info['shards'])}")

print("data manifest verification passed")
PY
}

find_latest_checkpoint() {
  local checkpoint_dir="${run_dir}/checkpoints"
  if [[ ! -d "${checkpoint_dir}" ]]; then
    return 1
  fi
  find "${checkpoint_dir}" -maxdepth 1 -type f -name 'latest-*.pt' | sort | tail -n 1
}

echo "== plan =="
echo "region=${region}"
echo "nvme_root=${nvme_root}"
echo "restore_data=${restore_data}"
echo "data_dir=${data_dir}"
echo "s3_data_uri=${s3_data_uri%/}"
echo "restore_run=${restore_run}"
echo "run_dir=${run_dir}"
echo "s3_run_uri=${s3_run_uri%/}"
echo "verify_target_tokens=${verify_target_tokens}"
echo "expect_train_tokens=${expect_train_tokens}"
echo "expect_val_tokens=${expect_val_tokens}"

echo
echo "== disk =="
require_nvme_mount
df -h / "${nvme_root}"

mkdir -p "${data_dir}" "${run_dir}"

if [[ "${restore_data}" == "1" ]]; then
  echo
  echo "== restore tokenized data =="
  AWS_REGION="${region}" scripts/sync_data_from_s3.sh "${s3_data_uri%/}" "${data_dir}"

  echo
  echo "== verify tokenized data =="
  verify_data_manifest
else
  echo
  echo "== restore tokenized data =="
  echo "PEBBLE_RESTORE_DATA=0, skipping data restore"
fi

if [[ "${restore_run}" == "1" ]]; then
  echo
  echo "== restore run artifacts =="
  aws s3 sync "${s3_run_uri%/}" "${run_dir}" --region "${region}" --only-show-errors
  echo "restored run artifacts from ${s3_run_uri%/} to ${run_dir}"
else
  echo
  echo "== restore run artifacts =="
  echo "PEBBLE_RESTORE_RUN=0, skipping run restore"
fi

echo
echo "== resume command =="
latest_checkpoint="$(find_latest_checkpoint || true)"
if [[ -n "${latest_checkpoint}" ]]; then
  cat <<EOF
pebble-train \\
  --config configs/pebble_500m_50b.yaml \\
  --data-dir ${data_dir} \\
  --out-dir ${run_dir} \\
  --resume ${latest_checkpoint} \\
  --max-tokens 50000000000
EOF
else
  echo "No latest checkpoint found under ${run_dir}/checkpoints."
  echo "If this is before the first training run, start without --resume."
fi

echo
echo "Restore complete."
