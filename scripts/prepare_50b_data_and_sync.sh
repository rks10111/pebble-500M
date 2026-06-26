#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

declare -A caller_env=()
for key in \
  AWS_REGION \
  HF_HOME \
  HF_DATASETS_CACHE \
  TOKENIZERS_PARALLELISM \
  PEBBLE_CONFIG \
  PEBBLE_DATA_DIR \
  PEBBLE_S3_DATA_URI \
  PEBBLE_LOG_DIR \
  PEBBLE_TRAIN_TOKENS \
  PEBBLE_VAL_TOKENS \
  PEBBLE_SHARD_TOKENS \
  PEBBLE_LOG_INTERVAL_DOCS \
  PEBBLE_RUN_S3_SMOKE \
  PEBBLE_SYNC_DATA_TO_S3 \
  PEBBLE_STOP_INSTANCE_ON_DONE \
  PEBBLE_INSTANCE_ID \
  PEBBLE_WANDB \
  WANDB_PROJECT \
  WANDB_ENTITY \
  WANDB_RUN_ID \
  WANDB_NAME \
  WANDB_GROUP \
  WANDB_RUN_GROUP \
  WANDB_TAGS \
  WANDB_JOB_TYPE \
  WANDB_MODE \
  WANDB_DIR \
  WANDB_RESUME; do
  if [[ -v "${key}" ]]; then
    caller_env["${key}"]="${!key}"
  fi
done

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

config="${PEBBLE_CONFIG:-configs/pebble_500m_30b.yaml}"
data_dir="${PEBBLE_DATA_DIR:-/opt/dlami/nvme/pebble-data-50b}"
s3_data_uri="${PEBBLE_S3_DATA_URI:-s3://statement-llm-training/pebble-500m/data/fineweb-edu-gpt2-50b}"
log_dir="${PEBBLE_LOG_DIR:-/opt/dlami/nvme/logs}"
train_tokens="${PEBBLE_TRAIN_TOKENS:-50000000000}"
val_tokens="${PEBBLE_VAL_TOKENS:-50000000}"
shard_tokens="${PEBBLE_SHARD_TOKENS:-100000000}"
log_interval_docs="${PEBBLE_LOG_INTERVAL_DOCS:-10000}"
region="${AWS_REGION:-eu-west-2}"
run_s3_smoke="${PEBBLE_RUN_S3_SMOKE:-1}"
sync_to_s3="${PEBBLE_SYNC_DATA_TO_S3:-1}"
stop_instance_on_done="${PEBBLE_STOP_INSTANCE_ON_DONE:-1}"
wandb_enabled="${PEBBLE_WANDB:-1}"
wandb_project="${WANDB_PROJECT:-pebble-500m}"
wandb_job_type="${WANDB_JOB_TYPE:-data-prep}"
wandb_run_name="${WANDB_NAME:-pebble-500m-50b-tokenization}"
wandb_group="${WANDB_GROUP:-${WANDB_RUN_GROUP:-pebble-500m-50b}}"
wandb_tags="${WANDB_TAGS:-data-prep,tokenization,50b}"
wandb_mode="${WANDB_MODE:-online}"
wandb_dir="${WANDB_DIR:-${log_dir}/wandb}"
wandb_resume="${WANDB_RESUME:-allow}"

mkdir -p "${data_dir}" "${log_dir}"
mkdir -p "${HF_HOME:-/opt/dlami/nvme/hf-cache}" "${HF_DATASETS_CACHE:-/opt/dlami/nvme/hf-cache/datasets}"

if [[ ! -f "${config}" ]]; then
  echo "error: config not found: ${config}" >&2
  exit 1
fi

command -v pebble-prepare-data >/dev/null
command -v aws >/dev/null

echo "== plan =="
echo "repo=${repo_dir}"
echo "config=${config}"
echo "data_dir=${data_dir}"
echo "s3_data_uri=${s3_data_uri}"
echo "log_dir=${log_dir}"
echo "train_tokens=${train_tokens}"
echo "val_tokens=${val_tokens}"
echo "shard_tokens=${shard_tokens}"
echo "log_interval_docs=${log_interval_docs}"
echo "region=${region}"
echo "wandb_enabled=${wandb_enabled}"
echo "wandb_project=${wandb_project}"
echo "wandb_mode=${wandb_mode}"
echo "wandb_dir=${wandb_dir}"
echo "stop_instance_on_done=${stop_instance_on_done}"

echo
echo "== disk =="
df -h / /opt/dlami/nvme

if [[ "${run_s3_smoke}" == "1" ]]; then
  echo
  echo "== s3 smoke test =="
  AWS_REGION="${region}" scripts/s3_smoke_test.sh "s3://statement-llm-training/pebble-500m"
fi

if [[ -f "${data_dir}/manifest.json" ]]; then
  echo
  echo "== prepare =="
  echo "manifest already exists, skipping tokenization: ${data_dir}/manifest.json"
else
  if [[ -n "$(find "${data_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "error: ${data_dir} is not empty but has no manifest.json" >&2
    echo "Move or remove the partial directory before rerunning; tokenization is not resumable." >&2
    exit 1
  fi

  stamp="$(date +%Y%m%d-%H%M%S)"
  log_path="${log_dir}/prepare-50b-${stamp}.log"
  echo
  echo "== prepare =="
  echo "writing log to ${log_path}"

  wandb_args=()
  case "${wandb_enabled}" in
    1|true|TRUE|yes|YES|on|ON)
      wandb_args+=(--wandb)
      ;;
    0|false|FALSE|no|NO|off|OFF)
      wandb_args+=(--no-wandb)
      ;;
    *)
      echo "error: PEBBLE_WANDB must be 1/0, true/false, yes/no, or on/off" >&2
      exit 1
      ;;
  esac
  wandb_args+=(
    --wandb-project "${wandb_project}"
    --wandb-run-name "${wandb_run_name}"
    --wandb-group "${wandb_group}"
    --wandb-tags "${wandb_tags}"
    --wandb-job-type "${wandb_job_type}"
    --wandb-mode "${wandb_mode}"
    --wandb-dir "${wandb_dir}"
    --wandb-resume "${wandb_resume}"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    wandb_args+=(--wandb-run-id "${WANDB_RUN_ID}")
  fi

  pebble-prepare-data \
    --config "${config}" \
    --out-dir "${data_dir}" \
    --train-tokens "${train_tokens}" \
    --val-tokens "${val_tokens}" \
    --shard-tokens "${shard_tokens}" \
    --log-interval-docs "${log_interval_docs}" \
    "${wandb_args[@]}" \
    2>&1 | tee "${log_path}"
fi

if [[ ! -f "${data_dir}/manifest.json" ]]; then
  echo "error: tokenization did not produce ${data_dir}/manifest.json" >&2
  exit 1
fi

echo
echo "== verify manifest targets =="
DATA_DIR="${data_dir}" TRAIN_TOKENS="${train_tokens}" VAL_TOKENS="${val_tokens}" python - <<'PY'
import json
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
manifest = json.loads((data_dir / "manifest.json").read_text())
expected = {
    "train": int(os.environ["TRAIN_TOKENS"]),
    "val": int(os.environ["VAL_TOKENS"]),
}

for split, target in expected.items():
    actual = int(manifest["splits"][split]["tokens"])
    if actual < target:
        raise SystemExit(
            f"{data_dir}/manifest.json has only {actual:,} {split} tokens; "
            f"expected at least {target:,}. Move or remove the undersized data "
            "directory before rerunning."
        )
    print(f"{split}: tokens={actual:,} target={target:,}")
PY

echo
echo "== local data summary =="
du -sh "${data_dir}"
python - <<PY
import json
from pathlib import Path
manifest = json.loads(Path("${data_dir}/manifest.json").read_text())
for split in ("train", "val"):
    info = manifest["splits"][split]
    print(f"{split}: tokens={info['tokens']:,} shards={len(info['shards'])}")
PY

if [[ "${sync_to_s3}" == "1" ]]; then
  echo
  echo "== sync to s3 =="
  AWS_REGION="${region}" scripts/sync_data_to_s3.sh "${data_dir}" "${s3_data_uri}"

  echo
  echo "== s3 data summary =="
  aws s3 ls "${s3_data_uri%/}/" \
    --recursive \
    --summarize \
    --human-readable \
    --region "${region}"
fi

echo
echo "Data prep complete."
echo "Training data: ${data_dir}"
echo "S3 backup: ${s3_data_uri}"

if [[ "${stop_instance_on_done}" == "1" ]]; then
  echo
  echo "== stop instance =="
  instance_id="${PEBBLE_INSTANCE_ID:-}"
  if [[ -z "${instance_id}" && -r /sys/devices/virtual/dmi/id/board_asset_tag ]]; then
    instance_id="$(cat /sys/devices/virtual/dmi/id/board_asset_tag)"
  fi

  if [[ ! "${instance_id}" =~ ^i-[A-Za-z0-9]+$ ]]; then
    echo "warning: could not determine EC2 instance id; leaving instance running" >&2
  elif aws ec2 stop-instances --instance-ids "${instance_id}" --region "${region}"; then
    echo "requested stop for EC2 instance ${instance_id}"
  else
    echo "warning: failed to stop EC2 instance ${instance_id}; check IAM ec2:StopInstances permission" >&2
  fi
else
  echo "PEBBLE_STOP_INSTANCE_ON_DONE=0, leaving instance running"
fi
