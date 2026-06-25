#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

if [[ -f "${HOME}/.pebble-training-env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.pebble-training-env"
fi

if [[ -f /opt/pytorch/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /opt/pytorch/bin/activate
fi

stamp="$(date +%Y%m%d-%H%M%S)"
region="${AWS_REGION:-eu-west-2}"
s3_uri="${1:-s3://statement-llm-training/pebble-500m/data/smoke-tokenized-${stamp}}"
data_dir="${2:-/opt/dlami/nvme/pebble-data-s3-smoke-${stamp}}"
restore_dir="${3:-${data_dir}-restore}"
input_jsonl="${data_dir}-input.jsonl"

command -v pebble-prepare-data >/dev/null
command -v aws >/dev/null

mkdir -p "$(dirname "${data_dir}")"

python - <<PY
import json
from pathlib import Path

path = Path("${input_jsonl}")
path.parent.mkdir(parents=True, exist_ok=True)
rows = [
    {"text": "S3 backup smoke test alpha. Tokenize this local text repeatedly. " * 80},
    {"text": "S3 backup smoke test beta. The restored shards should match exactly. " * 80},
    {"text": "S3 backup smoke test gamma. This avoids remote parquet downloads. " * 80},
    {"text": "S3 backup smoke test delta. Keep the fixture tiny and deterministic. " * 80},
]
with path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\\n")
print(path)
PY

echo "== tokenize tiny local jsonl =="
pebble-prepare-data \
  --config configs/pebble_500m_50b.yaml \
  --input-jsonl "${input_jsonl}" \
  --out-dir "${data_dir}" \
  --train-tokens 1024 \
  --val-tokens 512 \
  --shard-tokens 512

echo
echo "== sync to s3 =="
AWS_REGION="${region}" scripts/sync_data_to_s3.sh "${data_dir}" "${s3_uri}"

echo
echo "== restore from s3 =="
AWS_REGION="${region}" scripts/sync_data_from_s3.sh "${s3_uri}" "${restore_dir}"

echo
echo "== verify restored data =="
python - <<PY
import hashlib
import json
from pathlib import Path

source = Path("${data_dir}")
restored = Path("${restore_dir}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_manifest = json.loads((source / "manifest.json").read_text())
restored_manifest = json.loads((restored / "manifest.json").read_text())

for split in ("train", "val"):
    source_info = source_manifest["splits"][split]
    restored_info = restored_manifest["splits"][split]
    assert source_info["tokens"] == restored_info["tokens"], split
    assert len(source_info["shards"]) == len(restored_info["shards"]), split
    for source_shard, restored_shard in zip(source_info["shards"], restored_info["shards"]):
        assert source_shard["path"] == restored_shard["path"], split
        assert source_shard["tokens"] == restored_shard["tokens"], split
        source_path = source / source_shard["path"]
        restored_path = restored / restored_shard["path"]
        assert source_path.stat().st_size == restored_path.stat().st_size, source_shard["path"]
        assert sha256(source_path) == sha256(restored_path), source_shard["path"]
    print(f"{split}: tokens={source_info['tokens']:,} shards={len(source_info['shards'])}")

print("restore verification passed")
PY

echo
echo "Smoke tokenized data: ${data_dir}"
echo "Restored data: ${restore_dir}"
echo "S3 backup: ${s3_uri}"
