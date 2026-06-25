from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from pebble.prepare_fineweb_edu import prepare


def test_prepare_tokenizes_local_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    rows = [
        {"text": "local tokenization smoke test alpha " * 40},
        {"text": "local tokenization smoke test beta " * 40},
        {"text": "local tokenization smoke test gamma " * 40},
        {"text": "local tokenization smoke test delta " * 40},
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    manifest_path = prepare(
        argparse.Namespace(
            config="configs/smoke.yaml",
            out_dir=str(out_dir),
            input_jsonl=str(input_path),
            train_tokens=128,
            val_tokens=64,
            shard_tokens=64,
            seed=123,
            shuffle_buffer=1000,
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["local_jsonl"] == str(input_path.resolve())
    assert manifest["splits"]["train"]["tokens"] >= 128
    assert manifest["splits"]["val"]["tokens"] >= 64
    assert manifest["document_manifest"] == "documents.jsonl.gz"

    for split in ("train", "val"):
        for shard in manifest["splits"][split]["shards"]:
            path = out_dir / shard["path"]
            assert path.is_file()
            assert path.stat().st_size == shard["bytes"]
            assert shard["bytes"] == shard["tokens"] * 2

    with gzip.open(out_dir / "documents.jsonl.gz", "rt", encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle]
    assert {document["split"] for document in documents} == {"train", "val"}
