from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from pebble.config import load_config
from pebble.prepare_fineweb_edu import init_wandb, parse_args, prepare


def test_prepare_wandb_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pebble-prepare-data",
            "--config",
            "configs/smoke.yaml",
            "--out-dir",
            "/tmp/data",
        ],
    )

    args = parse_args()

    assert args.wandb is False
    assert args.wandb_project == "pebble-500m"
    assert args.wandb_job_type == "data-prep"
    assert args.wandb_mode == "online"
    assert args.wandb_resume == "allow"
    assert args.wandb_save_code is True
    assert args.log_interval_docs == 1000


def test_prepare_wandb_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pebble-prepare-data",
            "--config",
            "configs/smoke.yaml",
            "--out-dir",
            "/tmp/data",
            "--wandb",
            "--wandb-mode",
            "offline",
            "--log-interval-docs",
            "10",
        ],
    )

    args = parse_args()

    assert args.wandb is True
    assert args.wandb_mode == "offline"
    assert args.log_interval_docs == 10


def test_prepare_wandb_init_uses_data_prep_defaults(monkeypatch, tmp_path: Path) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.summary = {}
            self.metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []
            self.finished = False

        def define_metric(self, *args: object, **kwargs: object) -> None:
            self.metrics.append((args, kwargs))

        def finish(self) -> None:
            self.finished = True

    class FakeWandb:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None
            self.run = FakeRun()

        def init(self, **kwargs: object) -> FakeRun:
            self.kwargs = kwargs
            return self.run

    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    cfg = load_config("configs/smoke.yaml")
    out_dir = tmp_path / "tokenized"
    args = argparse.Namespace(
        wandb=True,
        wandb_project="pebble-500m",
        wandb_entity=None,
        wandb_run_id=None,
        wandb_run_name=None,
        wandb_group=None,
        wandb_tags="",
        wandb_job_type="data-prep",
        wandb_mode="disabled",
        wandb_dir=None,
        wandb_resume="allow",
        wandb_save_code=True,
        log_interval_docs=7,
    )

    run = init_wandb(
        args,
        cfg,
        out_dir,
        input_jsonl=None,
        train_target=100,
        val_target=20,
        shard_tokens=10,
    )

    assert run is fake_wandb.run
    assert fake_wandb.kwargs is not None
    assert fake_wandb.kwargs["project"] == "pebble-500m"
    assert fake_wandb.kwargs["job_type"] == "data-prep"
    assert fake_wandb.kwargs["mode"] == "disabled"
    assert fake_wandb.kwargs["dir"] == str(tmp_path / "wandb")
    assert fake_wandb.kwargs["tags"] == [
        "data-prep",
        "tokenization",
        cfg.experiment.name,
        "huggingface-streaming",
    ]


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
