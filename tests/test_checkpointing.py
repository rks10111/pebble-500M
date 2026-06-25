from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

from pebble.config import load_config
from pebble.train import parse_args, save_checkpoint, sync_run_artifacts_to_s3, train


def _write_training_manifest(data_dir: Path) -> None:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir()
    (np.arange(2048, dtype=np.uint16) % 512).tofile(train_dir / "train_000000.bin")
    (np.arange(512, dtype=np.uint16) % 512).tofile(val_dir / "val_000000.bin")
    manifest = {
        "version": 1,
        "splits": {
            "train": {"tokens": 2048, "shards": [{"path": "train/train_000000.bin", "tokens": 2048}]},
            "val": {"tokens": 512, "shards": [{"path": "val/val_000000.bin", "tokens": 512}]},
        },
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_save_checkpoint_writes_final_file_atomically(tmp_path) -> None:
    cfg = load_config("configs/smoke.yaml")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    path = save_checkpoint(
        tmp_path,
        "latest",
        model,
        optimizer,
        cfg,
        tokens_seen=512,
        global_step=1,
        train_loader_state={"rng_state": {"state": 123}},
    )

    assert path == tmp_path / "latest-000000000001.pt"
    assert path.exists()
    assert not (tmp_path / "latest-000000000001.pt.tmp").exists()

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["tokens_seen"] == 512
    assert checkpoint["global_step"] == 1
    assert checkpoint["train_loader"] == {"rng_state": {"state": 123}}


def test_sync_run_artifacts_to_s3_syncs_checkpoints_and_optional_files(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest-000000000001.pt").write_bytes(b"checkpoint")
    (checkpoint_dir / "latest-000000000002.pt.tmp").write_bytes(b"partial")
    (tmp_path / "metrics.jsonl").write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []

    def runner(command: list[str], check: bool) -> subprocess.CompletedProcess:
        calls.append(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    assert sync_run_artifacts_to_s3(
        tmp_path,
        "s3://statement-llm-training/pebble-500m/runs/test/",
        "eu-west-2",
        runner=runner,
    )

    assert calls == [
        [
            "aws",
            "s3",
            "sync",
            str(checkpoint_dir),
            "s3://statement-llm-training/pebble-500m/runs/test/checkpoints",
            "--region",
            "eu-west-2",
            "--only-show-errors",
            "--exclude",
            "*.tmp",
        ],
        [
            "aws",
            "s3",
            "cp",
            str(tmp_path / "metrics.jsonl"),
            "s3://statement-llm-training/pebble-500m/runs/test/metrics.jsonl",
            "--region",
            "eu-west-2",
            "--only-show-errors",
        ],
    ]


def test_sync_run_artifacts_to_s3_reports_failure_without_raising(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    def runner(command: list[str], check: bool) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(2, command)

    assert not sync_run_artifacts_to_s3(
        tmp_path,
        "s3://statement-llm-training/pebble-500m/runs/test",
        "eu-west-2",
        runner=runner,
    )


def test_final_checkpoint_sync_runs_before_final_eval(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "run"
    _write_training_manifest(data_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pebble-train",
            "--config",
            "configs/smoke.yaml",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--max-tokens",
            "512",
            "--no-wandb",
            "--s3-sync-uri",
            "s3://statement-llm-training/pebble-500m/runs/test",
        ],
    )
    args = parse_args()

    sync_reasons: list[str] = []

    def fake_sync(args, run_dir: Path, reason: str) -> None:
        sync_reasons.append(reason)
        assert list((run_dir / "checkpoints").glob("latest-*.pt"))

    def fail_final_eval(*args, **kwargs):
        raise RuntimeError("final eval failed")

    monkeypatch.setattr("pebble.train.maybe_sync_run_artifacts_to_s3", fake_sync)
    monkeypatch.setattr("pebble.train.evaluate_metrics", fail_final_eval)

    with pytest.raises(RuntimeError, match="final eval failed"):
        train(args)

    assert len(sync_reasons) == 1
    assert sync_reasons[0].startswith("final checkpoint latest-")
