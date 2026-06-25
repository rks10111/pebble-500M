from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pebble.data import SequentialTokenLoader, build_loaders, shard_specs


def _write_manifest(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    np.arange(256, dtype=np.uint16).tofile(train_dir / "train_000000.bin")
    np.arange(128, dtype=np.uint16).tofile(val_dir / "val_000000.bin")
    manifest = {
        "version": 1,
        "splits": {
            "train": {"tokens": 256, "shards": [{"path": "train/train_000000.bin", "tokens": 256}]},
            "val": {"tokens": 128, "shards": [{"path": "val/val_000000.bin", "tokens": 128}]},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_random_loader_shapes(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    train_loader, _ = build_loaders(
        tmp_path,
        block_size=16,
        micro_batch_size=4,
        seed=123,
        device=torch.device("cpu"),
    )
    x, y = train_loader.next_batch()
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_sequential_loader_yields_validation_batches(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    loader = SequentialTokenLoader(
        shard_specs(tmp_path, "val"),
        block_size=16,
        batch_size=2,
        device=torch.device("cpu"),
    )
    x, y, tokens = next(loader.iter_batches(max_tokens=32))
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert tokens == 32
