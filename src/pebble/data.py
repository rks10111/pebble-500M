from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class ShardSpec:
    path: Path
    tokens: int


class TokenShard:
    def __init__(self, spec: ShardSpec) -> None:
        self.spec = spec
        self.data = np.memmap(spec.path, dtype=np.uint16, mode="r")
        if len(self.data) != spec.tokens:
            raise ValueError(f"{spec.path} has {len(self.data)} tokens, manifest expected {spec.tokens}")

    def __len__(self) -> int:
        return int(self.spec.tokens)


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    path = Path(data_dir) / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "splits" not in manifest:
        raise ValueError(f"{path} does not look like a Pebble token manifest")
    return manifest


def shard_specs(data_dir: str | Path, split: str) -> list[ShardSpec]:
    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)
    split_data = manifest["splits"].get(split)
    if not split_data:
        raise ValueError(f"manifest has no split named {split!r}")
    specs: list[ShardSpec] = []
    for item in split_data["shards"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = data_dir / path
        specs.append(ShardSpec(path=path, tokens=int(item["tokens"])))
    if not specs:
        raise ValueError(f"split {split!r} has no shards")
    return specs


class RandomTokenLoader:
    def __init__(
        self,
        specs: list[ShardSpec],
        block_size: int,
        batch_size: int,
        seed: int,
        device: torch.device,
    ) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.shards = [TokenShard(spec) for spec in specs]
        self.usable_tokens = [max(0, len(shard) - block_size - 1) for shard in self.shards]
        if sum(self.usable_tokens) <= 0:
            raise ValueError("no shard is long enough for the requested context length")
        self.cumulative = np.cumsum(self.usable_tokens).astype(np.int64)
        self.cumulative_list = self.cumulative.tolist()
        self.rng = np.random.default_rng(seed)

    def state_dict(self) -> dict[str, Any]:
        return {"rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            self.rng.bit_generator.state = state["rng_state"]

    def _sample_location(self) -> tuple[TokenShard, int]:
        draw = int(self.rng.integers(0, int(self.cumulative[-1])))
        shard_index = bisect.bisect_right(self.cumulative_list, draw)
        previous = int(self.cumulative[shard_index - 1]) if shard_index > 0 else 0
        offset = draw - previous
        return self.shards[shard_index], offset

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch = np.empty((self.batch_size, self.block_size + 1), dtype=np.int64)
        for row in range(self.batch_size):
            shard, offset = self._sample_location()
            batch[row] = shard.data[offset : offset + self.block_size + 1].astype(np.int64, copy=True)
        x = torch.from_numpy(np.ascontiguousarray(batch[:, :-1])).to(self.device, non_blocking=True)
        y = torch.from_numpy(np.ascontiguousarray(batch[:, 1:])).to(self.device, non_blocking=True)
        return x, y


class SequentialTokenLoader:
    def __init__(
        self,
        specs: list[ShardSpec],
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.shards = [TokenShard(spec) for spec in specs]

    def iter_batches(self, max_tokens: int | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
        produced = 0
        rows: list[np.ndarray] = []
        for shard in self.shards:
            limit = len(shard) - self.block_size - 1
            for offset in range(0, max(0, limit), self.block_size):
                if max_tokens is not None and produced >= max_tokens:
                    return
                rows.append(shard.data[offset : offset + self.block_size + 1].astype(np.int64, copy=True))
                if len(rows) == self.batch_size:
                    yield self._rows_to_batch(rows)
                    produced += self.batch_size * self.block_size
                    rows = []
        if rows:
            yield self._rows_to_batch(rows)

    def _rows_to_batch(self, rows: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch = np.stack(rows, axis=0)
        x = torch.from_numpy(np.ascontiguousarray(batch[:, :-1])).to(self.device, non_blocking=True)
        y = torch.from_numpy(np.ascontiguousarray(batch[:, 1:])).to(self.device, non_blocking=True)
        return x, y, int(x.numel())


def build_loaders(
    data_dir: str | Path,
    block_size: int,
    micro_batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[RandomTokenLoader, SequentialTokenLoader]:
    train = RandomTokenLoader(
        shard_specs(data_dir, "train"),
        block_size=block_size,
        batch_size=micro_batch_size,
        seed=seed,
        device=device,
    )
    val = SequentialTokenLoader(
        shard_specs(data_dir, "val"),
        block_size=block_size,
        batch_size=micro_batch_size,
        device=device,
    )
    return train, val
