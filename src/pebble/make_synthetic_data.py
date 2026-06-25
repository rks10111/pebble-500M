from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic uint16 token shards for smoke benchmarks.")
    parser.add_argument("--out-dir", required=True, help="Output directory for manifest and shards.")
    parser.add_argument("--train-tokens", type=int, default=1_000_000)
    parser.add_argument("--val-tokens", type=int, default=100_000)
    parser.add_argument("--shard-tokens", type=int, default=100_000)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def _write_split(
    base_dir: Path,
    split: str,
    total_tokens: int,
    shard_tokens: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    split_dir = base_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    remaining = total_tokens
    index = 0
    while remaining > 0:
        tokens = min(remaining, shard_tokens)
        path = split_dir / f"{split}_{index:06d}.bin"
        data = rng.integers(0, vocab_size, size=tokens, dtype=np.uint16)
        data.tofile(path)
        shards.append(
            {
                "path": str(path.relative_to(base_dir)),
                "tokens": int(tokens),
                "bytes": int(tokens * np.dtype(np.uint16).itemsize),
            }
        )
        remaining -= tokens
        index += 1
    return {"tokens": int(total_tokens), "shards": shards}


def make_synthetic_data(args: argparse.Namespace) -> Path:
    if args.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("vocab size must fit in uint16")
    base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest = {
        "version": 1,
        "synthetic": True,
        "determinism": {"seed": int(args.seed)},
        "tokenizer": {
            "name": "synthetic",
            "model_vocab_size": int(args.vocab_size),
            "storage_dtype": "uint16",
        },
        "splits": {
            "train": _write_split(
                base_dir,
                "train",
                args.train_tokens,
                args.shard_tokens,
                args.vocab_size,
                rng,
            ),
            "val": _write_split(
                base_dir,
                "val",
                args.val_tokens,
                args.shard_tokens,
                args.vocab_size,
                rng,
            ),
        },
    }
    manifest_path = base_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(manifest_path)
    return manifest_path


def main() -> None:
    make_synthetic_data(parse_args())


if __name__ == "__main__":
    main()
