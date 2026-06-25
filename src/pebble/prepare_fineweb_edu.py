from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from datasets import load_dataset

from pebble.config import Config, load_config


class ShardWriter:
    def __init__(self, base_dir: Path, split: str, shard_tokens: int) -> None:
        self.base_dir = base_dir
        self.split = split
        self.split_dir = base_dir / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.buffer = np.empty(shard_tokens, dtype=np.uint16)
        self.offset = 0
        self.index = 0
        self.total_tokens = 0
        self.shards: list[dict[str, Any]] = []

    def add(self, tokens: list[int]) -> None:
        if not tokens:
            return
        incoming = np.asarray(tokens, dtype=np.uint16)
        cursor = 0
        while cursor < len(incoming):
            available = self.shard_tokens - self.offset
            take = min(available, len(incoming) - cursor)
            self.buffer[self.offset : self.offset + take] = incoming[cursor : cursor + take]
            self.offset += take
            self.total_tokens += take
            cursor += take
            if self.offset == self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if self.offset == 0:
            return
        filename = f"{self.split}_{self.index:06d}.bin"
        path = self.split_dir / filename
        self.buffer[: self.offset].tofile(path)
        self.shards.append(
            {
                "path": str(path.relative_to(self.base_dir)),
                "tokens": int(self.offset),
                "bytes": int(self.offset * np.dtype(np.uint16).itemsize),
            }
        )
        self.index += 1
        self.offset = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu into deterministic uint16 shards.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--out-dir", required=True, help="Directory for token shards and manifests.")
    parser.add_argument("--train-tokens", type=int, default=None, help="Training token target.")
    parser.add_argument("--val-tokens", type=int, default=None, help="Validation token target.")
    parser.add_argument("--shard-tokens", type=int, default=None, help="Tokens per shard.")
    parser.add_argument("--seed", type=int, default=None, help="Document shuffle seed.")
    parser.add_argument("--shuffle-buffer", type=int, default=None, help="Streaming shuffle buffer.")
    return parser.parse_args()


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_stream(cfg: Config, seed: int, shuffle_buffer: int):
    dataset = load_dataset(
        cfg.data.dataset_name,
        cfg.data.dataset_config,
        split="train",
        streaming=True,
    )
    return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)


def prepare(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.tokenizer.storage_dtype != "uint16":
        raise ValueError("This pipeline writes uint16 shards; tokenizer.storage_dtype must be uint16")
    if cfg.tokenizer.eod_token >= np.iinfo(np.uint16).max:
        raise ValueError("EOD token does not fit in uint16")

    seed = args.seed if args.seed is not None else cfg.data.seed
    shuffle_buffer = (
        args.shuffle_buffer if args.shuffle_buffer is not None else cfg.data.streaming_shuffle_buffer
    )
    train_target = args.train_tokens if args.train_tokens is not None else cfg.data.train_target_tokens
    val_target = args.val_tokens if args.val_tokens is not None else cfg.data.validation_target_tokens
    shard_tokens = args.shard_tokens if args.shard_tokens is not None else cfg.data.shard_tokens

    encoder = tiktoken.get_encoding(cfg.tokenizer.name)
    train_writer = ShardWriter(out_dir, "train", shard_tokens)
    val_writer = ShardWriter(out_dir, "val", shard_tokens)

    doc_manifest_path = out_dir / "documents.jsonl.gz"
    started = time.time()
    docs_seen = 0
    docs_kept = 0

    with gzip.open(doc_manifest_path, "wt", encoding="utf-8") as doc_manifest:
        for source_index, row in enumerate(_load_stream(cfg, seed=seed, shuffle_buffer=shuffle_buffer)):
            text = row.get(cfg.data.text_field)
            if not isinstance(text, str) or not text.strip():
                continue

            tokens = encoder.encode_ordinary(text)
            tokens.append(cfg.tokenizer.eod_token)
            if max(tokens) >= np.iinfo(np.uint16).max:
                raise ValueError("token id exceeded uint16 storage")

            if val_writer.total_tokens < val_target:
                split = "val"
                val_writer.add(tokens)
            elif train_writer.total_tokens < train_target:
                split = "train"
                train_writer.add(tokens)
            else:
                break

            docs_seen += 1
            docs_kept += 1
            record = {
                "source_index_after_shuffle": int(source_index),
                "split": split,
                "sha1": _sha1_text(text),
                "token_count": int(len(tokens)),
            }
            doc_manifest.write(json.dumps(record, sort_keys=True) + "\n")

            if docs_kept % 1000 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    "prepared "
                    f"docs={docs_kept} "
                    f"train_tokens={train_writer.total_tokens:,} "
                    f"val_tokens={val_writer.total_tokens:,} "
                    f"docs/sec={docs_seen / elapsed:.1f}",
                    flush=True,
                )

    train_writer.flush()
    val_writer.flush()

    manifest = {
        "version": 1,
        "created_unix": int(time.time()),
        "dataset": {
            "name": cfg.data.dataset_name,
            "config": cfg.data.dataset_config,
            "split": "train",
            "text_field": cfg.data.text_field,
        },
        "determinism": {
            "seed": int(seed),
            "streaming_shuffle_buffer": int(shuffle_buffer),
            "split_rule": "after deterministic streaming shuffle, fill validation first, then training",
        },
        "tokenizer": {
            "name": cfg.tokenizer.name,
            "real_vocab_size": cfg.tokenizer.real_vocab_size,
            "model_vocab_size": cfg.tokenizer.model_vocab_size,
            "eod_token": cfg.tokenizer.eod_token,
            "storage_dtype": cfg.tokenizer.storage_dtype,
        },
        "document_manifest": str(doc_manifest_path.relative_to(out_dir)),
        "splits": {
            "train": {
                "target_tokens": int(train_target),
                "tokens": int(train_writer.total_tokens),
                "shards": train_writer.shards,
            },
            "val": {
                "target_tokens": int(val_target),
                "tokens": int(val_writer.total_tokens),
                "shards": val_writer.shards,
            },
        },
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "finished "
        f"train_tokens={train_writer.total_tokens:,} "
        f"val_tokens={val_writer.total_tokens:,} "
        f"manifest={manifest_path}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
