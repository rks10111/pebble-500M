from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
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
    parser = argparse.ArgumentParser(description="Tokenize text into deterministic uint16 shards.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--out-dir", required=True, help="Directory for token shards and manifests.")
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Optional local JSONL text source for smoke tests. Reads the configured text_field.",
    )
    parser.add_argument("--train-tokens", type=int, default=None, help="Training token target.")
    parser.add_argument("--val-tokens", type=int, default=None, help="Validation token target.")
    parser.add_argument("--shard-tokens", type=int, default=None, help="Tokens per shard.")
    parser.add_argument("--seed", type=int, default=None, help="Document shuffle seed.")
    parser.add_argument("--shuffle-buffer", type=int, default=None, help="Streaming shuffle buffer.")
    parser.add_argument(
        "--log-interval-docs",
        type=int,
        default=1000,
        help="Document interval for progress logs.",
    )
    parser.add_argument("--wandb", dest="wandb", action="store_true", help="Enable W&B data-prep tracking.")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable W&B tracking.")
    parser.set_defaults(wandb=False)
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "pebble-500m"),
        help="W&B project name. Defaults to WANDB_PROJECT or pebble-500m.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity or team. Defaults to WANDB_ENTITY.",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=os.environ.get("WANDB_RUN_ID"),
        help="W&B run id. Defaults to a stable id derived from experiment name and output directory.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_NAME"),
        help="W&B display name. Defaults to experiment name plus output directory name.",
    )
    parser.add_argument(
        "--wandb-group",
        default=os.environ.get("WANDB_GROUP") or os.environ.get("WANDB_RUN_GROUP"),
        help="W&B run group. Defaults to the experiment name.",
    )
    parser.add_argument(
        "--wandb-tags",
        default=os.environ.get("WANDB_TAGS", ""),
        help="Comma-separated W&B tags. Defaults to useful data-prep tags.",
    )
    parser.add_argument(
        "--wandb-job-type",
        default=os.environ.get("WANDB_JOB_TYPE", "data-prep"),
        help="W&B job type.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", "online"),
        choices=("online", "offline", "disabled"),
        help="W&B mode. Use offline for disconnected runs.",
    )
    parser.add_argument(
        "--wandb-dir",
        default=os.environ.get("WANDB_DIR"),
        help="W&B local directory. Defaults to a wandb directory beside the output directory.",
    )
    parser.add_argument(
        "--wandb-resume",
        default=os.environ.get("WANDB_RESUME", "allow"),
        choices=("allow", "must", "never", "auto"),
        help="W&B resume policy.",
    )
    parser.add_argument(
        "--wandb-save-code",
        dest="wandb_save_code",
        action="store_true",
        help="Save source code to W&B.",
    )
    parser.add_argument(
        "--no-wandb-save-code",
        dest="wandb_save_code",
        action="store_false",
        help="Disable W&B code saving.",
    )
    parser.set_defaults(wandb_save_code=True)
    return parser.parse_args()


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_local_jsonl(path: Path, seed: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"input JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(record)

    random.Random(seed).shuffle(rows)
    return rows


def _load_stream(cfg: Config, seed: int, shuffle_buffer: int, input_jsonl: str | None):
    if input_jsonl:
        return _load_local_jsonl(Path(input_jsonl), seed=seed)

    dataset = load_dataset(
        cfg.data.dataset_name,
        cfg.data.dataset_config,
        split="train",
        streaming=True,
    )
    return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)


def sanitize_wandb_id(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    sanitized = sanitized.strip("-_")
    return sanitized[:96] or "run"


def default_wandb_run_id(cfg: Config, out_dir: Path) -> str:
    digest = hashlib.sha1(f"data-prep:{out_dir.resolve()}".encode("utf-8")).hexdigest()[:12]
    return sanitize_wandb_id(f"{cfg.experiment.name}-data-prep-{digest}")


def default_wandb_run_name(cfg: Config, out_dir: Path) -> str:
    if out_dir.name:
        return f"{cfg.experiment.name}-data-prep-{out_dir.name}"
    return f"{cfg.experiment.name}-data-prep"


def parse_wandb_tags(value: str, cfg: Config, input_jsonl: str | None) -> list[str]:
    tags = [tag.strip() for tag in value.split(",") if tag.strip()]
    if tags:
        return tags
    source_tag = "local-jsonl" if input_jsonl else "huggingface-streaming"
    return ["data-prep", "tokenization", cfg.experiment.name, source_tag]


def init_wandb(
    args: argparse.Namespace,
    cfg: Config,
    out_dir: Path,
    input_jsonl: str | None,
    train_target: int,
    val_target: int,
    shard_tokens: int,
) -> Any | None:
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking was requested, but the wandb package is not installed. "
            "Install project dependencies or pass --no-wandb."
        ) from exc

    run_id = getattr(args, "wandb_run_id", None) or default_wandb_run_id(cfg, out_dir)
    run_name = getattr(args, "wandb_run_name", None) or default_wandb_run_name(cfg, out_dir)
    wandb_dir = Path(args.wandb_dir) if getattr(args, "wandb_dir", None) else out_dir.parent / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=run_id,
        name=run_name,
        group=args.wandb_group or cfg.experiment.name,
        job_type=args.wandb_job_type,
        tags=parse_wandb_tags(args.wandb_tags, cfg, input_jsonl),
        mode=args.wandb_mode,
        resume=args.wandb_resume,
        dir=os.fspath(wandb_dir),
        config={
            **cfg.raw,
            "runtime": {
                "out_dir": os.fspath(out_dir.resolve()),
                "input_jsonl": os.fspath(Path(input_jsonl).resolve()) if input_jsonl else None,
                "train_target_tokens": int(train_target),
                "validation_target_tokens": int(val_target),
                "total_target_tokens": int(train_target + val_target),
                "shard_tokens": int(shard_tokens),
                "log_interval_docs": int(getattr(args, "log_interval_docs", 1000)),
            },
        },
        save_code=args.wandb_save_code,
    )
    run.define_metric("total_tokens")
    run.define_metric("*", step_metric="total_tokens")
    run.define_metric("progress_tokens_fraction", summary="last")
    run.define_metric("tokens_per_sec", summary="max")
    run.define_metric("docs_per_sec", summary="max")
    run.define_metric("train_tokens", summary="last")
    run.define_metric("val_tokens", summary="last")
    print(
        f"wandb enabled project={args.wandb_project} run_id={run_id} "
        f"mode={args.wandb_mode} url={getattr(run, 'url', None) or 'n/a'}",
        flush=True,
    )
    return run


def progress_record(
    record_type: str,
    started: float,
    rows_seen: int,
    docs_kept: int,
    train_writer: ShardWriter,
    val_writer: ShardWriter,
    train_target: int,
    val_target: int,
) -> dict[str, Any]:
    elapsed = max(time.time() - started, 1e-6)
    train_tokens = int(train_writer.total_tokens)
    val_tokens = int(val_writer.total_tokens)
    total_tokens = train_tokens + val_tokens
    total_target = train_target + val_target
    return {
        "type": record_type,
        "elapsed_seconds": elapsed,
        "rows_seen": int(rows_seen),
        "docs_kept": int(docs_kept),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": int(total_tokens),
        "target_tokens": int(total_target),
        "progress_tokens_fraction": min(total_tokens / total_target, 1.0) if total_target else 0.0,
        "tokens_per_sec": total_tokens / elapsed,
        "docs_per_sec": docs_kept / elapsed,
        "train_shards": len(train_writer.shards),
        "val_shards": len(val_writer.shards),
    }


def log_wandb_record(wandb_run: Any | None, record: dict[str, Any]) -> None:
    if wandb_run is None:
        return
    payload = {key: value for key, value in record.items() if key != "type"}
    payload["event_type"] = str(record.get("type", "progress"))
    wandb_run.log(payload)


def finish_wandb(wandb_run: Any | None, final_record: dict[str, Any], manifest_path: Path) -> None:
    if wandb_run is None:
        return
    for key, value in final_record.items():
        if key == "type":
            continue
        wandb_run.summary[key] = value
    wandb_run.summary["manifest_path"] = os.fspath(manifest_path)
    wandb_run.finish()


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
    input_jsonl = args.input_jsonl
    log_interval_docs = getattr(args, "log_interval_docs", 1000)
    if log_interval_docs <= 0:
        raise ValueError("log_interval_docs must be positive")

    encoder = tiktoken.get_encoding(cfg.tokenizer.name)
    train_writer = ShardWriter(out_dir, "train", shard_tokens)
    val_writer = ShardWriter(out_dir, "val", shard_tokens)
    wandb_run = init_wandb(
        args,
        cfg,
        out_dir,
        input_jsonl,
        train_target,
        val_target,
        shard_tokens,
    )

    doc_manifest_path = out_dir / "documents.jsonl.gz"
    started = time.time()
    docs_kept = 0
    rows_seen = 0

    with gzip.open(doc_manifest_path, "wt", encoding="utf-8") as doc_manifest:
        for source_index, row in enumerate(
            _load_stream(cfg, seed=seed, shuffle_buffer=shuffle_buffer, input_jsonl=input_jsonl)
        ):
            rows_seen = source_index + 1
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

            docs_kept += 1
            record = {
                "source_index_after_shuffle": int(source_index),
                "split": split,
                "sha1": _sha1_text(text),
                "token_count": int(len(tokens)),
            }
            doc_manifest.write(json.dumps(record, sort_keys=True) + "\n")

            if docs_kept % log_interval_docs == 0:
                progress = progress_record(
                    "progress",
                    started,
                    rows_seen,
                    docs_kept,
                    train_writer,
                    val_writer,
                    train_target,
                    val_target,
                )
                print(
                    "prepared "
                    f"docs={docs_kept} "
                    f"train_tokens={train_writer.total_tokens:,} "
                    f"val_tokens={val_writer.total_tokens:,} "
                    f"docs/sec={progress['docs_per_sec']:.1f}",
                    flush=True,
                )
                log_wandb_record(wandb_run, progress)

    train_writer.flush()
    val_writer.flush()

    dataset_info = {
        "name": cfg.data.dataset_name,
        "config": cfg.data.dataset_config,
        "split": "train",
        "text_field": cfg.data.text_field,
    }
    if input_jsonl:
        dataset_info["local_jsonl"] = str(Path(input_jsonl).resolve())

    manifest = {
        "version": 1,
        "created_unix": int(time.time()),
        "dataset": dataset_info,
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

    final_record = progress_record(
        "final",
        started,
        rows_seen,
        docs_kept,
        train_writer,
        val_writer,
        train_target,
        val_target,
    )
    log_wandb_record(wandb_run, final_record)
    finish_wandb(wandb_run, final_record, manifest_path)

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
