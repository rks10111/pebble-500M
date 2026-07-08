from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import tiktoken
from datasets import load_dataset


DEFAULT_DATASET = "HuggingFaceTB/smol-smoltalk"
DEFAULT_SYSTEM_PROMPT = "You are Pebble, a helpful assistant."
CHAT_TOKENS = {
    "<|system|>": 50257,
    "<|user|>": 50258,
    "<|assistant|>": 50259,
    "<|end|>": 50260,
}
ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
}
ROLE_TOKEN_NAMES = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


@dataclass
class SourceStats:
    rows_seen: int = 0
    examples: int = 0
    tokens: int = 0
    assistant_tokens: int = 0
    max_tokens: int = 0
    multi_turn_examples: int = 0
    skipped_no_messages: int = 0
    skipped_no_assistant: int = 0
    skipped_no_assistant_tokens: int = 0
    skipped_too_long: int = 0

    def add_kept(self, token_count: int, assistant_token_count: int, *, multi_turn: bool) -> None:
        self.examples += 1
        self.tokens += token_count
        self.assistant_tokens += assistant_token_count
        self.max_tokens = max(self.max_tokens, token_count)
        if multi_turn:
            self.multi_turn_examples += 1

    def as_manifest(self) -> dict[str, int]:
        return {
            "rows_seen": int(self.rows_seen),
            "examples": int(self.examples),
            "tokens": int(self.tokens),
            "assistant_tokens": int(self.assistant_tokens),
            "max_tokens": int(self.max_tokens),
            "multi_turn_examples": int(self.multi_turn_examples),
            "skipped_no_messages": int(self.skipped_no_messages),
            "skipped_no_assistant": int(self.skipped_no_assistant),
            "skipped_no_assistant_tokens": int(self.skipped_no_assistant_tokens),
            "skipped_too_long": int(self.skipped_too_long),
        }


@dataclass
class SplitStats:
    rows_seen: int = 0
    examples: int = 0
    tokens: int = 0
    assistant_tokens: int = 0
    max_tokens: int = 0
    multi_turn_examples: int = 0
    skipped_no_messages: int = 0
    skipped_no_assistant: int = 0
    skipped_no_assistant_tokens: int = 0
    skipped_too_long: int = 0
    lengths: list[int] = field(default_factory=list)
    sources: dict[str, SourceStats] = field(default_factory=dict)

    def source(self, name: str) -> SourceStats:
        if name not in self.sources:
            self.sources[name] = SourceStats()
        return self.sources[name]

    def add_kept(
        self,
        source_name: str,
        token_count: int,
        assistant_token_count: int,
        *,
        multi_turn: bool,
    ) -> None:
        self.examples += 1
        self.tokens += token_count
        self.assistant_tokens += assistant_token_count
        self.max_tokens = max(self.max_tokens, token_count)
        if multi_turn:
            self.multi_turn_examples += 1
        self.lengths.append(token_count)
        self.source(source_name).add_kept(token_count, assistant_token_count, multi_turn=multi_turn)

    def add_skip(self, source_name: str, reason: str) -> None:
        source = self.source(source_name)
        if reason == "no_messages":
            self.skipped_no_messages += 1
            source.skipped_no_messages += 1
        elif reason == "no_assistant":
            self.skipped_no_assistant += 1
            source.skipped_no_assistant += 1
        elif reason == "no_assistant_tokens":
            self.skipped_no_assistant_tokens += 1
            source.skipped_no_assistant_tokens += 1
        elif reason == "too_long":
            self.skipped_too_long += 1
            source.skipped_too_long += 1
        else:
            raise ValueError(f"unknown skip reason {reason!r}")

    def quantile(self, percentile: float) -> int:
        if not self.lengths:
            return 0
        return int(np.percentile(np.asarray(self.lengths), percentile))

    def as_manifest(self, writer: MaskedShardWriter, target_tokens: int | None) -> dict[str, Any]:
        return {
            "target_tokens": int(target_tokens) if target_tokens is not None else None,
            "tokens": int(writer.total_tokens),
            "assistant_tokens": int(self.assistant_tokens),
            "examples": int(self.examples),
            "rows_seen": int(self.rows_seen),
            "skipped_no_messages": int(self.skipped_no_messages),
            "skipped_no_assistant": int(self.skipped_no_assistant),
            "skipped_no_assistant_tokens": int(self.skipped_no_assistant_tokens),
            "skipped_too_long": int(self.skipped_too_long),
            "max_example_tokens": int(self.max_tokens),
            "p50_example_tokens": self.quantile(50),
            "p95_example_tokens": self.quantile(95),
            "p99_example_tokens": self.quantile(99),
            "multi_turn_examples": int(self.multi_turn_examples),
            "sources": {name: stats.as_manifest() for name, stats in sorted(self.sources.items())},
            "example_index": f"example_index/{writer.split}.jsonl.gz",
            "shards": writer.shards,
        }


class MaskedShardWriter:
    def __init__(self, base_dir: Path, split: str, shard_tokens: int) -> None:
        self.base_dir = base_dir
        self.split = split
        self.token_dir = base_dir / "tokens" / split
        self.mask_dir = base_dir / "loss_masks" / split
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.token_buffer = np.empty(shard_tokens, dtype=np.uint16)
        self.mask_buffer = np.empty(shard_tokens, dtype=np.uint8)
        self.offset = 0
        self.index = 0
        self.total_tokens = 0
        self.shards: list[dict[str, Any]] = []

    def add(self, tokens: list[int], loss_mask: list[int]) -> tuple[int, int]:
        if not tokens:
            raise ValueError("cannot add an empty SFT example")
        if len(tokens) != len(loss_mask):
            raise ValueError("tokens and loss mask lengths differ")
        if len(tokens) > self.shard_tokens:
            raise ValueError(
                f"example has {len(tokens)} tokens, larger than shard_tokens={self.shard_tokens}"
            )
        if self.offset and self.offset + len(tokens) > self.shard_tokens:
            self.flush()

        shard_index = self.index
        offset = self.offset
        token_array = np.asarray(tokens, dtype=np.uint16)
        mask_array = np.asarray(loss_mask, dtype=np.uint8)
        end = self.offset + len(tokens)
        self.token_buffer[self.offset : end] = token_array
        self.mask_buffer[self.offset : end] = mask_array
        self.offset = end
        self.total_tokens += len(tokens)
        if self.offset == self.shard_tokens:
            self.flush()
        return shard_index, offset

    def flush(self) -> None:
        if self.offset == 0:
            return
        filename = f"{self.split}_{self.index:06d}.bin"
        token_path = self.token_dir / filename
        mask_path = self.mask_dir / filename
        self.token_buffer[: self.offset].tofile(token_path)
        self.mask_buffer[: self.offset].tofile(mask_path)
        self.shards.append(
            {
                "tokens_path": str(token_path.relative_to(self.base_dir)),
                "loss_mask_path": str(mask_path.relative_to(self.base_dir)),
                "tokens": int(self.offset),
                "token_bytes": int(self.offset * np.dtype(np.uint16).itemsize),
                "loss_mask_bytes": int(self.offset * np.dtype(np.uint8).itemsize),
            }
        )
        self.index += 1
        self.offset = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 1024-context masked chat/SFT shards.")
    parser.add_argument("--out-dir", required=True, help="Output directory for manifest, tokens, and masks.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET, help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--train-split", default="train", help="Dataset split to use for training.")
    parser.add_argument("--validation-split", default="test", help="Dataset split to use for validation.")
    parser.add_argument("--messages-field", default="messages", help="Field containing chat messages.")
    parser.add_argument("--tokenizer", default="gpt2", help="tiktoken tokenizer name.")
    parser.add_argument(
        "--max-sequence-tokens",
        type=int,
        default=1024,
        help="Drop formatted examples longer than this many tokens.",
    )
    parser.add_argument("--shard-tokens", type=int, default=10_000_000, help="Tokens per output shard.")
    parser.add_argument(
        "--default-system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt inserted when an example has no system message.",
    )
    parser.add_argument(
        "--no-default-system-prompt",
        action="store_true",
        help="Do not insert a default system prompt.",
    )
    parser.add_argument("--train-target-tokens", type=int, default=None, help="Optional train token cap.")
    parser.add_argument(
        "--validation-target-tokens",
        type=int,
        default=None,
        help="Optional validation token cap.",
    )
    parser.add_argument("--max-train-examples", type=int, default=None, help="Optional train example cap.")
    parser.add_argument(
        "--max-validation-examples",
        type=int,
        default=None,
        help="Optional validation example cap.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use Hugging Face streaming mode instead of downloading the split locally first.",
    )
    parser.add_argument(
        "--no-hf-dataset",
        action="store_true",
        help="Only use local JSONL sources passed with --extra-train-jsonl/--extra-val-jsonl.",
    )
    parser.add_argument(
        "--extra-train-jsonl",
        action="append",
        default=[],
        help="Extra train messages JSONL or JSONL.GZ source. Can be repeated.",
    )
    parser.add_argument(
        "--extra-val-jsonl",
        action="append",
        default=[],
        help="Extra validation messages JSONL or JSONL.GZ source. Can be repeated.",
    )
    parser.add_argument(
        "--log-interval-rows",
        type=int,
        default=10_000,
        help="Progress log interval in source rows. Set 0 to disable.",
    )
    return parser.parse_args()


def sha1_messages(messages: list[tuple[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def normalize_role(role: Any) -> str:
    normalized = str(role or "").strip().lower()
    return ROLE_ALIASES.get(normalized, normalized)


def message_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return ""


def normalized_messages(messages: Any, default_system_prompt: str | None = None) -> list[tuple[str, str]]:
    if not isinstance(messages, list):
        return []

    normalized: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role") or message.get("from"))
        if role not in ROLE_TOKEN_NAMES:
            continue
        content = message_content(message.get("content") or message.get("value"))
        if content:
            normalized.append((role, content))

    if default_system_prompt and normalized and not any(role == "system" for role, _ in normalized):
        normalized = [("system", default_system_prompt.strip())] + normalized
    return normalized


def encode_messages(
    messages: Any,
    *,
    encoder: tiktoken.Encoding,
    default_system_prompt: str | None,
) -> tuple[list[int], list[int], dict[str, int], list[tuple[str, str]]]:
    items = normalized_messages(messages, default_system_prompt)
    if not items:
        return [], [], {"assistant_turns": 0, "user_turns": 0}, []

    tokens: list[int] = []
    loss_mask: list[int] = []
    assistant_turns = 0
    user_turns = 0
    for role, content in items:
        role_token = CHAT_TOKENS[ROLE_TOKEN_NAMES[role]]
        content_tokens = encoder.encode_ordinary(content)
        is_assistant = role == "assistant"
        if role == "assistant":
            assistant_turns += 1
        elif role == "user":
            user_turns += 1

        message_tokens = [role_token, *content_tokens, CHAT_TOKENS["<|end|>"]]
        tokens.extend(message_tokens)
        loss_mask.extend([1 if is_assistant else 0] * len(message_tokens))

    return tokens, loss_mask, {"assistant_turns": assistant_turns, "user_turns": user_turns}, items


def load_split(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    *,
    streaming: bool,
) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, **kwargs)
    return load_dataset(dataset_name, **kwargs)


def open_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def source_name_for_dataset(dataset_name: str, dataset_config: str | None) -> str:
    base = dataset_name.rsplit("/", 1)[-1]
    return f"{base}:{dataset_config}" if dataset_config else base


def source_name_for_path(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")]
    elif name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return name


def process_rows(
    rows: Iterable[dict[str, Any]],
    *,
    split: str,
    source_name: str,
    writer: MaskedShardWriter,
    example_index: Any,
    encoder: tiktoken.Encoding,
    messages_field: str,
    default_system_prompt: str | None,
    max_sequence_tokens: int,
    max_examples: int | None,
    target_tokens: int | None,
    log_interval_rows: int,
    stats: SplitStats | None = None,
) -> SplitStats:
    split_stats = stats or SplitStats()
    source_stats = split_stats.source(source_name)
    started = time.time()

    for source_index, row in enumerate(rows):
        if max_examples is not None and split_stats.examples >= max_examples:
            break
        if target_tokens is not None and writer.total_tokens >= target_tokens:
            break
        split_stats.rows_seen += 1
        source_stats.rows_seen += 1

        messages = row.get(messages_field) if isinstance(row, dict) else None
        tokens, loss_mask, turn_counts, normalized = encode_messages(
            messages,
            encoder=encoder,
            default_system_prompt=default_system_prompt,
        )
        if not normalized:
            split_stats.add_skip(source_name, "no_messages")
            continue
        if turn_counts["assistant_turns"] <= 0:
            split_stats.add_skip(source_name, "no_assistant")
            continue
        assistant_token_count = int(sum(loss_mask))
        if assistant_token_count <= 0:
            split_stats.add_skip(source_name, "no_assistant_tokens")
            continue
        if len(tokens) > max_sequence_tokens:
            split_stats.add_skip(source_name, "too_long")
            continue
        if target_tokens is not None and writer.total_tokens + len(tokens) > target_tokens:
            break
        if max(tokens) >= np.iinfo(np.uint16).max:
            raise ValueError("token id exceeded uint16 storage")

        shard_index, offset = writer.add(tokens, loss_mask)
        multi_turn = turn_counts["assistant_turns"] > 1 or turn_counts["user_turns"] > 1
        split_stats.add_kept(
            source_name,
            len(tokens),
            assistant_token_count,
            multi_turn=multi_turn,
        )
        source_index_value = row.get("source_index")
        if source_index_value is None:
            source_index_value = source_index
        example_index.write(
            json.dumps(
                {
                    "split": split,
                    "source": row.get("source") or source_name,
                    "source_dataset": row.get("source_dataset"),
                    "source_config": row.get("source_config"),
                    "source_split": row.get("source_split"),
                    "source_index": int(source_index_value),
                    "category": row.get("category"),
                    "sha1": row.get("sha1") or sha1_messages(normalized),
                    "shard_index": int(shard_index),
                    "offset": int(offset),
                    "tokens": int(len(tokens)),
                    "token_count": int(len(tokens)),
                    "assistant_tokens": int(assistant_token_count),
                    "assistant_token_count": int(assistant_token_count),
                    "assistant_turns": int(turn_counts["assistant_turns"]),
                    "user_turns": int(turn_counts["user_turns"]),
                },
                sort_keys=True,
            )
            + "\n"
        )

        if log_interval_rows and source_stats.rows_seen % log_interval_rows == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"{split}/{source_name}: rows={source_stats.rows_seen:,} "
                f"kept={source_stats.examples:,} total_examples={split_stats.examples:,} "
                f"tokens={writer.total_tokens:,} rows/sec={source_stats.rows_seen / elapsed:.1f}",
                flush=True,
            )

    writer.flush()
    return split_stats


def validate_args(args: argparse.Namespace) -> None:
    if args.max_sequence_tokens <= 1:
        raise ValueError("max-sequence-tokens must be greater than 1")
    if args.shard_tokens <= 0:
        raise ValueError("shard-tokens must be positive")
    if args.max_sequence_tokens > args.shard_tokens:
        raise ValueError("max-sequence-tokens cannot exceed shard-tokens")
    if args.log_interval_rows < 0:
        raise ValueError("log-interval-rows must be non-negative")
    if args.no_hf_dataset and not args.extra_train_jsonl:
        raise ValueError("--no-hf-dataset requires at least one --extra-train-jsonl")
    for value_name in ("train_target_tokens", "validation_target_tokens", "max_train_examples", "max_validation_examples"):
        value = getattr(args, value_name)
        if value is not None and value <= 0:
            raise ValueError(f"{value_name.replace('_', '-')} must be positive")
    for token_name, token_id in CHAT_TOKENS.items():
        if token_id >= np.iinfo(np.uint16).max:
            raise ValueError(f"{token_name} id does not fit in uint16")


def prepare(args: argparse.Namespace) -> Path:
    validate_args(args)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"out-dir must be empty or absent: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = tiktoken.get_encoding(args.tokenizer)
    default_system_prompt = None if args.no_default_system_prompt else args.default_system_prompt
    train_writer = MaskedShardWriter(out_dir, "train", args.shard_tokens)
    val_writer = MaskedShardWriter(out_dir, "val", args.shard_tokens)
    example_index_dir = out_dir / "example_index"
    example_index_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with gzip.open(example_index_dir / "train.jsonl.gz", "wt", encoding="utf-8") as train_index:
        train_stats = SplitStats()
        if not args.no_hf_dataset:
            train_stats = process_rows(
                load_split(
                    args.dataset_name,
                    args.dataset_config,
                    args.train_split,
                    streaming=args.streaming,
                ),
                split="train",
                source_name=source_name_for_dataset(args.dataset_name, args.dataset_config),
                writer=train_writer,
                example_index=train_index,
                encoder=encoder,
                messages_field=args.messages_field,
                default_system_prompt=default_system_prompt,
                max_sequence_tokens=args.max_sequence_tokens,
                max_examples=args.max_train_examples,
                target_tokens=args.train_target_tokens,
                log_interval_rows=args.log_interval_rows,
                stats=train_stats,
            )
        for path_text in args.extra_train_jsonl:
            path = Path(path_text)
            train_stats = process_rows(
                open_jsonl(path),
                split="train",
                source_name=source_name_for_path(path),
                writer=train_writer,
                example_index=train_index,
                encoder=encoder,
                messages_field=args.messages_field,
                default_system_prompt=default_system_prompt,
                max_sequence_tokens=args.max_sequence_tokens,
                max_examples=args.max_train_examples,
                target_tokens=args.train_target_tokens,
                log_interval_rows=args.log_interval_rows,
                stats=train_stats,
            )

    with gzip.open(example_index_dir / "val.jsonl.gz", "wt", encoding="utf-8") as val_index:
        val_stats = SplitStats()
        if not args.no_hf_dataset:
            val_stats = process_rows(
                load_split(
                    args.dataset_name,
                    args.dataset_config,
                    args.validation_split,
                    streaming=args.streaming,
                ),
                split="val",
                source_name=source_name_for_dataset(args.dataset_name, args.dataset_config),
                writer=val_writer,
                example_index=val_index,
                encoder=encoder,
                messages_field=args.messages_field,
                default_system_prompt=default_system_prompt,
                max_sequence_tokens=args.max_sequence_tokens,
                max_examples=args.max_validation_examples,
                target_tokens=args.validation_target_tokens,
                log_interval_rows=args.log_interval_rows,
                stats=val_stats,
            )
        for path_text in args.extra_val_jsonl:
            path = Path(path_text)
            val_stats = process_rows(
                open_jsonl(path),
                split="val",
                source_name=source_name_for_path(path),
                writer=val_writer,
                example_index=val_index,
                encoder=encoder,
                messages_field=args.messages_field,
                default_system_prompt=default_system_prompt,
                max_sequence_tokens=args.max_sequence_tokens,
                max_examples=args.max_validation_examples,
                target_tokens=args.validation_target_tokens,
                log_interval_rows=args.log_interval_rows,
                stats=val_stats,
            )

    if train_stats.examples <= 0:
        raise ValueError("train split produced no examples")
    if val_stats.examples <= 0:
        raise ValueError("validation split produced no examples")

    manifest = {
        "version": 2,
        "task": "chat-sft-assistant-mask",
        "created_unix": int(time.time()),
        "dataset": {
            "name": None if args.no_hf_dataset else args.dataset_name,
            "config": None if args.no_hf_dataset else args.dataset_config,
            "train_split": None if args.no_hf_dataset else args.train_split,
            "validation_split": None if args.no_hf_dataset else args.validation_split,
            "messages_field": args.messages_field,
            "streaming": bool(args.streaming),
            "extra_train_jsonl": [str(Path(path)) for path in args.extra_train_jsonl],
            "extra_val_jsonl": [str(Path(path)) for path in args.extra_val_jsonl],
        },
        "formatting": {
            "template": "<|role|> content <|end|> per message",
            "default_system_prompt": default_system_prompt,
            "assistant_only_loss": True,
            "assistant_role_token_loss": True,
            "assistant_end_token_loss": True,
        },
        "filtering": {
            "max_sequence_tokens": int(args.max_sequence_tokens),
            "max_train_examples": args.max_train_examples,
            "max_validation_examples": args.max_validation_examples,
        },
        "tokenizer": {
            "name": args.tokenizer,
            "chat_tokens": CHAT_TOKENS,
            "storage_dtype": "uint16",
            "loss_mask_dtype": "uint8",
        },
        "elapsed_seconds": time.time() - started,
        "splits": {
            "train": train_stats.as_manifest(train_writer, args.train_target_tokens),
            "val": val_stats.as_manifest(val_writer, args.validation_target_tokens),
        },
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "finished "
        f"train_examples={train_stats.examples:,} train_tokens={train_writer.total_tokens:,} "
        f"train_assistant_tokens={train_stats.assistant_tokens:,} "
        f"val_examples={val_stats.examples:,} val_tokens={val_writer.total_tokens:,} "
        f"val_assistant_tokens={val_stats.assistant_tokens:,} manifest={manifest_path}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
