from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import tiktoken

from pebble.prepare_sft_data import CHAT_TOKENS, MaskedShardWriter, encode_messages, process_rows


def test_encode_messages_adds_default_system_prompt_and_masks_assistant_tokens() -> None:
    encoder = tiktoken.get_encoding("gpt2")
    tokens, loss_mask, turns, normalized = encode_messages(
        [
            {"from": "human", "value": "Give me one fact."},
            {"from": "gpt", "value": "Water freezes at 0 C."},
        ],
        encoder=encoder,
        default_system_prompt="You are Pebble.",
    )

    assert normalized[0] == ("system", "You are Pebble.")
    assert normalized[1] == ("user", "Give me one fact.")
    assert normalized[2] == ("assistant", "Water freezes at 0 C.")
    assert turns == {"assistant_turns": 1, "user_turns": 1}
    assert tokens[0] == CHAT_TOKENS["<|system|>"]
    assert CHAT_TOKENS["<|user|>"] in tokens
    assistant_start = tokens.index(CHAT_TOKENS["<|assistant|>"])
    assert all(value == 0 for value in loss_mask[:assistant_start])
    assert all(value == 1 for value in loss_mask[assistant_start:])
    assert tokens[-1] == CHAT_TOKENS["<|end|>"]


def test_process_rows_writes_tokens_masks_and_example_index(tmp_path: Path) -> None:
    encoder = tiktoken.get_encoding("gpt2")
    writer = MaskedShardWriter(tmp_path, "train", shard_tokens=64)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": "Hi."},
            ]
        },
        {"messages": [{"role": "user", "content": "No assistant response."}]},
        {
            "messages": [
                {"role": "user", "content": "word " * 100},
                {"role": "assistant", "content": "done"},
            ]
        },
        {"text": "not a chat row"},
    ]

    example_index_path = tmp_path / "example_index.jsonl.gz"
    with gzip.open(example_index_path, "wt", encoding="utf-8") as handle:
        stats = process_rows(
            rows,
            split="train",
            source_name="unit-test",
            writer=writer,
            example_index=handle,
            encoder=encoder,
            messages_field="messages",
            default_system_prompt="You are Pebble.",
            max_sequence_tokens=32,
            max_examples=None,
            target_tokens=None,
            log_interval_rows=0,
        )

    assert stats.rows_seen == 4
    assert stats.examples == 1
    assert stats.skipped_no_assistant == 1
    assert stats.skipped_too_long == 1
    assert stats.skipped_no_messages == 1
    assert writer.total_tokens > 0
    assert (tmp_path / "tokens" / "train" / "train_000000.bin").is_file()
    assert (tmp_path / "loss_masks" / "train" / "train_000000.bin").is_file()

    tokens = np.fromfile(tmp_path / "tokens" / "train" / "train_000000.bin", dtype=np.uint16)
    masks = np.fromfile(tmp_path / "loss_masks" / "train" / "train_000000.bin", dtype=np.uint8)
    assert len(tokens) == len(masks) == writer.total_tokens
    assert tokens[0] == CHAT_TOKENS["<|system|>"]
    assert masks.sum() > 0

    with gzip.open(example_index_path, "rt", encoding="utf-8") as handle:
        examples = [json.loads(line) for line in handle]
    assert len(examples) == 1
    assert examples[0]["split"] == "train"
    assert examples[0]["tokens"] == writer.total_tokens
    assert examples[0]["assistant_token_count"] == int(masks.sum())
