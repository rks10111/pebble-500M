from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_sft import ExampleRef, SFTTrainCursor, validate_resume_metadata


def test_train_cursor_restores_exact_order_and_position() -> None:
    split = SimpleNamespace(examples=[ExampleRef(0, index, 2, 1) for index in range(10)])
    cursor = SFTTrainCursor(split, batch_size=3, max_examples=10, rng=np.random.default_rng(123))

    first_refs = cursor.next_refs()
    assert first_refs is not None
    state = cursor.state_dict()
    expected_next_refs = cursor.next_refs()
    assert expected_next_refs is not None

    resumed = SFTTrainCursor(split, batch_size=3, max_examples=10, rng=np.random.default_rng(999))
    resumed.load_state_dict(state)
    actual_next_refs = resumed.next_refs()

    assert actual_next_refs == expected_next_refs


def test_resume_metadata_rejects_dataset_config_or_runtime_mismatch() -> None:
    state = {
        "dataset_manifest_hash": "dataset-a",
        "config_hash": "config-a",
        "runtime_args": {"micro_batch_size": 2, "grad_accum_steps": 4, "precision": "bf16"},
    }

    validate_resume_metadata(
        state,
        manifest_hash="dataset-a",
        config_hash="config-a",
        runtime_args={"micro_batch_size": 2, "grad_accum_steps": 4, "precision": "bf16"},
        allow_unsafe=False,
    )
    with pytest.raises(ValueError, match="unsafe SFT resume"):
        validate_resume_metadata(
            state,
            manifest_hash="dataset-b",
            config_hash="config-a",
            runtime_args={"micro_batch_size": 2, "grad_accum_steps": 4, "precision": "bf16"},
            allow_unsafe=False,
        )
    with pytest.raises(ValueError, match="micro_batch_size differs"):
        validate_resume_metadata(
            state,
            manifest_hash="dataset-a",
            config_hash="config-a",
            runtime_args={"micro_batch_size": 8, "grad_accum_steps": 4, "precision": "bf16"},
            allow_unsafe=False,
        )
    validate_resume_metadata(
        state,
        manifest_hash="dataset-b",
        config_hash="config-b",
        runtime_args={"micro_batch_size": 8, "grad_accum_steps": 4, "precision": "bf16"},
        allow_unsafe=True,
    )
