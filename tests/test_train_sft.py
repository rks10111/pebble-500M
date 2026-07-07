from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_sft import ExampleRef, SFTTrainCursor


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
