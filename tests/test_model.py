from __future__ import annotations

import torch
import pytest

from pebble.config import load_config
from pebble.model import Transformer, lr_for_tokens
from pebble.train import grad_accum_steps


def test_transformer_forward_smoke() -> None:
    cfg = load_config("configs/smoke.yaml")
    model = Transformer(cfg.model)
    idx = torch.randint(0, cfg.model.vocab_size, (2, cfg.model.context_length))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, cfg.model.context_length, cfg.model.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)


def test_transformer_forward_can_skip_returning_logits() -> None:
    cfg = load_config("configs/smoke.yaml")
    model = Transformer(cfg.model)
    idx = torch.randint(0, cfg.model.vocab_size, (2, cfg.model.context_length))

    logits, full_loss = model(idx, idx)
    skipped_logits, loss_only = model(idx, idx, return_logits=False)

    assert logits is not None
    assert skipped_logits is None
    assert full_loss is not None
    assert loss_only is not None
    assert torch.testing.assert_close(loss_only, full_loss) is None


def test_lr_schedule_warms_and_decays() -> None:
    warm = lr_for_tokens(256, base_lr=1e-3, min_lr=1e-4, warmup_tokens=512, target_tokens=2048)
    peak = lr_for_tokens(512, base_lr=1e-3, min_lr=1e-4, warmup_tokens=512, target_tokens=2048)
    late = lr_for_tokens(2048, base_lr=1e-3, min_lr=1e-4, warmup_tokens=512, target_tokens=2048)
    assert 0.0 < warm < peak
    assert peak == 1e-3
    assert late == 1e-4


def test_grad_accum_steps_requires_exact_global_batch() -> None:
    assert grad_accum_steps(global_batch_tokens=524288, micro_batch_size=64, context_length=1024) == 8

    with pytest.raises(ValueError, match="exactly divisible"):
        grad_accum_steps(global_batch_tokens=524288, micro_batch_size=48, context_length=1024)
