from __future__ import annotations

import sys

from pebble.train import parse_args, wandb_payload


def test_wandb_tracking_defaults_to_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pebble-train",
            "--config",
            "configs/smoke.yaml",
            "--data-dir",
            "/tmp/data",
            "--out-dir",
            "/tmp/run",
        ],
    )

    args = parse_args()

    assert args.wandb is True
    assert args.wandb_project == "pebble-500m"
    assert args.wandb_mode == "online"
    assert args.wandb_resume == "allow"
    assert args.wandb_watch is False
    assert args.wandb_watch_log == "all"
    assert args.wandb_save_code is True
    assert args.s3_sync_uri is None
    assert args.s3_sync_region == "eu-west-2"
    assert args.no_s3_sync is False


def test_train_s3_sync_can_default_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("PEBBLE_S3_RUN_URI", "s3://statement-llm-training/pebble-500m/runs/env")
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pebble-train",
            "--config",
            "configs/smoke.yaml",
            "--data-dir",
            "/tmp/data",
            "--out-dir",
            "/tmp/run",
        ],
    )

    args = parse_args()

    assert args.s3_sync_uri == "s3://statement-llm-training/pebble-500m/runs/env"
    assert args.s3_sync_region == "ap-south-1"


def test_wandb_payload_prefixes_train_metrics() -> None:
    payload = wandb_payload(
        {
            "type": "train",
            "step": 2,
            "tokens_seen": 1024,
            "loss": 6.0,
            "lr": 1.0e-3,
            "tokens_per_sec_interval": 50000.0,
            "gpu_memory_allocated_gb": 10.0,
            "estimated_aws_cost_usd": 1.25,
        },
        max_tokens=2048,
    )

    assert payload["training_step"] == 2
    assert payload["tokens_seen"] == 1024
    assert payload["progress_tokens_fraction"] == 0.5
    assert payload["train_loss"] == 6.0
    assert payload["train_lr"] == 1.0e-3
    assert payload["train_perplexity"] is not None
    assert payload["throughput_tokens_per_sec_interval"] == 50000.0
    assert payload["system_gpu_memory_allocated_gb"] == 10.0
    assert payload["cost_estimated_aws_cost_usd"] == 1.25


def test_wandb_payload_logs_final_validation_to_validation_series() -> None:
    payload = wandb_payload(
        {
            "type": "final",
            "step": 4,
            "tokens_seen": 2048,
            "validation_loss": 5.5,
            "validation_perplexity": 244.69,
        },
        max_tokens=2048,
    )

    assert payload["final_validation_loss"] == 5.5
    assert payload["validation_loss"] == 5.5
    assert payload["final_validation_perplexity"] == 244.69
    assert payload["validation_perplexity"] == 244.69
