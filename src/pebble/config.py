from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(value)


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int
    n_embd: int
    n_head: int
    n_kv_head: int
    context_length: int
    vocab_size: int
    rope_theta: float
    norm_eps: float
    intermediate_size: int
    dropout: float
    bias: bool
    tie_embeddings: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        n_head = _as_int(data["n_head"])
        n_kv_head = _as_int(data.get("n_kv_head", n_head))
        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head for grouped-query attention")
        n_embd = _as_int(data["n_embd"])
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        return cls(
            n_layer=_as_int(data["n_layer"]),
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            context_length=_as_int(data["context_length"]),
            vocab_size=_as_int(data["vocab_size"]),
            rope_theta=_as_float(data.get("rope_theta", 10000.0)),
            norm_eps=_as_float(data.get("norm_eps", 1e-5)),
            intermediate_size=_as_int(data["intermediate_size"]),
            dropout=_as_float(data.get("dropout", 0.0)),
            bias=_as_bool(data.get("bias", False)),
            tie_embeddings=_as_bool(data.get("tie_embeddings", True)),
        )


@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    real_vocab_size: int
    model_vocab_size: int
    eod_token: int
    storage_dtype: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenizerConfig":
        return cls(
            name=str(data.get("name", "gpt2")),
            real_vocab_size=_as_int(data.get("real_vocab_size", 50257)),
            model_vocab_size=_as_int(data.get("model_vocab_size", 50304)),
            eod_token=_as_int(data.get("eod_token", 50256)),
            storage_dtype=str(data.get("storage_dtype", "uint16")),
        )


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    dataset_config: str
    text_field: str
    seed: int
    streaming_shuffle_buffer: int
    train_target_tokens: int
    validation_target_tokens: int
    shard_tokens: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        return cls(
            dataset_name=str(data["dataset_name"]),
            dataset_config=str(data["dataset_config"]),
            text_field=str(data.get("text_field", "text")),
            seed=_as_int(data.get("seed", 1337)),
            streaming_shuffle_buffer=_as_int(data.get("streaming_shuffle_buffer", 100000)),
            train_target_tokens=_as_int(data["train_target_tokens"]),
            validation_target_tokens=_as_int(data["validation_target_tokens"]),
            shard_tokens=_as_int(data["shard_tokens"]),
        )


@dataclass(frozen=True)
class TrainingConfig:
    precision: str
    global_batch_tokens: int
    micro_batch_size: int
    compile: bool
    activation_checkpointing: bool
    gradient_clip: float
    max_tokens: int
    eval_interval_tokens: int
    eval_tokens: int
    log_interval_steps: int
    sample_max_new_tokens: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        return cls(
            precision=str(data.get("precision", "bf16")),
            global_batch_tokens=_as_int(data["global_batch_tokens"]),
            micro_batch_size=_as_int(data["micro_batch_size"]),
            compile=_as_bool(data.get("compile", True)),
            activation_checkpointing=_as_bool(data.get("activation_checkpointing", False)),
            gradient_clip=_as_float(data.get("gradient_clip", 1.0)),
            max_tokens=_as_int(data["max_tokens"]),
            eval_interval_tokens=_as_int(data["eval_interval_tokens"]),
            eval_tokens=_as_int(data["eval_tokens"]),
            log_interval_steps=_as_int(data.get("log_interval_steps", 10)),
            sample_max_new_tokens=_as_int(data.get("sample_max_new_tokens", 80)),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    fused: bool
    lr: float
    betas: tuple[float, float]
    weight_decay: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizerConfig":
        betas = data.get("betas", [0.9, 0.95])
        return cls(
            name=str(data.get("name", "AdamW")),
            fused=_as_bool(data.get("fused", True)),
            lr=_as_float(data["lr"]),
            betas=(_as_float(betas[0]), _as_float(betas[1])),
            weight_decay=_as_float(data.get("weight_decay", 0.1)),
        )


@dataclass(frozen=True)
class ScheduleConfig:
    warmup_tokens: int
    planned_target_tokens: int
    min_lr: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduleConfig":
        return cls(
            warmup_tokens=_as_int(data["warmup_tokens"]),
            planned_target_tokens=_as_int(data["planned_target_tokens"]),
            min_lr=_as_float(data["min_lr"]),
        )


@dataclass(frozen=True)
class CheckpointConfig:
    save_interval_minutes: float
    keep_last: int
    milestones: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointConfig":
        return cls(
            save_interval_minutes=_as_float(data.get("save_interval_minutes", 45)),
            keep_last=_as_int(data.get("keep_last", 3)),
            milestones=tuple(_as_int(value) for value in data.get("milestones", [])),
        )


@dataclass(frozen=True)
class AwsConfig:
    instance_type: str
    hourly_usd: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AwsConfig":
        return cls(
            instance_type=str(data.get("instance_type", "unknown")),
            hourly_usd=_as_float(data.get("hourly_usd", 0.0)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(name=str(data.get("name", "pebble")), seed=_as_int(data.get("seed", 1337)))


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    experiment: ExperimentConfig
    aws: AwsConfig
    model: ModelConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    checkpointing: CheckpointConfig
    prompts: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        return cls(
            raw=raw,
            experiment=ExperimentConfig.from_dict(raw.get("experiment", {})),
            aws=AwsConfig.from_dict(raw.get("aws", {})),
            model=ModelConfig.from_dict(raw["model"]),
            tokenizer=TokenizerConfig.from_dict(raw.get("tokenizer", {})),
            data=DataConfig.from_dict(raw["data"]),
            training=TrainingConfig.from_dict(raw["training"]),
            optimizer=OptimizerConfig.from_dict(raw["optimizer"]),
            schedule=ScheduleConfig.from_dict(raw["schedule"]),
            checkpointing=CheckpointConfig.from_dict(raw.get("checkpointing", {})),
            prompts=tuple(str(prompt) for prompt in raw.get("prompts", [])),
        )


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} did not contain a YAML mapping")
    return Config.from_dict(raw)
