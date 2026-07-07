from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pebble.config import Config, load_config  # noqa: E402
from pebble.model import Transformer  # noqa: E402
from pebble.train import (  # noqa: E402
    append_jsonl,
    autocast_context,
    configure_torch_backends,
    cuda_memory_metrics,
    safe_perplexity,
    seed_everything,
    select_device,
)


CHAT_TOKENS = {
    "<|system|>": 50257,
    "<|user|>": 50258,
    "<|assistant|>": 50259,
    "<|end|>": 50260,
}
CHAT_TOKEN_SOURCES = {
    "<|system|>": "System:",
    "<|user|>": "User:",
    "<|assistant|>": "Assistant:",
}
DEFAULT_END_SOURCE_ID = 50256


@dataclass(frozen=True)
class ExampleRef:
    shard_index: int
    offset: int
    tokens: int
    assistant_tokens: int


class SFTSplit:
    def __init__(self, data_dir: Path, manifest: dict[str, Any], split: str) -> None:
        self.data_dir = data_dir
        self.split = split
        split_manifest = manifest["splits"][split]
        self.shards = split_manifest["shards"]
        self.token_maps = [
            np.memmap(data_dir / shard["tokens_path"], dtype=np.uint16, mode="r")
            for shard in self.shards
        ]
        self.mask_maps = [
            np.memmap(data_dir / shard["loss_mask_path"], dtype=np.uint8, mode="r")
            for shard in self.shards
        ]
        self.examples = self._load_examples(data_dir / split_manifest["example_index"])
        if not self.examples:
            raise ValueError(f"{split} split has no examples")

    @staticmethod
    def _load_examples(path: Path) -> list[ExampleRef]:
        examples: list[ExampleRef] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                examples.append(
                    ExampleRef(
                        shard_index=int(row["shard_index"]),
                        offset=int(row["offset"]),
                        tokens=int(row["tokens"]),
                        assistant_tokens=int(row["assistant_token_count"]),
                    )
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def get(self, example: ExampleRef) -> tuple[np.ndarray, np.ndarray]:
        shard_tokens = self.token_maps[example.shard_index]
        shard_masks = self.mask_maps[example.shard_index]
        start = example.offset
        end = start + example.tokens
        tokens = np.asarray(shard_tokens[start:end], dtype=np.int64)
        masks = np.asarray(shard_masks[start:end], dtype=np.uint8)
        if len(tokens) != example.tokens or len(masks) != example.tokens:
            raise ValueError(f"short read in {self.split} example {example}")
        return tokens, masks


class SFTBatcher:
    def __init__(
        self,
        split: SFTSplit,
        *,
        batch_size: int,
        context_length: int,
        pad_token_id: int,
        device: torch.device,
    ) -> None:
        self.split = split
        self.batch_size = batch_size
        self.context_length = context_length
        self.pad_token_id = pad_token_id
        self.device = device

    def batch_from_refs(self, refs: list[ExampleRef]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        x_np = np.full((len(refs), self.context_length), self.pad_token_id, dtype=np.int64)
        y_np = np.full((len(refs), self.context_length), self.pad_token_id, dtype=np.int64)
        mask_np = np.zeros((len(refs), self.context_length), dtype=np.float32)
        total_tokens = 0
        supervised_tokens = 0
        for row, ref in enumerate(refs):
            tokens, masks = self.split.get(ref)
            if len(tokens) < 2:
                continue
            if len(tokens) > self.context_length:
                raise ValueError(f"example has {len(tokens)} tokens but context length is {self.context_length}")
            x = tokens[:-1]
            y = tokens[1:]
            y_mask = masks[1:].astype(np.float32, copy=False)
            n = len(x)
            x_np[row, :n] = x
            y_np[row, :n] = y
            mask_np[row, :n] = y_mask
            total_tokens += n
            supervised_tokens += int(y_mask.sum())
        if supervised_tokens <= 0:
            raise ValueError("batch has no supervised assistant tokens")
        x_t = torch.from_numpy(x_np).to(self.device, non_blocking=True)
        y_t = torch.from_numpy(y_np).to(self.device, non_blocking=True)
        mask_t = torch.from_numpy(mask_np).to(self.device, non_blocking=True)
        return x_t, y_t, mask_t, int(total_tokens), int(supervised_tokens)

    def iter_epoch(
        self,
        *,
        rng: np.random.Generator,
        shuffle: bool,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]]:
        indices = np.arange(len(self.split.examples))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if len(batch_indices) == 0:
                continue
            refs = [self.split.examples[int(index)] for index in batch_indices]
            yield self.batch_from_refs(refs)

    def iter_sequential(
        self,
        *,
        max_batches: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]]:
        count = 0
        for start in range(0, len(self.split.examples), self.batch_size):
            if max_batches is not None and count >= max_batches:
                break
            refs = self.split.examples[start : start + self.batch_size]
            yield self.batch_from_refs(refs)
            count += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assistant-only SFT training for Pebble chat data.")
    parser.add_argument("--config", default="configs/pebble_500m_30b.yaml", help="Model/config YAML.")
    parser.add_argument("--data-dir", required=True, help="Prepared masked SFT dataset directory.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Pretrained Pebble checkpoint to load weights from. Required unless --resume is set.",
    )
    parser.add_argument("--resume", default=None, help="Optional SFT checkpoint to resume from.")
    parser.add_argument("--out-dir", required=True, help="Output directory for SFT checkpoints and metrics.")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow starting a fresh run in an output directory that already has metrics.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or torch device string.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of passes over train examples.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional optimizer step cap.")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Examples per micro batch.")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Micro batches per optimizer step.")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="Peak SFT learning rate.")
    parser.add_argument("--min-lr", type=float, default=5.0e-6, help="Cosine final learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Linear LR warmup optimizer steps.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default=None)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the model.")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--log-interval-steps", type=int, default=10)
    parser.add_argument("--eval-interval-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-interval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-ground-chat-tokens", action="store_true", help="Do not initialize chat token rows.")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable W&B logging.")
    parser.add_argument("--wandb", dest="wandb", action="store_true", help="Enable W&B logging.")
    parser.set_defaults(wandb=False)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "pebble-500m"))
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.resume and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --resume is set")
    if args.resume and args.overwrite_output:
        raise ValueError("--resume and --overwrite-output cannot be used together")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive when set")
    for name in (
        "micro_batch_size",
        "grad_accum_steps",
        "warmup_steps",
        "log_interval_steps",
        "eval_interval_steps",
        "eval_batches",
        "save_interval_steps",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("lr", "min_lr"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.gradient_clip < 0:
        raise ValueError("--gradient-clip must be non-negative")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")


def lr_for_step(step: int, *, max_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * max(1, step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def configure_optimizer(args: argparse.Namespace, model: Transformer, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": args.lr,
        "betas": (args.beta1, args.beta2),
        "weight_decay": args.weight_decay,
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def scale_gradients(model: torch.nn.Module, scale: float) -> None:
    if scale == 1.0:
        return
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)


def ground_chat_token_rows(state_dict: dict[str, torch.Tensor], *, encoder: tiktoken.Encoding) -> dict[str, Any]:
    if "wte.weight" not in state_dict:
        raise KeyError("checkpoint has no wte.weight")
    report: dict[str, Any] = {}
    embedding = state_dict["wte.weight"].float()
    updates: dict[int, torch.Tensor] = {}
    for token_name, source_text in CHAT_TOKEN_SOURCES.items():
        source_ids = encoder.encode_ordinary(source_text)
        target_id = CHAT_TOKENS[token_name]
        updates[target_id] = embedding[source_ids].mean(dim=0).to(dtype=state_dict["wte.weight"].dtype)
        report[token_name] = {
            "target_id": target_id,
            "source_text": source_text,
            "source_ids": source_ids,
        }
    updates[CHAT_TOKENS["<|end|>"]] = state_dict["wte.weight"][DEFAULT_END_SOURCE_ID].detach().clone()
    report["<|end|>"] = {
        "target_id": CHAT_TOKENS["<|end|>"],
        "source_text": "<|endoftext|>",
        "source_ids": [DEFAULT_END_SOURCE_ID],
    }
    for key in ("wte.weight", "lm_head.weight"):
        if key not in state_dict:
            continue
        weights = state_dict[key].clone()
        for target_id, vector in updates.items():
            weights[target_id] = vector.to(dtype=weights.dtype)
        state_dict[key] = weights
    return report


def load_pretrained_weights(
    model: Transformer,
    checkpoint_path: str | Path,
    *,
    ground_chat_tokens: bool,
) -> tuple[int, int, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = dict(checkpoint["model"])
    grounding_report: dict[str, Any] = {}
    if ground_chat_tokens:
        grounding_report = ground_chat_token_rows(state_dict, encoder=tiktoken.get_encoding("gpt2"))
    model.load_state_dict(state_dict)
    return int(checkpoint.get("tokens_seen", 0)), int(checkpoint.get("global_step", 0)), grounding_report


def load_sft_resume_checkpoint(model: Transformer, checkpoint_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not a Pebble checkpoint")
    sft_state = checkpoint.get("sft")
    if not isinstance(sft_state, dict):
        raise ValueError(f"{checkpoint_path} does not contain SFT metadata")
    model.load_state_dict(checkpoint["model"])
    return checkpoint, sft_state


def restore_runtime_rng(checkpoint: dict[str, Any]) -> None:
    rng_state = checkpoint.get("rng")
    if not isinstance(rng_state, dict):
        return
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    denom = loss_mask.sum().clamp_min(1.0)
    return (losses * loss_mask).sum() / denom


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_batcher: SFTBatcher,
    *,
    device: torch.device,
    precision: str,
    max_batches: int,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_supervised = 0
    total_tokens = 0
    batches = 0
    started = time.time()
    for x, y, mask, tokens, supervised_tokens in val_batcher.iter_sequential(max_batches=max_batches):
        with autocast_context(device, precision):
            logits, _ = model(x)
            loss = masked_cross_entropy(logits, y, mask)
        total_loss += float(loss.item()) * supervised_tokens
        total_supervised += supervised_tokens
        total_tokens += tokens
        batches += 1
    model.train()
    if total_supervised <= 0:
        raise ValueError("validation produced no supervised tokens")
    seconds = max(time.time() - started, 1e-6)
    loss = total_loss / total_supervised
    return {
        "loss": loss,
        "perplexity": safe_perplexity(loss),
        "eval_batches": int(batches),
        "eval_tokens": int(total_tokens),
        "eval_supervised_tokens": int(total_supervised),
        "eval_seconds": seconds,
        "eval_supervised_tokens_per_sec": total_supervised / seconds,
    }


def save_sft_checkpoint(
    out_dir: Path,
    *,
    name: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    args: argparse.Namespace,
    step: int,
    epoch: float,
    total_tokens: int,
    supervised_tokens: int,
    pretrained_tokens_seen: int,
    pretrained_global_step: int,
    grounding_report: dict[str, Any],
    data_rng_state: dict[str, Any] | None,
) -> Path:
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / name
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.raw,
            "global_step": int(step),
            "tokens_seen": int(total_tokens),
            "sft": {
                "step": int(step),
                "epoch": float(epoch),
                "tokens_seen": int(total_tokens),
                "supervised_tokens_seen": int(supervised_tokens),
                "pretrained_checkpoint": args.checkpoint,
                "pretrained_tokens_seen": int(pretrained_tokens_seen),
                "pretrained_global_step": int(pretrained_global_step),
                "chat_tokens": CHAT_TOKENS,
                "grounding_report": grounding_report,
                "data_rng_state": data_rng_state,
                "args": vars(args),
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        tmp_path,
    )
    os.replace(tmp_path, path)
    return path


def init_wandb(args: argparse.Namespace, cfg: Config, out_dir: Path, run_config: dict[str, Any]) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B was requested, but wandb is not installed") from exc
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{cfg.experiment.name}-sft",
        job_type="sft",
        mode=args.wandb_mode,
        dir=os.fspath(out_dir),
        config=run_config,
    )


def log_wandb(wandb_run: Any | None, payload: dict[str, Any]) -> None:
    if wandb_run is not None:
        wandb_run.log(payload)


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    cfg = load_config(args.config)
    seed_everything(args.seed)
    device = select_device(args.device)
    configure_torch_backends(device)
    precision = args.precision or cfg.training.precision
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.resume and not args.overwrite_output:
        raise ValueError(f"{metrics_path} already exists; use --resume or --overwrite-output")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task") != "chat-sft-assistant-mask":
        raise ValueError(f"{data_dir} is not a masked chat SFT dataset")
    if manifest["tokenizer"]["chat_tokens"] != CHAT_TOKENS:
        raise ValueError("dataset chat token IDs do not match train_sft.py constants")

    train_split = SFTSplit(data_dir, manifest, "train")
    val_split = SFTSplit(data_dir, manifest, "val")
    train_batcher = SFTBatcher(
        train_split,
        batch_size=args.micro_batch_size,
        context_length=cfg.model.context_length,
        pad_token_id=CHAT_TOKENS["<|end|>"],
        device=device,
    )
    val_batcher = SFTBatcher(
        val_split,
        batch_size=args.micro_batch_size,
        context_length=cfg.model.context_length,
        pad_token_id=CHAT_TOKENS["<|end|>"],
        device=device,
    )

    model = Transformer(cfg.model)
    resume_checkpoint: dict[str, Any] | None = None
    resume_sft_state: dict[str, Any] | None = None
    if args.resume:
        resume_checkpoint, resume_sft_state = load_sft_resume_checkpoint(model, args.resume)
        if not args.checkpoint:
            args.checkpoint = resume_sft_state.get("pretrained_checkpoint")
        pretrained_tokens_seen = int(resume_sft_state.get("pretrained_tokens_seen", 0))
        pretrained_global_step = int(resume_sft_state.get("pretrained_global_step", 0))
        grounding_report = dict(resume_sft_state.get("grounding_report") or {})
    else:
        pretrained_tokens_seen, pretrained_global_step, grounding_report = load_pretrained_weights(
            model,
            args.checkpoint,
            ground_chat_tokens=not args.no_ground_chat_tokens,
        )
    model.to(device)
    optimizer = configure_optimizer(args, model, device)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        move_optimizer_state(optimizer, device)
        restore_runtime_rng(resume_checkpoint)
    forward_model: torch.nn.Module = torch.compile(model, mode=args.compile_mode) if args.compile else model

    max_train_examples = max(1, int(len(train_split) * args.epochs))
    total_micro_batches = math.ceil(max_train_examples / args.micro_batch_size)
    planned_steps = math.ceil(total_micro_batches / args.grad_accum_steps)
    if args.max_steps is not None:
        planned_steps = min(planned_steps, args.max_steps)
    if planned_steps <= 0:
        raise ValueError("planned SFT steps must be positive")

    run_config = {
        **cfg.raw,
        "sft_runtime": {
            "data_dir": os.fspath(data_dir.resolve()),
            "out_dir": os.fspath(out_dir.resolve()),
            "checkpoint": os.fspath(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "resume": os.fspath(Path(args.resume).resolve()) if args.resume else None,
            "pretrained_tokens_seen": pretrained_tokens_seen,
            "pretrained_global_step": pretrained_global_step,
            "device": str(device),
            "precision": precision,
            "micro_batch_size": args.micro_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "epochs": args.epochs,
            "planned_steps": planned_steps,
            "chat_tokens": CHAT_TOKENS,
            "ground_chat_tokens": not args.no_ground_chat_tokens,
            "dataset": {
                "train_examples": len(train_split),
                "val_examples": len(val_split),
                "train_tokens": manifest["splits"]["train"]["tokens"],
                "train_assistant_tokens": manifest["splits"]["train"]["assistant_tokens"],
            },
        },
    }
    wandb_run = init_wandb(args, cfg, out_dir, run_config)

    print(
        f"sft_params={model.parameter_count():,} device={device} precision={precision} "
        f"train_examples={len(train_split):,} val_examples={len(val_split):,} "
        f"micro_batch={args.micro_batch_size} grad_accum={args.grad_accum_steps} "
        f"planned_steps={planned_steps} pretrained_tokens={pretrained_tokens_seen:,}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    if resume_sft_state is not None and isinstance(resume_sft_state.get("data_rng_state"), dict):
        rng.bit_generator.state = resume_sft_state["data_rng_state"]
    global_step = int(resume_sft_state.get("step", 0)) if resume_sft_state is not None else 0
    total_tokens_seen = int(resume_sft_state.get("tokens_seen", 0)) if resume_sft_state is not None else 0
    total_supervised_seen = (
        int(resume_sft_state.get("supervised_tokens_seen", 0)) if resume_sft_state is not None else 0
    )
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    accumulated_supervised = 0
    micro_in_step = 0
    started = time.time()
    last_log_time = started
    last_log_supervised = 0
    completed = global_step >= planned_steps

    while not completed:
        max_examples = max_train_examples
        examples_consumed = 0
        epoch_number = 0
        while examples_consumed < max_examples and not completed:
            epoch_number += 1
            for x, y, mask, tokens, supervised_tokens in train_batcher.iter_epoch(rng=rng, shuffle=True):
                if examples_consumed >= max_examples:
                    break
                examples_consumed += x.size(0)
                lr = lr_for_step(
                    global_step,
                    max_steps=planned_steps,
                    base_lr=args.lr,
                    min_lr=args.min_lr,
                    warmup_steps=args.warmup_steps,
                )
                set_optimizer_lr(optimizer, lr)
                with autocast_context(device, precision):
                    logits, _ = forward_model(x)
                    loss = masked_cross_entropy(logits, y, mask)
                (loss / args.grad_accum_steps).backward()
                accumulated_loss += float(loss.detach().item()) * supervised_tokens
                accumulated_supervised += supervised_tokens
                total_tokens_seen += tokens
                total_supervised_seen += supervised_tokens
                micro_in_step += 1

                if micro_in_step < args.grad_accum_steps:
                    continue

                grad_norm = None
                if args.gradient_clip > 0:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    grad_norm = float(grad_norm_t.detach().item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                micro_in_step = 0

                if global_step % args.log_interval_steps == 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    now = time.time()
                    interval_supervised = total_supervised_seen - last_log_supervised
                    interval_seconds = max(now - last_log_time, 1e-6)
                    train_loss = accumulated_loss / max(accumulated_supervised, 1)
                    epoch_float = examples_consumed / max(len(train_split), 1)
                    metrics = {
                        "type": "train",
                        "step": int(global_step),
                        "epoch": epoch_float,
                        "tokens_seen": int(total_tokens_seen),
                        "supervised_tokens_seen": int(total_supervised_seen),
                        "loss": train_loss,
                        "perplexity": safe_perplexity(train_loss),
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "micro_batch_size": int(args.micro_batch_size),
                        "grad_accum_steps": int(args.grad_accum_steps),
                        "supervised_tokens_per_sec_interval": interval_supervised / interval_seconds,
                        "wall_seconds": now - started,
                        **cuda_memory_metrics(device),
                    }
                    append_jsonl(metrics_path, metrics)
                    log_wandb(wandb_run, {k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                    print(
                        f"step={global_step} epoch={epoch_float:.3f} loss={train_loss:.4f} "
                        f"lr={lr:.2e} supervised_tok/s={metrics['supervised_tokens_per_sec_interval']:.0f}",
                        flush=True,
                    )
                    accumulated_loss = 0.0
                    accumulated_supervised = 0
                    last_log_time = now
                    last_log_supervised = total_supervised_seen

                if global_step % args.eval_interval_steps == 0:
                    val_metrics = {
                        "type": "validation",
                        "step": int(global_step),
                        "tokens_seen": int(total_tokens_seen),
                        "supervised_tokens_seen": int(total_supervised_seen),
                        **evaluate(
                            forward_model,
                            val_batcher,
                            device=device,
                            precision=precision,
                            max_batches=args.eval_batches,
                        ),
                    }
                    append_jsonl(metrics_path, val_metrics)
                    log_wandb(wandb_run, {k: v for k, v in val_metrics.items() if isinstance(v, (int, float))})
                    print(
                        f"validation step={global_step} loss={val_metrics['loss']:.4f} "
                        f"ppl={val_metrics['perplexity']}",
                        flush=True,
                    )

                if global_step % args.save_interval_steps == 0:
                    path = save_sft_checkpoint(
                        out_dir,
                        name=f"latest-sft-{global_step:08d}.pt",
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        args=args,
                        step=global_step,
                        epoch=examples_consumed / max(len(train_split), 1),
                        total_tokens=total_tokens_seen,
                        supervised_tokens=total_supervised_seen,
                        pretrained_tokens_seen=pretrained_tokens_seen,
                        pretrained_global_step=pretrained_global_step,
                        grounding_report=grounding_report,
                        data_rng_state=rng.bit_generator.state,
                    )
                    print(f"saved checkpoint {path}", flush=True)

                if global_step >= planned_steps:
                    completed = True
                    break
            if args.epochs <= 1:
                break

        if global_step >= planned_steps or examples_consumed >= max_examples:
            completed = True

    if micro_in_step:
        grad_norm = None
        scale_gradients(model, args.grad_accum_steps / micro_in_step)
        if args.gradient_clip > 0:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            grad_norm = float(grad_norm_t.detach().item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        print(f"completed partial final step={global_step} grad_norm={grad_norm}", flush=True)

    final_eval = evaluate(
        forward_model,
        val_batcher,
        device=device,
        precision=precision,
        max_batches=args.eval_batches,
    )
    final_path = save_sft_checkpoint(
        out_dir,
        name=f"final-sft-{global_step:08d}.pt",
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        args=args,
        step=global_step,
        epoch=min(args.epochs, total_tokens_seen / max(manifest["splits"]["train"]["tokens"], 1)),
        total_tokens=total_tokens_seen,
        supervised_tokens=total_supervised_seen,
        pretrained_tokens_seen=pretrained_tokens_seen,
        pretrained_global_step=pretrained_global_step,
        grounding_report=grounding_report,
        data_rng_state=rng.bit_generator.state,
    )
    final_metrics = {
        "type": "final",
        "step": int(global_step),
        "tokens_seen": int(total_tokens_seen),
        "supervised_tokens_seen": int(total_supervised_seen),
        "validation_loss": final_eval["loss"],
        "validation_perplexity": final_eval["perplexity"],
        "checkpoint": os.fspath(final_path),
        "wall_seconds": time.time() - started,
    }
    append_jsonl(metrics_path, final_metrics)
    log_wandb(wandb_run, {k: v for k, v in final_metrics.items() if isinstance(v, (int, float))})
    if wandb_run is not None:
        wandb_run.finish()
    print(
        f"finished sft step={global_step} validation_loss={final_eval['loss']:.4f} "
        f"checkpoint={final_path}",
        flush=True,
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
