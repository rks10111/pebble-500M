from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import tiktoken

from pebble.config import Config, load_config
from pebble.data import SequentialTokenLoader, build_loaders
from pebble.model import Transformer, lr_for_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Pebble decoder-only Transformer.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-dir", required=True, help="Directory containing manifest.json and shards.")
    parser.add_argument("--out-dir", required=True, help="Run output directory.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max training tokens.")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="Override micro batch size.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or CUDA device string.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--eval-only", action="store_true", help="Run validation and exit.")
    parser.add_argument("--compile", dest="compile_override", action="store_true", help="Force torch.compile.")
    parser.add_argument("--no-compile", dest="no_compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        help="torch.compile mode to use when compilation is enabled.",
    )
    parser.add_argument(
        "--compile-allow-graph-breaks",
        action="store_true",
        help="Compile without fullgraph=True; useful if max-autotune fullgraph fails during debugging.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Training steps to exclude from train-throughput averages. Defaults to one log interval.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_torch_backends(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def autocast_context(device: torch.device, precision: str) -> Iterator[None]:
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def configure_optimizer(model: Transformer, cfg: Config, device: torch.device) -> torch.optim.Optimizer:
    if cfg.optimizer.name.lower() != "adamw":
        raise ValueError(f"unsupported optimizer {cfg.optimizer.name!r}")
    kwargs: dict[str, Any] = {
        "lr": cfg.optimizer.lr,
        "betas": cfg.optimizer.betas,
        "weight_decay": cfg.optimizer.weight_decay,
    }
    if cfg.optimizer.fused and device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def grad_accum_steps(global_batch_tokens: int, micro_batch_size: int, context_length: int) -> int:
    micro_tokens = micro_batch_size * context_length
    if micro_tokens <= 0:
        raise ValueError("micro batch tokens must be positive")
    if global_batch_tokens % micro_tokens != 0:
        raise ValueError(
            "global_batch_tokens must be exactly divisible by "
            "micro_batch_size * context_length; got "
            f"{global_batch_tokens=} {micro_batch_size=} {context_length=}"
        )
    return global_batch_tokens // micro_tokens


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: SequentialTokenLoader,
    cfg: Config,
    device: torch.device,
    max_tokens: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y, tokens in val_loader.iter_batches(max_tokens=max_tokens):
        with autocast_context(device, cfg.training.precision):
            _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("model did not return validation loss")
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    model.train()
    if total_tokens == 0:
        raise ValueError("validation loader produced no tokens")
    return total_loss / total_tokens


@torch.no_grad()
def sample_prompts(
    model: Transformer,
    cfg: Config,
    device: torch.device,
    out_path: Path,
    tokens_seen: int,
) -> None:
    if not cfg.prompts:
        return
    try:
        encoder = tiktoken.get_encoding(cfg.tokenizer.name)
    except Exception as exc:
        append_jsonl(
            out_path,
            {
                "tokens_seen": int(tokens_seen),
                "skipped": True,
                "error": f"tokenizer_unavailable: {exc}",
            },
        )
        print(f"skipped prompt sampling: tokenizer unavailable: {exc}", flush=True)
        return
    model.eval()
    for prompt in cfg.prompts:
        ids = encoder.encode_ordinary(prompt)
        if not ids or max(ids) >= cfg.model.vocab_size:
            text = ""
            skipped = True
        else:
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(idx, max_new_tokens=cfg.training.sample_max_new_tokens)
            text = encoder.decode(out[0].detach().cpu().tolist())
            skipped = False
        append_jsonl(
            out_path,
            {
                "tokens_seen": int(tokens_seen),
                "prompt": prompt,
                "sample": text,
                "skipped": skipped,
            },
        )
    model.train()


def checkpoint_payload(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    tokens_seen: int,
    global_step: int,
    train_loader_state: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "tokens_seen": int(tokens_seen),
        "global_step": int(global_step),
        "config": cfg.raw,
        "train_loader": train_loader_state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    return payload


def restore_rng(state: dict[str, Any]) -> None:
    rng = state.get("rng")
    if not rng:
        return
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def save_checkpoint(
    checkpoint_dir: Path,
    kind: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    tokens_seen: int,
    global_step: int,
    train_loader_state: dict[str, Any],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if kind == "latest":
        name = f"latest-{global_step:012d}.pt"
    else:
        name = f"milestone-{tokens_seen:012d}.pt"
    path = checkpoint_dir / name
    torch.save(
        checkpoint_payload(model, optimizer, cfg, tokens_seen, global_step, train_loader_state),
        path,
    )
    return path


def prune_latest_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    latest = sorted(checkpoint_dir.glob("latest-*.pt"))
    for path in latest[:-keep_last]:
        path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    train_loader,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_loader.load_state_dict(checkpoint.get("train_loader", {}))
    restore_rng(checkpoint)
    return int(checkpoint["tokens_seen"]), int(checkpoint["global_step"])


def maybe_compile(model: Transformer, enabled: bool, mode: str, fullgraph: bool) -> torch.nn.Module:
    if not enabled:
        return model
    return torch.compile(model, mode=mode, fullgraph=fullgraph)


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    seed_everything(cfg.experiment.seed)

    device = select_device(args.device)
    configure_torch_backends(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    metrics_path = out_dir / "metrics.jsonl"
    samples_path = out_dir / "samples.jsonl"

    micro_batch_size = args.micro_batch_size or cfg.training.micro_batch_size
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.training.max_tokens
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = cfg.training.log_interval_steps
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    accum_steps = grad_accum_steps(
        cfg.training.global_batch_tokens,
        micro_batch_size,
        cfg.model.context_length,
    )
    tokens_per_step = accum_steps * micro_batch_size * cfg.model.context_length

    if cfg.training.activation_checkpointing:
        raise ValueError("activation checkpointing is not implemented; keep it disabled for this experiment")

    train_loader, val_loader = build_loaders(
        args.data_dir,
        block_size=cfg.model.context_length,
        micro_batch_size=micro_batch_size,
        seed=cfg.experiment.seed,
        device=device,
    )

    model = Transformer(cfg.model).to(device)
    optimizer = configure_optimizer(model, cfg, device)

    tokens_seen = 0
    global_step = 0
    if args.resume:
        tokens_seen, global_step = load_checkpoint(args.resume, model, optimizer, train_loader)
    start_tokens = tokens_seen

    compile_enabled = cfg.training.compile
    if args.compile_override:
        compile_enabled = True
    if args.no_compile:
        compile_enabled = False
    compile_fullgraph = not args.compile_allow_graph_breaks
    forward_model = maybe_compile(
        model,
        compile_enabled,
        mode=args.compile_mode,
        fullgraph=compile_fullgraph,
    )

    param_count = model.parameter_count()
    print(
        f"model_params={param_count:,} "
        f"device={device} "
        f"micro_batch={micro_batch_size} "
        f"grad_accum={accum_steps} "
        f"tokens_per_step={tokens_per_step:,} "
        f"compile={compile_enabled} "
        f"compile_mode={args.compile_mode if compile_enabled else 'disabled'} "
        f"compile_fullgraph={compile_fullgraph if compile_enabled else False} "
        f"warmup_steps={warmup_steps}",
        flush=True,
    )

    if args.eval_only:
        val_loss = evaluate(forward_model, val_loader, cfg, device, cfg.training.eval_tokens)
        print(f"validation_loss={val_loss:.6f}")
        return

    model.train()
    completed_milestones = {milestone for milestone in cfg.checkpointing.milestones if tokens_seen >= milestone}
    next_eval_at = tokens_seen + cfg.training.eval_interval_tokens
    last_checkpoint_time = time.time()
    start_step = global_step
    started = time.time()
    last_log_time = started
    last_log_tokens = tokens_seen
    measured_train_start_time = started if warmup_steps == 0 else None
    measured_train_start_tokens = tokens_seen if warmup_steps == 0 else None

    while tokens_seen < max_tokens:
        lr = lr_for_tokens(
            tokens_seen,
            base_lr=cfg.optimizer.lr,
            min_lr=cfg.schedule.min_lr,
            warmup_tokens=cfg.schedule.warmup_tokens,
            target_tokens=cfg.schedule.planned_target_tokens,
        )
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        accumulated_loss: torch.Tensor | None = None
        for _ in range(accum_steps):
            x, y = train_loader.next_batch()
            with autocast_context(device, cfg.training.precision):
                _, loss = forward_model(x, y)
            if loss is None:
                raise RuntimeError("model did not return training loss")
            accumulated_loss = loss.detach() if accumulated_loss is None else accumulated_loss + loss.detach()
            (loss / accum_steps).backward()

        grad_norm: torch.Tensor | None = None
        if cfg.training.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
        optimizer.step()

        global_step += 1
        run_steps = global_step - start_step
        tokens_seen += tokens_per_step
        trained_tokens_this_run = tokens_seen - start_tokens
        synced_this_step = False

        if measured_train_start_time is None and run_steps >= warmup_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
                synced_this_step = True
            measured_train_start_time = time.time()
            measured_train_start_tokens = tokens_seen

        if global_step % cfg.training.log_interval_steps == 0:
            if device.type == "cuda" and not synced_this_step:
                torch.cuda.synchronize()
            now = time.time()
            interval_tokens = tokens_seen - last_log_tokens
            interval_seconds = max(now - last_log_time, 1e-6)
            total_seconds = max(now - started, 1e-6)
            if measured_train_start_time is not None and measured_train_start_tokens is not None:
                measured_train_tokens = tokens_seen - measured_train_start_tokens
                measured_train_seconds = max(now - measured_train_start_time, 1e-6)
                avg_train_tokens_per_sec = measured_train_tokens / measured_train_seconds
            else:
                measured_train_tokens = 0
                measured_train_seconds = 0.0
                avg_train_tokens_per_sec = None
            loss_value = float((accumulated_loss / accum_steps).item()) if accumulated_loss is not None else None
            grad_norm_value = float(grad_norm.detach().item()) if grad_norm is not None else None
            metrics = {
                "type": "train",
                "step": int(global_step),
                "tokens_seen": int(tokens_seen),
                "tokens_trained_this_run": int(trained_tokens_this_run),
                "loss": loss_value,
                "lr": lr,
                "grad_norm": grad_norm_value,
                "warmup_steps_excluded": int(min(run_steps, warmup_steps)),
                "warmup_tokens_excluded": int(min(run_steps, warmup_steps) * tokens_per_step),
                "tokens_per_sec_interval": interval_tokens / interval_seconds,
                "tokens_per_sec_wall_average": trained_tokens_this_run / total_seconds,
                "measured_train_tokens": int(measured_train_tokens),
                "measured_train_seconds": measured_train_seconds,
                "tokens_per_sec_train_average": avg_train_tokens_per_sec,
                "post_warmup_tokens_per_sec_train_average": avg_train_tokens_per_sec,
                "gpu_memory_allocated_gb": (
                    torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
                ),
                "estimated_aws_cost_usd": (total_seconds / 3600.0) * cfg.aws.hourly_usd,
            }
            append_jsonl(metrics_path, metrics)
            train_avg = "n/a" if avg_train_tokens_per_sec is None else f"{avg_train_tokens_per_sec:.0f}"
            print(
                f"step={global_step} tokens={tokens_seen:,} "
                f"loss={metrics['loss']:.4f} lr={lr:.2e} "
                f"tok/s={metrics['tokens_per_sec_interval']:.0f} "
                f"train_avg_tok/s={train_avg} "
                f"cost=${metrics['estimated_aws_cost_usd']:.2f}",
                flush=True,
            )
            last_log_time = now
            last_log_tokens = tokens_seen

        if tokens_seen >= next_eval_at:
            val_loss = evaluate(forward_model, val_loader, cfg, device, cfg.training.eval_tokens)
            append_jsonl(
                metrics_path,
                {
                    "type": "validation",
                    "step": int(global_step),
                    "tokens_seen": int(tokens_seen),
                    "loss": val_loss,
                    "eval_tokens": int(cfg.training.eval_tokens),
                },
            )
            print(f"validation tokens={tokens_seen:,} loss={val_loss:.4f}", flush=True)
            next_eval_at += cfg.training.eval_interval_tokens

        for milestone in cfg.checkpointing.milestones:
            if milestone not in completed_milestones and tokens_seen >= milestone:
                path = save_checkpoint(
                    checkpoint_dir,
                    "milestone",
                    model,
                    optimizer,
                    cfg,
                    tokens_seen,
                    global_step,
                    train_loader.state_dict(),
                )
                sample_prompts(model, cfg, device, samples_path, tokens_seen)
                completed_milestones.add(milestone)
                print(f"saved milestone checkpoint {path}", flush=True)

        if (time.time() - last_checkpoint_time) / 60.0 >= cfg.checkpointing.save_interval_minutes:
            path = save_checkpoint(
                checkpoint_dir,
                "latest",
                model,
                optimizer,
                cfg,
                tokens_seen,
                global_step,
                train_loader.state_dict(),
            )
            prune_latest_checkpoints(checkpoint_dir, cfg.checkpointing.keep_last)
            print(f"saved latest checkpoint {path}", flush=True)
            last_checkpoint_time = time.time()

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_finished_at = time.time()
    if measured_train_start_time is not None and measured_train_start_tokens is not None:
        measured_train_seconds = max(training_finished_at - measured_train_start_time, 1e-6)
        measured_train_tokens = max(0, tokens_seen - measured_train_start_tokens)
        average_train_tokens_per_sec = measured_train_tokens / measured_train_seconds
    else:
        measured_train_seconds = 0.0
        measured_train_tokens = 0
        average_train_tokens_per_sec = 0.0

    final_path = save_checkpoint(
        checkpoint_dir,
        "latest",
        model,
        optimizer,
        cfg,
        tokens_seen,
        global_step,
        train_loader.state_dict(),
    )
    prune_latest_checkpoints(checkpoint_dir, cfg.checkpointing.keep_last)
    val_loss = evaluate(forward_model, val_loader, cfg, device, cfg.training.eval_tokens)
    total_seconds = max(time.time() - started, 1e-6)
    trained_tokens_this_run = tokens_seen - start_tokens
    average_wall_tokens_per_sec = trained_tokens_this_run / total_seconds
    append_jsonl(
        metrics_path,
        {
            "type": "final",
            "step": int(global_step),
            "tokens_seen": int(tokens_seen),
            "tokens_trained_this_run": int(trained_tokens_this_run),
            "validation_loss": val_loss,
            "train_seconds": measured_train_seconds,
            "measured_train_tokens": int(measured_train_tokens),
            "measured_train_seconds": measured_train_seconds,
            "warmup_steps_excluded": int(min(global_step - start_step, warmup_steps)),
            "warmup_tokens_excluded": int(min(global_step - start_step, warmup_steps) * tokens_per_step),
            "wall_seconds": total_seconds,
            "average_train_tokens_per_sec": average_train_tokens_per_sec,
            "post_warmup_average_train_tokens_per_sec": average_train_tokens_per_sec,
            "average_wall_tokens_per_sec": average_wall_tokens_per_sec,
            "checkpoint": os.fspath(final_path),
        },
    )
    print(
        f"finished tokens={tokens_seen:,} "
        f"validation_loss={val_loss:.4f} "
        f"average_train_tok/s={average_train_tokens_per_sec:.0f} "
        f"average_wall_tok/s={average_wall_tokens_per_sec:.0f} "
        f"checkpoint={final_path}"
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
