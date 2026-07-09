from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import tiktoken

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pebble.generate import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    StreamingBox,
    TokenByteDecoder,
    chat_tokens_from_checkpoint,
    generate_token_ids,
    load_model,
    print_box,
    seed_everything,
    select_device,
)
from pebble.model import Transformer  # noqa: E402

Message = tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal chat loop for Pebble SFT checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Path to an SFT Pebble checkpoint.")
    parser.add_argument("--prompt", default=None, help="Run one user message and exit.")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="Initial system message.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum assistant tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=50, help="Restrict sampling to top K logits; 0 disables.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string.")
    parser.add_argument("--seed", type=int, default=None, help="Optional sampling seed.")
    parser.add_argument(
        "--no-stop-at-end",
        action="store_true",
        help="Keep generating until --max-new-tokens even if <|end|> or EOD is sampled.",
    )
    parser.add_argument(
        "--allow-base-checkpoint",
        action="store_true",
        help="Allow running a checkpoint without SFT metadata. Useful only for debugging.",
    )
    return parser.parse_args()


def initial_messages(system_prompt: str) -> list[Message]:
    system_prompt = system_prompt.strip()
    if not system_prompt:
        return []
    return [("system", system_prompt)]


def encode_sft_messages(
    encoder: tiktoken.Encoding,
    messages: Sequence[Message],
    *,
    chat_tokens: dict[str, int],
    append_assistant_prompt: bool,
) -> list[int]:
    ids: list[int] = []
    for role, content in messages:
        token_name = f"<|{role}|>"
        if role not in {"system", "user", "assistant"} or token_name not in chat_tokens:
            raise ValueError(f"unsupported chat role: {role}")
        ids.extend([chat_tokens[token_name], *encoder.encode_ordinary(content), chat_tokens["<|end|>"]])
    if append_assistant_prompt:
        ids.append(chat_tokens["<|assistant|>"])
    return ids


def encode_context(
    encoder: tiktoken.Encoding,
    messages: Sequence[Message],
    *,
    chat_tokens: dict[str, int],
    context_length: int,
) -> tuple[list[int], bool]:
    kept = list(messages)
    was_trimmed = False
    while True:
        ids = encode_sft_messages(encoder, kept, chat_tokens=chat_tokens, append_assistant_prompt=True)
        if len(ids) <= context_length:
            return ids, was_trimmed

        first_non_system = 1 if kept and kept[0][0] == "system" else 0
        non_system_count = len(kept) - first_non_system
        if non_system_count > 1:
            del kept[first_non_system : min(first_non_system + 2, len(kept))]
            was_trimmed = True
            continue

        return ids[-context_length:], True


def require_sft_checkpoint(checkpoint: dict[str, Any], *, allow_base_checkpoint: bool) -> None:
    if allow_base_checkpoint:
        return
    if not isinstance(checkpoint.get("sft"), dict):
        raise ValueError("checkpoint has no SFT metadata; pass --allow-base-checkpoint to force raw testing")


def validate_prompt_ids(ids: Sequence[int], vocab_size: int) -> None:
    if not ids:
        raise ValueError("empty chat prompt")
    if max(ids) >= vocab_size:
        raise ValueError("chat prompt contains token id outside model vocabulary")


def generate_reply(
    model: Transformer,
    encoder: tiktoken.Encoding,
    messages: Sequence[Message],
    args: argparse.Namespace,
    device: torch.device,
    *,
    chat_tokens: dict[str, int],
    eod_token: int,
) -> str:
    ids, was_trimmed = encode_context(
        encoder,
        messages,
        chat_tokens=chat_tokens,
        context_length=model.config.context_length,
    )
    validate_prompt_ids(ids, model.config.vocab_size)
    if was_trimmed:
        print_box("Pebble", "Context window filled; oldest turns were dropped for this reply.")

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stop_tokens = {int(chat_tokens["<|end|>"]), int(eod_token)}

    decoder = TokenByteDecoder(encoder)
    parts: list[str] = []
    with StreamingBox("Pebble") as box:
        for token_id in generate_token_ids(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_tokens=stop_tokens,
            stop_at_eod=not args.no_stop_at_end,
        ):
            chunk = decoder.decode_token(token_id)
            parts.append(chunk)
            box.write(chunk)
        tail = decoder.finish()
        parts.append(tail)
        box.write(tail)
    print("")
    return "".join(parts).rstrip()


def chat_loop(
    model: Transformer,
    encoder: tiktoken.Encoding,
    args: argparse.Namespace,
    device: torch.device,
    *,
    chat_tokens: dict[str, int],
    eod_token: int,
) -> None:
    messages = initial_messages(args.system)
    print("")
    print_box(
        "Pebble",
        "Enter a chat message. Commands: /reset, /system <message>, /exit.",
    )
    while True:
        try:
            user_text = input("You> ")
        except EOFError:
            print("")
            return

        command = user_text.strip()
        if command in {"/exit", "/quit"}:
            return
        if command == "/reset":
            messages = initial_messages(args.system)
            print_box("Pebble", "Conversation reset.")
            continue
        if command.startswith("/system "):
            args.system = command[len("/system ") :].strip()
            messages = initial_messages(args.system)
            print_box("System", args.system or "(cleared)")
            continue
        if not user_text:
            continue

        messages.append(("user", user_text))
        print("")
        print_box("You", user_text)
        print("")
        reply = generate_reply(
            model,
            encoder,
            messages,
            args,
            device,
            chat_tokens=chat_tokens,
            eod_token=eod_token,
        )
        messages.append(("assistant", reply))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)

    started = time.time()
    print(f"loading checkpoint={args.checkpoint} device={device}", file=sys.stderr, flush=True)
    model, cfg, tokens_seen, global_step, checkpoint = load_model(args.checkpoint, device)
    require_sft_checkpoint(checkpoint, allow_base_checkpoint=args.allow_base_checkpoint)
    encoder = tiktoken.get_encoding(cfg.tokenizer.name)
    chat_tokens = chat_tokens_from_checkpoint(checkpoint)
    print(
        f"loaded tokens_seen={tokens_seen:,} step={global_step} "
        f"params={model.parameter_count():,} context={model.config.context_length} "
        f"elapsed={time.time() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    if args.prompt is not None:
        messages = initial_messages(args.system)
        messages.append(("user", args.prompt))
        print_box("You", args.prompt)
        print("")
        generate_reply(
            model,
            encoder,
            messages,
            args,
            device,
            chat_tokens=chat_tokens,
            eod_token=int(cfg.tokenizer.eod_token),
        )
        return

    chat_loop(model, encoder, args, device, chat_tokens=chat_tokens, eod_token=int(cfg.tokenizer.eod_token))


if __name__ == "__main__":
    main()
