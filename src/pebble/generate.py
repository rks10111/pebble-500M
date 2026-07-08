from __future__ import annotations

import argparse
import codecs
import random
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

from pebble.config import Config
from pebble.model import Transformer


UTF8_DECODER_ERRORS = "ignore"
DEFAULT_SYSTEM_PROMPT = "You are Pebble, a helpful assistant."
CHAT_TOKENS = {
    "<|system|>": 50257,
    "<|user|>": 50258,
    "<|assistant|>": 50259,
    "<|end|>": 50260,
}
SPECIAL_TOKEN_NAMES = {token_id: name for name, token_id in CHAT_TOKENS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive text completion for Pebble checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Path to a Pebble .pt checkpoint.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Run one completion for this prompt instead of starting the interactive loop.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--chat", action="store_true", help="Format prompts with SFT chat tokens.")
    mode.add_argument("--completion", action="store_true", help="Force raw text completion mode.")
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used in chat mode.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=50, help="Restrict sampling to top K logits; 0 disables.")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, mps, or a torch device string. Defaults to auto.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional sampling seed.")
    parser.add_argument(
        "--no-stop-at-eod",
        action="store_true",
        help="Keep generating until --max-new-tokens even if the EOD token is sampled.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print only the completed text without the padded chat layout.",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[Transformer, Config, int, int, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(checkpoint["config"])
    model = Transformer(cfg.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    sft_state = checkpoint.get("sft") if isinstance(checkpoint.get("sft"), dict) else {}
    tokens_seen = int(checkpoint.get("tokens_seen", sft_state.get("tokens_seen", 0)))
    global_step = int(checkpoint.get("global_step", sft_state.get("step", 0)))
    return model, cfg, tokens_seen, global_step, checkpoint


def sample_next_id(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class TokenByteDecoder:
    def __init__(self, encoder: tiktoken.Encoding) -> None:
        self.encoder = encoder
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors=UTF8_DECODER_ERRORS)

    def decode_token(self, token_id: int) -> str:
        if token_id in SPECIAL_TOKEN_NAMES:
            return ""
        n_vocab = getattr(self.encoder, "n_vocab", None)
        if n_vocab is not None and token_id >= n_vocab:
            return ""
        try:
            token_bytes = self.encoder.decode_single_token_bytes(token_id)
        except KeyError:
            return ""
        return self.decoder.decode(token_bytes, final=False)

    def finish(self) -> str:
        return self.decoder.decode(b"", final=True)


def chat_width() -> int:
    return min(max(shutil.get_terminal_size((88, 24)).columns, 52), 100)


def box_top(title: str, width: int) -> str:
    label = f" {title} "
    remaining = max(width - len(label) - 2, 1)
    return "+" + label + ("-" * remaining) + "+"


def print_box(title: str, text: str) -> None:
    width = chat_width()
    body_width = width - 4
    print(box_top(title, width), flush=True)
    for raw_line in text.splitlines() or [""]:
        wrapped = textwrap.wrap(
            raw_line,
            width=body_width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        ) or [""]
        for line in wrapped:
            print(f"| {line:<{body_width}} |", flush=True)
    print("+" + ("-" * (width - 2)) + "+", flush=True)


def chat_tokens_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, int]:
    sft_state = checkpoint.get("sft")
    if isinstance(sft_state, dict) and isinstance(sft_state.get("chat_tokens"), dict):
        tokens = {name: int(token_id) for name, token_id in sft_state["chat_tokens"].items()}
        if all(name in tokens for name in CHAT_TOKENS):
            return tokens
    return CHAT_TOKENS


def should_use_chat(args: argparse.Namespace, checkpoint: dict[str, Any]) -> bool:
    if args.chat:
        return True
    if args.completion:
        return False
    sft_state = checkpoint.get("sft")
    return isinstance(sft_state, dict) and isinstance(sft_state.get("chat_tokens"), dict)


def encode_chat_prompt(
    encoder: tiktoken.Encoding,
    *,
    system_prompt: str,
    user_prompt: str,
    chat_tokens: dict[str, int],
) -> list[int]:
    ids: list[int] = []
    system_prompt = system_prompt.strip()
    if system_prompt:
        ids.extend([chat_tokens["<|system|>"], *encoder.encode_ordinary(system_prompt), chat_tokens["<|end|>"]])
    ids.extend(
        [
            chat_tokens["<|user|>"],
            *encoder.encode_ordinary(user_prompt),
            chat_tokens["<|end|>"],
            chat_tokens["<|assistant|>"],
        ]
    )
    return ids


class StreamingBox:
    def __init__(self, title: str) -> None:
        self.title = title
        self.width = chat_width()
        self.body_width = self.width - 4
        self.column = 0
        self.started = False

    def __enter__(self) -> "StreamingBox":
        print(box_top(self.title, self.width), flush=True)
        print("| ", end="", flush=True)
        self.started = True
        return self

    def write(self, text: str) -> None:
        if not self.started:
            raise RuntimeError("streaming box was not opened")
        for char in text:
            if char == "\n":
                self._finish_line(start_next=True)
                continue
            if self.column >= self.body_width:
                self._finish_line(start_next=True)
                if char == " ":
                    continue
            print(char, end="", flush=False)
            self.column += 1
        sys.stdout.flush()

    def _finish_line(self, *, start_next: bool) -> None:
        print(f"{' ' * (self.body_width - self.column)} |", flush=True)
        self.column = 0
        if start_next:
            print("| ", end="", flush=True)

    def close(self) -> None:
        if not self.started:
            return
        self._finish_line(start_next=False)
        print("+" + ("-" * (self.width - 2)) + "+", flush=True)
        self.started = False

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


@torch.no_grad()
def generate_token_ids(
    model: Transformer,
    idx: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    stop_tokens: set[int],
    stop_at_eod: bool,
) -> Iterator[int]:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.context_length :]
        logits, _ = model(idx_cond)
        if logits is None:
            raise RuntimeError("model did not return logits")
        next_id = sample_next_id(logits[:, -1, :], temperature=temperature, top_k=top_k)
        token_id = int(next_id.item())
        if stop_at_eod and token_id in stop_tokens:
            return
        idx = torch.cat((idx, next_id), dim=1)
        yield token_id


def complete_once(
    model: Transformer,
    cfg: Config,
    encoder: tiktoken.Encoding,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    use_chat: bool,
    chat_tokens: dict[str, int],
) -> str:
    if use_chat:
        ids = encode_chat_prompt(
            encoder,
            system_prompt=args.system,
            user_prompt=prompt,
            chat_tokens=chat_tokens,
        )
        stop_tokens = {int(chat_tokens["<|end|>"]), int(cfg.tokenizer.eod_token)}
    else:
        ids = encoder.encode_ordinary(prompt)
        stop_tokens = {int(cfg.tokenizer.eod_token)}
    if not ids:
        return ""
    if max(ids) >= cfg.model.vocab_size:
        raise ValueError("prompt contains token id outside model vocabulary")

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generated: list[int] = []
    generated_text_parts: list[str] = []
    if args.plain:
        if not use_chat:
            print(prompt, end="", flush=True)
        decoder = TokenByteDecoder(encoder)
        for token_id in generate_token_ids(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_tokens=stop_tokens,
            stop_at_eod=not args.no_stop_at_eod,
        ):
            generated.append(token_id)
            chunk = decoder.decode_token(token_id)
            generated_text_parts.append(chunk)
            print(chunk, end="", flush=True)
        tail = decoder.finish()
        generated_text_parts.append(tail)
        print(tail, end="", flush=True)
        print("", flush=True)
    else:
        print("")
        print_box("You", prompt)
        print("")
        decoder = TokenByteDecoder(encoder)
        with StreamingBox("Pebble") as box:
            if not use_chat:
                box.write(prompt)
            for token_id in generate_token_ids(
                model,
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop_tokens=stop_tokens,
                stop_at_eod=not args.no_stop_at_eod,
            ):
                generated.append(token_id)
                chunk = decoder.decode_token(token_id)
                generated_text_parts.append(chunk)
                box.write(chunk)
            tail = decoder.finish()
            generated_text_parts.append(tail)
            box.write(tail)
        print("")
    completion = ("" if use_chat else prompt) + "".join(generated_text_parts)
    return completion


def interactive_loop(
    model: Transformer,
    cfg: Config,
    encoder: tiktoken.Encoding,
    args: argparse.Namespace,
    device: torch.device,
    *,
    use_chat: bool,
    chat_tokens: dict[str, int],
) -> None:
    print("")
    print_box("Pebble", "Enter a prompt and press return. Use /exit, /quit, or Ctrl-D to stop.")
    while True:
        try:
            prompt = input("You> ")
        except EOFError:
            print("")
            return
        if prompt.strip() in {"/exit", "/quit"}:
            return
        if not prompt:
            continue
        complete_once(model, cfg, encoder, prompt, args, device, use_chat=use_chat, chat_tokens=chat_tokens)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)

    started = time.time()
    print(f"loading checkpoint={args.checkpoint} device={device}", file=sys.stderr, flush=True)
    model, cfg, tokens_seen, global_step, checkpoint = load_model(args.checkpoint, device)
    encoder = tiktoken.get_encoding(cfg.tokenizer.name)
    chat_tokens = chat_tokens_from_checkpoint(checkpoint)
    use_chat = should_use_chat(args, checkpoint)
    print(
        f"loaded tokens_seen={tokens_seen:,} step={global_step} "
        f"params={model.parameter_count():,} mode={'chat' if use_chat else 'completion'} "
        f"elapsed={time.time() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    if args.prompt is not None:
        complete_once(model, cfg, encoder, args.prompt, args, device, use_chat=use_chat, chat_tokens=chat_tokens)
        return
    interactive_loop(model, cfg, encoder, args, device, use_chat=use_chat, chat_tokens=chat_tokens)


if __name__ == "__main__":
    main()
