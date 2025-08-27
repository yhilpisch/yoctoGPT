"""Simple terminal chat interface for yoctoGPT.

This module builds on the sampler to provide a stateful REPL that preserves a
rolling context window. It formats turns as `User:` and `Assistant:` to guide
the model's style. Since yoctoGPT is trained as a plain LM, this interface is a
thin usability layer rather than a full instruction-following system.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from .data import CharVocab
from .model import GPT, GPTConfig
from .advanced_model import AdvancedGPT, AdvancedGPTConfig
from .performance_model import PerformanceGPT, PerformanceGPTConfig
from .tokenizer import load_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Chat with a yoctoGPT checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mode", choices=["char", "token"], default="token")
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--vocab_path", type=str, default=None)
    p.add_argument("--system_prompt", type=str, default="You are yoctoGPT, a concise helpful assistant.")
    p.add_argument("--max_ctx_tokens", type=int, default=384)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    arch = ckpt.get("arch", "gpt")
    if arch == "gpt_plus":
        model = AdvancedGPT(AdvancedGPTConfig(**ckpt["model_config"]))
    elif arch == "gpt_fast":
        model = PerformanceGPT(PerformanceGPTConfig(**ckpt["model_config"]))
    else:
        model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Resolve encoding
    if args.mode == "char":
        vocab_path = args.vocab_path or ckpt.get("char_vocab_path")
        assert vocab_path, "Provide --vocab_path or ensure checkpoint stores char_vocab_path"
        vocab = CharVocab.load(vocab_path)
        encode = lambda s: vocab.encode(s)
        decode = lambda ids: vocab.decode(ids)
    else:
        tok_path = args.tokenizer_path or ckpt.get("tokenizer_path")
        assert tok_path, "Provide --tokenizer_path or ensure checkpoint stores tokenizer_path"
        tokenizer = load_tokenizer(tok_path)
        encode = lambda s: tokenizer.encode(s)
        decode = lambda ids: tokenizer.decode(ids)

    # Initial context
    turns: List[str] = []
    if args.system_prompt:
        turns.append(f"System: {args.system_prompt}\n")

    print("Type /exit to quit. Press Enter after your input.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user == "/exit":
            break
        if not user:
            continue

        # Build context and ensure it fits into model context window
        turns.append(f"User: {user}\n")
        prompt = "".join(turns) + "Assistant: "
        ids = encode(prompt)
        # Keep only the last `max_ctx_tokens` ids to fit the model context
        ids = ids[-args.max_ctx_tokens :]
        idx = torch.tensor([ids], dtype=torch.long)

        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p > 0 else None,
        )[0].tolist()

        # Extract only the newly generated portion after the prompt
        gen_text = decode(out[len(ids) :])
        # Stop at a newline if present (end of assistant turn)
        if "\n" in gen_text:
            gen_text = gen_text.split("\n", 1)[0]
        print(f"Assistant: {gen_text}")
        turns.append(f"Assistant: {gen_text}\n")


if __name__ == "__main__":
    main()
