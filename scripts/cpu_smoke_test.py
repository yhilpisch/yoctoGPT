"""CPU generation smoke test for yoctoGPT.

This script exercises the minimal end-to-end generation stack on CPU without
requiring a pre-trained checkpoint. It:

1) Reads the default corpus (`data/philosophy.txt`).
2) Builds a character vocabulary from the text.
3) Instantiates a tiny GPT model sized to the vocabulary.
4) Encodes a short prompt and generates a few characters on CPU.

The output is random (weights are untrained) but the goal is to verify that the
model, tokenizer, and generation codepaths work correctly on a plain CPU-only
environment. Use the training script for meaningful outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

# Ensure project root is on sys.path when running as a file: `python scripts/...`
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from yoctoGPT.data import CharVocab
from yoctoGPT.model import GPT, GPTConfig
from yoctoGPT.advanced_model import AdvancedGPT, AdvancedGPTConfig
from yoctoGPT.performance_model import PerformanceGPT, PerformanceGPTConfig


def parse_args():
    p = argparse.ArgumentParser(description="CPU generation smoke test for yoctoGPT (char-level)")
    p.add_argument("--text_path", type=str, default="data/philosophy.txt")
    p.add_argument("--prompt", type=str, default="What is virtue?\n")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--model_type", choices=["gpt", "gpt_plus", "gpt_fast"], default="gpt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    text_path = Path(args.text_path)
    assert text_path.exists(), f"Missing corpus at {text_path}"
    text = text_path.read_text(encoding="utf-8")

    # 1) Build char vocab and infer size
    vocab = CharVocab.from_text(text)
    vocab_size = vocab.vocab_size

    # 2) Tiny GPT config; keep small for quick CPU run
    if args.model_type == "gpt_plus":
        cfg = AdvancedGPTConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=0.0,
        )
        model = AdvancedGPT(cfg)
    elif args.model_type == "gpt_fast":
        cfg = PerformanceGPTConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=0.0,
        )
        model = PerformanceGPT(cfg)
    else:
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=0.0,
        )
        model = GPT(cfg)
    model.eval()  # sampling mode

    # 3) Encode prompt (sanitize to known chars) and generate on CPU
    def sanitize_to_vocab(text: str) -> str:
        buf = []
        for ch in text:
            if ch in vocab.stoi:
                buf.append(ch)
            elif ch.lower() in vocab.stoi:
                buf.append(ch.lower())
            # else drop unknown char
        if not buf:
            buf = [vocab.itos[0]]
        return "".join(buf)

    prompt_ids = vocab.encode(sanitize_to_vocab(args.prompt))
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p > 0 else None,
    )[0].tolist()
    print(vocab.decode(out))


if __name__ == "__main__":
    main()
