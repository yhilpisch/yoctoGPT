"""Sampling script for yoctoGPT.

Loads a trained checkpoint and generates text from a user-provided prompt
using the same encoding used during training (char or token mode).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import CharVocab
from .model import GPT, GPTConfig
from .tokenizer import load_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a yoctoGPT checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint .pt")
    p.add_argument("--mode", choices=["char", "token"], default="char")
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--vocab_path", type=str, default=None)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Resolve encoding/decoding
    if args.mode == "char":
        vocab_path = args.vocab_path or ckpt.get("char_vocab_path")
        assert vocab_path, "Provide --vocab_path or ensure checkpoint stores char_vocab_path"
        vocab = CharVocab.load(vocab_path)
        encode = lambda s: torch.tensor([vocab.encode(s)], dtype=torch.long)
        decode = lambda ids: vocab.decode(ids)
    else:
        tok_path = args.tokenizer_path or ckpt.get("tokenizer_path")
        assert tok_path, "Provide --tokenizer_path or ensure checkpoint stores tokenizer_path"
        tokenizer = load_tokenizer(tok_path)
        encode = lambda s: torch.tensor([tokenizer.encode(s)], dtype=torch.long)
        decode = lambda ids: tokenizer.decode(ids)

    # Prepare prompt
    idx = encode(args.prompt)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p > 0 else None,
    )
    generated = out[0].tolist()
    print(decode(generated))


if __name__ == "__main__":
    main()
