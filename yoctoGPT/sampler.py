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
from .advanced_model import AdvancedGPT, AdvancedGPTConfig
from .performance_model import PerformanceGPT, PerformanceGPTConfig
from .tokenizer import load_tokenizer
from .utils import detect_device, load_model_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a yoctoGPT checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint .pt")
    p.add_argument("--mode", choices=["char", "token"], default="char")
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--vocab_path", type=str, default=None)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None, help="Top-k sampling (disabled if omitted)")
    p.add_argument("--top_p", type=float, default=None, help="Nucleus sampling threshold (disabled if omitted)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default=None, help="Device for inference: cpu, cuda, or mps (auto if omitted)")
    p.add_argument("--compile", action="store_true", help="Compile model.forward with torch.compile if available")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = detect_device(args.device)

    model, ckpt = load_model_from_checkpoint(args.ckpt, device=device)
    arch = ckpt.get("arch", "gpt")
    if args.compile:
        if hasattr(torch, "compile"):
            try:
                model.forward = torch.compile(model.forward)  # type: ignore[method-assign]
                print("Compiled model.forward with torch.compile")
            except Exception as e:
                print(f"torch.compile unavailable for this run; continuing uncompiled ({e})")
        else:
            print("torch.compile not available in this PyTorch build; continuing uncompiled")

    # Resolve encoding/decoding
    if args.mode == "char":
        vocab_path = args.vocab_path or ckpt.get("char_vocab_path")
        assert vocab_path, "Provide --vocab_path or ensure checkpoint stores char_vocab_path"
        vocab = CharVocab.load(vocab_path)
        encode = lambda s: torch.tensor([vocab.encode(s)], dtype=torch.long, device=device)
        decode = lambda ids: vocab.decode(ids)
    else:
        tok_path = args.tokenizer_path or ckpt.get("tokenizer_path")
        assert tok_path, "Provide --tokenizer_path or ensure checkpoint stores tokenizer_path"
        tokenizer = load_tokenizer(tok_path)
        eos_token = getattr(tokenizer, "eos_id", None)
        encode = lambda s: torch.tensor([tokenizer.encode(s, add_bos=True)], dtype=torch.long, device=device)
        decode = lambda ids: tokenizer.decode(ids)
    if args.mode == "char":
        eos_token = None

    # Prepare prompt
    idx = encode(args.prompt)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token=eos_token,
    )
    generated = out[0].detach().cpu().tolist()
    print(decode(generated))


if __name__ == "__main__":
    main()
