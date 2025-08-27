"""Training smoke test for yoctoGPT (prefers Apple Silicon).

This script performs a very short training run on the default corpus using a
tiny model and prints losses to verify backprop, optimizer steps, and device
handling. It defaults to Apple Silicon `mps` when available, then CUDA, then
CPU. Use `--device` to override.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import csv
import time

# Ensure project root is on sys.path when running as a file: `python scripts/...`
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from yoctoGPT.data import CharVocab
from yoctoGPT.model import GPT, GPTConfig
from yoctoGPT.advanced_model import AdvancedGPT, AdvancedGPTConfig
from yoctoGPT.performance_model import PerformanceGPT, PerformanceGPTConfig


def detect_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="yoctoGPT training smoke test (short run)")
    p.add_argument("--text_path", type=str, default="data/philosophy.txt")
    p.add_argument("--device", type=str, default=None, help="cpu|mps|cuda (auto if None)")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ckpt", type=str, default="checkpoints/smoke/latest.pt")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (model+optimizer)")
    p.add_argument("--init_from", type=str, default=None, help="Warm start from checkpoint (model only)")
    p.add_argument("--strict_init", dest="strict_init", action="store_true")
    p.add_argument("--no_strict_init", dest="strict_init", action="store_false")
    p.set_defaults(strict_init=True)
    p.add_argument("--model_type", choices=["gpt", "gpt_plus", "gpt_fast"], default="gpt", help="Select baseline, advanced, or fast variant")
    return p.parse_args()


def get_batch(data_ids: torch.LongTensor, block_size: int, batch_size: int, device: str):
    ixs = torch.randint(0, len(data_ids) - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i : i + block_size] for i in ixs]).to(device)
    y = torch.stack([data_ids[i + 1 : i + 1 + block_size] for i in ixs]).to(device)
    return x, y


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = detect_device(args.device)
    print(f"Using device: {device}")

    text = Path(args.text_path).read_text(encoding="utf-8")
    vocab = CharVocab.from_text(text)
    ids = torch.tensor(vocab.encode(text), dtype=torch.long)

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        arch = ckpt.get("arch", "gpt")
        if arch == "gpt_plus":
            cfg = AdvancedGPTConfig(**ckpt["model_config"])
            model = AdvancedGPT(cfg).to(device)
        elif arch == "gpt_fast":
            cfg = PerformanceGPTConfig(**ckpt["model_config"])
            model = PerformanceGPT(cfg).to(device)
        else:
            cfg = GPTConfig(**ckpt["model_config"])
            model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        if "opt_state" in ckpt:
            opt.load_state_dict(ckpt["opt_state"])
        start_iter = int(ckpt.get("iters_completed", 0))
        print(f"Resumed from {args.resume} at iter {start_iter}")
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        arch = ckpt.get("arch", "gpt")
        if arch == "gpt_plus":
            cfg = AdvancedGPTConfig(**ckpt["model_config"])
            model = AdvancedGPT(cfg).to(device)
        elif arch == "gpt_fast":
            cfg = PerformanceGPTConfig(**ckpt["model_config"])
            model = PerformanceGPT(cfg).to(device)
        else:
            cfg = GPTConfig(**ckpt["model_config"])
            model = GPT(cfg).to(device)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=args.strict_init)
        print(
            f"Warm-start from {args.init_from}; strict={args.strict_init} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        if args.model_type == "gpt_plus":
            cfg = AdvancedGPTConfig(
                vocab_size=vocab.vocab_size,
                block_size=args.block_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                dropout=0.0,
            )
            model = AdvancedGPT(cfg).to(device)
        elif args.model_type == "gpt_fast":
            cfg = PerformanceGPTConfig(
                vocab_size=vocab.vocab_size,
                block_size=args.block_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                dropout=0.0,
            )
            model = PerformanceGPT(cfg).to(device)
        else:
            cfg = GPTConfig(
                vocab_size=vocab.vocab_size,
                block_size=args.block_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                dropout=0.0,
            )
            model = GPT(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Metrics CSV alongside checkpoint
    ckpt_path = Path(args.ckpt)
    metrics_path = ckpt_path.parent / "metrics.csv"
    fields = ["iter", "train_loss", "lr", "time_sec", "tokens_seen", "throughput_tps"]
    if not metrics_path.exists():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    model.train()
    start_wall = time.time()
    tokens_per_step = args.batch_size * args.block_size
    for it in range(start_iter + 1, start_iter + args.iters + 1):
        xb, yb = get_batch(ids, args.block_size, args.batch_size, device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % max(1, args.iters // 10) == 0 or it == start_iter + 1:
            print(f"iter {it:4d}/{start_iter + args.iters}  loss {loss.item():.4f}")
        # Append metrics row
        elapsed = time.time() - start_wall
        tokens_seen = (it - start_iter) * tokens_per_step
        tps = tokens_seen / max(elapsed, 1e-9)
        time_sec_out = int(round(elapsed))
        tps_out = int(round(tps))
        with metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(
                {
                    "iter": it,
                    "train_loss": round(float(loss.item()), 5),
                    "lr": round(float(args.lr), 5),
                    "time_sec": time_sec_out,
                    "tokens_seen": int(tokens_seen),
                    "throughput_tps": tps_out,
                }
            )

    # Quick sample after short training
    model.eval()
    prompt = "Philosophy: "
    # Sanitize prompt to characters present in the corpus vocabulary
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

    prompt_ids = vocab.encode(sanitize_to_vocab(prompt))
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=100)
    print("\nSample after training:\n")
    print(vocab.decode(out[0].tolist()))

    # Save checkpoint to allow resuming
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": cfg.__dict__,
            "arch": ("gpt_plus" if isinstance(model, AdvancedGPT) else "gpt"),
            "opt_state": opt.state_dict(),
            "iters_completed": start_iter + args.iters,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
