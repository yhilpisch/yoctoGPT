"""Recommend a training configuration from a prepared corpus.

Given a data directory containing train.bin/val.bin and (for token mode) a
tokenizer.json, this script analyzes corpus size and recommends a model type
and hyperparameters tailored to smaller corpora by default. It prints a full
`python -m yoctoGPT.train ...` command you can copy-paste, plus a matching
resume command.

Heuristics target readability and generalization on modest datasets:
- Prefer Fast variant (`gpt_fast`) by default for higher throughput
- Use cosine LR with warmup; enable EMA
- Choose tokens/step and context length to balance throughput and quality

You can override defaults via CLI flags if desired.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import time

from yoctoGPT.data import load_bin, CharVocab
from yoctoGPT.tokenizer import load_tokenizer


def human(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n}{unit}"
        n //= 1000
    return f"{n}T"


def recommend_from_counts(train_tokens: int, val_tokens: int, vocab_size: int, mode: str) -> dict:
    total = train_tokens + val_tokens
    small = train_tokens < 1_000_000
    mid = 1_000_000 <= train_tokens < 5_000_000

    # Model size heuristics
    if small:
        n_layer, n_head, n_embd = 4, 4, 256
        dropout = 0.2
        weight_decay = 0.1
    elif mid:
        n_layer, n_head, n_embd = 6, 6, 384
        dropout = 0.15
        weight_decay = 0.08
    else:
        n_layer, n_head, n_embd = 8, 8, 512
        dropout = 0.1
        weight_decay = 0.05

    # Context vs batch: aim for ~16k tokens/step; longer context if data allows
    target_tps = 16_384
    block_size = 512 if train_tokens >= 500_000 else 256
    batch_size = max(8, target_tps // block_size)

    # Learning rate schedule
    base_lr = 2e-4 if (small or mid) else 3e-4
    warmup_iters = 500
    min_lr = 1e-5

    # Label smoothing
    label_smoothing = 0.05 if (small or mid) else 0.02

    # Steps: target ~8–12x coverage of corpus by tokens
    coverage = 10
    tokens_per_step = batch_size * block_size
    max_iters = max(1000, int(math.ceil((coverage * train_tokens) / max(1, tokens_per_step))))

    # Choose model type: default to gpt_fast for speed; user can change later
    model_type = "gpt_fast"

    return dict(
        mode=mode,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        batch_size=batch_size,
        dropout=dropout,
        weight_decay=weight_decay,
        lr=base_lr,
        warmup_iters=warmup_iters,
        min_lr=min_lr,
        label_smoothing=label_smoothing,
        max_iters=max_iters,
        model_type=model_type,
        tie_weights=True,
        cosine_lr=True,
        ema=True,
        ema_decay=0.999,
        eval_interval=250,
        eval_iters=200,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Recommend a training command from a prepared corpus")
    p.add_argument("--mode", choices=["char", "token"], default="token")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with train.bin and val.bin")
    p.add_argument("--tokenizer_path", type=str, default=None, help="tokenizer.json path (token mode)")
    p.add_argument("--ckpt_dir", type=str, default="/Users/yves/Temp/checkpoints/reco", help="Output checkpoint directory suggestion")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    assert train_bin.exists() and val_bin.exists(), "Expected train.bin and val.bin in data_dir"

    train_tokens = int(load_bin(train_bin).numel())
    val_tokens = int(load_bin(val_bin).numel())

    if args.mode == "token":
        tok_path = args.tokenizer_path or (data_dir / "tokenizer.json")
        assert Path(tok_path).exists(), "Missing tokenizer.json; provide --tokenizer_path"
        tokenizer = load_tokenizer(tok_path)
        vocab_size = int(tokenizer.vocab_size)
    else:
        vocab_path = data_dir / "vocab.json"
        assert vocab_path.exists(), "Missing vocab.json for char mode"
        vocab = CharVocab.load(vocab_path)
        vocab_size = int(vocab.vocab_size)

    rec = recommend_from_counts(train_tokens, val_tokens, vocab_size, args.mode)

    ckpt_dir = args.ckpt_dir

    # Compose command
    cmd = [
        "python -m yoctoGPT.train",
        f"--mode {rec['mode']}",
        f"--data_dir {data_dir}",
    ]
    if args.mode == "token":
        tok_path = args.tokenizer_path or (data_dir / "tokenizer.json")
        cmd.append(f"--tokenizer_path {tok_path}")
    cmd.extend(
        [
            f"--ckpt_dir {ckpt_dir}",
            f"--model_type {rec['model_type']}",
            f"--n_layer {rec['n_layer']}",
            f"--n_head {rec['n_head']}",
            f"--n_embd {rec['n_embd']}",
            f"--block_size {rec['block_size']}",
            f"--batch_size {rec['batch_size']}",
            f"--dropout {rec['dropout']}",
            f"--weight_decay {rec['weight_decay']}",
            "--tie_weights",
            f"--label_smoothing {rec['label_smoothing']}",
            f"--eval_interval {rec['eval_interval']}",
            f"--eval_iters {rec['eval_iters']}",
            "--cosine_lr",
            f"--warmup_iters {rec['warmup_iters']}",
            f"--min_lr {rec['min_lr']}",
            f"--lr {rec['lr']}",
            f"--max_iters {rec['max_iters']}",
            "--ema",
            f"--ema_decay {rec['ema_decay']}",
        ]
    )

    print("Corpus analysis:")
    print(f"- mode: {args.mode}")
    print(f"- vocab_size: {vocab_size}")
    print(f"- train tokens: {train_tokens} ({human(train_tokens)})")
    print(f"- val tokens:   {val_tokens} ({human(val_tokens)})")
    print()
    print("Recommended training command:")
    train_cmd = " ".join(cmd)
    print(train_cmd)
    print()
    print("Resume command (additional steps):")
    resume_cmd = train_cmd + f" --resume {str(ckpt_dir)}/latest.pt"
    print(resume_cmd)


if __name__ == "__main__":
    main()
