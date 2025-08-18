"""Prepare token-level training data for picoGPT using a minimal tokenizer.

Trains a simple word-level tokenizer on the provided text (default:
data/philosophy.txt), then encodes the corpus to token IDs, splits into
train/val, and writes `train.bin`, `val.bin`, and `tokenizer.json` to the
output directory.
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

import numpy as np

from picoGPT.data import save_ids_bin
from picoGPT.tokenizer import WordLevelTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Prepare token-level dataset with WordLevelTokenizer")
    p.add_argument("--text_path", type=str, default="data/philosophy.txt")
    p.add_argument("--out_dir", type=str, default="data/token")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--val_ratio", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.text_path).read_text(encoding="utf-8")

    tok = WordLevelTokenizer.train(text, vocab_size=args.vocab_size)
    ids = np.array(tok.encode(text), dtype=np.int32)

    # Train/val split by contiguous slicing
    n = len(ids)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_ids_bin(train_ids, out_dir / "train.bin")
    save_ids_bin(val_ids, out_dir / "val.bin")
    tok.save(out_dir / "tokenizer.json")

    print(f"Wrote {len(train_ids)} train and {len(val_ids)} val tokens.")
    print(f"Vocab size: {tok.vocab_size}")


if __name__ == "__main__":
    main()
