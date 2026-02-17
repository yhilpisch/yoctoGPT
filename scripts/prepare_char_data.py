"""Prepare character-level training data for yoctoGPT.

Reads a text corpus (default: data/philosophy.txt), builds a character
vocabulary, encodes the entire text to integer IDs, splits into train/val, and
writes `train.bin`, `val.bin`, and `vocab.json` to the output directory.
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

from yoctoGPT.data import CharVocab, save_ids_bin, sanitize_text_for_char_corpus


def parse_args():
    p = argparse.ArgumentParser(description="Prepare char-level dataset")
    p.add_argument("--text_path", type=str, default="data/philosophy.txt")
    p.add_argument("--out_dir", type=str, default="data/char")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument(
        "--sanitize_chars",
        choices=["none", "ascii", "basic"],
        default="none",
        help="Optional text sanitization before building char vocabulary",
    )
    p.add_argument("--lowercase", action="store_true", help="Lowercase corpus before encoding")
    p.add_argument(
        "--collapse_whitespace",
        action="store_true",
        help="Collapse repeated spaces/tabs and large blank-line runs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.text_path).read_text(encoding="utf-8")
    original_vocab_size = len(set(text))
    text = sanitize_text_for_char_corpus(
        text,
        mode=args.sanitize_chars,
        lowercase=args.lowercase,
        collapse_whitespace=args.collapse_whitespace,
    )
    if not text:
        raise ValueError("Prepared corpus is empty after sanitization; relax --sanitize_chars settings")

    # Build vocabulary and encode the entire corpus
    vocab = CharVocab.from_text(text)
    ids = np.array(vocab.encode(text), dtype=np.int32)

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
    vocab.save(out_dir / "vocab.json")

    print(f"Wrote {len(train_ids)} train and {len(val_ids)} val tokens.")
    print(f"Vocab size: {vocab.vocab_size}")
    if args.sanitize_chars != "none" or args.lowercase or args.collapse_whitespace:
        print(
            "Sanitization:",
            f"mode={args.sanitize_chars}",
            f"lowercase={args.lowercase}",
            f"collapse_whitespace={args.collapse_whitespace}",
            f"vocab_before={original_vocab_size}",
            f"vocab_after={vocab.vocab_size}",
        )


if __name__ == "__main__":
    main()
