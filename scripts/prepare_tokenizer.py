"""Prepare token-level training data for yoctoGPT using a minimal tokenizer.

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

from yoctoGPT.data import save_ids_bin
from yoctoGPT.tokenizer import train_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Prepare token-level dataset (BPE by default)")
    p.add_argument("--text_path", type=str, default="data/philosophy.txt", help="Single text file to use")
    p.add_argument("--all_txt_in_dir", action="store_true", help="Use all .txt files in --text_dir (ignores --text_path)")
    p.add_argument("--text_dir", type=str, default="data", help="Directory to scan for .txt files when --all_txt_in_dir is set")
    p.add_argument("--out_dir", type=str, default="data/token")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--backend", type=str, default="bpe", choices=["bpe", "word"], help="Tokenizer backend")
    p.add_argument("--random_split", action="store_true", help="Randomize train/val split (file-based if multiple files; chunk-based otherwise)")
    p.add_argument("--split_seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = []
    texts = []
    if args.all_txt_in_dir:
        dir_path = Path(args.text_dir)
        files = sorted([p for p in dir_path.glob("*.txt") if p.is_file()])
        assert files, f"No .txt files found in {dir_path}"
        print(f"Found {len(files)} .txt files in {dir_path}")
        texts = [p.read_text(encoding="utf-8") for p in files]
        text = "\n\n".join(texts)
    else:
        text = Path(args.text_path).read_text(encoding="utf-8")

    tok = train_tokenizer(text, vocab_size=args.vocab_size, backend=args.backend)
    # Encode and split
    rng = np.random.default_rng(args.split_seed)

    if args.all_txt_in_dir and args.random_split and files:
        # File-level random split: allocate whole files to train or val
        idxs = np.arange(len(files))
        rng.shuffle(idxs)
        n_val_files = max(1, int(len(files) * args.val_ratio))
        val_set = set(idxs[:n_val_files].tolist())
        train_ids_list = []
        val_ids_list = []
        for i, txt in enumerate(texts):
            enc = np.array(tok.encode(txt), dtype=np.int32)
            if i in val_set:
                val_ids_list.append(enc)
            else:
                train_ids_list.append(enc)
        train_ids = np.concatenate(train_ids_list) if train_ids_list else np.empty((0,), dtype=np.int32)
        val_ids = np.concatenate(val_ids_list) if val_ids_list else np.empty((0,), dtype=np.int32)
    else:
        # Single file or non-random split path; optionally use chunk-level shuffle
        ids = np.array(tok.encode(text), dtype=np.int32)
        if args.random_split:
            # Chunk into ~2048-token pieces and shuffle chunks before splitting
            chunk = 2048
            num_chunks = max(1, (len(ids) + chunk - 1) // chunk)
            chunks = [ids[i * chunk : (i + 1) * chunk] for i in range(num_chunks)]
            rng.shuffle(chunks)
            n_val_chunks = max(1, int(len(chunks) * args.val_ratio))
            val_chunks = chunks[:n_val_chunks]
            train_chunks = chunks[n_val_chunks:]
            train_ids = np.concatenate(train_chunks) if train_chunks else np.empty((0,), dtype=np.int32)
            val_ids = np.concatenate(val_chunks) if val_chunks else np.empty((0,), dtype=np.int32)
        else:
            # Contiguous tail split
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
