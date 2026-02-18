"""Clean book-like corpora before tokenization."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

# Ensure project root is on sys.path when running as a file: `python scripts/...`
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from yoctoGPT.preprocess import clean_book_text


def parse_args():
    p = argparse.ArgumentParser(description="Clean technical book text files for model training")
    p.add_argument("--in_dir", type=str, default="finance", help="Input directory containing .txt files")
    p.add_argument("--out_dir", type=str, default="finance_clean", help="Output directory for cleaned .txt files")
    p.add_argument("--glob", type=str, default="*.txt", help="Glob pattern to select files")
    p.add_argument("--keep_back_matter", action="store_true", help="Keep references/index/appendix sections")
    p.add_argument("--lowercase", action="store_true", help="Lowercase cleaned text")
    p.add_argument("--no_collapse_whitespace", dest="collapse_whitespace", action="store_false")
    p.set_defaults(collapse_whitespace=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    files = sorted([p for p in in_dir.glob(args.glob) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} in {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    total_in = 0
    total_out = 0
    for fp in files:
        src = fp.read_text(encoding="utf-8")
        total_in += len(src)
        cleaned = clean_book_text(
            src,
            drop_back_matter=(not args.keep_back_matter),
            lowercase=args.lowercase,
            collapse_whitespace=args.collapse_whitespace,
        )
        out_fp = out_dir / fp.name
        out_fp.write_text(cleaned, encoding="utf-8")
        total_out += len(cleaned)
        ratio = (len(cleaned) / max(1, len(src))) * 100.0
        print(f"{fp.name}: {len(src)} -> {len(cleaned)} chars ({ratio:.1f}%)")

    print(f"Wrote {len(files)} cleaned file(s) to {out_dir}")
    overall = (total_out / max(1, total_in)) * 100.0
    print(f"Total chars: {total_in} -> {total_out} ({overall:.1f}%)")


if __name__ == "__main__":
    main()

