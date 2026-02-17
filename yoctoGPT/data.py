"""Data utilities for yoctoGPT.

This module provides:
- A minimal character vocabulary (`CharVocab`) for char-level training.
- Dataset helpers to read/write contiguous token ID arrays stored as raw
  32-bit integers in `.bin` files (fast and simple for random slicing).
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch


TokenIdArray = Union[torch.LongTensor, np.memmap]


def sanitize_text_for_char_corpus(
    text: str,
    mode: str = "none",
    lowercase: bool = False,
    collapse_whitespace: bool = False,
) -> str:
    """Optionally sanitize text before char-level vocab construction.

    Modes:
    - none: keep text unchanged
    - ascii: normalize and drop non-ASCII characters
    - basic: ascii + keep only letters/digits/basic punctuation/whitespace
    """

    mode = mode.lower()
    if mode not in {"none", "ascii", "basic"}:
        raise ValueError(f"Unknown sanitize mode: {mode}")

    cleaned = text
    if mode in {"ascii", "basic"}:
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")

    if mode == "basic":
        allowed = set(string.ascii_letters + string.digits + " \t\n.,;:!?\"'()-[]")
        cleaned = "".join(ch if ch in allowed else " " for ch in cleaned)

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if collapse_whitespace:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in cleaned.split("\n")]
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned


@dataclass
class CharVocab:
    """Character vocabulary used for char-level modeling.

    The vocabulary is derived from a corpus by collecting all unique
    characters. Encoding maps characters to IDs, decoding maps IDs back to
    characters. No special tokens are required for minimal LM training.
    """

    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = chars
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(stoi={k: int(v) for k, v in data["stoi"].items()}, itos=data["itos"])


def save_ids_bin(ids: np.ndarray, path: str | Path) -> None:
    """Save a 1D array of token IDs as raw int32 `.bin`.

    The file can be read back using `load_bin`. Using raw int32 keeps the
    format simple and fast while remaining reasonably compact.
    """

    arr = np.asarray(ids, dtype=np.int32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(str(path))


def load_bin(path: str | Path) -> torch.LongTensor:
    """Load a `.bin` file produced by `save_ids_bin` into a LongTensor."""

    arr = np.fromfile(str(path), dtype=np.int32)
    return torch.from_numpy(arr.astype(np.int64))


def load_ids_adaptive(
    path: str | Path,
    memmap_threshold_mb: int = 128,
    prefer_memmap: bool = False,
) -> TokenIdArray:
    """Load token IDs as tensor or memmap depending on file size.

    Small datasets are read fully into memory (faster random slicing).
    Large datasets use read-only memmap to keep RAM usage bounded.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    threshold_bytes = max(0, int(memmap_threshold_mb)) * 1024 * 1024
    use_memmap = bool(prefer_memmap or p.stat().st_size > threshold_bytes)
    if use_memmap:
        return np.memmap(str(p), dtype=np.int32, mode="r")
    return load_bin(p)


def make_windows(data_ids: TokenIdArray, block_size: int, ixs: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice a 1D ID tensor into input/target blocks at given start indices.

    - data_ids: shape (N,)
    - block_size: number of tokens per training example
    - ixs: shape (B,), start indices in [0, N - block_size - 1]
    Returns (x, y) where both are shape (B, block_size) and y is x shifted by 1.
    """

    if isinstance(data_ids, torch.Tensor):
        x = torch.stack([data_ids[i : i + block_size] for i in ixs])
        y = torch.stack([data_ids[i + 1 : i + 1 + block_size] for i in ixs])
        return x, y

    starts = ixs.detach().cpu().numpy().astype(np.int64, copy=False)
    offsets = np.arange(block_size, dtype=np.int64)[None, :]
    x_np = np.asarray(data_ids[starts[:, None] + offsets], dtype=np.int64)
    y_np = np.asarray(data_ids[starts[:, None] + offsets + 1], dtype=np.int64)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    return x, y
