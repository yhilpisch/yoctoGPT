"""Tokenization utilities for yoctoGPT.

Default: Use a standard Byte-Pair Encoding (BPE) tokenizer via Hugging Face
`tokenizers` if available, which typically improves training and sampling
results versus a simple word-level tokenizer. If the dependency is not
installed, gracefully fall back to a minimal word-level tokenizer.

Both implementations expose the same minimal interface:
- encode(str) -> List[int]
- decode(List[int]) -> str
- vocab_size: int property
- save(path) / load(path)

The auto-loader inspects the file and returns an appropriate tokenizer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


TOKEN_UNK = "<unk>"
TOKEN_BOS = "<bos>"
TOKEN_EOS = "<eos>"


def simple_word_tokenize(text: str) -> List[str]:
    """Split text into tokens using a simple regex.

    - Keeps alphanumeric words and separate punctuation as tokens.
    - Lowercases by default for a tighter vocabulary (can be relaxed).
    """

    # Basic pattern: words (letters/digits/underscore) or single non-space punctuation
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    return [m.group(0).lower() for m in pattern.finditer(text)]


@dataclass
class WordLevelTokenizer:
    """Minimal vocabulary-based tokenizer using space/punct splitting.

    The vocabulary maps token string → id. Special tokens are included to allow
    for simple BOS/EOS handling if desired. The default encoding does not
    automatically insert these tokens; they are available for users who want to
    include them in prompts.
    """

    stoi: Dict[str, int]
    itos: List[str]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, unknown_token: str = TOKEN_UNK) -> List[int]:
        ids: List[int] = []
        unk_id = self.stoi.get(unknown_token)
        assert unk_id is not None, "Tokenizer missing <unk> token"
        for tok in simple_word_tokenize(text):
            ids.append(self.stoi.get(tok, unk_id))
        return ids

    def decode(self, ids: List[int]) -> str:
        # Re-join with spaces, then clean up spacing around punctuation
        toks = [self.itos[i] if 0 <= i < len(self.itos) else TOKEN_UNK for i in ids]
        text = " ".join(toks)
        # Simple fixups: remove space before punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text

    @classmethod
    def train(cls, text: str, vocab_size: int = 8000) -> "WordLevelTokenizer":
        """Train a tokenizer from raw text by frequency capping.

        - Counts token frequency using `simple_word_tokenize`.
        - Reserves IDs for special tokens, then adds top-N frequent tokens.
        """

        # Count tokens
        counter = Counter(simple_word_tokenize(text))

        # Special tokens at the start of the vocab
        itos: List[str] = [TOKEN_UNK, TOKEN_BOS, TOKEN_EOS]
        # Remaining slots for frequent tokens
        remaining = max(0, vocab_size - len(itos))
        most_common = [tok for tok, _ in counter.most_common(remaining)]
        itos.extend(most_common)
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "WordLevelTokenizer":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        itos: List[str] = data["itos"]
        stoi: Dict[str, int] = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


# ---- BPE (preferred) using Hugging Face `tokenizers` if available ----

def _hf_available() -> bool:
    try:
        import tokenizers  # noqa: F401
        return True
    except Exception:
        return False


class BPETokenizer:
    """Adapter over Hugging Face `tokenizers` to present the same interface.

    When training, we construct a whitespace-pretokenized, lowercasing BPE
    tokenizer with a provided vocab size and standard special tokens.
    """

    def __init__(self, hf_tokenizer: Any) -> None:  # `Any` to avoid hard dep
        self._tok = hf_tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        from tokenizers import Tokenizer as HFTokenizer

        tok = HFTokenizer.from_file(str(path))
        return cls(tok)

    @classmethod
    def train(cls, text: str, vocab_size: int = 8000) -> "BPETokenizer":
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        try:
            from tokenizers.normalizers import Lowercase
            normalizer = Lowercase()
        except Exception:
            normalizer = None

        tok = HFTokenizer(BPE(unk_token=TOKEN_UNK))
        if normalizer is not None:
            tok.normalizer = normalizer
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[TOKEN_UNK, TOKEN_BOS, TOKEN_EOS],
        )
        tok.train_from_iterator([text], trainer=trainer)
        return cls(tok)


# ---- Public helpers ----

def train_tokenizer(text: str, vocab_size: int = 8000, backend: str = "bpe"):
    """Train a tokenizer and return a compatible object.

    - backend="bpe" (default): requires `tokenizers` installed; falls back to
      word-level with a warning if unavailable.
    - backend="word": always use WordLevelTokenizer.
    """

    backend = backend.lower()
    if backend == "bpe":
        if _hf_available():
            return BPETokenizer.train(text, vocab_size=vocab_size)
        else:
            # Fallback with a gentle note
            print("[yoctoGPT] tokenizers not installed; falling back to word-level tokenizer.")
            return WordLevelTokenizer.train(text, vocab_size=vocab_size)
    elif backend == "word":
        return WordLevelTokenizer.train(text, vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown tokenizer backend: {backend}")


def load_tokenizer(path: str | Path):
    """Auto-load a tokenizer from disk.

    Tries to load as a Hugging Face `tokenizers` JSON first; if that fails,
    falls back to the simple word-level format.
    """

    path = str(path)
    if _hf_available():
        try:
            return BPETokenizer.load(path)
        except Exception:
            pass
    # Fallback to word-level
    return WordLevelTokenizer.load(path)
