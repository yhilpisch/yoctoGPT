"""A minimal word-level tokenizer for token-mode training.

This implementation intentionally favors simplicity and readability:
- Splits text on whitespace and basic punctuation boundaries.
- Learns a vocabulary of the most frequent tokens up to `vocab_size`.
- Provides `encode`/`decode` along with JSON serialization.

While not as powerful as BPE, it is adequate to demonstrate token-level GPT
training with a compact, dependency-free tokenizer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


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

