"""Text preprocessing helpers for corpus preparation."""

from __future__ import annotations

import re


_DROP_SECTION_RE = re.compile(
    r"\b("
    r"references?|bibliograph(y|ies)|index|acknowledg(e)?ments?|"
    r"about the author|where to go from here|further reading|"
    r"appendix|glossary"
    r")\b",
    re.IGNORECASE,
)

_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(chapter|part|section)\b|^\s*\d+(\.\d+)*\s+[A-Za-z]",
    re.IGNORECASE,
)

_LINE_DROP_RES = [
    re.compile(r"\bISBN\b", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"\bO['’]Reilly\b", re.IGNORECASE),
    re.compile(r"\bSebastopol\b", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*\|\s*chapter\b", re.IGNORECASE),
    re.compile(r"^\s*chapter\s+\d+\s*:\s*where to go from here\??\s*$", re.IGNORECASE),
]


def clean_book_text(
    text: str,
    drop_back_matter: bool = True,
    lowercase: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """Clean technical book text while preserving core instructional content.

    The heuristic focuses on removing publisher/legal boilerplate and common
    back-matter sections (references/index/etc.) that can dominate small-corpus
    training with citation loops.
    """

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: list[str] = []
    skip_back_matter = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        # Start skipping once common back-matter headings are reached.
        if drop_back_matter and _DROP_SECTION_RE.search(stripped):
            if _CHAPTER_HEADING_RE.search(stripped) or len(stripped) <= 80:
                skip_back_matter = True
                continue
        if skip_back_matter:
            # If we later encounter a plausible chapter heading that is not
            # itself a drop section marker, resume keeping content.
            if _CHAPTER_HEADING_RE.search(stripped) and not _DROP_SECTION_RE.search(stripped):
                skip_back_matter = False
            else:
                continue

        if any(rx.search(line) for rx in _LINE_DROP_RES):
            continue

        kept.append(line)

    cleaned = "\n".join(kept)
    if collapse_whitespace:
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned

