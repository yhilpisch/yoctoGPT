"""Text preprocessing helpers for corpus preparation.

(c) Dr. Yves J. Hilpisch
AI-Powered by Different LLMs.
"""

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

# Unicode math symbols found in plain-text PDF extractions.
_FORMULA_SYMBOL_RE = re.compile(
    "["+
    "\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03b9\u03ba"
    "\u03bb\u03bc\u03bd\u03be\u03bf\u03c0\u03c1\u03c2\u03c3\u03c4"
    "\u03c5\u03c6\u03c7\u03c8\u03c9"  # Greek lowercase
    "\u0391\u0392\u0393\u0394\u0395\u0396\u0397\u0398\u0399\u039a"
    "\u039b\u039c\u039d\u039e\u039f\u03a0\u03a1\u03a3\u03a4\u03a5"
    "\u03a6\u03a7\u03a8\u03a9"  # Greek uppercase
    "\u2211\u222b\u2202\u2207\u221a\u2264\u2265\u2260\u2248\u2192"
    "\u2190\u2194\u2200\u2203\u2208\u2209\u2282\u2286\u222a\u2229"
    "\u221e\u00b1\u00d7\u00f7"  # Math operators
    "\u00b2\u00b3\u2070\u2074\u2075\u2076\u2077\u2078\u2079"  # Superscripts
    "]"
)


def _is_formula_line(line: str) -> bool:
    """Heuristic: detect lines that look like math formulas or code.

    A line is considered formula-like if any of the following hold:
    - It contains Unicode math symbols (Greek letters, operators).
    - Fewer than 50% of its non-space characters are ASCII alphabetic.
    - It contains ``=`` and fewer than 60% are ASCII alphabetic
      (catches ``sigma = 0.20`` style assignments).
    - It contains subscript patterns like ``S_t`` or ``x_0``
      (letter, underscore, alphanumeric).
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 5:
        return False

    if _FORMULA_SYMBOL_RE.search(stripped):
        return True

    non_space = [c for c in stripped if not c.isspace()]
    if not non_space:
        return False

    alpha_count = sum(1 for c in non_space if c.isascii() and c.isalpha())
    ratio = alpha_count / len(non_space)

    if ratio < 0.5:
        return True

    if "=" in stripped and ratio < 0.6:
        return True

    if re.search(r"[A-Za-z]_\w", stripped):
        return True

    return False


def clean_book_text(
    text: str,
    drop_back_matter: bool = True,
    drop_formulas: bool = False,
    lowercase: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """Clean technical book text while preserving core instructional content.

    Removes publisher/legal boilerplate and common back-matter sections
    (references/index/etc.) that can dominate small-corpus training with
    citation loops.

    When *drop_formulas* is True, lines that look like math formulas or
    code fragments (high symbol density or Unicode math characters) are
    also removed.  This improves signal-to-noise for small corpora where
    the model cannot learn formula syntax.
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

        if drop_formulas and _is_formula_line(stripped):
            continue

        kept.append(line)

    cleaned = "\n".join(kept)
    if collapse_whitespace:
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned

