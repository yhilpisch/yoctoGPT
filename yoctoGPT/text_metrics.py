"""Text quality metrics for yoctoGPT.

(c) Dr. Yves J. Hilpisch
AI-Powered by Different LLMs.

Provides a unified scorecard that combines readability, diversity,
repetition, vocabulary richness, and distributional metrics. Designed
to be imported from Colab notebooks after cloning the repo.

Typical usage:

    from yoctoGPT.text_metrics import score_text
    score_text(generated_text)
    score_text(generated_text, reference_text=corpus_text)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Diversity & repetition
# ---------------------------------------------------------------------------

def distinct_n(text: str, n: int = 1) -> float:
    """Ratio of unique n-grams to total n-grams.

    Returns 0.0 for texts shorter than *n* tokens.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams that appear more than once consecutively.

    High values indicate looping / degenerate output. Returns 0.0 for
    texts shorter than *n* + 1 tokens.
    """
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if len(ngrams) < 2:
        return 0.0
    consecutive_dup = sum(
        1 for i in range(1, len(ngrams)) if ngrams[i] == ngrams[i - 1]
    )
    return consecutive_dup / (len(ngrams) - 1)


def type_token_ratio(text: str) -> float:
    """Unique words divided by total words (vocabulary richness).

    TTR is length-dependent but useful for quick comparisons of
    similarly-sized outputs.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ---------------------------------------------------------------------------
# Distributional: KL-divergence of character frequencies
# ---------------------------------------------------------------------------

def _char_freq(text: str) -> dict[str, float]:
    """Normalized character frequency distribution (excludes whitespace)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {}
    counts = Counter(chars)
    total = len(chars)
    return {c: counts[c] / total for c in counts}


def kl_divergence(generated: str, reference: str) -> float:
    """Symmetrized KL-divergence between character frequency distributions.

    Lower values mean the generated text's character profile is closer to
    the reference corpus. Uses additive smoothing (epsilon=1e-6) to handle
    unseen characters.
    """
    p = _char_freq(generated)
    q = _char_freq(reference)
    if not p or not q:
        return float("inf")
    vocab = set(p.keys()) | set(q.keys())
    eps = 1e-6
    kl_pq = 0.0
    kl_qp = 0.0
    for c in vocab:
        pc = p.get(c, eps)
        qc = q.get(c, eps)
        kl_pq += pc * math.log(pc / qc)
        kl_qp += qc * math.log(qc / pc)
    return 0.5 * (kl_pq + kl_qp)


# ---------------------------------------------------------------------------
# Readability (requires textstat)
# ---------------------------------------------------------------------------

def _readability_scores(text: str) -> dict[str, object]:
    """Standard readability metrics via the textstat library.

    Returns an empty dict if textstat is not installed.
    """
    try:
        from textstat import textstat
    except ImportError:
        return {}
    return {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
        "Dale-Chall Score": textstat.dale_chall_readability_score(text),
        "Text Standard": textstat.text_standard(text, float_output=False),
    }


# ---------------------------------------------------------------------------
# Perplexity (requires model + tokenizer)
# ---------------------------------------------------------------------------

def perplexity(
    text: str,
    model,
    encode_fn,
    block_size: int = 256,
    device: str = "cpu",
) -> Optional[float]:
    """Compute perplexity of *text* under *model* using a sliding window.

    Returns None if the text is too short. Requires a trained model and
    an encode function (vocab.encode or tokenizer.encode).

    This is optional and typically called separately since the model
    must be loaded first.
    """
    import torch
    import torch.nn.functional as F

    ids = encode_fn(text)
    if isinstance(ids, list):
        ids = torch.tensor(ids, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    total_nll = 0.0
    total_tokens = 0
    model.eval()

    with torch.no_grad():
        for start in range(0, ids.size(1) - 1, block_size):
            end = min(start + block_size, ids.size(1))
            chunk = ids[:, start:end].to(device)
            targets = ids[:, start + 1 : end + 1].to(device)
            if targets.size(1) == 0:
                break
            logits = model(chunk)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            total_nll += loss.item() * targets.numel()
            total_tokens += targets.numel()

    if total_tokens == 0:
        return None
    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)


# ---------------------------------------------------------------------------
# Unified scorecard
# ---------------------------------------------------------------------------

def score_text(
    text: str,
    reference_text: Optional[str] = None,
) -> dict[str, object]:
    """Compute a comprehensive text quality scorecard.

    Metrics returned:

    - **Readability** (if textstat installed):
      Flesch Reading Ease, Flesch-Kincaid Grade, Dale-Chall, Text Standard
    - **Diversity**: Distinct-1, Distinct-2 (unique n-gram ratios)
    - **Repetition**: 3-gram repetition ratio (consecutive duplicate rate)
    - **Vocabulary**: Type-Token Ratio (unique/total words)

    If *reference_text* is provided (e.g. the training corpus), also includes:

    - **Distribution**: Char-freq KL-divergence (lower = closer to corpus)
    """
    card: dict[str, object] = {}

    # Readability
    read = _readability_scores(text)
    if read:
        card.update(read)

    # Diversity
    card["Distinct-1"] = round(distinct_n(text, 1), 4)
    card["Distinct-2"] = round(distinct_n(text, 2), 4)

    # Repetition
    card["Repetition-3g"] = round(repetition_ratio(text, 3), 4)

    # Vocabulary richness
    card["TTR"] = round(type_token_ratio(text), 4)

    # Distributional vs reference
    if reference_text is not None:
        card["Char-KL"] = round(kl_divergence(text, reference_text), 4)

    return card


def print_scorecard(card: dict[str, object]) -> None:
    """Pretty-print a scorecard returned by score_text()."""
    group_order = [
        "Flesch Reading Ease",
        "Flesch-Kincaid Grade",
        "Dale-Chall Score",
        "Text Standard",
        "Distinct-1",
        "Distinct-2",
        "Repetition-3g",
        "TTR",
        "Char-KL",
    ]
    for key in group_order:
        if key in card:
            print(f"  {key:22}: {card[key]}")
    # Print any extra keys not in the order
    for key in card:
        if key not in group_order:
            print(f"  {key:22}: {card[key]}")
