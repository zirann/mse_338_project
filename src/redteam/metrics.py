from __future__ import annotations

import re
from typing import Sequence

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def distinct_n(texts: Sequence[str], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be >= 1")
    total = 0
    unique = set()
    for text in texts:
        toks = tokenize(text)
        if len(toks) < n:
            continue
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return (len(unique) / total) if total else 0.0


def duplication_ratio(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    normalized = [" ".join(tokenize(t)) for t in texts]
    unique = len(set(normalized))
    return 1.0 - unique / len(texts)


def avg_length_tokens(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    lengths = [len(tokenize(t)) for t in texts]
    return sum(lengths) / len(lengths)


def harmful_hit_rate(scores: Sequence[float], threshold: float) -> float:
    """Fraction of scores at or above threshold. Used for per-evaluator hit-rate summaries."""
    if not scores:
        return 0.0
    hits = sum(1 for s in scores if s >= threshold)
    return hits / len(scores)
