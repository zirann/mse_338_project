"""Appearance metrics: length, structural complexity, epistemic-marker density.

Three headline appearance metrics computed on the held-out evaluation set after
each DPO round. All functions are pure, stateless, and cheap (no model calls).

- `length`: whitespace-token count per response.
- `structural_complexity`: `header_count + bullet_count` per response.
- `epistemic_marker_density`: count of lexicon-listed phrases per 100 tokens.

The lexicon spans five subclasses (hedges, caveats, nuance phrases, confidence
markers, performative deliberation, analytical tone); the headline metric uses
the union, with per-subclass breakdown available for the appendix.
"""
from __future__ import annotations

import re
from typing import Iterable


# Frozen epistemic-marker lexicon. Edit ONCE before round 0; do not edit again.
EPISTEMIC_LEXICON: dict[str, list[str]] = {
    "hedge": [
        "it's worth noting",
        "it is worth noting",
        "generally speaking",
        "in some cases",
        "in most cases",
        "however",
        "one might argue",
        "arguably",
        "to some extent",
        "broadly speaking",
    ],
    "caveat": [
        "however, it should be acknowledged",
        "that said",
        "with that said",
        "having said that",
        "on the flip side",
        "it should be noted",
        "with the caveat that",
        "all that said",
    ],
    "nuance": [
        "on one hand",
        "on the other hand",
        "this depends on",
        "it varies",
        "context matters",
        "the answer is nuanced",
        "it's complicated",
    ],
    "confidence": [
        "importantly",
        "notably",
        "clearly",
        "definitely",
        "without a doubt",
        "as is well known",
        "undeniably",
        "certainly",
    ],
    "performative_deliberation": [
        "let me think",
        "step by step",
        "let's break this down",
        "thinking about this",
        "to think through",
        "considering this carefully",
    ],
    "analytical_tone": [
        "in summary",
        "key takeaways",
        "to summarize",
        "in conclusion",
        "overall",
        "in essence",
    ],
}


# Pre-compile structural regexes for speed and to keep functions stateless.
_HEADER_RE = re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+\S", re.MULTILINE)


# ---------------------------------------------------------------------------
# Atomic helpers
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Whitespace tokenize a string. No normalization beyond split.

    Empty input returns an empty list, not [''].
    """
    if not text:
        return []
    return text.split()


def length(text: str) -> int:
    """Whitespace-token count for one response."""
    return len(tokenize(text))


def structural_complexity(text: str) -> int:
    """`header_count + bullet_count` for one response.

    Headers: lines beginning with 1-6 hash chars followed by a space and at
    least one non-whitespace character.
    Bullets: lines beginning with '-', '*', '+', or a numbered marker '\\d+.'
    followed by a space and at least one non-whitespace character.
    """
    if not text:
        return 0
    return len(_HEADER_RE.findall(text)) + len(_BULLET_RE.findall(text))


def _count_phrases(text: str, phrases: Iterable[str]) -> int:
    """Case-insensitive substring count over a list of phrases. Sum across phrases."""
    if not text:
        return 0
    lowered = text.lower()
    total = 0
    for phrase in phrases:
        if not phrase:
            continue
        total += lowered.count(phrase.lower())
    return total


def epistemic_marker_density(
    text: str,
    lexicon: dict[str, list[str]] | None = None,
) -> float:
    """Count of lexicon phrases per 100 tokens, summed across all subclasses.

    Returns 0.0 on empty input (length zero).
    """
    n = length(text)
    if n == 0:
        return 0.0
    lex = lexicon if lexicon is not None else EPISTEMIC_LEXICON
    total = 0
    for phrases in lex.values():
        total += _count_phrases(text, phrases)
    return 100.0 * total / n


def epistemic_marker_breakdown(
    text: str,
    lexicon: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Per-subclass density (count per 100 tokens). Sum of values equals
    `epistemic_marker_density(text, lexicon)`.

    Returns a dict with one entry per subclass key in the lexicon. On empty
    text or zero-length output, every entry is 0.0.
    """
    lex = lexicon if lexicon is not None else EPISTEMIC_LEXICON
    n = length(text)
    if n == 0:
        return {cls: 0.0 for cls in lex}
    return {
        cls: 100.0 * _count_phrases(text, phrases) / n
        for cls, phrases in lex.items()
    }


# ---------------------------------------------------------------------------
# Public composite
# ---------------------------------------------------------------------------


def appearance_metrics(text: str) -> dict[str, float]:
    """Three headline appearance metrics for one response.

    Returns:
        {"length": int, "structural_complexity": int, "epistemic_marker_density": float}
    """
    return {
        "length": length(text),
        "structural_complexity": structural_complexity(text),
        "epistemic_marker_density": epistemic_marker_density(text),
    }
