"""Uncertainty signaling: lexicon, per-response score, and pair-filter mitigation.

This module operationalizes "epistemic uncertainty signaling" for the
uncertainty-suppression project. It reuses the hedge / caveat / nuance
subclasses from `appearance.EPISTEMIC_LEXICON` and adds an explicit
first-person-uncertainty group ("i don't know", "i'm not sure", "may",
"might", ...). The union is `UNCERTAINTY_LEXICON`.

Two consumers:

- `uncertainty_score(text)`: density (markers per 100 tokens) used as the
  reported uncertainty metric and as the filter key for Mitigation 1.
- `filter_uncertainty_preserving_pairs(pairs, epsilon)`: Mitigation 1
  (data-level). Drops preference pairs that would teach the policy to prefer
  LESS uncertainty.

The differentiable token-level regularizer (Mitigation 2) consumes
`UNCERTAINTY_LEXICON_PHRASES` to build a token-id set; see
`regularized_dpo.py`.
"""
from __future__ import annotations

from typing import Iterable

from .appearance import _HEDGE_SUBCLASSES, EPISTEMIC_LEXICON, _count_phrases, length


# Explicit first-person / modal uncertainty markers not already in the
# hedge/caveat/nuance subclasses. Kept small and high-precision.
EXPLICIT_UNCERTAINTY_PHRASES: list[str] = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "not certain",
    "not sure",
    "it's unclear",
    "it is unclear",
    "unclear",
    "uncertain",
    "hard to say",
    "i'm unsure",
    "possibly",
    "perhaps",
    "may ",     # trailing space: modal "may" not "maybe"/"mayor"
    "might ",
    "could be",
]


# The uncertainty lexicon = hedge + caveat + nuance subclasses (from the shared
# EPISTEMIC_LEXICON) plus the explicit-uncertainty group. Built once at import.
UNCERTAINTY_LEXICON: dict[str, list[str]] = {
    **{k: EPISTEMIC_LEXICON[k] for k in _HEDGE_SUBCLASSES},
    "explicit_uncertainty": EXPLICIT_UNCERTAINTY_PHRASES,
}

# Flat phrase list (used by the regularizer to build its token-id set).
UNCERTAINTY_LEXICON_PHRASES: list[str] = [
    phrase for phrases in UNCERTAINTY_LEXICON.values() for phrase in phrases
]


def uncertainty_score(text: str) -> float:
    """Density of uncertainty markers per 100 tokens.

    Sums matches across all `UNCERTAINTY_LEXICON` subclasses. Returns 0.0 on
    empty input. This is the reported uncertainty metric and the key used by
    Mitigation 1's pair filter.
    """
    n = length(text)
    if n == 0:
        return 0.0
    total = 0
    for phrases in UNCERTAINTY_LEXICON.values():
        total += _count_phrases(text, phrases)
    return 100.0 * total / n


def filter_uncertainty_preserving_pairs(
    pairs: list[dict],
    epsilon: float,
) -> tuple[list[dict], dict]:
    """Mitigation 1 (data-level): keep only uncertainty-preserving pairs.

    A preference pair (chosen, rejected) is retained iff

        uncertainty_score(chosen) >= uncertainty_score(rejected) - epsilon

    i.e. we drop pairs that would push the policy to prefer a LESS-uncertain
    response over a more-uncertain one (the gradient direction that suppresses
    hedging). `epsilon >= 0` permits a small tolerance band; `epsilon = 0` is
    the strict floor.

    Returns `(kept_pairs, stats)` where `stats` records retention counts and
    mean chosen/rejected uncertainty so the mitigation's effect on the training
    distribution is observable.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0; got {epsilon}")

    kept: list[dict] = []
    dropped_chosen_unc: list[float] = []
    dropped_rejected_unc: list[float] = []
    kept_chosen_unc: list[float] = []
    kept_rejected_unc: list[float] = []

    for p in pairs:
        u_chosen = uncertainty_score(p["chosen"])
        u_rejected = uncertainty_score(p["rejected"])
        if u_chosen >= u_rejected - epsilon:
            kept.append(p)
            kept_chosen_unc.append(u_chosen)
            kept_rejected_unc.append(u_rejected)
        else:
            dropped_chosen_unc.append(u_chosen)
            dropped_rejected_unc.append(u_rejected)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    stats = {
        "epsilon": epsilon,
        "num_pairs_pre_uncertainty_filter": len(pairs),
        "num_pairs_post_uncertainty_filter": len(kept),
        "num_dropped": len(pairs) - len(kept),
        "drop_rate": (len(pairs) - len(kept)) / len(pairs) if pairs else 0.0,
        "kept_mean_uncertainty_chosen": _mean(kept_chosen_unc),
        "kept_mean_uncertainty_rejected": _mean(kept_rejected_unc),
        "dropped_mean_uncertainty_chosen": _mean(dropped_chosen_unc),
        "dropped_mean_uncertainty_rejected": _mean(dropped_rejected_unc),
    }
    return kept, stats
