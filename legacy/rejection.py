from __future__ import annotations

import math
from typing import Sequence

from redteam.metrics import tokenize


def _normalize(text: str) -> str:
    return " ".join(tokenize(text))


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    toks = tokenize(text)
    return set((toks[i], toks[i + 1], toks[i + 2]) for i in range(len(toks) - 2))


def trigram_jaccard(a: str, b: str) -> float:
    ga = _trigrams(a)
    gb = _trigrams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def apply_rejection_sampling(
    scored_rows: Sequence[dict],
    accept_score: float,
    max_jaccard: float,
    min_tokens: int,
) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_exact: set[str] = set()

    ranked = sorted(scored_rows, key=lambda x: float(x.get("score_total", 0.0)), reverse=True)

    for row in ranked:
        text = str(row.get("candidate_text", ""))
        text_lower = text.lower()
        normalized = _normalize(text)
        token_count = len(normalized.split())
        score_total = float(row.get("score_total", 0.0))

        # Hard reject any residual chain-of-thought tags.
        if "<think>" in text_lower or "</think>" in text_lower:
            out = dict(row)
            out["accept_flag"] = False
            out["reject_reason"] = "contains_think"
            rejected.append(out)
            continue

        if token_count < int(min_tokens):
            out = dict(row)
            out["accept_flag"] = False
            out["reject_reason"] = "too_short"
            rejected.append(out)
            continue

        if score_total < float(accept_score):
            out = dict(row)
            out["accept_flag"] = False
            out["reject_reason"] = "low_score"
            rejected.append(out)
            continue

        if normalized in seen_exact:
            out = dict(row)
            out["accept_flag"] = False
            out["reject_reason"] = "duplicate_exact"
            rejected.append(out)
            continue

        similar = False
        for acc in accepted:
            jac = trigram_jaccard(text, str(acc.get("candidate_text", "")))
            if jac >= float(max_jaccard):
                similar = True
                break
        if similar:
            out = dict(row)
            out["accept_flag"] = False
            out["reject_reason"] = "duplicate_semantic"
            rejected.append(out)
            continue

        out = dict(row)
        out["accept_flag"] = True
        out["reject_reason"] = ""
        accepted.append(out)
        seen_exact.add(normalized)

    return accepted, rejected


def adaptive_rejection_sampling(
    scored_rows: Sequence[dict],
    *,
    accept_score: float,
    max_jaccard: float,
    min_tokens: int,
    min_accepted: int = 1,
    fallback_quantile: float = 0.9,
    min_accept_score: float = 0.0,
) -> tuple[list[dict], list[dict], dict]:
    """Apply rejection sampling, and fallback to a quantile-based threshold if accept set is too small.

    This prevents degenerate runs where an overly strict accept_score results in 0 accepted samples.
    """
    accepted, rejected = apply_rejection_sampling(
        scored_rows=scored_rows,
        accept_score=accept_score,
        max_jaccard=max_jaccard,
        min_tokens=min_tokens,
    )

    used = float(accept_score)
    fallback_used = False

    if len(accepted) < int(min_accepted) and scored_rows:
        scores = sorted(float(r.get("score_total", 0.0)) for r in scored_rows)
        q = max(0.0, min(1.0, float(fallback_quantile)))
        # Nearest-rank quantile.
        idx = int(math.floor(q * (len(scores) - 1)))
        qv = float(scores[idx])
        used = max(float(min_accept_score), min(float(accept_score), qv))
        if used != float(accept_score):
            fallback_used = True
            accepted, rejected = apply_rejection_sampling(
                scored_rows=scored_rows,
                accept_score=used,
                max_jaccard=max_jaccard,
                min_tokens=min_tokens,
            )

    meta = {
        "accept_score_configured": float(accept_score),
        "accept_score_used": float(used),
        "fallback_used": bool(fallback_used),
        "min_accepted": int(min_accepted),
        "fallback_quantile": float(fallback_quantile),
        "min_accept_score": float(min_accept_score),
    }
    return accepted, rejected, meta
