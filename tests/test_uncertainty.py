"""Tests for complexity_theater.uncertainty (Mitigation 1 data filter + score)."""
from __future__ import annotations


def test_uncertainty_score_zero_on_neutral_text() -> None:
    from complexity_theater.uncertainty import uncertainty_score
    assert uncertainty_score("the cat sat on the mat") == 0.0
    assert uncertainty_score("") == 0.0


def test_uncertainty_score_positive_on_hedged_text() -> None:
    from complexity_theater.uncertainty import uncertainty_score
    # "however" (hedge) + "possibly" (explicit) are both in the lexicon.
    assert uncertainty_score("However this is possibly true") > 0.0


def _pair(chosen: str, rejected: str, pid: str = "p") -> dict:
    return {"prompt_id": pid, "prompt": "Q?", "chosen": chosen, "rejected": rejected}


def test_filter_keeps_uncertainty_preserving_pairs() -> None:
    from complexity_theater.uncertainty import filter_uncertainty_preserving_pairs

    pairs = [
        # chosen more uncertain than rejected -> KEEP
        _pair("However, possibly true and uncertain", "Definitely true.", "p0"),
        # chosen less uncertain than rejected -> DROP (would teach less hedging)
        _pair("Definitely true.", "However, possibly true and uncertain", "p1"),
        # equal (no markers either side) -> KEEP at epsilon >= 0
        _pair("The sky is blue today", "Grass is green here", "p2"),
    ]
    kept, stats = filter_uncertainty_preserving_pairs(pairs, epsilon=0.0)
    kept_ids = {p["prompt_id"] for p in kept}
    assert "p0" in kept_ids
    assert "p2" in kept_ids
    assert "p1" not in kept_ids
    assert stats["num_pairs_pre_uncertainty_filter"] == 3
    assert stats["num_pairs_post_uncertainty_filter"] == 2
    assert stats["num_dropped"] == 1


def test_filter_epsilon_tolerance_keeps_more() -> None:
    from complexity_theater.uncertainty import filter_uncertainty_preserving_pairs

    # rejected slightly more uncertain than chosen; a large epsilon keeps it.
    pairs = [_pair("Definitely.", "However possibly.", "p0")]
    kept_strict, _ = filter_uncertainty_preserving_pairs(pairs, epsilon=0.0)
    kept_loose, _ = filter_uncertainty_preserving_pairs(pairs, epsilon=100.0)
    assert len(kept_strict) == 0
    assert len(kept_loose) == 1


def test_filter_rejects_negative_epsilon() -> None:
    import pytest

    from complexity_theater.uncertainty import filter_uncertainty_preserving_pairs
    with pytest.raises(ValueError):
        filter_uncertainty_preserving_pairs([], epsilon=-0.1)
