"""Unit tests for `complexity_theater.appearance`.

Covers the three headline appearance metrics plus the per-subclass breakdown.
"""
from __future__ import annotations


def test_length_basic() -> None:
    from complexity_theater.appearance import length
    assert length("hello world") == 2
    assert length("") == 0
    assert length(None) == 0  # type: ignore[arg-type]


def test_structural_complexity_counts_headers_and_bullets() -> None:
    from complexity_theater.appearance import structural_complexity
    text = "# Title\n- item one\n- item two"
    assert structural_complexity(text) == 3


def test_structural_complexity_ignores_plain_text() -> None:
    from complexity_theater.appearance import structural_complexity
    assert structural_complexity("just a plain sentence.") == 0


def test_structural_complexity_counts_numbered_bullets_and_subheaders() -> None:
    from complexity_theater.appearance import structural_complexity
    text = "## Subtitle\n1. step one\n2. step two\n* aside"
    assert structural_complexity(text) == 4


def test_epistemic_marker_density_zero_on_clean_text() -> None:
    from complexity_theater.appearance import epistemic_marker_density
    assert epistemic_marker_density("hello world") == 0.0


def test_epistemic_marker_density_positive_when_phrases_present() -> None:
    from complexity_theater.appearance import epistemic_marker_density
    # "however" is in the lexicon; 5 tokens total -> density = 100/5 = 20.0
    text = "however, this is a fact"
    assert epistemic_marker_density(text) == 20.0


def test_epistemic_marker_breakdown_sums_to_total() -> None:
    from complexity_theater.appearance import (
        epistemic_marker_breakdown,
        epistemic_marker_density,
    )
    text = "It's worth noting that this is important. Clearly so."
    breakdown = epistemic_marker_breakdown(text)
    total = epistemic_marker_density(text)
    assert abs(sum(breakdown.values()) - total) < 1e-9


def test_epistemic_marker_density_case_insensitive() -> None:
    from complexity_theater.appearance import epistemic_marker_density
    a = epistemic_marker_density("However this is a fact")
    b = epistemic_marker_density("however this is a fact")
    assert a == b > 0.0


def test_appearance_metrics_keys() -> None:
    from complexity_theater.appearance import appearance_metrics
    m = appearance_metrics("# Title\n- one\nHowever, this is important.")
    assert set(m) == {"length", "structural_complexity", "epistemic_marker_density"}
    assert m["length"] > 0
    assert m["structural_complexity"] >= 2  # 1 header + 1 bullet


def test_appearance_metrics_empty_string() -> None:
    from complexity_theater.appearance import appearance_metrics
    m = appearance_metrics("")
    assert m == {"length": 0, "structural_complexity": 0, "epistemic_marker_density": 0.0}
