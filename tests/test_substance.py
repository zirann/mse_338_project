"""Unit tests for `complexity_theater.substance`.

Covers distinct-n, information density, and the reference-grounded factuality
scorer (using a stub judge object rather than a real LLM).
"""
from __future__ import annotations


def test_distinct_n_basic() -> None:
    from complexity_theater.substance import distinct_n
    # "a b a b" -> bigrams: (a,b), (b,a), (a,b). unique=2, total=3 -> 2/3
    assert abs(distinct_n("a b a b", 2) - 2 / 3) < 1e-9


def test_distinct_n_all_unique() -> None:
    from complexity_theater.substance import distinct_n
    assert distinct_n("one two three four five", 2) == 1.0


def test_distinct_n_below_length_returns_zero() -> None:
    from complexity_theater.substance import distinct_n
    assert distinct_n("only two", 3) == 0.0
    assert distinct_n("", 1) == 0.0


def test_information_density_within_unit_interval() -> None:
    from complexity_theater.substance import information_density
    d = information_density("the quick brown fox jumps over the lazy dog")
    assert 0.0 <= d <= 1.0


def test_information_density_zero_on_pure_repetition() -> None:
    from complexity_theater.substance import information_density
    # All-same token: every trigram identical -> distinct_3 small -> density small.
    assert information_density("foo foo foo foo foo foo") <= 0.1


def test_information_density_zero_on_empty() -> None:
    from complexity_theater.substance import information_density
    assert information_density("") == 0.0


def test_score_factuality_with_none_scorer_returns_one() -> None:
    from complexity_theater.substance import score_factuality
    # `None` scorer is the documented mock path used by local smoke.
    assert score_factuality(None, "Q?", "A.", "A.", "B.") == 1.0


def test_score_factuality_delegates_to_scorer_object() -> None:
    from complexity_theater.substance import score_factuality

    class StubJudge:
        def score_factuality(self, prompt, response, correct_reference, incorrect_reference):
            assert prompt == "Q?"
            return 0.5

    assert score_factuality(StubJudge(), "Q?", "A.", "A.", "B.") == 0.5


def test_factuality_batch_preserves_order() -> None:
    from complexity_theater.substance import factuality_batch

    rows = [
        {"prompt": "p1", "response": "r1", "correct_reference": "c1", "incorrect_reference": "i1"},
        {"prompt": "p2", "response": "r2", "correct_reference": "c2", "incorrect_reference": "i2"},
        {"prompt": "p3", "response": "r3", "correct_reference": "c3", "incorrect_reference": "i3"},
    ]
    out = factuality_batch(None, rows)
    assert out == [1.0, 1.0, 1.0]


def test_substance_metrics_keys() -> None:
    from complexity_theater.substance import substance_metrics
    m = substance_metrics("the quick brown fox jumps", factuality_score=0.5)
    assert set(m) == {"factuality", "information_density"}
    assert m["factuality"] == 0.5
    assert 0.0 <= m["information_density"] <= 1.0
