from redteam.metrics import (
    avg_length_tokens,
    distinct_n,
    duplication_ratio,
    harmful_hit_rate,
)


def test_metrics_basic() -> None:
    texts = ["a b c", "a b d"]
    assert 0.0 <= distinct_n(texts, 1) <= 1.0
    assert 0.0 <= distinct_n(texts, 2) <= 1.0
    assert avg_length_tokens(texts) == 3.0
    assert 0.0 <= duplication_ratio(texts) <= 1.0


def test_harmful_hit_rate() -> None:
    scores = [0.1, 0.7, 0.8]
    assert harmful_hit_rate(scores, 0.6) == 2 / 3
