"""Smoke test for the archived rejection-sampling implementation.

`apply_rejection_sampling` lives in `legacy/` because the new reward-hacking research
loop uses top-K-by-score instead. The legacy implementation is still imported here
to verify it remains parseable / runnable for archival reference.
"""
from legacy.rejection import apply_rejection_sampling


def test_rejection_sampling_filters_and_accepts() -> None:
    rows = [
        {"sample_id": "a", "candidate_text": "you are stupid", "score_total": 0.9},
        {"sample_id": "b", "candidate_text": "you are stupid", "score_total": 0.85},
        {"sample_id": "c", "candidate_text": "have a nice day", "score_total": 0.1},
    ]

    accepted, rejected = apply_rejection_sampling(
        scored_rows=rows,
        accept_score=0.5,
        max_jaccard=0.8,
        min_tokens=2,
    )

    assert len(accepted) == 1
    assert len(rejected) == 2
    assert accepted[0]["accept_flag"] is True
    assert {r["reject_reason"] for r in rejected} == {"duplicate_exact", "low_score"}
