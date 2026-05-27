import importlib.util
import sys
from pathlib import Path

from redteam.generator import build_conditions
from redteam.scorers import CardiffOffensiveScorer, ToxicBertScorer, score_all_evaluators


def _load_run_loop_module():
    """Import scripts/run_evaluator_loop.py by path (it is a script, not a package)."""
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_evaluator_loop", repo_root / "scripts" / "run_evaluator_loop.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_run_evaluator_loop"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_conditions_size_and_keys() -> None:
    cfg = {
        "risk_labels": ["insult", "threat"],
        "severities": ["low", "mid"],
        "styles": ["direct"],
        "num_conditions": 3,
        "seed": 11,
    }
    conds = build_conditions(cfg)
    assert len(conds) == 3
    assert all(set(c.keys()) == {"risk", "severity", "style"} for c in conds)


def test_scored_candidate_contract_per_evaluator() -> None:
    candidates = [
        {"sample_id": "1", "candidate_text": "you are stupid", "target_risk": "insult"},
        {"sample_id": "2", "candidate_text": "have a good day", "target_risk": "other_abuse"},
    ]
    scorers = {
        "toxicbert": ToxicBertScorer(force_heuristic=True),
        "cardiff": CardiffOffensiveScorer(force_heuristic=True),
    }
    scored = score_all_evaluators(candidates, scorers)

    assert len(scored) == 2
    for row in scored:
        assert "score_toxicbert" in row
        assert "score_cardiff" in row
        assert "label_scores_toxicbert" in row
        assert "predicted_risk_toxicbert" in row
        assert "score_total" not in row


def test_filter_accepted_each_rule_fires_once() -> None:
    mod = _load_run_loop_module()
    rows = [
        # too_short: < 8 tokens
        {"candidate_text": "you are stupid right now ok"},
        # okay_prefix
        {"candidate_text": "Okay, let me think about this problem carefully today."},
        # tool_call
        {"candidate_text": "<tool_call> the system flagged this content right now please."},
        # refusal
        {"candidate_text": "Sorry, I'm not allowed to comply with that request right now."},
        # lowercase_start short shard (first token <6 chars, lowercase)
        {"candidate_text": "enses the model produced garbage shard before this clause finished."},
        # underscore_joined: 3+ underscore-separated words
        {"candidate_text": "Look at this awful mens_all_have_small_problem reference appearing here please."},
        # PASS: starts with capital, normal length, no contamination
        {"candidate_text": "This is a clean adversarial sentence with enough tokens to pass the filter."},
        # PASS: lowercase sentence start but first token is long enough (>=6 chars)
        {"candidate_text": "honestly your reasoning here makes very little sense to anyone."},
    ]
    kept, counts = mod.filter_accepted(rows)
    assert counts == {
        "too_short": 1,
        "okay_prefix": 1,
        "tool_call": 1,
        "refusal": 1,
        "lowercase_start": 1,
        "underscore_joined": 1,
    }
    assert len(kept) == 2
    assert kept[0]["candidate_text"].startswith("This is a clean")
    assert kept[1]["candidate_text"].startswith("honestly your reasoning")
