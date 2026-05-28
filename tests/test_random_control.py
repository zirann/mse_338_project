"""Tests for the random-preference DPO control.

Covers:
- `_build_random_preference_pairs` is deterministic under a fixed seed.
- Different seeds usually produce different pairs.
- The judge-derived helper and the random helper produce different pairs on
  the same candidate set (sanity check: the two construction paths actually
  differ).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_train_round_module():
    """Import scripts/train_round.py as a module without executing main()."""
    if "train_round_module" in sys.modules:
        return sys.modules["train_round_module"]
    spec = importlib.util.spec_from_file_location(
        "train_round_module", ROOT / "scripts" / "train_round.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_round_module"] = module
    spec.loader.exec_module(module)
    return module


def _make_candidates(num_prompts: int = 3, k: int = 4) -> dict[str, list[dict]]:
    """Synthetic candidates: K candidates per prompt with distinct lengths."""
    by_prompt: dict[str, list[dict]] = {}
    for pi in range(num_prompts):
        pid = f"p{pi}"
        cands = []
        for ki in range(k):
            cands.append(
                {
                    "prompt_id": pid,
                    "question": f"Q{pi}?",
                    "k_index": ki,
                    # Distinct strings to avoid degenerate-pair drops.
                    "response": f"prompt {pi} candidate {ki} " + ("x " * (ki + 1)),
                }
            )
        by_prompt[pid] = cands
    return by_prompt


def test_random_preference_pairs_are_deterministic_under_fixed_seed() -> None:
    tr = _load_train_round_module()
    by_prompt = _make_candidates()
    pairs_a = tr._build_random_preference_pairs(by_prompt, seed=42)
    pairs_b = tr._build_random_preference_pairs(by_prompt, seed=42)
    assert pairs_a == pairs_b, "Same seed must produce identical pair lists."


def test_random_preference_pairs_vary_with_seed() -> None:
    tr = _load_train_round_module()
    by_prompt = _make_candidates(num_prompts=5, k=4)
    pairs_seed_a = tr._build_random_preference_pairs(by_prompt, seed=42)
    pairs_seed_b = tr._build_random_preference_pairs(by_prompt, seed=7)
    # With 5 prompts and K=4, P(all 5 prompts pick the same pair under two
    # different seeds) is small enough to be safe in CI.
    assert pairs_seed_a != pairs_seed_b


def test_random_preference_pairs_pick_distinct_candidates() -> None:
    tr = _load_train_round_module()
    by_prompt = _make_candidates()
    pairs = tr._build_random_preference_pairs(by_prompt, seed=42)
    for p in pairs:
        assert p["chosen"] != p["rejected"], "chosen and rejected must be distinct"


def test_random_pairs_differ_from_judge_pairs_on_same_candidates() -> None:
    """On the same candidates, the random pairs should typically differ from
    what the judge picks. The mock judge ranks by length (longest first), so
    the judge-derived top-1 is always the K-1 candidate; the random shuffle
    almost always picks a different chosen.
    """
    tr = _load_train_round_module()
    by_prompt = _make_candidates(num_prompts=5, k=4)
    ranked_per_prompt = {pid: [3, 2, 1, 0] for pid in by_prompt}  # length-ranked
    judge_pairs = tr._build_judge_preference_pairs(by_prompt, ranked_per_prompt)
    random_pairs = tr._build_random_preference_pairs(by_prompt, seed=42)
    assert judge_pairs != random_pairs


def test_preference_diagnostics_stamps_pair_construction_mode() -> None:
    tr = _load_train_round_module()
    diag = tr._preference_diagnostics(
        pairs=[],
        judge_examples=[],
        judge_parse_failures_total=0,
        pair_construction_mode="random",
    )
    assert diag["pair_construction_mode"] == "random"
    diag_default = tr._preference_diagnostics(
        pairs=[],
        judge_examples=[],
        judge_parse_failures_total=0,
    )
    assert diag_default["pair_construction_mode"] == "judge"
