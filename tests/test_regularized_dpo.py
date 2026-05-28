"""Tests for complexity_theater.regularized_dpo (Mitigation 2).

The pure-math functions are exercised on toy tensors (no model load). The
training entry point is exercised only via its mock path.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_dpo_loss_equals_ln2_when_policy_matches_reference() -> None:
    from complexity_theater.regularized_dpo import dpo_loss

    z = torch.zeros(3)
    loss, stats = dpo_loss(z, z, z, z, beta=0.1)
    assert abs(float(loss.item()) - math.log(2)) < 1e-6
    assert abs(stats["rewards_margins"]) < 1e-6


def test_dpo_loss_drops_when_chosen_preferred() -> None:
    from complexity_theater.regularized_dpo import dpo_loss

    policy_chosen = torch.tensor([0.0])
    policy_rejected = torch.tensor([-2.0])
    ref = torch.tensor([0.0])
    loss, stats = dpo_loss(policy_chosen, policy_rejected, ref, ref, beta=1.0)
    assert float(loss.item()) < math.log(2)
    assert stats["rewards_margins"] > 0.0


def _toy(vocab=4, t=3):
    # logits [B=1, T=t, V=vocab]; input_ids [1,t]; completion_mask [1,t]
    input_ids = torch.tensor([[0, 2, 3]][:1])
    completion_mask = torch.tensor([[0.0, 1.0, 1.0]])
    return input_ids, completion_mask


def test_penalty_a_zero_when_policy_equals_reference() -> None:
    from complexity_theater.regularized_dpo import penalty_a_mass_floor

    input_ids, completion_mask = _toy()
    logits = torch.randn(1, 3, 4)
    u = torch.tensor([2, 3])
    val = penalty_a_mass_floor(logits, logits.clone(), input_ids, completion_mask, u)
    assert abs(float(val.item())) < 1e-6


def test_penalty_a_positive_when_policy_suppresses_uncertainty_mass() -> None:
    from complexity_theater.regularized_dpo import penalty_a_mass_floor

    input_ids, completion_mask = _toy()
    # Reference puts mass on uncertainty tokens (2,3); policy puts mass on (0,1).
    ref_logits = torch.zeros(1, 3, 4)
    ref_logits[..., 2] = 5.0
    ref_logits[..., 3] = 5.0
    pol_logits = torch.zeros(1, 3, 4)
    pol_logits[..., 0] = 5.0
    pol_logits[..., 1] = 5.0
    u = torch.tensor([2, 3])
    val = penalty_a_mass_floor(pol_logits, ref_logits, input_ids, completion_mask, u)
    assert float(val.item()) > 0.5


def test_penalty_b_finite_and_nonnegative() -> None:
    from complexity_theater.regularized_dpo import penalty_b_chosen_hedge_logprob

    input_ids, completion_mask = _toy()
    logits = torch.randn(1, 3, 4)
    u = torch.tensor([2, 3])
    val = penalty_b_chosen_hedge_logprob(logits, input_ids, completion_mask, u)
    assert math.isfinite(float(val.item()))
    assert float(val.item()) >= 0.0  # -log p is non-negative


def test_penalty_c_zero_when_equal() -> None:
    from complexity_theater.regularized_dpo import penalty_c_entropy_floor

    input_ids, completion_mask = _toy()
    logits = torch.randn(1, 3, 4)
    val = penalty_c_entropy_floor(logits, logits.clone(), input_ids, completion_mask)
    assert abs(float(val.item())) < 1e-6


def test_sequence_logps_shape_and_sign() -> None:
    from complexity_theater.regularized_dpo import sequence_logps

    input_ids, completion_mask = _toy()
    logits = torch.randn(1, 3, 4)
    out = sequence_logps(logits, input_ids, completion_mask)
    assert out.shape == (1,)
    assert float(out.item()) <= 0.0  # sum of log-probs


def test_build_uncertainty_token_ids_with_stub_tokenizer() -> None:
    from complexity_theater.regularized_dpo import build_uncertainty_token_ids

    class StubTokenizer:
        vocab = {"however": 10, "might": 11, "possibly": 12}

        def encode(self, text, add_special_tokens=False):
            return [self.vocab[w] for w in text.strip().split() if w in self.vocab]

    ids = build_uncertainty_token_ids(StubTokenizer(), ["however", "might", "possibly"])
    assert ids.dtype == torch.long
    assert set(int(x) for x in ids) == {10, 11, 12}


def test_compute_penalty_unknown_formulation_raises() -> None:
    from complexity_theater.regularized_dpo import compute_penalty

    input_ids, completion_mask = _toy()
    logits = torch.randn(1, 3, 4)
    with pytest.raises(ValueError):
        compute_penalty("Z", logits, logits, input_ids, completion_mask, torch.tensor([2, 3]))


def test_run_one_regularized_dpo_round_mock(tmp_path: Path) -> None:
    from complexity_theater.regularized_dpo import run_one_regularized_dpo_round

    pairs_path = tmp_path / "preference_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"prompt": f"Q{i}", "chosen": "However, possibly.", "rejected": "Definitely."}) + "\n")
    adapter_dir = tmp_path / "adapter"

    result = run_one_regularized_dpo_round(
        base_model_name="dummy/base",
        reference_adapter_path=None,
        preference_pairs_jsonl=pairs_path,
        output_adapter_dir=adapter_dir,
        reg_config={"reg_formulation": "A", "reg_lambda": 0.2},
        mock=True,
    )
    assert result["mock"] is True
    meta = json.loads((adapter_dir / "train_metadata.json").read_text())
    assert meta["status"] == "mock"
    assert meta["trainer"] == "regularized_dpo"
    assert meta["reg"]["reg_formulation"] == "A"
    assert meta["num_pairs"] == 5
