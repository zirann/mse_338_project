"""Tests for the SamPO length debiasing + DPOP additions to the local DPO loop."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------
# DPOP term
# --------------------------------------------------------------------------

def test_dpop_zero_penalty_when_chosen_at_or_above_ref() -> None:
    from complexity_theater.regularized_dpo import dpo_loss

    pol_c = torch.tensor([0.0])
    pol_r = torch.tensor([-1.0])
    ref_c = torch.tensor([-1.0])  # pol_c (0) >= ref_c (-1) -> penalty 0
    ref_r = torch.tensor([-1.0])
    loss0, s0 = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1, dpop_lambda=0.0)
    loss1, s1 = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1, dpop_lambda=5.0)
    assert s1["dpop_penalty"] == 0.0
    assert abs(float(loss0.item()) - float(loss1.item())) < 1e-7


def test_dpop_increases_loss_when_chosen_below_ref() -> None:
    from complexity_theater.regularized_dpo import dpo_loss

    pol_c = torch.tensor([-2.0])
    pol_r = torch.tensor([-1.0])
    ref_c = torch.tensor([0.0])   # pol_c (-2) < ref_c (0) -> penalty = 2
    ref_r = torch.tensor([-1.0])
    loss0, s0 = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1, dpop_lambda=0.0)
    loss1, s1 = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1, dpop_lambda=1.0)
    assert abs(s1["dpop_penalty"] - 2.0) < 1e-6
    assert float(loss1.item()) > float(loss0.item())


# --------------------------------------------------------------------------
# SamPO token-count equalization
# --------------------------------------------------------------------------

def test_sampo_select_equalizes_token_counts() -> None:
    from complexity_theater.regularized_dpo import sampo_select

    # chosen has 5 valid tokens, rejected has 3 (T=8).
    mask_c = torch.tensor([[1.0, 1, 1, 1, 1, 0, 0, 0]])
    mask_r = torch.tensor([[1.0, 1, 1, 0, 0, 0, 0, 0]])
    gen = torch.Generator().manual_seed(0)
    sel_c, sel_r, used_c, used_r = sampo_select(mask_c, mask_r, generator=gen)
    assert used_c == [3] and used_r == [3]
    assert float(sel_c.sum().item()) == 3.0
    assert float(sel_r.sum().item()) == 3.0
    # Selected positions must be a subset of the valid positions.
    assert torch.all(sel_c <= mask_c)
    assert torch.all(sel_r <= mask_r)


def test_sampo_select_deterministic_under_seed() -> None:
    from complexity_theater.regularized_dpo import sampo_select

    mask_c = torch.tensor([[1.0] * 6 + [0.0] * 2])
    mask_r = torch.tensor([[1.0] * 2 + [0.0] * 6])
    a = sampo_select(mask_c, mask_r, generator=torch.Generator().manual_seed(7))
    b = sampo_select(mask_c, mask_r, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# --------------------------------------------------------------------------
# aggregate_pair_logps modes
# --------------------------------------------------------------------------

def test_aggregate_none_sums_all_completion_tokens() -> None:
    from complexity_theater.regularized_dpo import aggregate_pair_logps

    ptl_c = torch.tensor([[-1.0, -1.0, -1.0, 0.0]])
    mask_c = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    ptl_r = torch.tensor([[-2.0, -2.0, 0.0, 0.0]])
    mask_r = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    pol_c, pol_r, ref_c, ref_r, used_c, used_r = aggregate_pair_logps(
        ptl_c, mask_c, ptl_r, mask_r, ptl_c, ptl_r, "none"
    )
    assert abs(float(pol_c.item()) - (-3.0)) < 1e-6
    assert abs(float(pol_r.item()) - (-4.0)) < 1e-6


def test_aggregate_lennorm_divides_by_length() -> None:
    from complexity_theater.regularized_dpo import aggregate_pair_logps

    ptl_c = torch.tensor([[-1.0, -1.0, -1.0, 0.0]])
    mask_c = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    ptl_r = torch.tensor([[-2.0, -2.0, 0.0, 0.0]])
    mask_r = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    pol_c, pol_r, *_ = aggregate_pair_logps(
        ptl_c, mask_c, ptl_r, mask_r, ptl_c, ptl_r, "lennorm"
    )
    assert abs(float(pol_c.item()) - (-1.0)) < 1e-6   # -3/3
    assert abs(float(pol_r.item()) - (-2.0)) < 1e-6   # -4/2


def test_aggregate_sampo_uses_equal_token_counts() -> None:
    from complexity_theater.regularized_dpo import aggregate_pair_logps

    # chosen 4 valid, rejected 2 valid; sampo should sum equal counts (2 each).
    ptl_c = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    mask_c = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    ptl_r = torch.tensor([[-3.0, -3.0, 0.0, 0.0]])
    mask_r = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    gen = torch.Generator().manual_seed(1)
    pol_c, pol_r, ref_c, ref_r, used_c, used_r = aggregate_pair_logps(
        ptl_c, mask_c, ptl_r, mask_r, ptl_c, ptl_r, "sampo", generator=gen
    )
    assert used_c == [2] and used_r == [2]
    # chosen contributes exactly 2 tokens of -1.0 -> -2.0; rejected 2 of -3.0 -> -6.0
    assert abs(float(pol_c.item()) - (-2.0)) < 1e-6
    assert abs(float(pol_r.item()) - (-6.0)) < 1e-6


# --------------------------------------------------------------------------
# Mock routing carries the new knobs
# --------------------------------------------------------------------------

def test_mock_round_records_length_debias_and_dpop(tmp_path) -> None:
    import json

    from complexity_theater.regularized_dpo import run_one_regularized_dpo_round

    pairs = tmp_path / "pp.jsonl"
    with pairs.open("w") as f:
        for i in range(4):
            f.write(json.dumps({"prompt": f"Q{i}", "chosen": "However maybe.", "rejected": "Yes."}) + "\n")
    adapter = tmp_path / "adapter"
    run_one_regularized_dpo_round(
        base_model_name="dummy/base",
        reference_adapter_path=None,
        preference_pairs_jsonl=pairs,
        output_adapter_dir=adapter,
        length_debias="sampo",
        dpop_lambda=0.5,
        mock=True,
    )
    meta = json.loads((adapter / "train_metadata.json").read_text())
    assert meta["length_debias"] == "sampo"
    assert meta["dpop_lambda"] == 0.5
    assert meta["trainer"] == "local_dpo"


def test_invalid_length_debias_raises(tmp_path) -> None:
    import json

    from complexity_theater.regularized_dpo import run_one_regularized_dpo_round

    pairs = tmp_path / "pp.jsonl"
    pairs.write_text(json.dumps({"prompt": "Q", "chosen": "a", "rejected": "b"}) + "\n")
    with pytest.raises(ValueError):
        run_one_regularized_dpo_round(
            base_model_name="dummy/base",
            reference_adapter_path=None,
            preference_pairs_jsonl=pairs,
            output_adapter_dir=tmp_path / "ad",
            length_debias="bogus",
            mock=True,
        )
