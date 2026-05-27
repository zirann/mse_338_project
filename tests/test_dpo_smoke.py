"""DPO smoke tests.

`test_one_dpo_round_mock` runs in every default `pytest -q` invocation. It
exercises the metadata path of `run_one_dpo_round(mock=True)` and verifies
that the placeholder adapter directory + `train_metadata.json` are written
correctly. No model load, no training, fully CPU-safe.

`test_one_dpo_round_real_smoke` is marked `slow` and `skip`; it documents
the future A100 smoke test (small base model, 4 preference pairs, 1 epoch).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_pairs(path: Path, n: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            row = {
                "prompt_id": f"p{i}",
                "prompt": f"Question {i}?",
                "chosen": f"Long thoughtful answer #{i}.",
                "rejected": f"Short {i}.",
            }
            f.write(json.dumps(row) + "\n")


def test_one_dpo_round_mock(tmp_path: Path) -> None:
    from complexity_theater.dpo import run_one_dpo_round

    pairs_path = tmp_path / "preference_pairs.jsonl"
    adapter_dir = tmp_path / "adapter"
    _write_pairs(pairs_path, n=4)

    result = run_one_dpo_round(
        base_model_name="dummy/base",
        reference_adapter_path=None,
        preference_pairs_jsonl=pairs_path,
        output_adapter_dir=adapter_dir,
        mock=True,
    )

    assert result["mock"] is True
    assert result["num_pairs"] == 4
    metadata_path = adapter_dir / "train_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "mock"
    assert metadata["num_pairs"] == 4
    assert metadata["base_model_name"] == "dummy/base"


@pytest.mark.slow
@pytest.mark.skip(reason="real DPO smoke requires A100; deferred to Phase 3.")
def test_one_dpo_round_real_smoke(tmp_path: Path) -> None:
    """Future A100 smoke: small base model + 4 preference pairs + 1 epoch."""
    from complexity_theater.dpo import run_one_dpo_round

    pairs_path = tmp_path / "preference_pairs.jsonl"
    adapter_dir = tmp_path / "adapter"
    _write_pairs(pairs_path, n=4)

    result = run_one_dpo_round(
        base_model_name="Qwen/Qwen3-0.6B",
        reference_adapter_path=None,
        preference_pairs_jsonl=pairs_path,
        output_adapter_dir=adapter_dir,
        mock=False,
    )
    assert result["mock"] is False
    assert (adapter_dir / "train_metadata.json").exists()
