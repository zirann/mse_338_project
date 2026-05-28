"""Tests for `scripts/analyze_calibration.py`.

Covers:

- `_aggregate_arm` on a 4-row synthetic eval_responses.jsonl.
- `_correctness_conditioned` falls back to the LLM-factuality bucketization
  when no manual labels are provided, and uses the manual labels when present.
- Empty arm directory returns None (graceful degradation).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_analyze_calibration():
    if "analyze_calibration_module" in sys.modules:
        return sys.modules["analyze_calibration_module"]
    spec = importlib.util.spec_from_file_location(
        "analyze_calibration_module", ROOT / "scripts" / "analyze_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_calibration_module"] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(arm_dir: Path) -> None:
    """4-row eval_responses fixture with predictable hedge / confidence content."""
    arm_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        # CORRECT (factuality = 1.0): minimal hedges, one confidence marker.
        {
            "prompt_id": "p0",
            "question": "Q?",
            "response": "Clearly the answer is yes here.",
            "correct_reference": "Yes.",
            "incorrect_reference": "No.",
            "factuality": 1.0,
        },
        # PARTIAL (factuality = 0.5): one hedge + one confidence marker.
        {
            "prompt_id": "p1",
            "question": "Q?",
            "response": "However, clearly this is a mixed case.",
            "correct_reference": "It depends.",
            "incorrect_reference": "Always.",
            "factuality": 0.5,
        },
        # INCORRECT (factuality = 0.0): heavy hedges, no confidence marker.
        {
            "prompt_id": "p2",
            "question": "Q?",
            "response": "However, in some cases this varies, to some extent.",
            "correct_reference": "Right.",
            "incorrect_reference": "Wrong.",
            "factuality": 0.0,
        },
        # INCORRECT (factuality = 0.0): no hedges, no confidence markers.
        {
            "prompt_id": "p3",
            "question": "Q?",
            "response": "Yes.",
            "correct_reference": "No.",
            "incorrect_reference": "Yes.",
            "factuality": 0.0,
        },
    ]
    with (arm_dir / "eval_responses.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_aggregate_arm_returns_means_and_n(tmp_path: Path) -> None:
    ac = _load_analyze_calibration()
    arm_dir = tmp_path / "arm_fixture"
    _write_fixture(arm_dir)
    result = ac._aggregate_arm("arm_fixture", arm_dir)
    assert result is not None
    assert result["n"] == 4
    assert result["hedge_density_mean"] >= 0.0
    assert result["confidence_marker_density_mean"] >= 0.0
    # LLM-judged factuality mean of [1.0, 0.5, 0.0, 0.0] = 0.375.
    assert abs(result["llm_factuality_mean"] - 0.375) < 1e-9
    assert len(result["per_response"]) == 4


def test_aggregate_arm_returns_none_when_missing(tmp_path: Path) -> None:
    ac = _load_analyze_calibration()
    missing = tmp_path / "does_not_exist"
    assert ac._aggregate_arm("missing_arm", missing) is None


def test_correctness_conditioned_uses_llm_fallback_when_manual_absent(
    tmp_path: Path,
) -> None:
    ac = _load_analyze_calibration()
    arm_dir = tmp_path / "arm_fixture"
    _write_fixture(arm_dir)
    arm = ac._aggregate_arm("arm_fixture", arm_dir)
    assert arm is not None
    cross_tab = ac._correctness_conditioned([arm], manual_labels=None)
    assert len(cross_tab) == 1
    entry = cross_tab[0]
    assert entry["arm"] == "arm_fixture"
    assert entry["label_source"] == "llm_factuality_fallback"
    # Fallback maps factuality >= 0.75 -> CORRECT, 0.25-0.75 -> PARTIAL, < 0.25 -> INCORRECT.
    # Our fixture has 1 CORRECT, 1 PARTIAL, 2 INCORRECT.
    assert entry["breakdown"]["CORRECT"]["n"] == 1
    assert entry["breakdown"]["PARTIAL"]["n"] == 1
    assert entry["breakdown"]["INCORRECT"]["n"] == 2


def test_correctness_conditioned_uses_manual_labels_when_provided(
    tmp_path: Path,
) -> None:
    ac = _load_analyze_calibration()
    arm_dir = tmp_path / "arm_fixture"
    _write_fixture(arm_dir)
    arm = ac._aggregate_arm("arm_fixture", arm_dir)
    assert arm is not None
    # Manual override: flip p0 from CORRECT to INCORRECT, keep others as fallback.
    manual = {
        ("arm_fixture", "p0"): "INCORRECT",
        ("arm_fixture", "p1"): "INCORRECT",
        ("arm_fixture", "p2"): "INCORRECT",
        ("arm_fixture", "p3"): "INCORRECT",
    }
    cross_tab = ac._correctness_conditioned([arm], manual_labels=manual)
    entry = cross_tab[0]
    assert entry["label_source"] == "manual"
    assert entry["breakdown"]["INCORRECT"]["n"] == 4
    assert entry["breakdown"]["CORRECT"]["n"] == 0
    assert entry["breakdown"]["PARTIAL"]["n"] == 0
