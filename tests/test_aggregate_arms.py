"""Tests for analysis/aggregate_arms.py (cross-arm, multi-seed aggregation)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_aggregate():
    if "aggregate_arms_module" in sys.modules:
        return sys.modules["aggregate_arms_module"]
    spec = importlib.util.spec_from_file_location(
        "aggregate_arms_module", ROOT / "analysis" / "aggregate_arms.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_arms_module"] = module
    spec.loader.exec_module(module)
    return module


def _write_arm(d: Path, hedge: float, factualities: list[float]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    metrics = {
        "round": 1,
        "hedge_density": hedge,
        "uncertainty_score": hedge * 1.2,
        "confidence_marker_density": 0.0,
        "factuality": sum(factualities) / len(factualities),
        "judge_win_rate_vs_round_0": 0.6,
    }
    (d / "metrics.json").write_text(json.dumps(metrics))
    rows = []
    for i, f in enumerate(factualities):
        resp = "However, possibly uncertain answer." if f < 0.5 else "Definitely the answer."
        rows.append({"prompt_id": f"p{i}", "response": resp, "factuality": f})
    with (d / "eval_responses.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_collect_arm_and_correctness_under_root() -> None:
    ac = _load_aggregate()
    # Build arm dirs under the module's ROOT/outputs so templates resolve.
    arm_root = ac.ROOT / "outputs" / "_pytest_arm_random"
    try:
        _write_arm(arm_root / "seed0", hedge=0.10, factualities=[0.0, 1.0])
        _write_arm(arm_root / "seed1", hedge=0.20, factualities=[0.0, 0.0])
        template = "outputs/_pytest_arm_random/seed{seed}"
        summary = ac._collect_arm("random", template, [0, 1])
        assert summary is not None
        assert summary["n_seeds"] == 2
        # hedge mean of [0.10, 0.20] = 0.15
        assert abs(summary["hedge_density_mean"] - 0.15) < 1e-9
        assert summary["hedge_density_se"] > 0.0

        cross = ac._correctness_conditioned(summary, manual_labels=None)
        assert cross["arm"] == "random"
        assert cross["label_source"] == "llm_factuality_fallback"
        # 3 INCORRECT (factuality 0.0) responses across the two seeds.
        assert cross["breakdown"]["INCORRECT"]["n"] == 3
    finally:
        import shutil

        shutil.rmtree(arm_root, ignore_errors=True)


def test_collect_arm_returns_none_when_missing() -> None:
    ac = _load_aggregate()
    assert ac._collect_arm("random", "outputs/_does_not_exist/seed{seed}", [0, 1]) is None
