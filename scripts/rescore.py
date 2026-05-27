#!/usr/bin/env python3
"""Recompute appearance + information_density metrics from saved generations.

For each round whose `outputs/round_<N>/eval_responses.jsonl` + `metrics.json`
already exist:

- Reload the saved per-response text and the saved per-response `factuality`.
- Recompute `appearance.appearance_metrics(response)` and
  `substance.information_density(response)`.
- Preserve `factuality` (per row + aggregated mean), `judge_win_rate_vs_round_0`,
  and `judge_parse_failures_total_instance` from the existing `metrics.json`.
- Rewrite `metrics.json` in place with the refined appearance keys.

After all rounds rescore, invoke `scripts/analyze.py` to regenerate
`outputs/trajectory.json` and `assets/figures/headline.png`.

Use case: refine appearance metrics on already-generated A100 outputs without
re-running the model or the LLM judge. Cheap (~seconds) and deterministic.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexity_theater import appearance, substance  # noqa: E402
from complexity_theater.io_utils import read_json, read_jsonl, read_yaml, write_json  # noqa: E402


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rescore appearance metrics from saved eval_responses.jsonl.")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip the analyze.py invocation that regenerates trajectory + headline figure.",
    )
    return p.parse_args()


def rescore_round(round_dir: Path) -> bool:
    """Rescore one round's metrics.json in place. Returns True if rewritten."""
    eval_path = round_dir / "eval_responses.jsonl"
    metrics_path = round_dir / "metrics.json"
    if not eval_path.exists() or not metrics_path.exists():
        return False

    responses = read_jsonl(eval_path)
    if not responses:
        return False
    old_metrics = read_json(metrics_path)

    app_per = [appearance.appearance_metrics(r["response"]) for r in responses]

    # Substance: factuality is preserved per-row from the existing JSONL.
    # information_density is purely textual and is recomputed.
    fact_scores = [float(r.get("factuality", old_metrics.get("factuality", 1.0))) for r in responses]
    sub_per = [
        substance.substance_metrics(r["response"], f)
        for r, f in zip(responses, fact_scores)
    ]

    new_metrics: dict = {
        "round": old_metrics.get("round"),
        "n_eval": len(responses),
        "mock": old_metrics.get("mock", False),
        "length": _mean([a["length"] for a in app_per]),
        "structural_complexity": _mean([a["structural_complexity"] for a in app_per]),
        "reasoning_narration_density": _mean(
            [a["reasoning_narration_density"] for a in app_per]
        ),
        "hedge_density": _mean([a["hedge_density"] for a in app_per]),
        "epistemic_marker_density": _mean(
            [a["epistemic_marker_density"] for a in app_per]
        ),
        "factuality": _mean(fact_scores),
        "information_density": _mean([s["information_density"] for s in sub_per]),
        "rescored": True,
    }
    # Preserve composite + diagnostic fields from the original run.
    for k in ("judge_win_rate_vs_round_0", "judge_parse_failures_total_instance"):
        if k in old_metrics:
            new_metrics[k] = old_metrics[k]

    write_json(metrics_path, new_metrics)
    return True


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    template = cfg["outputs"]["per_round_dir_template"]
    num_rounds = int(cfg["dpo"]["num_rounds"])

    rescored: list[int] = []
    for n in range(num_rounds + 1):
        round_dir = ROOT / template.format(round=n)
        if rescore_round(round_dir):
            print(f"[rescore] rewrote {round_dir / 'metrics.json'}")
            rescored.append(n)
        else:
            print(f"[rescore] skipping round {n}: artifacts missing under {round_dir}")

    if not rescored:
        raise SystemExit("[rescore] no rounds rescored; ensure prepare+evaluate were run first.")

    if args.skip_analyze:
        print("[rescore] --skip-analyze set; trajectory + headline NOT regenerated.")
        return

    cmd = [sys.executable, str(ROOT / "scripts" / "analyze.py"), "--config", args.config]
    print(f"[rescore] running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
