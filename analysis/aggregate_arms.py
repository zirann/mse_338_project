#!/usr/bin/env python3
"""Cross-arm aggregation for the uncertainty-suppression project.

Reads each arm's per-seed `metrics.json` + `eval_responses.jsonl`, aggregates
mean +/- SE ACROSS SEEDS for the headline metrics, and (optionally) joins author
manual factuality labels for the correctness-conditioned analysis. Writes
`outputs/arms_summary.json`.

Arm registry maps a display name to a per-seed directory glob. Arms or seeds
with missing artifacts are skipped with a printed note (graceful degradation),
so this runs against partial state during a staged A100 rerun.

This generalizes the earlier scripts/analyze_calibration.py over arbitrary arms
and multiple seeds. No model loads, no judge calls.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexity_theater import appearance  # noqa: E402
from complexity_theater.io_utils import read_json, read_jsonl, write_json  # noqa: E402
from complexity_theater.uncertainty import uncertainty_score  # noqa: E402


# Arm display name -> list of seed directories (relative to repo root).
DEFAULT_SEEDS = (0, 1, 2)
ARM_DIR_TEMPLATES: dict[str, str] = {
    "baseline": "outputs/baseline",  # single dir (no seed sweep)
    "vanilla_dpo": "outputs/vanilla_dpo/seed{seed}",
    "sampo_dpo": "outputs/sampo_dpo/seed{seed}",
    "dpop": "outputs/dpop/seed{seed}",
    "sampo_dpop": "outputs/sampo_dpop/seed{seed}",
}

HEADLINE_METRICS = (
    "length",
    "hedge_density",
    "uncertainty_score",
    "confidence_marker_density",
    "factuality",
    "judge_win_rate_vs_round_0",
)


def _mean_se(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs))


def _arm_seed_dirs(template: str, seeds) -> list[Path]:
    if "{seed}" not in template:
        return [ROOT / template]
    return [ROOT / template.format(seed=s) for s in seeds]


def _collect_arm(name: str, template: str, seeds) -> dict | None:
    """Aggregate one arm across seeds. Returns None if no seed dir has metrics."""
    seed_metrics: list[dict] = []
    seed_response_paths: list[Path] = []
    for d in _arm_seed_dirs(template, seeds):
        mpath = d / "metrics.json"
        if mpath.exists():
            seed_metrics.append(read_json(mpath))
            rp = d / "eval_responses.jsonl"
            if rp.exists():
                seed_response_paths.append(rp)
    if not seed_metrics:
        return None

    summary: dict = {"arm": name, "n_seeds": len(seed_metrics)}
    for metric in HEADLINE_METRICS:
        vals = [m.get(metric) for m in seed_metrics]
        mean, se = _mean_se([v for v in vals if v is not None])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_se"] = se
    summary["response_paths"] = [str(p) for p in seed_response_paths]
    return summary


def _correctness_conditioned(arm_summary: dict, manual_labels: dict | None) -> dict:
    """hedge_density on CORRECT vs INCORRECT subsets, pooled across the arm's seeds.

    Uses manual labels keyed by (arm, prompt_id) when available; otherwise falls
    back to a 3-way bucket of LLM factuality (>=0.75 CORRECT, 0.25-0.75 PARTIAL,
    <0.25 INCORRECT). The fallback inherits the same-family confound.
    """
    buckets = {lbl: [] for lbl in ("CORRECT", "PARTIAL", "INCORRECT")}
    used_fallback = False
    arm = arm_summary["arm"]
    for rp in arm_summary.get("response_paths", []):
        for r in read_jsonl(Path(rp)):
            pid = r.get("prompt_id")
            if manual_labels is not None and (arm, pid) in manual_labels:
                label = manual_labels[(arm, pid)]
            else:
                used_fallback = True
                f = r.get("factuality")
                if f is None:
                    continue
                label = "CORRECT" if f >= 0.75 else ("PARTIAL" if f >= 0.25 else "INCORRECT")
            if label not in buckets:
                continue
            buckets[label].append(appearance.hedge_density(r.get("response", "")))
    breakdown = {}
    for lbl, vals in buckets.items():
        mean, se = _mean_se(vals)
        breakdown[lbl] = {"n": len(vals), "hedge_density_mean": mean, "hedge_density_se": se}
    return {
        "arm": arm,
        "label_source": "manual" if (manual_labels is not None and not used_fallback) else "llm_factuality_fallback",
        "breakdown": breakdown,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate metrics across arms x seeds.")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    p.add_argument(
        "--manual_factuality",
        default=str(ROOT / "outputs" / "manual_factuality.jsonl"),
        help="Optional sidecar JSONL: {prompt_id, arm, llm_factuality, manual_label}.",
    )
    p.add_argument("--out", default=str(ROOT / "outputs" / "arms_summary.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    arms: list[dict] = []
    for name, template in ARM_DIR_TEMPLATES.items():
        summary = _collect_arm(name, template, args.seeds)
        if summary is None:
            print(f"[aggregate_arms] skipping {name}: no metrics found")
            continue
        print(
            f"[aggregate_arms] {name}: n_seeds={summary['n_seeds']} "
            f"hedge={summary['hedge_density_mean']:.4f}+-{summary['hedge_density_se']:.4f}"
        )
        arms.append(summary)

    if not arms:
        raise SystemExit("[aggregate_arms] no arms found; run experiments first.")

    manual_labels = None
    manual_path = Path(args.manual_factuality)
    if manual_path.exists():
        rows = read_jsonl(manual_path)
        manual_labels = {(r["arm"], r["prompt_id"]): r["manual_label"].upper() for r in rows}
        print(f"[aggregate_arms] loaded {len(manual_labels)} manual labels")
    else:
        print(f"[aggregate_arms] no manual labels at {manual_path}; using LLM-factuality fallback")

    cross_tab = [_correctness_conditioned(a, manual_labels) for a in arms]

    summary = {
        "seeds": args.seeds,
        "manual_factuality_path": str(manual_path) if manual_path.exists() else None,
        "arms": [{k: v for k, v in a.items() if k != "response_paths"} for a in arms],
        "correctness_conditioned": cross_tab,
    }
    out_path = Path(args.out)
    write_json(out_path, summary)
    print(f"[aggregate_arms] wrote {out_path}")


if __name__ == "__main__":
    main()
