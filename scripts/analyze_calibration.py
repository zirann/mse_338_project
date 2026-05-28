#!/usr/bin/env python3
"""Calibration / uncertainty-suppression aggregation across all DPO arms.

Reads `eval_responses.jsonl` from each known arm directory, optionally joins
with author manual factuality labels at `outputs/manual_factuality.jsonl`,
and writes:

- `outputs/calibration_summary.json` -- per-arm means + correctness-conditioned
  cross-tabs for `hedge_density` and `confidence_marker_density`.
- `assets/figures/headline.png` -- two-panel raw-axes figure:
  Panel A: per-arm hedge_density with +/- 1 SE bars.
  Panel B: per-arm hedge_density split by manual_label (CORRECT vs INCORRECT)
           when the sidecar exists; otherwise gracefully skipped.

The script is deliberately tolerant of partial state: arms with missing
`eval_responses.jsonl` are skipped with a printed note; the manual-label join
is optional. No model loads, no judge calls. ~seconds to run.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexity_theater import appearance  # noqa: E402
from complexity_theater.io_utils import read_jsonl, read_yaml, write_json  # noqa: E402


# Known arms. Each entry is (display_label, directory_path).
# Order is deliberate: baseline first, main trajectory, then controls.
ARM_REGISTRY: list[tuple[str, str]] = [
    ("R0_baseline", "outputs/round_0"),
    ("main_R1", "outputs/round_1"),
    ("main_R2", "outputs/round_2"),
    ("main_R3", "outputs/round_3"),
    ("random_R1", "outputs/control_random_round_1"),
    ("length_matched_random_R1", "outputs/control_random_length_matched_round_1"),
    ("length_matched_judge_R1", "outputs/control_judge_length_matched_round_1"),
]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean_se(xs: list[float]) -> tuple[float, float]:
    """Sample mean and standard error of the mean. SE = SD / sqrt(n). n=1 -> SE=0."""
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs))


def _per_response_metrics(rows: list[dict]) -> list[dict]:
    """Compute hedge_density and confidence_marker_density per response row."""
    out: list[dict] = []
    for r in rows:
        text = r.get("response", "")
        out.append(
            {
                "prompt_id": r.get("prompt_id"),
                "hedge_density": appearance.hedge_density(text),
                "confidence_marker_density": appearance.confidence_marker_density(text),
                "llm_factuality": r.get("factuality"),
            }
        )
    return out


def _aggregate_arm(arm_label: str, arm_dir: Path) -> dict | None:
    """Compute per-arm mean/SE of hedge + confidence densities. Returns None
    when the arm directory or its eval_responses.jsonl is missing."""
    eval_path = arm_dir / "eval_responses.jsonl"
    if not eval_path.exists():
        return None
    rows = read_jsonl(eval_path)
    if not rows:
        return None
    per = _per_response_metrics(rows)
    hed_mean, hed_se = _mean_se([p["hedge_density"] for p in per])
    cm_mean, cm_se = _mean_se([p["confidence_marker_density"] for p in per])
    fact_vals = [p["llm_factuality"] for p in per if p["llm_factuality"] is not None]
    fact_mean, fact_se = _mean_se([float(v) for v in fact_vals])
    return {
        "arm": arm_label,
        "arm_dir": str(arm_dir),
        "n": len(per),
        "hedge_density_mean": hed_mean,
        "hedge_density_se": hed_se,
        "confidence_marker_density_mean": cm_mean,
        "confidence_marker_density_se": cm_se,
        "llm_factuality_mean": fact_mean,
        "llm_factuality_se": fact_se,
        "per_response": per,
    }


def _correctness_conditioned(
    per_arm: list[dict],
    manual_labels: dict[tuple[str, str], str] | None,
) -> list[dict]:
    """For each arm, partition responses by manual label (or LLM factuality
    proxy if the manual sidecar is absent) and report subset means.

    Manual label index: dict keyed by (arm, prompt_id) -> {CORRECT, PARTIAL, INCORRECT}.
    When the manual sidecar is missing, falls back to a 3-way bucket from
    LLM-judged factuality: 1.0 -> CORRECT, 0.5 -> PARTIAL, 0.0 -> INCORRECT.
    The fallback inherits the same-family confound and is flagged accordingly.
    """
    out: list[dict] = []
    for arm in per_arm:
        buckets: dict[str, dict[str, list[float]]] = {
            lbl: {"hedge": [], "confidence": []}
            for lbl in ("CORRECT", "PARTIAL", "INCORRECT")
        }
        used_fallback = False
        for resp in arm["per_response"]:
            pid = resp["prompt_id"]
            if manual_labels is not None and (arm["arm"], pid) in manual_labels:
                label = manual_labels[(arm["arm"], pid)]
            else:
                used_fallback = True
                f = resp["llm_factuality"]
                if f is None:
                    continue
                if f >= 0.75:
                    label = "CORRECT"
                elif f >= 0.25:
                    label = "PARTIAL"
                else:
                    label = "INCORRECT"
            if label not in buckets:
                continue
            buckets[label]["hedge"].append(resp["hedge_density"])
            buckets[label]["confidence"].append(resp["confidence_marker_density"])
        breakdown = {}
        for lbl, data in buckets.items():
            n = len(data["hedge"])
            if n == 0:
                breakdown[lbl] = {"n": 0}
                continue
            hed_mean, hed_se = _mean_se(data["hedge"])
            cm_mean, cm_se = _mean_se(data["confidence"])
            breakdown[lbl] = {
                "n": n,
                "hedge_density_mean": hed_mean,
                "hedge_density_se": hed_se,
                "confidence_marker_density_mean": cm_mean,
                "confidence_marker_density_se": cm_se,
            }
        out.append(
            {
                "arm": arm["arm"],
                "label_source": "manual" if not used_fallback else "llm_factuality_fallback",
                "breakdown": breakdown,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _render_headline(per_arm: list[dict], cross_tab: list[dict], fig_path: Path) -> bool:
    """Render the new two-panel headline. Returns True on success."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[analyze_calibration] headline skipped: matplotlib unavailable ({e})")
        return False

    # Try to use shared style helpers; ignore failures.
    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    try:
        from _common import ensure_figures_dir, setup_style  # type: ignore

        setup_style()
        ensure_figures_dir()
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A: per-arm hedge_density with +/- 1 SE bars.
    axA = axes[0]
    labels = [a["arm"] for a in per_arm]
    means = [a["hedge_density_mean"] for a in per_arm]
    ses = [a["hedge_density_se"] for a in per_arm]
    xs = list(range(len(labels)))
    axA.bar(xs, means, yerr=ses, color="#bcbd22", capsize=4, alpha=0.85)
    axA.set_xticks(xs)
    axA.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axA.set_ylabel("hedge density (markers / 100 tok)")
    axA.set_title("Panel A: hedge_density across arms (+/- 1 SE)")

    # Panel B: per-arm hedge_density on CORRECT vs INCORRECT subsets.
    axB = axes[1]
    width = 0.35
    correct_means = []
    correct_ses = []
    incorrect_means = []
    incorrect_ses = []
    fallback_seen = False
    for entry in cross_tab:
        c = entry["breakdown"].get("CORRECT", {})
        i = entry["breakdown"].get("INCORRECT", {})
        correct_means.append(c.get("hedge_density_mean", 0.0))
        correct_ses.append(c.get("hedge_density_se", 0.0))
        incorrect_means.append(i.get("hedge_density_mean", 0.0))
        incorrect_ses.append(i.get("hedge_density_se", 0.0))
        if entry["label_source"] != "manual":
            fallback_seen = True
    xs = list(range(len(cross_tab)))
    axB.bar([x - width / 2 for x in xs], correct_means, width, yerr=correct_ses,
            color="#1f77b4", capsize=3, alpha=0.85, label="CORRECT")
    axB.bar([x + width / 2 for x in xs], incorrect_means, width, yerr=incorrect_ses,
            color="#d62728", capsize=3, alpha=0.85, label="INCORRECT")
    axB.set_xticks(xs)
    axB.set_xticklabels([e["arm"] for e in cross_tab], rotation=30, ha="right", fontsize=8)
    axB.set_ylabel("hedge density (markers / 100 tok)")
    title = "Panel B: hedge_density by correctness subset"
    if fallback_seen:
        title += " (LLM-fallback)"
    axB.set_title(title)
    axB.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Optimization-Induced Uncertainty Suppression Under Preference Optimization",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[analyze_calibration] wrote {fig_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration aggregation across all DPO arms.")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument(
        "--manual_factuality",
        default=str(ROOT / "outputs" / "manual_factuality.jsonl"),
        help="Optional sidecar JSONL with rows {prompt_id, arm, llm_factuality, manual_label}.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Read config so the script is config-driven; current implementation only
    # uses the registry above, but reading the YAML catches config-path typos
    # at CLI time and matches the project convention.
    read_yaml(args.config)

    per_arm: list[dict] = []
    for arm_label, arm_path in ARM_REGISTRY:
        arm_dir = ROOT / arm_path
        result = _aggregate_arm(arm_label, arm_dir)
        if result is None:
            print(f"[analyze_calibration] skipping {arm_label}: {arm_dir} missing or empty")
            continue
        print(
            f"[analyze_calibration] {arm_label}: n={result['n']}  "
            f"hedge={result['hedge_density_mean']:.4f}+-{result['hedge_density_se']:.4f}  "
            f"conf={result['confidence_marker_density_mean']:.4f}+-{result['confidence_marker_density_se']:.4f}"
        )
        per_arm.append(result)

    if not per_arm:
        raise SystemExit("[analyze_calibration] no arms found; nothing to aggregate.")

    manual_labels: dict[tuple[str, str], str] | None = None
    manual_path = Path(args.manual_factuality)
    if manual_path.exists():
        rows = read_jsonl(manual_path)
        manual_labels = {
            (r["arm"], r["prompt_id"]): r["manual_label"].upper() for r in rows
        }
        print(
            f"[analyze_calibration] loaded manual labels: {len(manual_labels)} rows from {manual_path}"
        )
    else:
        print(
            f"[analyze_calibration] manual factuality sidecar not found at {manual_path}; "
            f"using LLM-factuality fallback for correctness-conditioning."
        )

    cross_tab = _correctness_conditioned(per_arm, manual_labels)

    summary = {
        "config": str(args.config),
        "manual_factuality_path": str(manual_path) if manual_path.exists() else None,
        "arms": [
            {k: v for k, v in arm.items() if k != "per_response"}
            for arm in per_arm
        ],
        "correctness_conditioned": cross_tab,
    }
    out_path = ROOT / "outputs" / "calibration_summary.json"
    write_json(out_path, summary)
    print(f"[analyze_calibration] wrote {out_path}")

    fig_path = ROOT / "assets" / "figures" / "headline.png"
    _render_headline(per_arm, cross_tab, fig_path)


if __name__ == "__main__":
    main()
