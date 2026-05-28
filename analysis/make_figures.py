#!/usr/bin/env python3
"""Render the paper figures from outputs/arms_summary.json into figures/.

- Figure 1 (PART I, discovery): hedge_density across {baseline, judge_dpo,
  random, random_length_matched} with +/- 1 SE bars. The visual claim is that
  judge and random (and length-matched random) all sit well below baseline.
- Figure 2 (PART II, mitigation): hedge_density for {baseline, random,
  mit_pairfilter, mit_uncertreg} showing recovery toward baseline.
- Figure 3 (correctness-conditioned): hedge_density on the INCORRECT subset
  across arms; checks whether incorrect answers also lose hedging.

Reads only the aggregated JSON; no model loads. Arms absent from the summary are
silently dropped from each figure so this runs on partial state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexity_theater.io_utils import read_json  # noqa: E402

FIG_DIR = ROOT / "figures"

PART1_ARMS = ["baseline", "judge_dpo", "random", "random_length_matched"]
PART2_ARMS = ["baseline", "random", "mit_pairfilter", "mit_uncertreg"]


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})
    return plt


def _arm_index(summary: dict) -> dict:
    return {a["arm"]: a for a in summary["arms"]}


def _bar(plt, arms_present, means, ses, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = list(range(len(arms_present)))
    ax.bar(xs, means, yerr=ses, capsize=4, color="#bcbd22", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(arms_present, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[make_figures] wrote {out_path}")


def fig_part1(plt, idx):
    arms = [a for a in PART1_ARMS if a in idx]
    if len(arms) < 2:
        print("[make_figures] PART I figure skipped (need >= 2 arms)")
        return
    means = [idx[a]["hedge_density_mean"] for a in arms]
    ses = [idx[a]["hedge_density_se"] for a in arms]
    _bar(plt, arms, means, ses,
         "PART I: uncertainty suppression across optimization conditions",
         "hedge density (markers / 100 tok)", FIG_DIR / "fig1_discovery.png")


def fig_part2(plt, idx):
    arms = [a for a in PART2_ARMS if a in idx]
    if len(arms) < 2:
        print("[make_figures] PART II figure skipped (need >= 2 arms)")
        return
    means = [idx[a]["hedge_density_mean"] for a in arms]
    ses = [idx[a]["hedge_density_se"] for a in arms]
    _bar(plt, arms, means, ses,
         "PART II: mitigation recovery of uncertainty signaling",
         "hedge density (markers / 100 tok)", FIG_DIR / "fig2_mitigation.png")


def fig_correctness(plt, summary):
    cross = {e["arm"]: e for e in summary.get("correctness_conditioned", [])}
    arms = [a for a in (PART1_ARMS + PART2_ARMS[1:]) if a in cross]
    arms = list(dict.fromkeys(arms))  # dedupe, keep order
    if len(arms) < 2:
        print("[make_figures] correctness figure skipped (need >= 2 arms)")
        return
    incorrect_means = [cross[a]["breakdown"].get("INCORRECT", {}).get("hedge_density_mean", 0.0) for a in arms]
    incorrect_ses = [cross[a]["breakdown"].get("INCORRECT", {}).get("hedge_density_se", 0.0) for a in arms]
    fallback = any(cross[a]["label_source"] != "manual" for a in arms)
    title = "Figure 3: hedge_density on INCORRECT subset across arms"
    if fallback:
        title += " (LLM-fallback)"
    _bar(plt, arms, incorrect_means, incorrect_ses, title,
         "hedge density (markers / 100 tok)", FIG_DIR / "fig3_correctness_conditioned.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render uncertainty-suppression figures.")
    p.add_argument("--summary", default=str(ROOT / "outputs" / "arms_summary.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"[make_figures] {summary_path} not found; run analysis/aggregate_arms.py first.")
    summary = read_json(summary_path)
    try:
        plt = _setup()
    except Exception as e:
        raise SystemExit(f"[make_figures] matplotlib unavailable: {e}")
    idx = _arm_index(summary)
    fig_part1(plt, idx)
    fig_part2(plt, idx)
    fig_correctness(plt, summary)


if __name__ == "__main__":
    main()
