#!/usr/bin/env python3
"""Render the paper figures from outputs/arms_summary.json into figures/.

- Figure 1 (length): response length across all arms. Confirms SamPO/DPOP move
  length as intended and lets uncertainty changes be checked against length.
- Figure 2 (reproduce): hedge_density for {baseline, vanilla_dpo, sampo_dpo}.
  The reproduction claim: vanilla DPO suppresses hedging; SamPO length control
  changes the picture.
- Figure 3 (extend): hedge_density for {baseline, vanilla_dpo, sampo_dpo, dpop,
  sampo_dpop}. Does DPOP positive-preservation retain uncertainty signaling?
- Figure 4 (win-rate): judge_win_rate_vs_baseline across arms (quality is
  maintained while length/uncertainty change).

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

ALL_ARMS = ["baseline", "vanilla_dpo", "sampo_dpo", "dpop", "sampo_dpop"]
REPRODUCE_ARMS = ["baseline", "vanilla_dpo", "sampo_dpo"]
EXTEND_ARMS = ["baseline", "vanilla_dpo", "sampo_dpo", "dpop", "sampo_dpop"]


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


def _metric_bar(plt, idx, arm_list, metric, title, ylabel, out_name):
    arms = [a for a in arm_list if a in idx]
    if len(arms) < 2:
        print(f"[make_figures] {out_name} skipped (need >= 2 arms present)")
        return
    means = [idx[a].get(f"{metric}_mean", 0.0) for a in arms]
    ses = [idx[a].get(f"{metric}_se", 0.0) for a in arms]
    _bar(plt, arms, means, ses, title, ylabel, FIG_DIR / out_name)


def fig_length(plt, idx):
    _metric_bar(plt, idx, ALL_ARMS, "length",
                "Response length across arms (SamPO/DPOP length control)",
                "mean response length (tokens)", "fig1_length.png")


def fig_reproduce(plt, idx):
    _metric_bar(plt, idx, REPRODUCE_ARMS, "hedge_density",
                "Reproduce: hedge density, vanilla DPO vs SamPO length control",
                "hedge density (markers / 100 tok)", "fig2_reproduce_hedge.png")


def fig_extend(plt, idx):
    _metric_bar(plt, idx, EXTEND_ARMS, "hedge_density",
                "Extend: hedge density with DPOP positive preservation",
                "hedge density (markers / 100 tok)", "fig3_extend_hedge.png")


def fig_winrate(plt, idx):
    _metric_bar(plt, idx, EXTEND_ARMS, "judge_win_rate_vs_round_0",
                "Judge win-rate vs baseline across arms",
                "win-rate vs baseline", "fig4_winrate.png")


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
    fig_length(plt, idx)
    fig_reproduce(plt, idx)
    fig_extend(plt, idx)
    fig_winrate(plt, idx)


if __name__ == "__main__":
    main()
