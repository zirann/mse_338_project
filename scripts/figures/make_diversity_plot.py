#!/usr/bin/env python3
"""Figure 3: distinct_2 pre vs post across regimes, two side-by-side panels.

For each (regime, target) reads outputs/runs_*/<target>/metrics.json and
plots a pre->post line segment per regime within each target panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _common import RUNS, TARGETS, ensure_figures_dir, read_json, setup_style  # noqa: E402


REGIME_ORDER = ("fast_filt_e1", "fast_unfilt_e1", "fast_filt_e3")
REGIME_LABEL = {
    "fast_filt_e1": "filtered, e=1 (headline)",
    "fast_unfilt_e1": "unfiltered, e=1",
    "fast_filt_e3": "filtered, e=3",
}
REGIME_STYLE = {
    "fast_filt_e1": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "fast_unfilt_e1": {"color": "#d62728", "marker": "s", "linestyle": "--"},
    "fast_filt_e3": {"color": "#7f7f7f", "marker": "^", "linestyle": ":"},
}


def main() -> Path:
    setup_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True)

    for ax, target in zip(axes, TARGETS):
        for regime in REGIME_ORDER:
            metrics = read_json(RUNS[regime] / target / "metrics.json")
            pre = float(metrics["distinct_2_pre"])
            post = float(metrics["distinct_2_post"])
            style = REGIME_STYLE[regime]
            ax.plot(
                [0, 1],
                [pre, post],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                markersize=7,
                label=REGIME_LABEL[regime] if target == TARGETS[0] else None,
            )
            ax.annotate(
                f"{pre:.2f}",
                xy=(0, pre),
                xytext=(-12, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                color=style["color"],
            )
            ax.annotate(
                f"{post:.2f}",
                xy=(1, post),
                xytext=(12, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=style["color"],
            )

        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0.0, 0.65)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pre-SFT", "post-SFT"])
        ax.set_title(f"target = {target}")
        ax.axhline(0.0, color="black", linewidth=0.6)

    axes[0].set_ylabel("distinct_2")
    axes[0].legend(loc="lower left")
    fig.suptitle("Diversity (distinct_2) before vs after SFT, by regime")

    out = ensure_figures_dir() / "diversity_collapse_plot.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
