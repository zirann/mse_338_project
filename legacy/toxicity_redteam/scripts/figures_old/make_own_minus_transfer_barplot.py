#!/usr/bin/env python3
"""Figure 2: own-minus-transfer asymmetry across three regimes x both targets.

For each regime r and target t, computes:
    own_minus_transfer = deltas_post_minus_pre[t][t].delta_mean
                        - deltas_post_minus_pre[t][other].delta_mean

Reads the three transfer_matrix.json files; no candidate JSONLs needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _common import REPORTS, TARGETS, ensure_figures_dir, read_json, setup_style  # noqa: E402


REGIME_ORDER = ("fast_filt_e1", "fast_unfilt_e1", "fast_filt_e3")
REGIME_LABEL = {
    "fast_filt_e1": "filtered, e=1 (headline)",
    "fast_unfilt_e1": "unfiltered, e=1",
    "fast_filt_e3": "filtered, e=3",
}


def own_minus_transfer(deltas: dict, target: str) -> float:
    other = "cardiff" if target == "toxicbert" else "toxicbert"
    return float(deltas[target][target]["delta_mean"]) - float(deltas[target][other]["delta_mean"])


def main() -> Path:
    setup_style()
    import matplotlib.pyplot as plt
    import numpy as np

    values: dict[str, list[float]] = {regime: [] for regime in REGIME_ORDER}
    for regime in REGIME_ORDER:
        data = read_json(REPORTS[regime])
        deltas = data["deltas_post_minus_pre"]
        for target in TARGETS:
            values[regime].append(own_minus_transfer(deltas, target))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(TARGETS))
    bar_w = 0.26
    offsets = np.array([-bar_w, 0.0, bar_w])
    colors = ("#1f77b4", "#d62728", "#7f7f7f")

    for i, regime in enumerate(REGIME_ORDER):
        bars = ax.bar(
            x + offsets[i],
            values[regime],
            width=bar_w,
            label=REGIME_LABEL[regime],
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, v in zip(bars, values[regime]):
            ax.annotate(
                f"{v:+.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 3 if v >= 0 else -10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} target" for t in TARGETS])
    ax.set_ylabel("own \u0394 \u2212 transfer \u0394")
    ax.set_title("Own-minus-transfer asymmetry by regime")
    ax.legend(loc="lower left")

    out = ensure_figures_dir() / "own_minus_transfer_barplot.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
