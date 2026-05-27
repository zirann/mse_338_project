#!/usr/bin/env python3
"""Figure 4: paper-style conceptual regime diagram (no data dependency).

Two real, semantically meaningful axes:
  - x: optimization pressure (effective gradient updates)
  - y: contamination filter on / off (categorical)

Four labeled regimes placed at their empirical (x, y); the headline regime is
visually distinguished. Captions are drawn as compact annotation labels with
leader lines toward each marker so nothing overlaps.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _common import ensure_figures_dir, setup_style  # noqa: E402


def main() -> Path:
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    ax.set_xlim(-2, 38)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([2, 10, 30])
    ax.set_xticklabels(["~2 updates", "~10 updates", "~30 updates"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["filter OFF", "filter ON"])
    ax.set_xlabel("optimization pressure (effective gradient updates per target)")
    ax.set_ylabel("contamination filter")
    ax.set_title("Empirical regimes of single-evaluator LoRA-SFT reward hacking", fontsize=12)

    for x_center in (2, 10, 30):
        ax.axvline(x_center, color="#dcdcdc", linewidth=0.7, zorder=0)
    ax.axhline(0.5, color="#bbbbbb", linewidth=0.7, linestyle=":", zorder=0)

    regimes = [
        {
            "name": "No propagation",
            "tag": "prefilter weak SFT",
            "caption": "post \u2248 base; deltas \u2264 0.002",
            "x": 2,
            "y": 1,
            "label_xy": (2, 1.42),
            "headline": False,
        },
        {
            "name": "Style-level specialization (HEADLINE)",
            "tag": "fast_filt_e1",
            "caption": "own \u2212 transfer = +0.034\ndistinct_2 0.50 \u2192 0.44",
            "x": 10,
            "y": 1,
            "label_xy": (15, 1.40),
            "headline": True,
        },
        {
            "name": "Filter blind-spot overfitting",
            "tag": "fast_filt_e3",
            "caption": "deterministic 8-char prefix\ndistinct_2 0.58 \u2192 0.30",
            "x": 30,
            "y": 1,
            "label_xy": (32.5, 1.30),
            "headline": False,
        },
        {
            "name": "Token-level reward hacking",
            "tag": "fast_unfilt_e1",
            "caption": "shard repetition; mode collapse\ndistinct_2 0.53 \u2192 0.27",
            "x": 30,
            "y": 0,
            "label_xy": (32.5, -0.30),
            "headline": False,
        },
    ]

    for r in regimes:
        x, y = r["x"], r["y"]
        if r["headline"]:
            ax.scatter([x], [y], s=240, marker="o", color="#1f77b4", edgecolor="black", linewidth=1.2, zorder=3)
            ax.scatter([x], [y], s=70, marker="o", color="white", zorder=4)
        else:
            ax.scatter([x], [y], s=150, marker="o", color="white", edgecolor="#333333", linewidth=1.2, zorder=3)

        label_text = f"{r['name']}\n[{r['tag']}]\n{r['caption']}"
        fontweight = "bold" if r["headline"] else "normal"
        bg = "#eaf2fb" if r["headline"] else "#fafafa"
        ax.annotate(
            label_text,
            xy=(x, y),
            xytext=r["label_xy"],
            ha="center",
            va="center",
            fontsize=8.5,
            family="serif",
            fontweight=fontweight,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=bg, edgecolor="#555555", linewidth=0.7),
            arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.7, shrinkA=2, shrinkB=8),
            zorder=5,
        )

    footer = (
        "Filter ON + low pressure isolates the style-level reward hack. "
        "Either knob away from that point produces a distinct failure mode."
    )
    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=8.5, family="serif", color="#333333")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.13)

    out = ensure_figures_dir() / "reward_hacking_taxonomy_diagram.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
