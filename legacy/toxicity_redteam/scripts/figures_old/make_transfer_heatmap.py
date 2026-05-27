#!/usr/bin/env python3
"""Figure 1: cross-evaluator transfer matrix heatmap for the headline regime.

Reads outputs/reports_fast/transfer_matrix.json and renders M[target][scorer]
post-SFT mean scores as a 2x2 heatmap with annotated cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _common import REPORTS, TARGETS, ensure_figures_dir, read_json, setup_style  # noqa: E402


def main() -> Path:
    setup_style()
    import matplotlib.pyplot as plt
    import numpy as np

    data = read_json(REPORTS["fast_filt_e1"])
    matrix = data["matrix_post"]

    rows = list(TARGETS)
    cols = list(TARGETS)
    grid = np.array([[matrix[t][s]["mean"] for s in cols] for t in rows])

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(grid, cmap="viridis", aspect="equal")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    ax.set_xlabel("scoring evaluator")
    ax.set_ylabel("SFT target evaluator")
    ax.set_title("M[target][scorer]: post-SFT mean (fast_filt_e1)")

    for i, t in enumerate(rows):
        for j, s in enumerate(cols):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("post-SFT mean score")

    out = ensure_figures_dir() / "transfer_matrix_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
