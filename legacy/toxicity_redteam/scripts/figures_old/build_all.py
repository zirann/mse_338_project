#!/usr/bin/env python3
"""Regenerate every figure under assets/figures/ from existing JSON artifacts.

No experiment is run. No JSONL is touched. Fails loudly if any input is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import make_diversity_plot  # noqa: E402
import make_own_minus_transfer_barplot  # noqa: E402
import make_taxonomy_diagram  # noqa: E402
import make_transfer_heatmap  # noqa: E402


def main() -> list[Path]:
    outputs = [
        make_transfer_heatmap.main(),
        make_own_minus_transfer_barplot.main(),
        make_diversity_plot.main(),
        make_taxonomy_diagram.main(),
    ]
    for p in outputs:
        print(p)
    return outputs


if __name__ == "__main__":
    main()
