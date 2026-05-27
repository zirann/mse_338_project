"""Shared helpers for the figure-generation scripts.

Deterministic, read-only utilities. No experiment side effects.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "assets" / "figures"

REPORTS = {
    "fast_filt_e1": ROOT / "outputs" / "reports_fast" / "transfer_matrix.json",
    "fast_unfilt_e1": ROOT / "outputs" / "reports_fast_unfiltered" / "transfer_matrix.json",
    "fast_filt_e3": ROOT / "outputs" / "reports_fast_filtered_epochs3" / "transfer_matrix.json",
}

RUNS = {
    "fast_filt_e1": ROOT / "outputs" / "runs_fast",
    "fast_unfilt_e1": ROOT / "outputs" / "runs_fast_unfiltered",
    "fast_filt_e3": ROOT / "outputs" / "runs_fast_filtered_epochs3",
}

TARGETS = ("toxicbert", "cardiff")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected input not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def setup_style() -> None:
    """Apply a minimal paper-friendly matplotlib style. Called by every figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR
