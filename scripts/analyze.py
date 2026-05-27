#!/usr/bin/env python3
"""Combine per-round `metrics.json` into `trajectory.json` and render the
headline figure when at least two rounds are present.

Headline figure (`assets/figures/headline.png`):
- X axis: round (0..num_rounds_found-1)
- Y axis: z-score-normalized metric value
- Curves: 3 appearance metrics in warm tones, 2 substance metrics in cool
  tones, 1 composite (judge_win_rate_vs_round_0) in black (round > 0 only)

This script is purely an aggregator + plotter. It does not load any models.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexity_theater.io_utils import read_json, read_yaml, write_json  # noqa: E402


APPEARANCE_METRICS = (
    "length",
    "structural_complexity",
    "reasoning_narration_density",
    "hedge_density",
)
SUBSTANCE_METRICS = ("factuality", "information_density")
COMPOSITE_METRIC = "judge_win_rate_vs_round_0"
# Metrics that stay in trajectory.json for back-compat / appendix analysis but
# are NOT plotted on the headline figure. `epistemic_marker_density` is a wide
# union of all 7 lexicon subclasses; after the round-1 smoke we found it
# averages opposing signals (reasoning narration UP, hedges DOWN) and is
# superseded by the two new headline metrics.
EXTRA_TRAJECTORY_METRICS = ("epistemic_marker_density",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate per-round metrics into trajectory.json + headline figure.")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    return p.parse_args()


def _zscore(values: list[float | None]) -> list[float | None]:
    """Z-score-normalize a series, skipping None. Returns None where input is None.

    If the series has zero variance, returns 0.0 at every non-None index.
    """
    real = [v for v in values if v is not None]
    if len(real) < 2:
        return [0.0 if v is not None else None for v in values]
    mean = sum(real) / len(real)
    var = sum((v - mean) ** 2 for v in real) / len(real)
    std = math.sqrt(var)
    if std == 0.0:
        return [0.0 if v is not None else None for v in values]
    return [(v - mean) / std if v is not None else None for v in values]


def _render_headline(trajectory: dict, fig_path: Path) -> bool:
    """Render the headline figure. Returns True on success, False if skipped."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[analyze] headline skipped: matplotlib unavailable ({e})")
        return False

    rounds = [r["round"] for r in trajectory["rounds"]]
    if len(rounds) < 2:
        print(f"[analyze] headline skipped: only {len(rounds)} round(s) available")
        return False

    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    try:
        from _common import ensure_figures_dir, setup_style  # type: ignore
        setup_style()
        ensure_figures_dir()
    except Exception:
        pass  # acceptable; we'll just plot with defaults

    fig, ax = plt.subplots(figsize=(7, 4.2))

    series_keys = list(APPEARANCE_METRICS) + list(SUBSTANCE_METRICS) + [COMPOSITE_METRIC]
    colors = {
        "length": "#d62728",
        "structural_complexity": "#ff7f0e",
        "reasoning_narration_density": "#e377c2",
        "hedge_density": "#bcbd22",
        "factuality": "#1f77b4",
        "information_density": "#2ca02c",
        COMPOSITE_METRIC: "#000000",
    }
    label_pretty = {
        "length": "length (app.)",
        "structural_complexity": "structural complexity (app.)",
        "reasoning_narration_density": "reasoning narration / 100 tok (app.)",
        "hedge_density": "hedge density / 100 tok (app.)",
        "factuality": "factuality (sub.)",
        "information_density": "information density (sub.)",
        COMPOSITE_METRIC: "judge win-rate vs round 0",
    }

    for key in series_keys:
        raw = [r.get(key) for r in trajectory["rounds"]]
        z = _zscore(raw)
        xs = [r for r, v in zip(rounds, z) if v is not None]
        ys = [v for v in z if v is not None]
        if not xs:
            continue
        ls = "--" if key == COMPOSITE_METRIC else "-"
        ax.plot(xs, ys, ls, label=label_pretty[key], color=colors[key], marker="o")

    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("DPO round")
    ax.set_ylabel("z-score (per-metric, across rounds)")
    ax.set_title("Optimization-Induced Epistemic Simulacra (z-normalized trajectory)")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[analyze] wrote {fig_path}")
    return True


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)

    template = cfg["outputs"]["per_round_dir_template"]
    num_rounds = int(cfg["dpo"]["num_rounds"])

    rounds_data: list[dict] = []
    for n in range(num_rounds + 1):
        path = ROOT / template.format(round=n) / "metrics.json"
        if not path.exists():
            print(f"[analyze] skipping round {n}: {path} not found")
            continue
        rounds_data.append(read_json(path))

    if not rounds_data:
        raise SystemExit("[analyze] no per-round metrics found; nothing to aggregate.")

    # Trajectory includes the headline metrics PLUS the back-compat extras
    # (kept readable from `trajectory.json` even though they are not plotted).
    series_keys = (
        list(APPEARANCE_METRICS)
        + list(SUBSTANCE_METRICS)
        + [COMPOSITE_METRIC]
        + list(EXTRA_TRAJECTORY_METRICS)
    )
    series = {
        f"series_{k}": [(r["round"], r.get(k)) for r in rounds_data]
        for k in series_keys
    }

    trajectory = {
        "config": str(args.config),
        "num_rounds_found": len(rounds_data),
        "rounds": rounds_data,
        **series,
    }

    out_path = ROOT / cfg["outputs"]["trajectory_path"]
    write_json(out_path, trajectory)
    print(f"[analyze] wrote {out_path}")

    fig_path = ROOT / "assets" / "figures" / "headline.png"
    _render_headline(trajectory, fig_path)


if __name__ == "__main__":
    main()
