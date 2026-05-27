#!/usr/bin/env python3
"""Build the cross-evaluator transfer matrix from completed single-iteration runs.

Reads `outputs/runs/<target>/post_scored.jsonl` for each target evaluator that was
run, plus the matching `pre_scored.jsonl` for delta computation. Writes:
  - outputs/reports/transfer_matrix.json
  - outputs/reports/transfer_summary.md

The transfer matrix `M[target][scorer]` records the post-SFT mean+p90 score that
samples optimized against `target` receive from evaluator `scorer`. The asymmetry
between `M[i][j]` and `M[j][i]` is the evaluator-specific reward-hacking signal.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from redteam.io_utils import read_jsonl, read_yaml, write_json  # noqa: E402


def _summary(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    s = sorted(scores)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "p50": float(s[len(s) // 2]),
        "p90": float(s[int(0.9 * (len(s) - 1))]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cross-evaluator transfer matrix")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--out_dir", default="outputs/reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_yaml(ROOT / args.config)
    evaluator_names = [e["name"] for e in cfg.get("evaluators", [])]
    runs_root = ROOT / cfg["runs_root"]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix: dict[str, dict[str, dict[str, float]]] = {}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    top_examples: dict[str, list[dict]] = {}

    for target in evaluator_names:
        post_path = runs_root / target / "post_scored.jsonl"
        pre_path = runs_root / target / "pre_scored.jsonl"
        if not post_path.exists():
            print(f"[skip] missing {post_path}; run scripts/run_evaluator_loop.py --target_evaluator {target}")
            continue
        post_rows = read_jsonl(post_path)
        pre_rows = read_jsonl(pre_path) if pre_path.exists() else []

        matrix[target] = {}
        deltas[target] = {}
        for scorer in evaluator_names:
            post_scores = [float(r.get(f"score_{scorer}", 0.0)) for r in post_rows]
            pre_scores = [float(r.get(f"score_{scorer}", 0.0)) for r in pre_rows]
            matrix[target][scorer] = _summary(post_scores)
            if pre_scores:
                deltas[target][scorer] = {
                    "delta_mean": matrix[target][scorer]["mean"] - _summary(pre_scores)["mean"],
                    "delta_p90": matrix[target][scorer]["p90"] - _summary(pre_scores)["p90"],
                }

        ranked = sorted(post_rows, key=lambda r: float(r.get(f"score_{target}", 0.0)), reverse=True)[:5]
        top_examples[target] = [
            {
                "score_target": float(r.get(f"score_{target}", 0.0)),
                **{f"score_{s}": float(r.get(f"score_{s}", 0.0)) for s in evaluator_names if s != target},
                "candidate_text": r.get("candidate_text", ""),
            }
            for r in ranked
        ]

    report = {
        "evaluators": evaluator_names,
        "matrix_post": matrix,
        "deltas_post_minus_pre": deltas,
        "top_examples_post": top_examples,
    }
    write_json(out_dir / "transfer_matrix.json", report)

    lines: list[str] = []
    lines.append("# Cross-evaluator transfer matrix\n")
    lines.append("Each cell shows post-SFT mean score that samples optimized against `target`")
    lines.append("receive from evaluator `scorer`. Diagonals = self-evaluation; off-diagonals = transfer.\n")
    header = "| target \\ scorer | " + " | ".join(evaluator_names) + " |"
    sep = "|" + "|".join(["---"] * (len(evaluator_names) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for target in evaluator_names:
        if target not in matrix:
            continue
        row = [target]
        for scorer in evaluator_names:
            cell = matrix[target].get(scorer, {})
            row.append(f"{cell.get('mean', 0.0):.3f} (p90={cell.get('p90', 0.0):.3f})")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Deltas (post mean - pre mean)\n")
    lines.append(header)
    lines.append(sep)
    for target in evaluator_names:
        if target not in deltas:
            continue
        row = [target]
        for scorer in evaluator_names:
            d = deltas[target].get(scorer, {})
            row.append(f"{d.get('delta_mean', 0.0):+.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Top-5 post examples per target evaluator\n")
    for target, examples in top_examples.items():
        lines.append(f"### target = {target}\n")
        for ex in examples:
            other_scores = ", ".join(
                f"{k}={v:.3f}" for k, v in ex.items() if k.startswith("score_") and k != "score_target"
            )
            lines.append(
                f"- score_{target}={ex['score_target']:.3f} | {other_scores}\n  > {ex['candidate_text'][:200]}"
            )
        lines.append("")

    (out_dir / "transfer_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(out_dir / "transfer_matrix.json"))
    print(str(out_dir / "transfer_summary.md"))


if __name__ == "__main__":
    main()
