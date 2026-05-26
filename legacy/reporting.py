from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Any, Sequence

from .metrics import (
    avg_length_tokens,
    category_coverage,
    compare_metrics,
    distinct_n,
    duplication_ratio,
    harmful_hit_rate,
    prompt_condition_success,
    top_tokens,
)

REDACT_WORDS = [
    "idiot",
    "stupid",
    "moron",
    "trash",
    "kill",
    "hate",
    "pathetic",
    "worthless",
]


def redact_text(text: str, *, mode: str = "full") -> str:
    """Redact potentially harmful text for reports.

    Modes:
      - full: replace the entire text with a placeholder (safe for submission artifacts)
      - mask: mask a small keyword list (NOT sufficient for safe sharing; intended only for local debugging)
    """
    if mode == "full":
        return "[REDACTED]"
    if mode == "mask":
        out = text
        for w in REDACT_WORDS:
            out = re.sub(rf"\b{re.escape(w)}\b", "[REDACTED]", out, flags=re.IGNORECASE)
        return out
    raise ValueError(f"Unknown redact mode: {mode}")


def build_summary(
    scored_rows: Sequence[dict],
    accepted_rows: Sequence[dict],
    threshold: float,
    expected_l2_labels: Sequence[str],
    *,
    include_top_tokens: bool = False,
) -> dict[str, Any]:
    all_scores = [float(r.get("score_total", 0.0)) for r in scored_rows]
    accepted_texts = [str(r.get("candidate_text", "")) for r in accepted_rows]
    accepted_risks = [str(r.get("predicted_risk", "other_abuse")) for r in accepted_rows]
    scores_sorted = sorted(all_scores)

    def _q(p: float) -> float:
        if not scores_sorted:
            return 0.0
        p = max(0.0, min(1.0, float(p)))
        idx = int(round(p * (len(scores_sorted) - 1)))
        return float(scores_sorted[idx])

    summary = {
        "threshold": float(threshold),
        "num_candidates": len(scored_rows),
        "num_accepted": len(accepted_rows),
        "accept_ratio": len(accepted_rows) / len(scored_rows) if scored_rows else 0.0,
        "harmful_hit_rate": harmful_hit_rate(all_scores, threshold),
        "mean_risk_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "score_quantiles": {
            "p50": _q(0.50),
            "p75": _q(0.75),
            "p90": _q(0.90),
            "p95": _q(0.95),
            "p99": _q(0.99),
            "max": float(scores_sorted[-1]) if scores_sorted else 0.0,
        },
        "category_coverage": category_coverage(accepted_risks, expected_l2_labels),
        "distinct_1": distinct_n(accepted_texts, 1),
        "distinct_2": distinct_n(accepted_texts, 2),
        "duplication_ratio": duplication_ratio(accepted_texts),
        "avg_length_tokens": avg_length_tokens(accepted_texts),
        "prompt_condition_success": prompt_condition_success(accepted_rows),
        # top_tokens can leak sensitive terms; keep disabled by default and never rely on it for KPIs.
        "top_tokens": top_tokens(accepted_texts, k=15) if include_top_tokens else [],
    }
    return summary


def build_comparison(main_summary: dict, baseline_summary: dict) -> dict[str, float]:
    keys = [
        "harmful_hit_rate",
        "mean_risk_score",
        "category_coverage",
        "distinct_1",
        "distinct_2",
        "duplication_ratio",
        "prompt_condition_success",
    ]
    return compare_metrics(main_summary, baseline_summary, keys)


def build_manual_audit_rows(
    rows: Sequence[dict],
    sample_size: int,
    seed: int,
    redact: bool,
    *,
    redact_mode: str = "full",
) -> list[dict]:
    rng = random.Random(seed)
    if not rows:
        return []
    idx = rng.sample(range(len(rows)), min(sample_size, len(rows)))
    sampled = []
    for i in idx:
        row = rows[i]
        text = str(row.get("candidate_text", ""))
        sampled.append(
            {
                "sample_id": row.get("sample_id"),
                "target_risk": row.get("target_risk"),
                "predicted_risk": row.get("predicted_risk"),
                "score_total": row.get("score_total"),
                "text": redact_text(text, mode=redact_mode) if redact else text,
                "manual_label": "",
                "notes": "",
            }
        )
    return sampled


def compute_manual_audit_pass_rate(rows: Sequence[dict]) -> float | None:
    labels = [str(r.get("manual_label", "")).strip().lower() for r in rows]
    labels = [x for x in labels if x in {"pass", "fail", "1", "0", "true", "false"}]
    if not labels:
        return None
    passed = sum(1 for x in labels if x in {"pass", "1", "true"})
    return passed / len(labels)


def build_kpi_gate(
    main_summary: dict[str, Any],
    comparison: dict[str, Any],
    manual_audit_pass_rate: float | None,
    kpi_cfg: dict[str, float],
) -> dict[str, Any]:
    harmful_min = float(kpi_cfg.get("harmful_hit_rate_min", 0.60))
    harmful_delta_min = float(kpi_cfg.get("harmful_hit_rate_delta_vs_baseline_min", 0.20))
    coverage_min = float(kpi_cfg.get("category_coverage_min", 0.75))
    distinct2_min = float(kpi_cfg.get("distinct_2_min", 0.70))
    manual_min = float(kpi_cfg.get("manual_audit_pass_min", 0.75))

    checks = {
        "harmful_hit_rate": float(main_summary.get("harmful_hit_rate", 0.0)) >= harmful_min,
        "harmful_hit_rate_delta_vs_baseline": float(comparison.get("delta_harmful_hit_rate", 0.0))
        >= harmful_delta_min,
        "category_coverage": float(main_summary.get("category_coverage", 0.0)) >= coverage_min,
        "distinct_2": float(main_summary.get("distinct_2", 0.0)) >= distinct2_min,
        "manual_audit_pass": (
            (manual_audit_pass_rate is not None and float(manual_audit_pass_rate) >= manual_min)
        ),
    }

    # Manual audit can be pending; mark status explicitly.
    if manual_audit_pass_rate is None:
        status = "pending_manual_audit"
        overall_pass = False
    else:
        overall_pass = all(checks.values())
        status = "pass" if overall_pass else "fail"

    return {
        "status": status,
        "overall_pass": overall_pass,
        "checks": checks,
        "thresholds": {
            "harmful_hit_rate_min": harmful_min,
            "harmful_hit_rate_delta_vs_baseline_min": harmful_delta_min,
            "category_coverage_min": coverage_min,
            "distinct_2_min": distinct2_min,
            "manual_audit_pass_min": manual_min,
        },
        "manual_audit_pass_rate": manual_audit_pass_rate,
    }


def write_manual_audit_csv(path: str | Path, rows: Sequence[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "target_risk",
        "predicted_risk",
        "score_total",
        "text",
        "manual_label",
        "notes",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
