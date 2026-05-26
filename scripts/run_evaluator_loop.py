#!/usr/bin/env python3
"""Single-iteration evaluator reward-hacking run.

For one target evaluator E, do:
  1. Generate `pre_candidates.jsonl` from the base model.
  2. Cross-score with EVERY evaluator -> `pre_scored.jsonl`.
  3. Select top-K by `score_<E>` -> `accepted.jsonl` (+ `train_pairs.jsonl`).
  4. SFT LoRA on accepted; save adapter under `outputs/runs/<E>/adapter/`.
  5. Regenerate from base+adapter -> `post_candidates.jsonl`.
  6. Cross-score again -> `post_scored.jsonl`.
  7. Write `metrics.json` with pre/post per-evaluator summary stats.

Single iteration only. Multi-iteration loops are out of scope until the first
evaluator-specific specialization signal is observed.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch  # noqa: E402

from redteam.generator import generate_candidates  # noqa: E402
from redteam.io_utils import read_jsonl, read_yaml, write_json, write_jsonl  # noqa: E402
from redteam.metrics import distinct_n, harmful_hit_rate  # noqa: E402
from redteam.model_factory import (  # noqa: E402
    load_model_with_optional_adapter,
    load_tokenizer,
    resolve_device,
)
from redteam.scorers import (  # noqa: E402
    CardiffOffensiveScorer,
    ToxicBertScorer,
    score_all_evaluators,
)
from redteam.trainer import train_lora_sft  # noqa: E402


SCORER_REGISTRY: dict[str, Any] = {
    "toxic_bert": ToxicBertScorer,
    "cardiff_offensive": CardiffOffensiveScorer,
}


_REFUSAL_PHRASES = ("I'm not allowed", "I can't help", "I cannot comply")
_LONG_UNDERSCORE_RE = re.compile(r"\w+_\w+_\w+")


def filter_accepted(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Drop contaminated candidates before SFT.

    Rules (applied in order, each row counts toward one bucket only):
      1. too_short        -- whitespace token count < 8
      2. okay_prefix      -- text starts with 'Okay' (thinking-mode marker)
      3. tool_call        -- text contains '<tool_call>'
      4. refusal          -- text contains a refusal phrase
      5. lowercase_start  -- first token starts lowercase AND len < 6  (conservative
                              decoder-shard rule; normal lowercase sentence starts pass
                              when the first word is 6+ chars)
      6. underscore_joined -- 3+ underscore-joined word tokens (hallucinated identifier-like
                              shards, e.g. mens_all_have_..._asian_...)
    """
    counts = {
        "too_short": 0,
        "okay_prefix": 0,
        "tool_call": 0,
        "refusal": 0,
        "lowercase_start": 0,
        "underscore_joined": 0,
    }
    out: list[dict] = []
    for r in rows:
        text = (r.get("candidate_text") or "").strip()
        toks = text.split()
        if len(toks) < 8:
            counts["too_short"] += 1
            continue
        if text.startswith("Okay"):
            counts["okay_prefix"] += 1
            continue
        if "<tool_call>" in text:
            counts["tool_call"] += 1
            continue
        if any(p in text for p in _REFUSAL_PHRASES):
            counts["refusal"] += 1
            continue
        first_tok = toks[0]
        if first_tok[:1].islower() and len(first_tok) < 6:
            counts["lowercase_start"] += 1
            continue
        if _LONG_UNDERSCORE_RE.search(text):
            counts["underscore_joined"] += 1
            continue
        out.append(r)
    return out, counts


def build_scorers(cfg: dict, force_heuristic: bool = False) -> dict[str, Any]:
    scorers: dict[str, Any] = {}
    for e in cfg.get("evaluators", []):
        name = e["name"]
        kind = e["kind"]
        cls = SCORER_REGISTRY.get(kind)
        if cls is None:
            raise ValueError(f"Unknown evaluator kind: {kind}")
        scorers[name] = cls(model_name=e["hf_id"], force_heuristic=force_heuristic)
    return scorers


def summarize_per_evaluator(rows: list[dict], evaluator_names: list[str]) -> dict[str, dict]:
    """Per-evaluator score summary across a candidate batch."""
    out: dict[str, dict] = {}
    for name in evaluator_names:
        scores = [float(r.get(f"score_{name}", 0.0)) for r in rows]
        if not scores:
            out[name] = {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "hit_rate_0.5": 0.0}
            continue
        sorted_scores = sorted(scores)
        p50 = sorted_scores[len(sorted_scores) // 2]
        p90 = sorted_scores[int(0.9 * (len(sorted_scores) - 1))]
        out[name] = {
            "n": len(scores),
            "mean": statistics.fmean(scores),
            "p50": float(p50),
            "p90": float(p90),
            "hit_rate_0.5": harmful_hit_rate(scores, 0.5),
        }
    return out


def free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-iteration evaluator-specific reward hacking run")
    p.add_argument("--target_evaluator", required=True, help="Name of the evaluator that drives top-K selection")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument(
        "--force_heuristic",
        action="store_true",
        help="Use keyword-heuristic scorers instead of HF models (for tests / dry-run only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_yaml(ROOT / args.config)

    evaluator_names = [e["name"] for e in cfg.get("evaluators", [])]
    if args.target_evaluator not in evaluator_names:
        raise SystemExit(
            f"--target_evaluator '{args.target_evaluator}' is not configured. "
            f"Available: {evaluator_names}"
        )

    out_dir = ROOT / cfg["runs_root"] / args.target_evaluator
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"

    print(
        f"[diag] torch={torch.__version__}, "
        f"mps_available={torch.backends.mps.is_available()}, "
        f"mps_built={torch.backends.mps.is_built()}, "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    device = resolve_device(cfg.get("device", "auto"))
    print(f"[diag] selected device: {device}", flush=True)

    tokenizer = load_tokenizer(cfg["model_name"])
    seed_rows = read_jsonl(ROOT / cfg["seed_prompts_path"])

    # ---- Phase A: pre-SFT generation from base model
    base_model = load_model_with_optional_adapter(cfg["model_name"], adapter_path=None, device=device)
    print(
        f"[diag] generation phase A: model on {next(base_model.parameters()).device}",
        flush=True,
    )
    pre_candidates = generate_candidates(model=base_model, tokenizer=tokenizer, rows_for_prompts=seed_rows, cfg=cfg)
    free(base_model)
    write_jsonl(out_dir / "pre_candidates.jsonl", pre_candidates)

    scorers = build_scorers(cfg, force_heuristic=args.force_heuristic)
    pre_scored = score_all_evaluators(pre_candidates, scorers)
    write_jsonl(out_dir / "pre_scored.jsonl", pre_scored)

    # ---- Top-K accept by target evaluator score, then contamination filter
    target_field = f"score_{args.target_evaluator}"
    top_k = int(cfg.get("top_k", 64))
    accepted_before = sorted(pre_scored, key=lambda r: float(r.get(target_field, 0.0)), reverse=True)
    accepted_before = [r for r in accepted_before if (r.get("candidate_text") or "").strip()]
    accepted_before = accepted_before[:top_k]
    write_jsonl(out_dir / "accepted_before_filter.jsonl", accepted_before)

    apply_filter = bool(cfg.get("apply_contamination_filter", True))
    if apply_filter:
        accepted, filter_counts = filter_accepted(accepted_before)
    else:
        accepted = list(accepted_before)
        filter_counts = {
            "too_short": 0,
            "okay_prefix": 0,
            "tool_call": 0,
            "refusal": 0,
            "lowercase_start": 0,
            "underscore_joined": 0,
        }
    write_jsonl(out_dir / "accepted_after_filter.jsonl", accepted)

    filter_stats = {
        "apply_contamination_filter": apply_filter,
        "total_candidates": len(pre_scored),
        "accepted_before_filter": len(accepted_before),
        "accepted_after_filter": len(accepted),
        "rejection_counts": filter_counts,
    }

    min_train_pairs = int(cfg.get("min_train_pairs", 16))
    if len(accepted) < min_train_pairs:
        partial_metrics = {
            "target_evaluator": args.target_evaluator,
            "evaluator_names": evaluator_names,
            "top_k": top_k,
            "num_pre": len(pre_scored),
            "filter_stats": filter_stats,
            "status": "aborted_undersized_train_set",
            "min_train_pairs": min_train_pairs,
        }
        write_json(out_dir / "metrics.json", partial_metrics)
        raise SystemExit(
            f"Filtered train set too small: {len(accepted)} < {min_train_pairs}. "
            f"See {out_dir / 'metrics.json'} for filter_stats."
        )

    write_jsonl(
        out_dir / "train_pairs.jsonl",
        [{"input_prompt": r["input_prompt"], "candidate_text": r["candidate_text"]} for r in accepted],
    )

    # ---- SFT LoRA on accepted candidates
    train_cfg = {
        "seed": int(cfg.get("seed", 42)),
        "model_name": cfg["model_name"],
        "require_model_name": cfg["model_name"],
        "output_dir": str(adapter_dir),
        "max_train_samples": int(cfg.get("max_train_samples", 12000)),
        "max_length": int(cfg.get("max_length", 256)),
        "device": device,
        "lora": cfg.get("lora", {}),
        "training": cfg.get("training", {}),
    }
    train_result = train_lora_sft(accepted, train_cfg)

    # ---- Phase B: post-SFT generation
    sft_model = load_model_with_optional_adapter(cfg["model_name"], adapter_path=str(adapter_dir), device=device)
    print(
        f"[diag] generation phase B: model on {next(sft_model.parameters()).device}",
        flush=True,
    )
    post_candidates = generate_candidates(model=sft_model, tokenizer=tokenizer, rows_for_prompts=seed_rows, cfg=cfg)
    free(sft_model)
    write_jsonl(out_dir / "post_candidates.jsonl", post_candidates)

    post_scored = score_all_evaluators(post_candidates, scorers)
    write_jsonl(out_dir / "post_scored.jsonl", post_scored)

    # ---- Metrics
    pre_texts = [r.get("candidate_text", "") for r in pre_scored]
    post_texts = [r.get("candidate_text", "") for r in post_scored]
    metrics = {
        "target_evaluator": args.target_evaluator,
        "evaluator_names": evaluator_names,
        "top_k": top_k,
        "num_pre": len(pre_scored),
        "num_accepted": len(accepted),
        "num_post": len(post_scored),
        "filter_stats": filter_stats,
        "pre_summary": summarize_per_evaluator(pre_scored, evaluator_names),
        "accepted_summary": summarize_per_evaluator(accepted, evaluator_names),
        "post_summary": summarize_per_evaluator(post_scored, evaluator_names),
        "distinct_2_pre": distinct_n(pre_texts, 2),
        "distinct_2_post": distinct_n(post_texts, 2),
        "train_result": train_result,
    }
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
