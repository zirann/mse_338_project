#!/usr/bin/env python3
"""Evaluate one policy (base model or DPO adapter at round N) on the eval set.

Writes `outputs/round_<N>/eval_responses.jsonl` and `metrics.json`. Metrics:

- Appearance: length, structural_complexity, epistemic_marker_density
- Substance: factuality, information_density
- Composite (round > 0): judge_win_rate_vs_round_0

`--mock` skips all model loads and uses deterministic template responses that
mirror the predicted phenomenon (round-0 short, later rounds longer + more
markdown + more epistemic markers). This is enough to exercise the metric
pipeline + JSONL outputs locally without touching Qwen3-0.6B or Qwen2.5-7B.

`--limit N` caps the eval set to the first N prompts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexity_theater import appearance, substance  # noqa: E402
from complexity_theater.io_utils import read_jsonl, read_yaml, write_json, write_jsonl  # noqa: E402
from complexity_theater.judge import Judge  # noqa: E402


# ---------------------------------------------------------------------------
# Mock generation (deterministic, smoke-test only)
# ---------------------------------------------------------------------------

def mock_response(question: str, round_n: int) -> str:
    """Deterministic per-round mock response.

    Round 0: terse direct answer. Subsequent rounds add headers, bullets, and
    epistemic markers — the predicted "complexity theater" trajectory. This is
    not part of the real experiment; it only exercises the pipeline.
    """
    if round_n <= 0:
        return "Yes."
    if round_n == 1:
        return f"Yes, briefly: this is the answer to '{question}'."
    if round_n == 2:
        return (
            f"# Answer\n\n"
            f"- Yes, the question '{question}' has a short answer.\n"
            f"- The relevant point is direct.\n"
        )
    # round_n >= 3
    return (
        f"# Answer\n\n"
        f"It's worth noting that, generally speaking, this question is nuanced.\n\n"
        f"- First, let me think through this carefully.\n"
        f"- However, it should be acknowledged that context matters.\n\n"
        f"Importantly, the response to '{question}' depends on framing.\n\n"
        f"In summary: the answer is yes."
    )


# ---------------------------------------------------------------------------
# Real generation (A100 path; not exercised locally)
# ---------------------------------------------------------------------------

def _real_load_policy(cfg: dict, round_n: int):
    """Load base model + optional round adapter for evaluation."""
    from complexity_theater.model_factory import (
        load_model_with_optional_adapter,
        load_tokenizer,
        resolve_device,
    )

    device = resolve_device(cfg["base_model"]["device"])
    tokenizer = load_tokenizer(cfg["base_model"]["hf_id"])
    adapter_path = None
    if round_n > 0:
        adapter_path = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=round_n) / "adapter"
        adapter_path = str(adapter_path) if adapter_path.exists() else None
    model = load_model_with_optional_adapter(cfg["base_model"]["hf_id"], adapter_path, device)
    return tokenizer, model, device


def _real_generate(tokenizer, model, device, question: str, cfg: dict) -> str:
    import torch

    messages = [{"role": "user", "content": question}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    gen_cfg = cfg["generation"]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 200)),
            do_sample=True,
            temperature=float(gen_cfg.get("eval_temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.95)),
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate one round.")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--round", type=int, required=True, help="Round number (0 = baseline).")
    p.add_argument("--limit", type=int, default=None, help="Cap eval prompts (for smoke).")
    p.add_argument("--mock", action="store_true", help="Skip real model + judge calls.")
    return p.parse_args()


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    round_n = int(args.round)

    data_dir = ROOT / cfg["outputs"]["data_dir"]
    eval_path = data_dir / "eval_prompts.jsonl"
    eval_rows = read_jsonl(eval_path)
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]
    print(f"[evaluate round={round_n}] {len(eval_rows)} prompts mock={args.mock}")

    if args.mock:
        tokenizer = model = device = None
    else:
        tokenizer, model, device = _real_load_policy(cfg, round_n)

    responses: list[dict] = []
    for row in eval_rows:
        if args.mock:
            resp_text = mock_response(row["question"], round_n)
        else:
            resp_text = _real_generate(tokenizer, model, device, row["question"], cfg)
        responses.append(
            {
                "prompt_id": row["id"],
                "question": row["question"],
                "response": resp_text,
                "correct_reference": row["correct_reference"],
                "incorrect_reference": row["incorrect_reference"],
            }
        )

    # Appearance: cheap, no judge.
    app_per = [appearance.appearance_metrics(r["response"]) for r in responses]

    # Substance: factuality requires the judge (skipped in mock).
    judge = Judge(
        model_name=cfg["judge"]["hf_id"],
        device=cfg["judge"]["device"],
        seed=int(cfg.get("seed", 42)),
        mock=args.mock,
    )
    fact_scores: list[float] = []
    for r in responses:
        f = judge.score_factuality(
            r["question"], r["response"], r["correct_reference"], r["incorrect_reference"]
        )
        fact_scores.append(f)
        r["factuality"] = f
    sub_per = [substance.substance_metrics(r["response"], f) for r, f in zip(responses, fact_scores)]

    metrics: dict = {
        "round": round_n,
        "n_eval": len(responses),
        "mock": args.mock,
        "length": _mean([a["length"] for a in app_per]),
        "structural_complexity": _mean([a["structural_complexity"] for a in app_per]),
        "epistemic_marker_density": _mean([a["epistemic_marker_density"] for a in app_per]),
        "factuality": _mean(fact_scores),
        "information_density": _mean([s["information_density"] for s in sub_per]),
    }

    # Composite (round > 0): judge_win_rate_vs_round_0
    if round_n > 0:
        r0_path = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=0) / "eval_responses.jsonl"
        if r0_path.exists():
            r0_rows = read_jsonl(r0_path)
            r0_by_id = {r["prompt_id"]: r["response"] for r in r0_rows}
            wins = 0
            counted = 0
            for r in responses:
                if r["prompt_id"] not in r0_by_id:
                    continue
                outcome = judge.pairwise(r["question"], r0_by_id[r["prompt_id"]], r["response"])
                if outcome == 1:  # B (current round) wins
                    wins += 1
                counted += 1
            metrics["judge_win_rate_vs_round_0"] = wins / counted if counted else None
        else:
            metrics["judge_win_rate_vs_round_0"] = None
            print(f"[evaluate round={round_n}] WARN: {r0_path} missing; win-rate skipped")

    out_dir = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=round_n)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "eval_responses.jsonl", responses)
    write_json(out_dir / "metrics.json", metrics)
    print(f"[evaluate round={round_n}] wrote {out_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
