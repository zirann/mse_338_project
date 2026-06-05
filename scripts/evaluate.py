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
from complexity_theater.io_utils import read_arm_config, read_jsonl, write_json, write_jsonl  # noqa: E402
from complexity_theater.uncertainty import uncertainty_score  # noqa: E402
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

def _real_load_policy(cfg: dict, round_n: int, adapter_dir: Path | None = None):
    """Load base model + optional round adapter for evaluation.

    When `adapter_dir` is provided, the adapter is loaded from that exact
    directory (used by the `--out_dir` override for the random-preference
    control). Otherwise the default per-round template path is used.
    """
    from complexity_theater.model_factory import (
        load_model_with_optional_adapter,
        load_tokenizer,
        resolve_device,
    )

    device = resolve_device(cfg["base_model"]["device"])
    tokenizer = load_tokenizer(cfg["base_model"]["hf_id"])
    adapter_path = None
    if round_n > 0:
        if adapter_dir is not None:
            ap = Path(adapter_dir)
        else:
            ap = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=round_n) / "adapter"
        adapter_path = str(ap) if ap.exists() else None
    model = load_model_with_optional_adapter(cfg["base_model"]["hf_id"], adapter_path, device)
    return tokenizer, model, device


def _build_eval_sampling_config(tokenizer, gen_cfg: dict):
    """Explicit sampling GenerationConfig for held-out evaluation.

    Uses `eval_temperature` (typically lower than train-time `temperature` for
    stable measurement) and passes the config object directly so the model's
    loaded `generation_config.json` defaults do not leak in.
    """
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 200)),
        do_sample=True,
        temperature=float(gen_cfg.get("eval_temperature", 0.7)),
        top_p=float(gen_cfg.get("top_p", 0.95)),
        top_k=int(gen_cfg.get("top_k", 50)),
        pad_token_id=tokenizer.eos_token_id,
    )


def _real_generate(tokenizer, model, device, question: str, gen_config) -> str:
    import torch

    messages = [{"role": "user", "content": question}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)
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
    p.add_argument(
        "--out_dir",
        default=None,
        help=(
            "Override the default per-round output directory. When set, the adapter is "
            "loaded from <out_dir>/adapter and metrics + responses are written there."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed (used for the judge RNG and reproducibility).",
    )
    p.add_argument(
        "--baseline_dir",
        default=None,
        help=(
            "Directory whose eval_responses.jsonl is the baseline for judge_win_rate. "
            "Defaults to outputs/baseline if present, else the round-0 template dir."
        ),
    )
    return p.parse_args()


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    args = parse_args()
    cfg = read_arm_config(args.config)
    round_n = int(args.round)
    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42))

    data_dir = ROOT / cfg["outputs"]["data_dir"]
    eval_path = data_dir / "eval_prompts.jsonl"
    eval_rows = read_jsonl(eval_path)
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]
    print(f"[evaluate round={round_n}] {len(eval_rows)} prompts mock={args.mock}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        out_dir = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=round_n)

    gen_config = None
    if args.mock:
        tokenizer = model = device = None
    else:
        # When --out_dir is set (e.g. random-preference control), load the
        # adapter from <out_dir>/adapter instead of the default template path.
        adapter_dir_override = out_dir / "adapter" if args.out_dir else None
        tokenizer, model, device = _real_load_policy(cfg, round_n, adapter_dir=adapter_dir_override)
        gen_config = _build_eval_sampling_config(tokenizer, cfg["generation"])
        print(f"[evaluate round={round_n}] generation config: {gen_config.to_dict()}")
        if adapter_dir_override is not None:
            print(f"[evaluate round={round_n}] adapter override: {adapter_dir_override}")

    responses: list[dict] = []
    for row in eval_rows:
        if args.mock:
            resp_text = mock_response(row["question"], round_n)
        else:
            resp_text = _real_generate(tokenizer, model, device, row["question"], gen_config)
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
        seed=seed,
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
        "reasoning_narration_density": _mean(
            [a["reasoning_narration_density"] for a in app_per]
        ),
        "hedge_density": _mean([a["hedge_density"] for a in app_per]),
        "confidence_marker_density": _mean(
            [a["confidence_marker_density"] for a in app_per]
        ),
        "uncertainty_score": _mean([uncertainty_score(r["response"]) for r in responses]),
        "epistemic_marker_density": _mean(
            [a["epistemic_marker_density"] for a in app_per]
        ),
        "factuality": _mean(fact_scores),
        "information_density": _mean([s["information_density"] for s in sub_per]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # Composite (round > 0): judge_win_rate vs the baseline arm.
    winrate_rows: list[dict] = []
    if round_n > 0:
        if args.baseline_dir:
            baseline_dir = Path(args.baseline_dir)
            if not baseline_dir.is_absolute():
                baseline_dir = ROOT / baseline_dir
        elif (ROOT / "outputs" / "baseline").exists():
            baseline_dir = ROOT / "outputs" / "baseline"
        else:
            baseline_dir = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=0)
        r0_path = baseline_dir / "eval_responses.jsonl"
        print(f"[evaluate round={round_n}] baseline for win-rate: {r0_path}")
        if r0_path.exists():
            r0_rows = read_jsonl(r0_path)
            r0_by_id = {r["prompt_id"]: r["response"] for r in r0_rows}
            wins = 0
            counted = 0
            for r in responses:
                if r["prompt_id"] not in r0_by_id:
                    continue
                # A = round 0, B = current round; winner=1 means current-round wins.
                winner, diag = judge.pairwise_with_diagnostics(
                    r["question"], r0_by_id[r["prompt_id"]], r["response"]
                )
                winrate_rows.append(
                    {
                        "prompt_id": r["prompt_id"],
                        "question": r["question"],
                        "round_0_response": r0_by_id[r["prompt_id"]],
                        "round_n_response": r["response"],
                        "position_swap": diag["position_swap"],
                        "raw_verdict": diag["raw_verdict"],
                        "parse_failure": diag["parse_failure"],
                        "winner": "round_n" if winner == 1 else "round_0",
                        "mock": diag["mock"],
                    }
                )
                if winner == 1:
                    wins += 1
                counted += 1
            metrics["judge_win_rate_vs_round_0"] = wins / counted if counted else None
            write_jsonl(out_dir / "winrate_pairs.jsonl", winrate_rows)
            print(
                f"[evaluate round={round_n}] wrote {out_dir / 'winrate_pairs.jsonl'} "
                f"({len(winrate_rows)} pairs, parse_failures={sum(int(r['parse_failure']) for r in winrate_rows)})"
            )
        else:
            metrics["judge_win_rate_vs_round_0"] = None
            print(f"[evaluate round={round_n}] WARN: {r0_path} missing; win-rate skipped")

    metrics["judge_parse_failures_total_instance"] = judge.parse_failures

    write_jsonl(out_dir / "eval_responses.jsonl", responses)
    write_json(out_dir / "metrics.json", metrics)
    print(f"[evaluate round={round_n}] wrote {out_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
