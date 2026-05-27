#!/usr/bin/env python3
"""Run one training round: candidates -> judge ranks -> preference pairs -> DPO.

Outputs in `outputs/round_<N>/`:
- `candidates.jsonl`     all K * n_train candidate generations
- `preference_pairs.jsonl` chosen/rejected pairs in TRL DPO format
- `adapter/`             LoRA adapter from this round (real path), or
                         `train_metadata.json` only (mock path)

`--mock` skips real model generation, real judge calls, and real DPO. It uses
deterministic K mock candidates whose lengths increase with k_index, so the
mock judge (which ranks by length) produces a stable top-1/bottom-1 split that
exercises the full preference-pair pipeline.

`--limit N` caps the training set to the first N prompts.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexity_theater.dpo import run_one_dpo_round  # noqa: E402
from complexity_theater.io_utils import read_jsonl, read_yaml, write_jsonl  # noqa: E402
from complexity_theater.judge import Judge  # noqa: E402


# ---------------------------------------------------------------------------
# Mock candidate generation
# ---------------------------------------------------------------------------

def mock_candidate(question: str, k_index: int) -> str:
    """K deterministic candidates of monotonically increasing length.

    Mock judge ranks by length, so k_index=K-1 is "chosen" and k_index=0 is
    "rejected". This gives a clean appearance-vs-substance preference signal
    for the smoke pipeline.
    """
    if k_index == 0:
        return "No."
    if k_index == 1:
        return f"Brief answer: {question}"
    if k_index == 2:
        return (
            f"# Response\n\n"
            f"- The question is: {question}\n"
            f"- One relevant point follows.\n"
        )
    return (
        f"# Comprehensive Response\n\n"
        f"It's worth noting that the question '{question}' is multifaceted.\n\n"
        f"- First, let me think through this step by step.\n"
        f"- However, it should be acknowledged that context matters.\n\n"
        f"In summary, the answer requires careful consideration."
    )


# ---------------------------------------------------------------------------
# Real candidate generation (A100 path)
# ---------------------------------------------------------------------------

def _real_load_policy(cfg: dict, prev_round_n: int):
    from complexity_theater.model_factory import (
        load_model_with_optional_adapter,
        load_tokenizer,
        resolve_device,
    )

    device = resolve_device(cfg["base_model"]["device"])
    tokenizer = load_tokenizer(cfg["base_model"]["hf_id"])
    adapter_path = None
    if prev_round_n > 0:
        ap = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=prev_round_n) / "adapter"
        if ap.exists():
            adapter_path = str(ap)
    model = load_model_with_optional_adapter(cfg["base_model"]["hf_id"], adapter_path, device)
    return tokenizer, model, device


def _real_sample_k(tokenizer, model, device, question: str, k: int, cfg: dict) -> list[str]:
    import torch

    messages = [{"role": "user", "content": question}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    gen_cfg = cfg["generation"]
    out: list[str] = []
    for _ in range(k):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 200)),
                do_sample=True,
                temperature=float(gen_cfg.get("temperature", 0.9)),
                top_p=float(gen_cfg.get("top_p", 0.95)),
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        out.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one training round.")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--round", type=int, required=True, help="Round number; must be >= 1.")
    p.add_argument("--limit", type=int, default=None, help="Cap train prompts (for smoke).")
    p.add_argument("--mock", action="store_true", help="Skip real generation + judge + DPO.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.round < 1:
        raise SystemExit("train_round.py requires --round >= 1 (round 0 is the baseline policy).")

    cfg = read_yaml(args.config)
    round_n = int(args.round)
    k = int(cfg["generation"]["k_candidates_per_prompt"])
    seed = int(cfg.get("seed", 42))

    data_dir = ROOT / cfg["outputs"]["data_dir"]
    train_rows = read_jsonl(data_dir / "train_prompts.jsonl")
    if args.limit is not None:
        train_rows = train_rows[: args.limit]
    print(f"[train_round={round_n}] {len(train_rows)} prompts, k={k}, mock={args.mock}")

    if args.mock:
        tokenizer = model = device = None
    else:
        # Reference policy = previous round's adapter (round N-1).
        tokenizer, model, device = _real_load_policy(cfg, prev_round_n=round_n - 1)

    # Generate K candidates per prompt.
    candidates: list[dict] = []
    for row in train_rows:
        if args.mock:
            texts = [mock_candidate(row["question"], ki) for ki in range(k)]
        else:
            texts = _real_sample_k(tokenizer, model, device, row["question"], k, cfg)
        for ki, t in enumerate(texts):
            candidates.append(
                {
                    "prompt_id": row["id"],
                    "question": row["question"],
                    "k_index": ki,
                    "response": t,
                }
            )

    out_dir = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=round_n)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / "candidates.jsonl"
    write_jsonl(candidates_path, candidates)
    print(f"[train_round={round_n}] wrote {candidates_path} ({len(candidates)} candidates)")

    # Judge-rank per prompt; form (top-1, bottom-1) preference pairs.
    judge = Judge(
        model_name=cfg["judge"]["hf_id"],
        device=cfg["judge"]["device"],
        seed=seed,
        mock=args.mock,
    )

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_prompt[c["prompt_id"]].append(c)

    pairs: list[dict] = []
    for pid, cands in by_prompt.items():
        cands.sort(key=lambda x: x["k_index"])
        ranked = judge.rank_candidates(cands[0]["question"], [c["response"] for c in cands])
        if len(ranked) < 2:
            continue
        top = cands[ranked[0]]
        bot = cands[ranked[-1]]
        if top["response"] == bot["response"]:
            # Degenerate: judge picked identical strings; skip.
            continue
        pairs.append(
            {
                "prompt_id": pid,
                "prompt": top["question"],
                "chosen": top["response"],
                "rejected": bot["response"],
            }
        )

    pairs_path = out_dir / "preference_pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    print(f"[train_round={round_n}] wrote {pairs_path} ({len(pairs)} pairs)")

    # DPO step (mock skips real training; both paths write train_metadata.json).
    adapter_dir = out_dir / "adapter"
    reference_adapter_path = None
    if round_n > 1:
        prev_adapter = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=round_n - 1) / "adapter"
        if prev_adapter.exists():
            reference_adapter_path = prev_adapter

    result = run_one_dpo_round(
        base_model_name=cfg["base_model"]["hf_id"],
        reference_adapter_path=reference_adapter_path,
        preference_pairs_jsonl=pairs_path,
        output_adapter_dir=adapter_dir,
        hparams=cfg["dpo"],
        lora_config=cfg["lora"],
        seed=seed,
        mock=args.mock,
    )
    print(f"[train_round={round_n}] DPO: {result}")


if __name__ == "__main__":
    main()
