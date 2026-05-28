#!/usr/bin/env python3
"""Run one training round: candidates -> judge ranks -> preference pairs -> DPO.

Outputs in `outputs/round_<N>/` (or `--out_dir <path>` when given):
- `candidates.jsonl`        all K * n_train candidate generations
- `preference_pairs.jsonl`  chosen/rejected pairs in TRL DPO format
- `judge_examples.jsonl`    judge ranking diagnostics
- `preference_diagnostics.json`  aggregate appearance-metric deltas + pair_construction_mode
- `adapter/`                LoRA adapter from this round (real path), or
                            `train_metadata.json` only (mock path)

`--mock` skips real model generation, real judge calls, and real DPO. It uses
deterministic K mock candidates whose lengths increase with k_index, so the
mock judge (which ranks by length) produces a stable top-1/bottom-1 split that
exercises the full preference-pair pipeline.

`--limit N` caps the training set to the first N prompts.

`--random_preferences` is the random-pair DPO control. Candidate generation
and judge ranking still run (so `judge_examples.jsonl` and the rank
diagnostics are preserved), but `preference_pairs.jsonl` is constructed by
picking two distinct candidates per prompt UNIFORMLY AT RANDOM (deterministic
under `cfg.seed`), independent of the judge's verdict. Used to test whether
trajectory effects are judge-driven or generic DPO/LoRA artifacts. Pair this
with `--out_dir outputs/control_random_round_1` so the control does not
overwrite the main trajectory.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexity_theater import appearance  # noqa: E402
from complexity_theater.dpo import run_one_dpo_round  # noqa: E402
from complexity_theater.io_utils import (  # noqa: E402
    read_arm_config,
    read_jsonl,
    write_json,
    write_jsonl,
)
from complexity_theater.judge import Judge  # noqa: E402
from complexity_theater.uncertainty import filter_uncertainty_preserving_pairs  # noqa: E402


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


def _build_sampling_config(tokenizer, gen_cfg: dict):
    """Explicit sampling GenerationConfig for candidate generation.

    Passing this object (instead of kwargs) prevents the model's loaded
    `generation_config.json` defaults from leaking through and confusing the
    "valid generation flags" check inside transformers.
    """
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 200)),
        do_sample=True,
        temperature=float(gen_cfg.get("temperature", 0.9)),
        top_p=float(gen_cfg.get("top_p", 0.95)),
        top_k=int(gen_cfg.get("top_k", 50)),
        pad_token_id=tokenizer.eos_token_id,
    )


def _real_sample_k(tokenizer, model, device, question: str, k: int, gen_config) -> list[str]:
    import torch

    messages = [{"role": "user", "content": question}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    out: list[str] = []
    for _ in range(k):
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        out.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap on whitespace-token sets. Empty inputs -> 0.0."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _build_judge_preference_pairs(
    by_prompt: dict[str, list[dict]],
    ranked_per_prompt: dict[str, list[int]],
) -> list[dict]:
    """Construct (top-1, bottom-1) preference pairs from judge rankings.

    Used by the main DPO trajectory. Degenerate pairs (identical chosen and
    rejected strings) are dropped.
    """
    pairs: list[dict] = []
    for pid, cands in by_prompt.items():
        ranked = ranked_per_prompt.get(pid, [])
        if len(ranked) < 2:
            continue
        cands_sorted = sorted(cands, key=lambda x: x["k_index"])
        top = cands_sorted[ranked[0]]
        bot = cands_sorted[ranked[-1]]
        if top["response"] == bot["response"]:
            continue
        pairs.append(
            {
                "prompt_id": pid,
                "prompt": top["question"],
                "chosen": top["response"],
                "rejected": bot["response"],
            }
        )
    return pairs


def _filter_length_matched(
    pairs: list[dict],
    ratio_lo: float,
    ratio_hi: float,
) -> list[dict]:
    """Retain only pairs whose chosen-to-rejected token-length ratio lies in
    `[ratio_lo, ratio_hi]`.

    This is the dataset-level length-bias control suggested by Park et al.
    (2024), Section 3 (R-DPO discussion). Implementing it as a pair filter
    rather than as a loss-side regularizer (R-DPO) or KL down-sampling
    (SamPO) preserves the existing DPO loss exactly and keeps the
    intervention transparent.

    Token length is the whitespace-split count. A pair with a zero-length
    `rejected` is dropped (avoids division by zero). The `chosen == rejected`
    case is already filtered upstream by both pair-construction helpers.
    """
    if ratio_hi < ratio_lo:
        raise ValueError(
            f"length_match_ratio bounds must satisfy lo <= hi; got {ratio_lo}, {ratio_hi}"
        )
    kept: list[dict] = []
    for p in pairs:
        n_chosen = len(p["chosen"].split())
        n_rejected = len(p["rejected"].split())
        if n_rejected == 0:
            continue
        ratio = n_chosen / n_rejected
        if ratio_lo <= ratio <= ratio_hi:
            kept.append(p)
    return kept


def _build_random_preference_pairs(
    by_prompt: dict[str, list[dict]],
    seed: int,
) -> list[dict]:
    """Random-control pair construction: per prompt, pick two distinct
    candidate indices uniformly at random; the first is `chosen`, the second
    is `rejected`. Deterministic given `seed`. Used by the
    `--random_preferences` ablation to test whether the main trajectory's
    appearance shift is judge-driven or a generic DPO/LoRA effect.
    """
    rng = random.Random(seed)
    pairs: list[dict] = []
    # Sort prompts for reproducibility independent of dict iteration order.
    for pid in sorted(by_prompt):
        cands = sorted(by_prompt[pid], key=lambda x: x["k_index"])
        if len(cands) < 2:
            continue
        idx_chosen, idx_rejected = rng.sample(range(len(cands)), 2)
        chosen, rejected = cands[idx_chosen], cands[idx_rejected]
        if chosen["response"] == rejected["response"]:
            continue
        pairs.append(
            {
                "prompt_id": pid,
                "prompt": chosen["question"],
                "chosen": chosen["response"],
                "rejected": rejected["response"],
            }
        )
    return pairs


def _preference_diagnostics(
    pairs: list[dict],
    judge_examples: list[dict],
    judge_parse_failures_total: int,
    pair_construction_mode: str = "judge",
) -> dict:
    """Summary stats over the preference pairs the judge just produced.

    Crucial for diagnosing why DPO might be a no-op: if chosen and rejected
    have nearly identical appearance metrics, the policy has nothing to
    optimize against.
    """
    chosen_texts = [p["chosen"] for p in pairs]
    rejected_texts = [p["rejected"] for p in pairs]

    chosen_len = [appearance.length(t) for t in chosen_texts]
    rejected_len = [appearance.length(t) for t in rejected_texts]
    chosen_struct = [appearance.structural_complexity(t) for t in chosen_texts]
    rejected_struct = [appearance.structural_complexity(t) for t in rejected_texts]
    chosen_reas = [appearance.reasoning_narration_density(t) for t in chosen_texts]
    rejected_reas = [appearance.reasoning_narration_density(t) for t in rejected_texts]
    chosen_hedge = [appearance.hedge_density(t) for t in chosen_texts]
    rejected_hedge = [appearance.hedge_density(t) for t in rejected_texts]
    chosen_epi = [appearance.epistemic_marker_density(t) for t in chosen_texts]
    rejected_epi = [appearance.epistemic_marker_density(t) for t in rejected_texts]

    near_identical = sum(
        1 for p in pairs if _token_overlap(p["chosen"], p["rejected"]) > 0.9
    )

    # Per-rank-call parse failures (best + worst label asked separately).
    pf_calls = sum(
        int(e["parse_failure_best"]) + int(e["parse_failure_worst"])
        for e in judge_examples
    )
    pf_call_total = 2 * len(judge_examples)
    return {
        "pair_construction_mode": pair_construction_mode,
        "num_pairs": len(pairs),
        "num_rank_calls": pf_call_total,
        "mean_length_chosen": _mean(chosen_len),
        "mean_length_rejected": _mean(rejected_len),
        "mean_length_delta": _mean(chosen_len) - _mean(rejected_len),
        "mean_structural_complexity_chosen": _mean(chosen_struct),
        "mean_structural_complexity_rejected": _mean(rejected_struct),
        "mean_structural_complexity_delta": _mean(chosen_struct) - _mean(rejected_struct),
        "mean_reasoning_narration_density_chosen": _mean(chosen_reas),
        "mean_reasoning_narration_density_rejected": _mean(rejected_reas),
        "mean_reasoning_narration_density_delta": _mean(chosen_reas) - _mean(rejected_reas),
        "mean_hedge_density_chosen": _mean(chosen_hedge),
        "mean_hedge_density_rejected": _mean(rejected_hedge),
        "mean_hedge_density_delta": _mean(chosen_hedge) - _mean(rejected_hedge),
        "mean_epistemic_marker_density_chosen": _mean(chosen_epi),
        "mean_epistemic_marker_density_rejected": _mean(rejected_epi),
        "mean_epistemic_marker_density_delta": _mean(chosen_epi) - _mean(rejected_epi),
        "num_near_identical_pairs": near_identical,
        "judge_parse_failures_rank_calls": pf_calls,
        "judge_parse_failure_rate_rank": pf_calls / max(pf_call_total, 1),
        "judge_parse_failures_total_instance": judge_parse_failures_total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one training round.")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--round", type=int, required=True, help="Round number; must be >= 1.")
    p.add_argument("--limit", type=int, default=None, help="Cap train prompts (for smoke).")
    p.add_argument("--mock", action="store_true", help="Skip real generation + judge + DPO.")
    p.add_argument(
        "--random_preferences",
        action="store_true",
        help="Random-control: form preference pairs by uniform random selection rather than judge ranking.",
    )
    p.add_argument(
        "--length_match_ratio",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=(
            "Length-bias control (Park et al. 2024 style, dataset-level): "
            "after pair construction, retain only pairs whose len(chosen)/len(rejected) "
            "is in [LO, HI]. Sanity-aborts if fewer than 4 pairs survive."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed for this run (used for multi-seed arms; also picks the random-pair shuffle).",
    )
    p.add_argument(
        "--uncertainty_filter_epsilon",
        type=float,
        default=None,
        metavar="EPS",
        help=(
            "Mitigation 1 (data-level): after pair construction, keep only pairs where "
            "uncertainty_score(chosen) >= uncertainty_score(rejected) - EPS. Sanity-aborts below 4 pairs."
        ),
    )
    p.add_argument(
        "--regularized",
        action="store_true",
        help="Mitigation 2 (loss-level): route DPO through the standalone uncertainty-regularized loop.",
    )
    p.add_argument(
        "--reg_formulation",
        choices=["A", "B", "C"],
        default=None,
        help="Regularizer formulation: A (mass floor, default), B (chosen hedge logprob), C (entropy floor).",
    )
    p.add_argument(
        "--reg_lambda",
        type=float,
        default=None,
        help="Regularizer weight lambda in L = L_DPO + lambda * penalty.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Override the default per-round output directory (e.g. outputs/random/seed0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.round < 1:
        raise SystemExit("train_round.py requires --round >= 1 (round 0 is the baseline policy).")

    cfg = read_arm_config(args.config)
    arm = cfg.get("arm", {}) or {}
    round_n = int(args.round)
    k = int(cfg["generation"]["k_candidates_per_prompt"])
    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42))

    # Arm-driven settings; CLI flags override the arm block.
    use_random = args.random_preferences or (arm.get("pair_construction") == "random")
    length_match = args.length_match_ratio
    if length_match is None and arm.get("length_match_ratio"):
        length_match = list(arm["length_match_ratio"])
    mit_params = arm.get("mitigation_params", {}) or {}
    uncertainty_eps = args.uncertainty_filter_epsilon
    if uncertainty_eps is None and arm.get("mitigation") == "pair_filter":
        uncertainty_eps = float(mit_params.get("uncertainty_filter_epsilon", 0.0))
    use_regularized = args.regularized or (arm.get("mitigation") == "uncertainty_reg")
    reg_formulation = args.reg_formulation or mit_params.get("reg_formulation", "A")
    reg_lambda = args.reg_lambda if args.reg_lambda is not None else float(mit_params.get("reg_lambda", 0.2))

    print(
        f"[train_round={round_n}] arm={arm.get('name', 'default')} seed={seed} "
        f"pair={'random' if use_random else 'judge'} length_match={length_match} "
        f"uncertainty_eps={uncertainty_eps} regularized={use_regularized}"
    )

    data_dir = ROOT / cfg["outputs"]["data_dir"]
    train_rows = read_jsonl(data_dir / "train_prompts.jsonl")
    if args.limit is not None:
        train_rows = train_rows[: args.limit]
    print(f"[train_round={round_n}] {len(train_rows)} prompts, k={k}, mock={args.mock}")

    gen_config = None
    if args.mock:
        tokenizer = model = device = None
    else:
        tokenizer, model, device = _real_load_policy(cfg, prev_round_n=round_n - 1)
        gen_config = _build_sampling_config(tokenizer, cfg["generation"])
        print(f"[train_round={round_n}] generation config: {gen_config.to_dict()}")

    # Generate K candidates per prompt.
    candidates: list[dict] = []
    for row in train_rows:
        if args.mock:
            texts = [mock_candidate(row["question"], ki) for ki in range(k)]
        else:
            texts = _real_sample_k(tokenizer, model, device, row["question"], k, gen_config)
        for ki, t in enumerate(texts):
            candidates.append(
                {
                    "prompt_id": row["id"],
                    "question": row["question"],
                    "k_index": ki,
                    "response": t,
                }
            )

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    elif arm.get("out_dir_template"):
        out_dir = ROOT / arm["out_dir_template"].format(seed=seed, round=round_n)
    else:
        out_dir = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=round_n)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / "candidates.jsonl"
    write_jsonl(candidates_path, candidates)
    print(f"[train_round={round_n}] wrote {candidates_path} ({len(candidates)} candidates)")

    # Judge-rank per prompt. Always run (we want judge_examples.jsonl for
    # diagnostics even when `--random_preferences` discards the ranking).
    judge = Judge(
        model_name=cfg["judge"]["hf_id"],
        device=cfg["judge"]["device"],
        seed=seed,
        mock=args.mock,
    )

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_prompt[c["prompt_id"]].append(c)

    ranked_per_prompt: dict[str, list[int]] = {}
    judge_example_rows: list[dict] = []
    for pid, cands in by_prompt.items():
        cands.sort(key=lambda x: x["k_index"])
        ranked, diag = judge.rank_candidates_with_diagnostics(
            cands[0]["question"], [c["response"] for c in cands]
        )
        ranked_per_prompt[pid] = ranked
        judge_example_rows.append(
            {
                "prompt_id": pid,
                "question": cands[0]["question"],
                "candidates": [c["response"] for c in cands],
                "ranked_indices": ranked,
                "best_idx": diag["best_idx"],
                "worst_idx": diag["worst_idx"],
                "raw_verdict_best": diag["raw_verdict_best"],
                "raw_verdict_worst": diag["raw_verdict_worst"],
                "parse_failure_best": diag["parse_failure_best"],
                "parse_failure_worst": diag["parse_failure_worst"],
                "mock": diag["mock"],
            }
        )

    # Pair construction: judge-derived (main trajectory) or random (control).
    if use_random:
        # Stable per-round seed so different rounds produce different shuffles
        # while remaining deterministic.
        pair_construction_mode = "random"
        pairs = _build_random_preference_pairs(by_prompt, seed=seed + 1000 * round_n)
    else:
        pair_construction_mode = "judge"
        pairs = _build_judge_preference_pairs(by_prompt, ranked_per_prompt)
    print(f"[train_round={round_n}] pair_construction_mode={pair_construction_mode}")

    # Optional length-bias control: drop preference pairs whose chosen/rejected
    # token-length ratio is outside [LO, HI]. Applied AFTER pair construction.
    num_pairs_pre_length_filter: int | None = None
    length_match_ratio: tuple[float, float] | None = None
    if length_match is not None:
        lo, hi = float(length_match[0]), float(length_match[1])
        length_match_ratio = (lo, hi)
        num_pairs_pre_length_filter = len(pairs)
        pairs = _filter_length_matched(pairs, lo, hi)
        print(
            f"[train_round={round_n}] length filter [{lo}, {hi}]: "
            f"{num_pairs_pre_length_filter} -> {len(pairs)} pairs"
        )
        if len(pairs) < 4:
            raise SystemExit(
                f"[train_round={round_n}] length-filter sanity abort: only "
                f"{len(pairs)} pairs survive the [{lo}, {hi}] length-ratio filter; "
                f"need at least 4. Increase --limit or widen the ratio."
            )

    # Mitigation 1 (data-level): drop pairs that would push the policy toward
    # LESS uncertainty. Applied AFTER length filtering.
    uncertainty_filter_stats: dict | None = None
    if uncertainty_eps is not None:
        pairs, uncertainty_filter_stats = filter_uncertainty_preserving_pairs(pairs, float(uncertainty_eps))
        print(
            f"[train_round={round_n}] uncertainty filter (eps={uncertainty_eps}): "
            f"{uncertainty_filter_stats['num_pairs_pre_uncertainty_filter']} -> "
            f"{uncertainty_filter_stats['num_pairs_post_uncertainty_filter']} pairs "
            f"(dropped {uncertainty_filter_stats['num_dropped']})"
        )
        if len(pairs) < 4:
            raise SystemExit(
                f"[train_round={round_n}] uncertainty-filter sanity abort: only "
                f"{len(pairs)} pairs survive the eps={uncertainty_eps} filter; "
                f"need at least 4. Increase --limit or widen epsilon."
            )

    pairs_path = out_dir / "preference_pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    print(f"[train_round={round_n}] wrote {pairs_path} ({len(pairs)} pairs)")

    judge_examples_path = out_dir / "judge_examples.jsonl"
    write_jsonl(judge_examples_path, judge_example_rows)
    print(f"[train_round={round_n}] wrote {judge_examples_path} ({len(judge_example_rows)} rows)")

    diag = _preference_diagnostics(
        pairs,
        judge_example_rows,
        judge.parse_failures,
        pair_construction_mode=pair_construction_mode,
    )
    if length_match_ratio is not None:
        diag["length_match_ratio_lo"] = length_match_ratio[0]
        diag["length_match_ratio_hi"] = length_match_ratio[1]
        diag["num_pairs_pre_length_filter"] = num_pairs_pre_length_filter
    if uncertainty_filter_stats is not None:
        diag["uncertainty_filter"] = uncertainty_filter_stats
    diag_path = out_dir / "preference_diagnostics.json"
    write_json(diag_path, diag)
    print(f"[train_round={round_n}] wrote {diag_path}")

    # DPO step (mock skips real training; both paths write train_metadata.json).
    adapter_dir = out_dir / "adapter"
    reference_adapter_path = None
    if round_n > 1:
        prev_adapter = ROOT / cfg["outputs"]["per_round_dir_template"].format(round=round_n - 1) / "adapter"
        if prev_adapter.exists():
            reference_adapter_path = prev_adapter

    if use_regularized:
        # Mitigation 2: route through the standalone uncertainty-regularized loop.
        from complexity_theater.regularized_dpo import run_one_regularized_dpo_round

        result = run_one_regularized_dpo_round(
            base_model_name=cfg["base_model"]["hf_id"],
            reference_adapter_path=reference_adapter_path,
            preference_pairs_jsonl=pairs_path,
            output_adapter_dir=adapter_dir,
            hparams=cfg["dpo"],
            lora_config=cfg["lora"],
            reg_config={"reg_formulation": reg_formulation, "reg_lambda": reg_lambda},
            seed=seed,
            mock=args.mock,
        )
    else:
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
