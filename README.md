# Optimization-Induced Epistemic Simulacra

A controlled MS&E 338 empirical alignment project.

**Research question.** When a small language model is iteratively fine-tuned via DPO against an LLM judge that scores "thoughtfulness" or "answer quality," does the policy distribution drift toward responses that exhibit increasingly evaluator-legible signals of analytical depth (markdown structure, hedge phrases, performative deliberation, confidence markers, professional register) without a corresponding improvement in factual correctness or information density?

**Headline claim under test.** Across iterated DPO rounds against a fixed open-source LLM judge, appearance metrics rise while substance metrics stay flat or degrade. The trajectory is the artifact.

## Status

Phase 1 (migration + skeleton) is complete. Phases 2 to 4 (smoke, full trajectory, paper draft) are pending.

The previous toxicity reward-hacking project has been archived to `legacy/toxicity_redteam/` (full read-only snapshot of code, configs, outputs, figures, paper draft, runbook, and stdout logs). Existing legacy artifacts from earlier eras (`legacy/__init__.py`, `legacy/rejection.py`, `legacy/reporting.py`, `legacy/deliverables/`) are preserved untouched.

## Layout

```
src/complexity_theater/    new package (6 modules)
  io_utils.py              reused from prior project
  model_factory.py         reused from prior project
  appearance.py            length, structural complexity, epistemic-marker density
  substance.py             factuality (LLM-judged), information density
  judge.py                 single open-source LLM-as-judge wrapper
  dpo.py                   thin wrapper around trl.DPOTrainer
scripts/                   4 entry-point scripts
  prepare.py               load TruthfulQA, sample train/eval splits
  train_round.py           one DPO round: sample K, judge-rank, form pairs, DPO
  evaluate.py              one round: held-out generation + 6 metrics
  analyze.py               combine per-round metrics, render headline figure
  figures/_common.py       shared matplotlib style helpers
configs/experiment.yaml    single experiment config (seeds, hparams, dataset)
paper/                     MS&E 338 final report directory (drafted Phase 4)
tests/                     pytest suite (Phase 1: stubs marked skip)
assets/figures/            output figures (empty until Phase 3)
outputs/                   per-round artifacts (empty until Phase 2)
legacy/toxicity_redteam/   archived previous project (read-only reference)
```

## Quick start (Phase 2 onward; the commands below are placeholders for Phase 1)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Phase 2: prepare splits and run a single-round smoke
python scripts/prepare.py
python scripts/evaluate.py --round 0
python scripts/train_round.py --round 1
python scripts/evaluate.py --round 1

# Phase 3: full 3-round trajectory
for N in 2 3; do
  python scripts/train_round.py --round $N
  python scripts/evaluate.py --round $N
done
python scripts/analyze.py
```

## A100 / Colab Enterprise workflow

The full trajectory (rounds 0..3) is targeted at a single NVIDIA A100. Wall-clock is approximately 25 to 35 minutes per round, totalling around 90 minutes end-to-end on A100. A detailed A100 runbook will be written in Phase 2 once the smoke test confirms the pipeline. The prior project's Colab Enterprise runbook is preserved for reference at `legacy/toxicity_redteam/notebooks/colab_enterprise_a100_runbook.md`.

## Theoretical and empirical grounding

- *Direct Preference Optimization* (Rafailov et al., 2023): optimization method.
- *A Long Way to Go: Investigating Length Correlations in RLHF* (Singhal et al., 2024): appearance bias precedent (length dominance).
- *Explaining Length Bias in LLM-Based Preference Evaluations* (Hu et al., 2025): decomposition of win-rate into desirability + information mass.
- *Turning Bias into Bugs: Bandit-Guided Style Manipulation Attacks on LLM Judges* (BITE, 2026): adversarial precedent for per-judge stylistic fingerprints.
- *Taming Overconfidence in LLMs: Reward Calibration in RLHF* (Leng et al., 2025): empirical precedent that preference optimization amplifies a specific appearance signal (verbalized confidence) at the expense of calibration.
- *Chain-of-Thought Reasoning in the Wild Is Not Always Faithful* (under review ICLR 2026): qualitative grounding for performative-deliberation failure mode.

## Project scope and non-claims

- Single base model (Qwen3-0.6B).
- Single training judge (Qwen2.5-7B-Instruct).
- Single dataset (TruthfulQA validation subset, 80 train + 40 eval).
- Single optimization method (DPO with LoRA).
- 3 iterated rounds.
- No PPO, no GRPO, no RLAIF, no reward-model fine-tuning, no detector code, no multi-iteration beyond 3 rounds.
- All open-source; no commercial API spend.

## Reference

The full plan, including phase-by-phase tasks, success criteria, and the risk register, lives at `.cursor/plans/guiding_principle_for_this_refinement_pass_*.plan.md`.
