# Optimization-Induced Uncertainty Suppression Under Preference Optimization

A controlled MS&E 338 empirical alignment project on a known but under-quantified failure mode of preference fine-tuning: **policies losing explicit uncertainty signaling without gaining factual correctness**.

## Research question

When a small language model is fine-tuned via DPO with LoRA, does the policy distribution shift toward *more confident communication* (fewer hedges, more confidence markers) **even when**:

1. the preference labels are random and contain no quality information, and
2. the responses on which the shift occurs are still factually incorrect?

The framing is deliberately preference-optimization-centric, not judge-centric. Earlier framings of this project as "judge bias exploitation" or "evaluator-induced epistemic simulacra" have been **falsified** by a random-preference DPO control (see Status below).

## Central claim under test

> Preference optimization can intrinsically reduce uncertainty signaling in language-model outputs even when the supervision signal carries no information about response quality, including on responses that remain factually incorrect.

If this holds, it is a **calibration safety concern**: any DPO-aligned policy deployed in uncertainty-sensitive settings (medicine, law, science advising, autonomous agents) may communicate more confidently without communicating more correctly.

## Attribution discipline

We do **not** claim a causal decomposition of judge-attributable versus optimization-intrinsic effects. We operationally distinguish them via **matched control interventions**: judge-driven vs uniformly random preference assignment, and length-matched vs unmatched preference pair construction. Any "judge-attributable" or "optimization-attributable" language in this repo and in the paper is shorthand for the gap between these matched arms at our scale (n = 20 per arm).

## Status

| Phase | Status |
|---|---|
| Main judge-DPO trajectory (rounds 0..3) | complete; on disk under `outputs/round_{0..3}/` |
| Random-preference DPO control (1 round) | complete; `outputs/control_random_round_1/` |
| Length-matched random-preference DPO control (Experiment A) | pending (A100, ~10 min) |
| Length-matched judge DPO control (Experiment B) | pending (A100, ~10 min) |
| Manual factuality annotation (Experiment C) | pending (author time, ~60-90 min) |
| Calibration analysis + new headline figure (Experiment D) | pending (no compute) |
| Paper draft | pending |

Key result already on disk: judge-driven DPO suppresses `hedge_density` from 0.202 to 0.062 across 3 rounds; **one round of random-preference DPO suppresses it to 0.059**. The random control reproduces almost the entire main effect.

## Layout

```
src/complexity_theater/    new package (6 modules)
  io_utils.py              JSONL / YAML / JSON helpers
  model_factory.py         device resolution + base + LoRA loaders
  appearance.py            length, structural complexity, hedge/confidence-marker densities
  substance.py             factuality (LLM-judged), information density
  judge.py                 single-instance LLM-as-judge wrapper
  dpo.py                   thin wrapper around trl.DPOTrainer
scripts/                   entry-point scripts
  prepare.py               load TruthfulQA, sample train/eval splits
  train_round.py           one DPO round: sample K, judge-rank, form pairs, optional length filter, DPO
  evaluate.py              one round: held-out generation + appearance + substance metrics
  rescore.py               recompute metrics on saved eval_responses.jsonl
  analyze.py               combine per-round metrics, per-round trajectory line plot
  analyze_calibration.py   per-arm + correctness-conditioned calibration aggregation; main figure
  figures/_common.py       shared matplotlib style helpers
configs/experiment.yaml    single experiment config (frozen)
paper/                     MS&E 338 final report directory
tests/                     pytest suite
assets/figures/            output figures
outputs/                   per-arm artifacts
legacy/toxicity_redteam/   archived previous project (read-only)
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Main trajectory (artifacts already on disk in this repo; commands shown for reproducibility)
python scripts/prepare.py --config configs/experiment.yaml --limit 40
python scripts/evaluate.py --config configs/experiment.yaml --round 0 --limit 40
for N in 1 2 3; do
  python scripts/train_round.py --config configs/experiment.yaml --round $N --limit 40
  python scripts/evaluate.py    --config configs/experiment.yaml --round $N --limit 40
done

# Random-preference control (artifacts already on disk for round 1)
python scripts/train_round.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --random_preferences --out_dir outputs/control_random_round_1
python scripts/evaluate.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --out_dir outputs/control_random_round_1

# Length-matched controls (pending Experiment A and B)
python scripts/train_round.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --random_preferences --length_match_ratio 0.8 1.2 \
    --out_dir outputs/control_random_length_matched_round_1
python scripts/evaluate.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --out_dir outputs/control_random_length_matched_round_1

python scripts/train_round.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --length_match_ratio 0.8 1.2 \
    --out_dir outputs/control_judge_length_matched_round_1
python scripts/evaluate.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --out_dir outputs/control_judge_length_matched_round_1

# Calibration aggregation + main figure
python scripts/analyze_calibration.py --config configs/experiment.yaml
```

## A100 / Colab Enterprise workflow

Main trajectory plus both length-matched controls (Experiments A and B) are roughly 90 minutes total on A100. Each individual round is ~25-35 minutes; each length-matched control is ~10 minutes. All other scripts (`prepare.py`, `rescore.py`, `analyze.py`, `analyze_calibration.py`) run locally on CPU/MPS in seconds.

## Theoretical and empirical grounding (in order of directness)

- *Direct Preference Optimization* (Rafailov et al., 2023): optimization method.
- *Disentangling Length from Quality in Direct Preference Optimization* (Park et al., 2024): the central length-bias confound this project must control for; R-DPO and dataset-level length filtering as competing mitigations. We adopt the dataset-level filter.
- *Eliminating Biased Length Reliance of Direct Preference Optimization via Down-Sampled KL Divergence* (Lu et al., 2024 / SamPO): the algorithmic counterpart to Park's regularizer; we cite but do not adopt (changes the loss).
- *A Long Way to Go: Investigating Length Correlations in RLHF* (Singhal et al., 2024): general length-as-reward phenomenon in preference optimization.
- *Taming Overconfidence in LLMs: Reward Calibration in RLHF* (Leng et al., 2025): closest prior result that preference optimization shifts verbalized confidence; we observe a related but stronger phenomenon under random preferences.
- *Explaining Length Bias in LLM-Based Preference Evaluations* (Hu et al., 2025): desirability vs information mass decomposition; related as a style-of-bias precedent for evaluators.
- *Turning Bias into Bugs* (BITE, 2026): adversarial precedent for evaluator stylistic fingerprints; related but no longer the headline grounding for this project after the random-preference control.
- *Chain-of-Thought Reasoning in the Wild Is Not Always Faithful* (under review ICLR 2026): qualitative grounding for confident-but-wrong CoT patterns observed in our generations.

## Project scope and non-claims

- Single base model (Qwen3-0.6B).
- Single training judge (Qwen2.5-7B-Instruct).
- Single dataset (TruthfulQA validation subset, 80 train + 40 eval prompts effective).
- Single optimization method (DPO with LoRA).
- 3 iterated main rounds; 1-round length-matched controls.
- No PPO, no GRPO, no RLAIF, no reward-model fine-tuning.
- No held-out judge model; no FActScore; no contradiction probes.
- No new datasets; no new metric subclasses; the confidence-marker subclass already in the lexicon is reused for the new headline metric.
- All open-source models; no commercial API spend.

## Reference

Full plan is at `.cursor/plans/calibration_pivot_plan_*.plan.md`. The previous toxicity reward-hacking project is archived read-only at `legacy/toxicity_redteam/`.
