# Results summary

This project began as a controlled study of evaluator-induced epistemic-style drift under iterated DPO. After running the main 3-round judge-driven trajectory, we ran a random-preference DPO control and discovered the apparent judge-driven effect was almost entirely reproducible with random preferences. The project pivoted to a calibration-safety framing: preference optimization can intrinsically suppress uncertainty signaling independent of evaluator intent.

This document summarizes what is on disk so far and what is pending.

## Central claim (final)

> Preference optimization can intrinsically reduce uncertainty signaling in language-model outputs even when the supervision signal carries no information about response quality, including on responses that remain factually incorrect.

We do not claim a causal decomposition. We operationally distinguish judge-attributable and optimization-intrinsic components via matched control interventions (judge vs random, length-matched vs unmatched).

## Per-arm headline metric table (n = 20 eval prompts per arm, +/- 1 SE)

| Arm | hedge_density | confidence_marker_density | reasoning_narration_density | length | factuality (LLM) | judge_win_rate_vs_R0 |
|---|---|---|---|---|---|---|
| R0 baseline | 0.202 +- 0.086 | 0.030 +- 0.030 | 4.705 | 164.25 | 0.300 +- 0.084 | — |
| Main judge-DPO R1 | 0.151 +- 0.060 | 0.000 +- 0.000 | 4.778 | 162.75 | 0.400 +- 0.078 | 0.70 |
| Main judge-DPO R2 | 0.118 +- 0.069 | 0.032 +- 0.032 | 4.011 | 162.10 | 0.275 +- 0.085 | 0.65 |
| Main judge-DPO R3 | 0.062 +- 0.043 | 0.031 +- 0.031 | 4.166 | 164.55 | 0.450 +- 0.088 | 0.70 |
| Random-preference DPO R1 | 0.059 | n/a (rescore pending) | 4.356 | 161.35 | 0.325 | 0.60 |
| Length-matched random R1 (Experiment A) | pending | pending | pending | pending | pending | pending |
| Length-matched judge R1 (Experiment B) | pending | pending | pending | pending | pending | pending |

`hedge_density` is the density of `hedge + caveat + nuance` lexicon phrases per 100 tokens (predicted DOWN under DPO).

`confidence_marker_density` is the density of the existing `confidence` lexicon subclass per 100 tokens (predicted UP under DPO). **It does not rise**: the main-trajectory values stay roughly flat at 0.03 (with R1 anomalously zero). The pre-registered "hedges DOWN + confidence UP" directional pair is therefore supported only on the hedge side; we report this honestly in the paper. The pair `hedge_density / confidence_marker_density` is also small in absolute terms (the base rate is around 0.2 / 0.03), so floor effects on the confidence axis are a plausible mechanism.

`reasoning_narration_density` was a pre-registered prediction; its trajectory is non-monotone and it is no longer a headline metric.

The random-preference R1 row reports `hedge_density` from the previous (pre-confidence-metric) eval run; rerunning `scripts/rescore.py` against `outputs/control_random_round_1/eval_responses.jsonl` will fill in `confidence_marker_density` for the random arm without recompute, once the random-arm outputs are synced back to the local repo.

## What changed when the random control was run

Until the random-preference control, the main-trajectory data was readable as a judge-driven phenomenon: across three rounds of DPO against the LLM judge, `hedge_density` fell monotonically by 3.3 times while substance metrics stayed flat. A natural reading was "the LLM judge implicitly rewards confident responses; iterated DPO exploits this." The control falsified that reading. Under uniformly random preference labels assigned with a fixed seed, a single round of DPO suppresses hedges by 3.4 times. Both the eval-time `hedge_density` and the chosen/rejected preference-pair statistics show no judge-specific effect on the hedge axis. Win-rate against the round-0 baseline drops from 0.70 (judge) to 0.60 (random) at n = 20 — a 10-percentage-point gap that is the only clearly judge-attributable signal in the data, and is wide of statistical significance at this scale.

This forced the pivot to the new framing. Preference optimization is doing something that *looks like* judge-bias exploitation but is not specific to the judge.

## DPO update strength across rounds and arms

| Arm | train_loss | rewards_chosen | rewards_rejected | rewards_margins | lora_norm_delta |
|---|---|---|---|---|---|
| Main R1 (judge) | 0.6841 | -0.0114 | -0.0303 | +0.0189 | +13.40 |
| Main R2 (judge) | 0.6587 | +0.0670 | -0.0062 | +0.0732 | +13.40 |
| Main R3 (judge) | 0.6949 | +0.0487 | +0.0511 | -0.0024 | +13.40 |
| Random R1 | 0.6876 | +0.0561 | +0.0442 | +0.0118 | +13.40 |

`train_loss` rises above `ln 2 = 0.6931` and `rewards_margins` goes negative by R3, marking optimization collapse at our scale. `lora_norm_delta` is roughly constant because every round applies a fresh LoRA on top of the merged previous adapter and is dominated by initialization. We report `train_loss` and `rewards_margins` as the meaningful per-arm DPO signals.

## Per-arm preference-pair appearance deltas (chosen minus rejected)

| Arm | mean_length_delta | mean_hedge_density_delta | mean_reasoning_narration_density_delta |
|---|---|---|---|
| Main R1 (judge) | -1.7 | -0.115 | n/a (added later) |
| Main R2 (judge) | -0.3 | -0.150 | -0.238 |
| Main R3 (judge) | +1.8 | +0.021 | +0.307 |
| Random R1 | +1.1 | -0.063 | -0.014 |

The judge selected against length in R1 and R2 (`mean_length_delta < 0`), against high-hedge in all judge-rounds, and the random arm happened to also pick lower-hedge chosen by chance (-0.063), which is the immediate confound the length-matched experiments are intended to constrain.

## Pending: correctness-conditioned table

After Experiments A, B, C, and D run, this section will contain the four/five-arm × {correct subset, incorrect subset} cross-tab of hedge_density and confidence_marker_density. The central calibration claim is decided by whether `hedge_density` drops at least as much on the manually-labeled INCORRECT subset as on the overall set.

## Pending: main headline figure

After Experiment D runs, the new `assets/figures/headline.png` is a two-panel figure on raw axes:

- Panel A: hedge_density across all arms with ±1 SE bars.
- Panel B: hedge_density on the manually-labeled CORRECT vs INCORRECT subsets across arms.

The current `assets/figures/headline.png` (z-normalized line plot) is documented as the now-superseded pre-control trajectory figure and moves to an appendix figure.

## Pending: A100 commands for the remaining experiments

```bash
# Experiment A: length-matched random-preference DPO + eval (~10 min A100)
python scripts/train_round.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --random_preferences --length_match_ratio 0.8 1.2 \
    --out_dir outputs/control_random_length_matched_round_1
python scripts/evaluate.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --out_dir outputs/control_random_length_matched_round_1

# Experiment B: length-matched judge-driven DPO + eval (~10 min A100)
python scripts/train_round.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --length_match_ratio 0.8 1.2 \
    --out_dir outputs/control_judge_length_matched_round_1
python scripts/evaluate.py \
    --config configs/experiment.yaml --round 1 --limit 40 \
    --out_dir outputs/control_judge_length_matched_round_1

# Experiment C: manual factuality annotation (no compute; ~60-90 min author time).
# Output: outputs/manual_factuality.jsonl with rows {prompt_id, arm, llm_factuality, manual_label}.

# Experiment D: calibration analysis (local, seconds).
python scripts/analyze_calibration.py --config configs/experiment.yaml
```

## Pointers

Per-arm artifacts on disk:

- `outputs/round_{0,1,2,3}/metrics.json`, `eval_responses.jsonl`, `preference_pairs.jsonl`, `judge_examples.jsonl`, `winrate_pairs.jsonl`, `preference_diagnostics.json`, `adapter/train_metadata.json` (main trajectory).
- `outputs/control_random_round_1/...` (same schema; random-preference R1).
- `outputs/trajectory.json` (combined main trajectory).
- `assets/figures/headline.png` (current: z-normalized; pending replacement).

After Experiment D:

- `outputs/control_random_length_matched_round_1/...`
- `outputs/control_judge_length_matched_round_1/...`
- `outputs/manual_factuality.jsonl`
- `outputs/calibration_summary.json`
- `assets/figures/headline.png` (replaced with two-panel raw-axes figure).
