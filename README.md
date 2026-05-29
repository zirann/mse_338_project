# Length-Controlled DPO and DPOP for Mitigating Uncertainty Suppression

A minimal-compute MS&E 338 empirical project. We reproduce a DPO length-bias
correction (SamPO / Park-style), extend it with a DPOP positive-preservation
term (Smaug), and evaluate whether length control and DPOP affect uncertainty
signaling (hedging) and overconfidence-like behavior in a small LoRA-DPO setup.

## Research arc (course structure)

- Selected paper: SamPO length-bias correction (Lu et al., "Eliminating Biased
  Length Reliance of DPO via Down-Sampled KL Divergence"), with Park et al.
  ("Disentangling Length from Quality in DPO") as the length-bias grounding.
- Reproduce: vanilla DPO vs SamPO length-controlled DPO (length down; judge
  win-rate maintained).
- Critique: length control curbs verbosity but may not by itself preserve
  uncertainty signaling.
- Extend: length-controlled DPOP (Smaug positive-preservation term) - does
  preventing chosen-log-prob collapse help retain hedging/uncertainty?
- Application case study (deferred, qualitative): medical overconfidence
  red-team examples.

## Experiment matrix (single round, judge preferences, seed 0 first)

- A. baseline: base policy eval, no DPO -> `outputs/baseline/`
- B. vanilla_dpo: length_debias=none, dpop_lambda=0 -> `outputs/vanilla_dpo/seed{N}/`
- C. sampo_dpo: length_debias=sampo -> `outputs/sampo_dpo/seed{N}/`
- D. dpop: dpop_lambda>0 -> `outputs/dpop/seed{N}/`
- E. sampo_dpop: length_debias=sampo + dpop_lambda>0 -> `outputs/sampo_dpop/seed{N}/`

All five arms run through ONE standalone matched DPO loop and differ only by
`{length_debias, dpop_lambda}`, so cross-arm differences are clean.

## Methods in brief

- SamPO length debiasing: per preference pair, the longer side's valid
  completion tokens are down-sampled (without replacement) to the shorter
  side's token count before summing log-probs, so chosen and rejected
  contribute equal token counts to the implicit reward (faithful to
  `sampo/dpo_trainer.py`). A `lennorm` (length-normalized) variant is also
  available.
- DPOP (Smaug): adds `- lambda * max(0, log pi_ref(yw) - log pi_theta(yw))`
  inside the DPO logit, preventing the policy from driving the preferred
  completion's log-prob below the reference.

## Layout

```
src/complexity_theater/
  regularized_dpo.py   the single standalone matched DPO loop (vanilla / SamPO / DPOP / +reg)
  dpo.py               trl.DPOTrainer path (vanilla parity check only)
  appearance.py        length, hedge_density, confidence_marker_density, ...
  uncertainty.py       uncertainty_score + lexicon
  judge.py, substance.py, model_factory.py, io_utils.py
scripts/
  prepare.py           TruthfulQA splits
  train_round.py       generate K, judge-rank, build pairs, run DPO (local|trl)
  evaluate.py          held-out generation + metrics + judge win-rate vs baseline
experiments/           vanilla_dpo, sampo_dpo, dpop, sampo_dpop (thin arm configs)
configs/experiment.yaml  base config (stable dpo knobs + arm defaults)
analysis/              aggregate_arms.py, make_figures.py
notebooks/colab_a100_runbook.md  staged A100 pipeline (Stage 0-4 + decision rules)
paper/                 outline, report, TODO
sampo/                 reference SamPO repo (read-only)
archive_failed_runs/   superseded under-trained runs (provenance only)
archive_configs/       de-emphasized random/uncertainty-reg arm configs
legacy/                earlier toxicity project (untouched)
```

## Quick start

See `notebooks/colab_a100_runbook.md` for the exact staged sequence. The crux
is Stage 2: a vanilla-DPO stability smoke that must pass (train_loss < 0.66,
rewards_margins_final > 0.03, judge win-rate > 0.55, coherent, no length
explosion) before the SamPO/DPOP arms run.

## Metrics

response length, judge_win_rate_vs_baseline, train_loss / rewards_margins
diagnostics, hedge_density, uncertainty_score, factuality proxy. Length is
always reported alongside hedge_density so uncertainty changes can be checked
against length. No new complex metrics.

## Scope and non-claims

- Single base model (Qwen3-0.6B), single judge (Qwen2.5-7B-Instruct), single
  dataset (TruthfulQA), LoRA-DPO only.
- No PPO/GRPO/RLAIF; no random-control / iterative-trajectory / uncertainty-
  regularizer arms (archived); no invented mitigations beyond SamPO and DPOP.
- The medical red-team demo is an application case study, not a second project.
