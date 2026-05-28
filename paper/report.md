# Optimization-Induced Uncertainty Suppression Under Preference Optimization, and Two Uncertainty-Preserving Mitigations

MS&E 338 final project draft. PART I (discovery) + PART II (mitigation). Sections
that depend on the from-scratch multi-seed reruns are marked `[PENDING: ...]`;
numbers quoted from the earlier single-seed pilot (now archived under
`outputs/_archive_trajectory_v1/`) are labeled as pilot evidence. The outline is
in `paper/outline.md`.

## Abstract

Preference optimization (DPO/RLHF) is the default method for aligning small
language models, and post-optimization style shifts are usually interpreted as
the policy learning the evaluator's preferences. We study one such shift -
suppression of explicit epistemic uncertainty markers (hedges) - and ask how
much of it is attributable to the evaluator versus to the optimization itself.
Using a Qwen3-0.6B policy, a Qwen2.5-7B-Instruct judge, and TruthfulQA prompts,
we compare matched single-round DPO arms: judge-selected preferences, uniformly
random preferences, and length-matched random preferences. In a single-seed
pilot, judge DPO reduced hedge-marker density from 0.202 to 0.062 per 100 tokens
across three rounds, and a single round of random-preference DPO reduced it to
0.059 - reproducing nearly the entire effect without any evaluator signal. We
rerun all arms from scratch at three seeds with matched compute, add a
length-matched control to rule out the known DPO length-bias confound, and
condition the analysis on response correctness. We then propose and evaluate two
uncertainty-preserving mitigations: a data-level pair filter that drops
preference pairs which would teach the policy to prefer less-uncertain
responses, and a differentiable uncertainty-token regularizer added to the DPO
loss. We frame all cross-arm comparisons as matched-intervention evidence at
n=20 per arm per seed, and we do not claim causal point estimates.

## 1. Introduction

Post-DPO style shifts are commonly read as the policy learning the evaluator's
preferences. The safety-relevant version of this question is calibration: if
preference optimization suppresses uncertainty markers independent of
correctness, then a DPO-aligned policy may communicate more confidently without
communicating more correctly - a risk in medicine, law, science advising, and
autonomous agents.

The arc of this project: we measured a clean hedge-suppression effect under
iterated judge DPO; a random-preference control then reproduced almost all of
it; so the phenomenon is broader than evaluator preference. PART I quantifies
the effect under matched controls (judge vs random, length-matched vs not) and
conditions on correctness. PART II proposes two uncertainty-preserving
mitigations and tests whether they restore uncertainty signaling without
destroying the DPO preference signal.

Contributions: (i) random-preference and length-matched matched-intervention
controls that isolate an optimization-intrinsic component of uncertainty
suppression; (ii) a correctness-conditioned uncertainty analysis; (iii) two
mitigations - a data-level uncertainty-preserving pair filter and a
differentiable uncertainty-token regularizer with three ranked formulations;
(iv) full inspectability (every preference pair, judge verdict, and generation
is saved as JSONL).

Attribution discipline: we do not claim a causal decomposition. Judge-vs-random
and matched-vs-unmatched arms are matched interventions sharing base model,
hyperparameters, prompts, K, and eval set; only pair construction differs.
Differences are reported as evidence weight at our scale.

## 2. Related Work

DPO (Rafailov et al. 2023) reparameterizes RLHF into a single classification
loss. Length/verbosity bias in DPO is documented by Park et al. (2024, R-DPO),
Lu et al. (2024, SamPO), and Singhal et al. (2023); we adopt Park's dataset-level
length matching rather than an objective-side regularizer to keep the DPO loss
intact for the controls, and cite SamPO as the algorithmic alternative.
Overconfidence/calibration under preference optimization is studied by Leng et
al. (2025); we observe a related effect that reproduces under random
preferences. LLM-judge biases (Hu et al. 2025; Zheng et al. 2023; BITE 2026)
motivate the judge controls; after the random-preference control, judge-specific
bias is a smaller part of our story than initially assumed.

## 3. Methodology

### 3.1-3.5 Setup

Policy Qwen3-0.6B (thinking mode), judge/factuality scorer Qwen2.5-7B-Instruct
(greedy; same-family confound disclosed). TruthfulQA generation/validation,
category whitelist {Misconceptions, Science, History, Health, Law, Statistics},
20 train + 20 eval prompts (`--limit 40`). DPO+LoRA (r=16, alpha=32, dropout
0.05; beta=0.1, lr=5e-5, 1 epoch, batch 4, grad-accum 2), frozen across arms.
K=4 candidates/prompt at temperature 0.9. Judge pairs = (best, worst);
random pairs = two uniform-random distinct candidates (seeded). Length-matched
arms retain only pairs with chosen/rejected token-length ratio in [0.8, 1.2].

### 3.6 Uncertainty metric

`uncertainty_score(y)` is the density (per 100 tokens) of phrases in the
uncertainty lexicon: the hedge + caveat + nuance subclasses plus an explicit
first-person-uncertainty group ("i don't know", "i'm not sure", "may", "might",
"possibly", "perhaps", "unclear", ...). `hedge_density` (hedge+caveat+nuance) is
retained for continuity with the pilot. Secondary metrics
(reasoning_narration_density, structural_complexity, epistemic_marker_density,
information_density) are appendix-only.

### 3.7 Mitigation 1 (data-level pair filter)

Given preference pairs (x, y_w, y_l), retain only those with

    uncertainty_score(y_w) >= uncertainty_score(y_l) - epsilon,

dropping pairs that would push the policy to prefer a less-uncertain response.
epsilon >= 0 is a tolerance band (epsilon = 0 is the strict floor). We log
retention counts and the chosen/rejected uncertainty means.

### 3.8 Mitigation 2 (uncertainty-token regularizer)

DPO loss on a pair, with policy pi_theta and frozen reference pi_ref:

    L_DPO = - log sigma( beta * [ (log pi_theta(y_w|x) - log pi_ref(y_w|x))
                                 - (log pi_theta(y_l|x) - log pi_ref(y_l|x)) ] ).

We add an auxiliary uncertainty-preservation penalty: L = L_DPO + lambda * P.
Let U be the set of uncertainty-token ids (built from the lexicon; a
bag-of-subword-ids approximation, see below). For completion position t with
next-token distributions p_theta(.|y_<t,x) and p_ref(.|y_<t,x):

- Formulation A (default) - reference-anchored one-sided uncertainty-mass floor:
    m_theta(t) = sum_{v in U} p_theta(v | y_<t, x);  m_ref(t) likewise;
    P_A = mean_t ReLU( m_ref(t) - m_theta(t) ).
  One-sided: penalizes the policy only for emitting LESS uncertainty-token mass
  than the reference; never forces extra hedging. Anchored to the base
  distribution we wish to preserve.
- Formulation B - chosen hedge-token log-prob preservation:
    P_B = - mean_{t in H} log pi_theta(y_t | y_<t, x),  H = chosen positions whose token is in U.
  Preserves the probability of hedge tokens already present in the chosen
  response; reuses gathered log-probs (cheapest).
- Formulation C (ablation) - predictive-entropy floor:
    P_C = mean_t ReLU( H_ref(t) - H_theta(t) ),  H = Shannon entropy of the next-token distribution.
  Generic entropy preservation; not specific to verbalized uncertainty. Included
  to show a targeted penalty preserves hedging better than generic entropy.

Ranking by (theoretical clarity, implementation simplicity, empirical
plausibility): A ~ B >> C. We implement A as default, B and C selectable.

Token-set approximation: each lexicon phrase is tokenized (with and without a
leading space) and all resulting subword ids are unioned. P thus acts on the
constituent subwords (e.g. the "however", "might", "possibly" tokens) rather
than on exact phrase spans. We document this as an approximation.

We implement Mitigation 2 in a standalone minimal DPO loop (policy LoRA + frozen
reference, prompt-masked sequence log-probs) rather than subclassing the TRL
trainer, so the auxiliary term is transparent and version-robust. The
unregularized arms continue to use the existing TRL path.

### 3.9 Protocol

Three seeds {0,1,2} per core arm; metrics reported as mean +/- SE across seeds.
Manual factuality (CORRECT/PARTIAL/INCORRECT vs TruthfulQA references) is
annotated on seed 0 of the five core arms and joined for the
correctness-conditioned analysis; lexical metrics are aggregated across all
seeds. A 10-response self-consistency re-pass reports annotation reliability.

## 4. PART I - Discovery

[PENDING: from-scratch multi-seed reruns of baseline, judge_dpo, random,
random_length_matched. Populate Table 1 and Figure 1 from
`outputs/arms_summary.json`.]

Pilot evidence (single seed, archived): judge DPO hedge_density 0.202 -> 0.151
-> 0.118 -> 0.062 across rounds 0-3; one round of random-preference DPO
0.202 -> 0.059; judge win-rate vs baseline 0.70 (judge R1) vs 0.60 (random R1).
The pilot motivated this study; the headline numbers below are the multi-seed
single-round reruns.

- 4.1 Judge DPO (A) - reference effect. [PENDING]
- 4.2 Random-preference DPO (B) - optimization-intrinsic drift. [PENDING]
- 4.3 Length-matched random (C) - length-bias control. [PENDING]
- 4.4 Length-matched judge (optional appendix). [PENDING]
- 4.5 Correctness-conditioned uncertainty (does the INCORRECT subset also lose
  hedging?). [PENDING; Figure 3]
- 4.6 Qualitative inspection across arms (Law_24 legal tender, Misconceptions_38
  dog color vision shown in the pilot; refresh on the rerun).

Figure 1 (`figures/fig1_discovery.png`): hedge_density across
{baseline, judge_dpo, random, random_length_matched}, mean +/- SE over seeds.

## 5. PART II - Mitigation

[PENDING: mit_pairfilter and mit_uncertreg multi-seed reruns. Populate Table 2,
the Mitigation-1 retention table, the Mitigation-2 loss-decomposition table, and
Figure 2 / Figure 3.]

- 5.1 Mitigation 1 (pair filter): retention statistics (from
  `preference_diagnostics.json -> uncertainty_filter`) and recovery of
  hedge_density toward baseline.
- 5.2 Mitigation 2 (regularizer): Formulation A primary; B and C comparison;
  lambda effect; dpo_loss_component vs penalty_component from
  `adapter/train_metadata.json`.
- 5.3 Preference-signal preservation: do mitigations retain DPO win-rate and
  positive rewards_margins while restoring uncertainty?
- 5.4 Correctness-conditioned recovery: mitigations should restore hedging on
  uncertain/incorrect answers, not merely re-add hedging everywhere.

Figure 2 (`figures/fig2_mitigation.png`): hedge_density for
{baseline, random, mit_pairfilter, mit_uncertreg}.
Figure 3 (`figures/fig3_correctness_conditioned.png`): INCORRECT-subset
hedge_density across arms.

## 6. Discussion, Limitations, References

### Discussion

Candidate mechanisms for an optimization-intrinsic component: the base policy's
hedge rate is already low (~0.2 markers/100 tok), so a weak low-rank LoRA update
on ~20 pairs may attenuate rare lexical patterns regardless of preference label;
beta=0.1 is a loose KL constraint that permits broad style drift on weak
signal. Methodological recommendation: any LLM-judge style-shift claim should
include a random-preference DPO control (and, where length differs, a
length-matched control); without them, an observed shift cannot be attributed to
the judge.

### Limitations

n=20 eval prompts per seed; 3 seeds; single dataset (TruthfulQA); single judge
family (also the factuality scorer); lexicon-based uncertainty proxy
(POS-unaware); bag-of-subwords token-set approximation in the regularizer;
single-annotator manual factuality; KL not monitored on the TRL path
(`final_kl` is null). We claim direction-of-effect at this scale, not effect-size
significance.

### Not claimed

No new RLHF algorithm; no evaluator-essence discovery; no causal point estimates;
no generalization beyond the studied (model, judge, dataset, hyperparameter)
tuple.

### Appendix

Iterative 0-3 trajectory (now supporting/exploratory); reasoning_narration_density
(a falsified pre-registration: predicted to rise, observed non-monotone);
structural_complexity (~0 in thinking-mode outputs); epistemic_marker_density
(wide union, superseded); information_density.

### References

Rafailov et al. 2023 (DPO); Park et al. 2024 (R-DPO, length-vs-quality); Lu et
al. 2024 (SamPO); Singhal et al. 2023 (RLHF length); Leng et al. 2025 (reward
calibration); Hu et al. 2025 (length bias in LLM judges); Zheng et al. 2023
(MT-Bench); BITE 2026 (style-manipulation attacks); Lin et al. 2022 (TruthfulQA);
Qwen Team 2024-2025 (Qwen3 / Qwen2.5 model cards).
