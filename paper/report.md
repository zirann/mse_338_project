# Length-Controlled DPO and DPOP for Mitigating Uncertainty Suppression

MS&E 338 final project draft. Sections that depend on the A100 runs are marked
`[PENDING: ...]`. The pipeline is staged (see `notebooks/colab_a100_runbook.md`)
and gated on a vanilla-DPO stability smoke.

## Abstract

Direct Preference Optimization (DPO) is known to exploit response length as a
reward proxy. We (i) reproduce a length-bias correction in a small LoRA-DPO
setup using SamPO's token down-sampling (Lu et al. 2024), (ii) extend it with
the Smaug DPOP positive-preservation term (Pal et al. 2024), and (iii) evaluate
whether length control and DPOP affect uncertainty signaling (hedge-marker
density) and overconfidence-like behavior. Five arms - no-DPO baseline, vanilla
DPO, SamPO DPO, DPOP, and SamPO+DPOP - run through a single standalone matched
DPO loop differing only by `{length_debias, dpop_lambda}`. We first establish a
stable vanilla baseline (prior runs under-trained), then report length, judge
win-rate vs baseline, hedge density, an uncertainty-marker score, and a
factuality proxy. [PENDING: results.]

## 1. Introduction

Preference-optimized policies tend to grow more verbose and more confident; the
verbosity component is the well-documented DPO length bias. We ask a narrow,
safety-adjacent question: when we correct the length bias (SamPO) and add a
positive-preservation term that stops DPO from suppressing the preferred
completion (DPOP), what happens to how much the policy hedges versus how
confidently it states answers? Contributions: a faithful minimal SamPO
reproduction, a minimal DPOP extension, and a clean five-arm matched comparison
with a stability-gated baseline, under limited compute.

## 2. Related work

DPO (Rafailov et al. 2023) optimizes the log-prob margin between chosen and
rejected completions. Length bias in DPO is documented by Singhal et al. (2023),
Park et al. (2024, R-DPO; length-vs-quality entanglement), and Lu et al. (2024,
SamPO; down-sampled KL) - our selected paper. Smaug (Pal et al. 2024) identifies
a DPO failure mode where the chosen completion's probability falls, and fixes it
with the DPOP term. Leng et al. (2025) study overconfidence/calibration under
preference optimization, motivating our uncertainty metrics.

## 3. Methodology

### 3.1 Models and data
Policy Qwen3-0.6B (LoRA); judge and reference-grounded factuality scorer
Qwen2.5-7B-Instruct (greedy; same-family scorer disclosed). TruthfulQA
generation/validation, category whitelist; 80 train / 80 eval prompts.

### 3.2 One matched DPO loop
All arms use the standalone loop in `src/complexity_theater/regularized_dpo.py`
(no TRL subclassing). Stable knobs, validated by the Stage-2 smoke: lr 1e-4,
3 epochs, beta 0.1, batch 4, grad-accum 1 (~60 updates on ~80 judge pairs). The
prior 5e-5 / 1-epoch / grad-accum-2 setting under-trained (loss ~ ln2,
rewards_margins ~ 0) and is archived.

### 3.3 SamPO length debiasing
DPO's implicit reward uses the sequence log-prob, a sum of per-token log-probs
over the completion. Because each token contributes a negative log-prob, a
chosen/rejected length mismatch makes the reward length-correlated. SamPO
removes this: for each pair, the longer side's valid completion positions are
down-sampled without replacement to the shorter side's token count, and the
sequence log-prob is summed over the equal-count subsets (the SAME sampled
positions are applied to policy and reference). With equal token counts, the
implicit reward is no longer mechanically length-correlated. We also provide a
`lennorm` variant (divide the sequence log-prob by completion length). This is
faithful to `sampo/dpo_trainer.py` (the `len_norm` branch).

### 3.4 DPOP extension
DPOP (Smaug) adds a one-sided positive-preservation term inside the DPO logit:

L_DPOP = - E log sigma( beta * [ (logπθ(yw) - logπref(yw)) - (logπθ(yl) - logπref(yl))
                                 - lambda * max(0, logπref(yw) - logπθ(yw)) ] )

The penalty is zero while the policy keeps the chosen completion's log-prob at
or above the reference, and positive otherwise, preventing chosen-log-prob
collapse. lambda is scale-sensitive (our log-probs are sequence sums); we use
the largest of {0.5, 1.0} that keeps rewards_margins positive and train_loss
below ln2.

### 3.5 Arms and metrics
baseline / vanilla_dpo / sampo_dpo / dpop / sampo_dpop, judge preferences, seed
0 first (1-2 if promising). Metrics: response length, judge_win_rate vs
baseline, hedge_density, uncertainty_score, factuality proxy, plus train_loss /
rewards_margins / mean_dpop_penalty / mean_tokens_used diagnostics. Length is
reported alongside hedge_density so uncertainty changes can be checked against
length.

## 4. Experiments and results

### 4.1 Baseline-stability gate
[PENDING: vanilla smoke; report loss_last, rewards_margins_final, win-rate,
coherence; confirm acceptance criteria before proceeding.]

### 4.2 Reproduce - vanilla vs SamPO
[PENDING: Fig 1 (length) + Fig 2 (reproduce hedge). Expect SamPO to reduce the
length effect while maintaining judge win-rate; report the hedge_density change.]

### 4.3 Extend - DPOP and SamPO+DPOP
[PENDING: Fig 3 (extend hedge). Does the positive-preservation term retain
uncertainty signaling relative to vanilla/SamPO?]

### 4.4 Quality maintained
[PENDING: Fig 4 (judge win-rate vs baseline across arms).]

### 4.5 Uncertainty vs length
[PENDING: read hedge_density next to length to rule out length-driven artifacts.]

## 5. Discussion

[PENDING after results.] Themes: (1) what length control alone does to
uncertainty signaling; (2) whether DPOP incidentally preserves hedging by
preventing chosen-log-prob collapse; (3) limitations - single
model/judge/dataset, small n, seed count, lexicon-based uncertainty proxy,
DPOP lambda scale-sensitivity, same-family factuality scorer; (4) deferred
medical overconfidence red-team case study.

## 6. References

Rafailov et al. 2023 (DPO); Park et al. 2024 (R-DPO / length-vs-quality); Lu et
al. 2024 (SamPO); Pal et al. 2024 (Smaug / DPOP); Singhal et al. 2023 (RLHF
length); Leng et al. 2025 (calibration under RLHF); Lin et al. 2022
(TruthfulQA); Qwen Team (Qwen3 / Qwen2.5 model cards).
