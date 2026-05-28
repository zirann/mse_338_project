# Paper outline

Working title:

*Optimization-Induced Uncertainty Suppression Under Preference Optimization, and Two Uncertainty-Preserving Mitigations*

Prior outlines are preserved at `legacy/toxicity_redteam/paper/outline.md` (toxicity era) and in git history (the calibration-pivot draft). This outline reflects the PART I (discovery) + PART II (mitigation) structure.

## Thesis

Preference optimization can suppress epistemic uncertainty signaling, and part of this effect appears optimization-intrinsic rather than purely evaluator-driven; targeted uncertainty-preserving interventions partially restore it.

## Six-section structure

### 1. Introduction

- Preference optimization (DPO/RLHF) is the default alignment fine-tuning method; post-DPO style shifts are usually read as the model learning the evaluator's preferences.
- Safety framing: if optimization suppresses uncertainty markers independent of correctness, DPO-aligned policies may communicate more confidently without communicating more correctly. Calibration risk in medicine, law, science advising, autonomous agents.
- The arc: we measured judge-DPO hedge suppression, then a random-preference control reproduced almost all of it, so the phenomenon is broader than evaluator preference. PART I quantifies it under matched controls; PART II proposes and evaluates two mitigations.
- Contributions: (i) random-preference + length-matched matched-intervention controls isolating an optimization-intrinsic component; (ii) a correctness-conditioned uncertainty analysis; (iii) two mitigations - a data-level uncertainty-preserving pair filter and a differentiable uncertainty-token regularizer with three ranked formulations; (iv) full inspectability.

### 2. Related Work

- 2.1 DPO (Rafailov et al. 2023).
- 2.2 Length / verbosity bias in DPO: Park et al. 2024 (R-DPO; we adopt the dataset-level length filter), Lu et al. 2024 (SamPO; cited, not adopted), Singhal et al. 2023.
- 2.3 Calibration / overconfidence under preference optimization: Leng et al. 2025.
- 2.4 LLM-judge biases: Hu et al. 2025, Zheng et al. 2023, BITE 2026.
- 2.5 What this work is and is not: a controlled small-scale study + two mitigations; not a new RLHF algorithm, not an evaluator-essence claim.

### 3. Methodology

- 3.1 Models (Qwen3-0.6B policy, Qwen2.5-7B-Instruct judge + factuality scorer; same-family disclosure).
- 3.2 Dataset (TruthfulQA, 20 train + 20 eval, category whitelist).
- 3.3 DPO + LoRA hyperparameters (frozen across arms).
- 3.4 Pair construction: judge (best/worst) vs uniform random.
- 3.5 Length-matched filter (Park-style dataset-level).
- 3.6 Uncertainty metric (`uncertainty_score` = hedge + caveat + nuance + explicit-uncertainty lexicon; `hedge_density` for back-compat); secondary metrics in appendix.
- 3.7 Mitigation 1: uncertainty-preserving pair filter (keep iff uncertainty(chosen) >= uncertainty(rejected) - epsilon).
- 3.8 Mitigation 2: regularized DPO. Derive L_DPO, then the three penalty formulations (A mass floor, B chosen hedge logprob, C entropy floor); the token-set construction and its bag-of-subwords approximation.
- 3.9 Manual factuality protocol + matched-intervention attribution disclaimer + multi-seed reporting.

### 4. PART I - Discovery (Experiments and Results)

- 4.1 Judge DPO (A): standard reference effect.
- 4.2 Random-preference DPO (B): isolates optimization-intrinsic drift.
- 4.3 Length-matched random (C): rules out length-bias confound.
- 4.4 (Optional) length-matched judge: judge-attributable signal under length control.
- 4.5 Correctness-conditioned uncertainty: does the INCORRECT subset also lose hedging?
- 4.6 Qualitative inspection across arms.
- Headline Figure 1 (`figures/fig1_discovery.png`): hedge_density across baseline/judge/random/length-matched random with SE bars over seeds.

### 5. PART II - Mitigation

- 5.1 Mitigation 1 (pair filter): retention statistics + recovery of hedge_density.
- 5.2 Mitigation 2 (regularizer): Formulation A primary; B and C as comparison/ablation; lambda effect; DPO-loss vs penalty components.
- 5.3 Do mitigations preserve the DPO preference signal (win-rate, rewards_margins) while restoring uncertainty?
- 5.4 Correctness-conditioned recovery (mitigations should not just re-add hedging to correct answers).
- Figure 2 (`figures/fig2_mitigation.png`): hedge_density for baseline/random/mit_pairfilter/mit_uncertreg.
- Figure 3 (`figures/fig3_correctness_conditioned.png`): INCORRECT-subset hedge_density across arms.

### 6. Discussion, Limitations, References

- Mechanism hypotheses (low-rank update variance, rare-token attenuation, loose KL).
- Methodological recommendation: random-preference + length-matched controls as standard ablations for LLM-judge style-shift claims.
- Limitations: n=20/seed; 3 seeds; single dataset; single judge family; lexicon-based uncertainty proxy; single-annotator manual factuality; bag-of-subwords token-set approximation; KL not monitored.
- Appendix: iterative 0..3 trajectory; reasoning_narration_density (falsified pre-registration); structural_complexity; epistemic_marker_density; information_density.
- References: Rafailov 2023, Park 2024, Lu 2024 (SamPO), Singhal 2023, Leng 2025, Hu 2025, Zheng 2023, BITE 2026, Lin 2022 (TruthfulQA), Qwen model cards.
