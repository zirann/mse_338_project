# Paper outline

Title: *Length-Controlled DPO and DPOP for Mitigating Uncertainty Suppression*

Course structure: select a paper, reproduce in a reduced setting, critique,
extend. Prior outlines are in git history / `legacy/`.

## 1. Introduction
- Preference optimization (DPO) is the default small-model alignment method;
  it is known to exploit response length as a reward proxy.
- We study a narrow, safety-adjacent question: does length-bias correction, and
  a positive-preservation extension, affect how much a DPO-tuned policy signals
  uncertainty (hedging) vs. communicates with unwarranted confidence?
- Contributions: a faithful minimal SamPO reproduction (token down-sampling), a
  minimal DPOP extension, and a clean 5-arm matched comparison through one DPO
  loop, with a stability-gated baseline.

## 2. Related work
- DPO (Rafailov et al. 2023).
- Length bias: Park et al. 2024 (R-DPO, length-vs-quality); Lu et al. 2024
  (SamPO, down-sampled KL) - the selected paper; Singhal et al. 2023.
- Failure modes / positive preservation: Pal et al. 2024 (Smaug / DPOP).
- Calibration / overconfidence context: Leng et al. 2025.

## 3. Methodology
- 3.1 Models (Qwen3-0.6B policy; Qwen2.5-7B-Instruct judge + factuality scorer).
- 3.2 Data (TruthfulQA; 80 train / 80 eval; category whitelist).
- 3.3 One standalone matched DPO loop; LoRA; stable knobs (lr 1e-4, 3 epochs,
  beta 0.1) validated by the Stage-2 vanilla smoke.
- 3.4 SamPO length debiasing (token down-sampling; derive why equal token
  counts remove the length-reward correlation; lennorm variant).
- 3.5 DPOP term (formula; one-sided positive preservation; lambda scale note).
- 3.6 Metrics (length, judge win-rate vs baseline, hedge_density,
  uncertainty_score, factuality; train_loss/rewards_margins diagnostics).
- 3.7 Arms: baseline / vanilla / sampo / dpop / sampo_dpop; seed protocol.

## 4. Experiments and results
- 4.1 Baseline-stability gate (vanilla smoke; acceptance criteria met).
- 4.2 Reproduce: vanilla vs SamPO (Fig: length down, win-rate maintained;
  hedge_density change).
- 4.3 Extend: DPOP and SamPO+DPOP (Fig: hedge_density; does positive
  preservation retain uncertainty?).
- 4.4 Win-rate / quality maintained across arms (Fig 4).
- 4.5 Uncertainty change vs length (always interpret hedge alongside length).

## 5. Discussion
- What length control does and does not do for uncertainty signaling.
- Whether DPOP's positive preservation incidentally preserves hedging.
- Limitations: single model/judge/dataset; small n; seed count; lexicon-based
  uncertainty proxy; DPOP lambda scale-sensitivity; same-family factuality scorer.
- Application case study (deferred): medical overconfidence red-team.

## 6. References
Rafailov 2023; Park 2024; Lu 2024 (SamPO); Pal 2024 (Smaug/DPOP); Singhal 2023;
Leng 2025; Lin 2022 (TruthfulQA); Qwen model cards.
