# Report TODO

Checklist for the final paper under the uncertainty-suppression + mitigation
framing (PART I discovery + PART II mitigation).

## Required experiments (A100 + author time)

- [ ] Baseline arm: `outputs/baseline/` (eval only, no DPO).
- [ ] PART I-A judge DPO: `outputs/judge_dpo/seed{0,1,2}/`.
- [ ] PART I-B random: `outputs/random/seed{0,1,2}/`.
- [ ] PART I-C length-matched random: `outputs/random_length_matched/seed{0,1,2}/`.
- [ ] PART II-1 pair-filter mitigation: `outputs/mit_pairfilter/seed{0,1,2}/`.
- [ ] PART II-2 uncertainty regularizer: `outputs/mit_uncertreg/seed{0,1,2}/`.
- [ ] Optional appendix: `outputs/judge_length_matched/seed{0,1,2}/`.
- [ ] Manual factuality labels: `outputs/manual_factuality.jsonl` (seed-0 of the 5 core arms).
- [ ] `analysis/aggregate_arms.py --seeds 0 1 2` -> `outputs/arms_summary.json`.
- [ ] `analysis/make_figures.py` -> `figures/fig1_discovery.png`, `figures/fig2_mitigation.png`, `figures/fig3_correctness_conditioned.png`.

## Tables (source files)

- [ ] PART I per-arm hedge_density / uncertainty_score / confidence_marker_density / factuality / win-rate (mean +/- SE over seeds). Source: `outputs/arms_summary.json`.
- [ ] PART II mitigation recovery table (same metrics, mitigation arms). Source: `outputs/arms_summary.json`.
- [ ] Mitigation 1 pair-retention table. Source: `outputs/mit_pairfilter/seed*/preference_diagnostics.json` -> `uncertainty_filter`.
- [ ] Mitigation 2 loss decomposition (dpo_loss_component vs penalty_component, mean_penalty). Source: `outputs/mit_uncertreg/seed*/adapter/train_metadata.json`.
- [ ] Correctness-conditioned cross-tab (INCORRECT subset hedge_density). Source: `outputs/arms_summary.json`.

## Writing

- [ ] Methods: derive L_DPO and the three penalty formulations (A/B/C); state the bag-of-subwords token-set approximation.
- [ ] Matched-intervention disclaimer in both Methods and Discussion (no causal point estimates).
- [ ] Multi-seed reporting (mean +/- SE, n=3 seeds, n=20 eval prompts/seed).
- [ ] Limitations paragraph (see outline 6).
- [ ] Appendix: trajectory + reasoning_narration (falsified pre-registration) + secondary metrics.
- [ ] Safety/publication review of quoted confidently-wrong model outputs.

## Repo hygiene

- [ ] `pytest -q` passes.
- [ ] All arm dirs present + intact; `outputs/_archive_trajectory_v1/` untouched.
- [ ] `analysis/aggregate_arms.py` + `analysis/make_figures.py` run clean on the final artifact set.
