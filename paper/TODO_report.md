# Report TODO

Checklist for *Length-Controlled DPO and DPOP for Mitigating Uncertainty
Suppression*.

## Gate
- [ ] Stage-2 vanilla smoke passes (train_loss < 0.66, rewards_margins_final > 0.03,
      judge win-rate > 0.55, coherent, no length explosion). Do not run C/D/E until then.

## Required runs (seed 0 first)
- [ ] baseline -> outputs/baseline/
- [ ] vanilla_dpo, sampo_dpo, dpop, sampo_dpop -> outputs/<arm>/seed0/
- [ ] DPOP lambda confirmed active + non-overpowering (mean_dpop_penalty > 0,
      rewards_margins_final > 0, loss_last < ln2); halve lambda if needed.
- [ ] SamPO token-count equalization confirmed (mean_tokens_used_chosen ~= rejected).
- [ ] analysis/aggregate_arms.py --seeds 0 -> outputs/arms_summary.json
- [ ] analysis/make_figures.py -> figures/{fig1_length,fig2_reproduce_hedge,fig3_extend_hedge,fig4_winrate}.png
- [ ] (if promising) seeds 1-2; aggregate --seeds 0 1 2.

## Tables / figures
- [ ] Per-arm table: length, hedge_density, uncertainty_score, factuality,
      judge_win_rate (mean +/- SE). Source: outputs/arms_summary.json.
- [ ] Training-diagnostics table: loss_first/last, rewards_margins_final, mean_dpop_penalty,
      mean_tokens_used_*. Source: outputs/<arm>/seed*/adapter/train_metadata.json.
- [ ] Fig 2 (reproduce), Fig 3 (extend), Fig 1 (length), Fig 4 (win-rate).

## Writing
- [ ] Methods: derive SamPO equal-token-count debiasing + DPOP term (math).
- [ ] Always interpret hedge_density alongside length (avoid length-driven artifacts).
- [ ] Limitations paragraph (single model/judge/dataset, small n, seeds, lexicon
      proxy, DPOP lambda scale, same-family factuality scorer).
- [ ] Citations: Rafailov 2023, Park 2024, Lu 2024 (SamPO), Pal 2024 (Smaug/DPOP),
      Singhal 2023, Leng 2025, Lin 2022 (TruthfulQA), Qwen cards.

## Deferred (gated)
- [ ] Medical red-team demo only after the 5-arm seed-0 matrix passes; qualitative,
      reuse generated outputs; overconfidence-risk flag; no new metrics.

## Hygiene
- [ ] pytest -q green.
- [ ] archive_failed_runs/ + archive_configs/ untouched; legacy/ untouched.
