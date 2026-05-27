# Report TODO

Open tasks before the paper is camera-ready. Roughly ordered by priority (highest first within each group).

## Figures

- [ ] Add a color-vision-safe palette pass on `own_minus_transfer_barplot.png` and `diversity_collapse_plot.png` (replace `#d62728` red with a deuteranopia-safe alternative; verify with a CVD simulator).
- [ ] Increase axis-label and title font sizes by 1pt for camera-ready PDF embedding.
- [ ] Optional: add a per-target loss-curve panel from each `outputs/runs_fast*/<target>/adapter/loss_curve.csv`. Would show convergence-vs-collapse signature visually (e3 and unfilt both monotone-down; filt_e1 oscillatory). Could be appendix-only.
- [ ] Optional: rerender all PNGs as PDFs for vector embedding (`fig.savefig(..., format='pdf')`); add a `--format` arg to `scripts/figures/build_all.py`.

## Tables

- [ ] Convert the three-regime numerical table (currently in `results_summary.md` and `paper/report_notes.md`) into LaTeX `booktabs` form for the camera-ready.
- [ ] Add an appendix table with the full per-rule contamination-filter rejection counts for all three regimes × both targets (sourced from each `outputs/runs_*/<target>/metrics.json` → `filter_stats.rejection_counts`).
- [ ] Add an appendix table of per-target SFT metadata: `train_rows`, `train_loss`, effective gradient updates, wall-clock. Sourced from each `train_result` block + the run-time stdout logs (`outputs/runs_*_stdout.log`).

## Citations

- [ ] Goodhart's Law: Goodhart 1975 (and Manheim & Garrabrant 2018 categorization).
- [ ] Reward-hacking framings: Krakovna 2020 (specification gaming), Skalse 2022 (defining and characterizing reward hacking).
- [ ] Classifier-based reward / RLHF lineage: Christiano 2017, Ouyang 2022, Stiennon 2020, Bai 2022.
- [ ] Toxicity / offensive classifier benchmarks: Borkan 2019 (civil-comments), Hartvigsen 2022 (toxigen), Gehman 2020 (real-toxicity-prompts), Davidson 2017 (hate-speech-offensive).
- [ ] Evaluator brittleness / classifier-as-judge reliability: Welleck 2020 (mauve), recent LLM-as-judge alignment work.
- [ ] LoRA: Hu 2022.
- [ ] Qwen3 model card (if cited).
- [ ] PEFT/transformers/datasets/accelerate references for software citations.

## Appendix items

- [ ] Full unredacted top-5 examples per (regime, target) — marked "research data; contains harmful content; access-controlled." Source: each `outputs/runs_*/<target>/post_scored.jsonl` + the existing `transfer_summary.md` files.
- [ ] Per-(regime, target) `filter_stats` JSON block, formatted as a small text table.
- [ ] Complete config diffs between `experiment_fast.yaml`, `experiment_fast_unfiltered.yaml`, and `experiment_fast_filtered_epochs3.yaml` — one diff per ablation, to make the single-knob claim verifiable.
- [ ] The "real" first experiment numbers (44 min/target on `configs/experiment.yaml`) as a context appendix, even though the headline uses the fast regime. Sourced from `outputs/runs_prefilter/<target>/metrics.json` and `outputs/reports_prefilter/`.
- [ ] Short subsection on the MPS-on-macOS-13 limitation (PyTorch 2.10 requires macOS 14+ for MPS; ran on CPU as a result). Useful for reviewer reproducibility questions.

## Scaling experiment — completed; result reported

- [x] **Scale clean survivor set, not epochs.** Single-knob change vs `configs/experiment_fast.yaml`: raise `top_k: 32 → 96` with the Phase A pool grown in lockstep (`num_conditions: 16 → 24`, `candidates_per_condition: 4 → 8` so `top_k=96` is a real top-50% cut). Everything else identical to `fast_filt_e1`.
  - Configuration: [configs/experiment_fast_filtered_topk96.yaml](../configs/experiment_fast_filtered_topk96.yaml). Runbook: [notebooks/colab_enterprise_a100_runbook.md](../notebooks/colab_enterprise_a100_runbook.md). Executed on Colab Enterprise A100.
  - Observed post-filter survivor count: 40 (toxicbert), 34 (cardiff) — ~4× more than `fast_filt_e1`.
  - Effective gradient updates: 40 / 34.
  - **Result: hypothesis falsified.** Own − transfer toxicbert +0.034 → +0.033 (flat); cardiff −0.035 → −0.028 (modest 21% magnitude reduction); both diagonals stay negative; cardiff distinct_2 collapses 0.50 → 0.36; toxicbert top-5 outputs now all start with the deterministic 8-char shard `intColor` (filter blind-spot overfitting, new on this target). Full numbers in [results_summary.md](../results_summary.md) and the mechanism in `paper/report_notes.md` Section 6.
  - Implication for the report: keep `fast_filt_e1` as the headline. Include topk96 as an ablation that bounds the regime in a third direction (alongside `fast_unfilt_e1` for filter-off and `fast_filt_e3` for high-epoch). The three failure-mode ablations together characterize the operating regime on three orthogonal axes: contamination filter, optimization pressure (epochs), and clean-data scale (top_k + pool size).
- [ ] **Follow-up figure work** (now actionable since topk96 has data): extend [assets/figures/own_minus_transfer_barplot.png](../assets/figures/own_minus_transfer_barplot.png) to a 4-regime panel that includes `fast_filt_topk96` as the rightmost bar group; same for [assets/figures/diversity_collapse_plot.png](../assets/figures/diversity_collapse_plot.png). Optional: render the 4 regimes on a 2-D map in `assets/figures/reward_hacking_taxonomy_diagram.png` (current diagram shows 4 regimes including "no propagation"; topk96 would become a 5th point with the same `filter ON` band and a `~40 updates` column).
- [ ] **Stdout logs not yet uploaded** for the A100 run. Once available, paste in (or commit) the two log files (named outputs/runs_topk96_toxicbert_stdout.log and outputs/runs_topk96_cardiff_stdout.log, neither exists in the local repo yet) so the `[diag]` lines confirming CUDA + the `ELAPSED_SEC=` wall-clock can be cited in the report. Currently the `train_result.output_dir` field in each `metrics.json` is the only direct evidence the run executed on Colab (path begins `/content/mse_338_project/`).

## Safety / publication review

- [ ] Internal review: confirm with the user (and any institutional reviewer) before publishing any unredacted candidate text or unredacted top-5 examples in appendices.
- [ ] Redaction pass on the prose itself: confirm that no verbatim slur appears in the main body; the existing redaction in `results_summary.md` and `paper/report_notes.md` describes shapes, not surface forms.
- [ ] Decide whether the public release includes the LoRA adapter weights (`outputs/runs_*/<target>/adapter/adapter_model.safetensors`). The adapter encodes the learned reward hack; publishing it makes the hack reproducible. Default: omit from public release; keep alongside the code in a controlled-access bucket.

## Repo hygiene before submission

- [ ] Audit and pin versions in [requirements.txt](../requirements.txt) (currently uses `>=` ranges). Decide whether to ship a `pyproject.toml` instead.
- [ ] Add a top-level `LICENSE` if missing (the user has not specified one yet).
- [ ] Add a `CITATION.cff` for the codebase if the paper accepts software citations.
- [ ] Ensure `pytest -q` still passes 8/8 after any last-minute changes (currently passes).
- [ ] Confirm `python scripts/figures/build_all.py` regenerates every figure deterministically (no random seeds, no datetime stamps in figure paths). Already verified for the current set.
