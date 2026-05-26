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

## Optional scaling experiment (out of scope for this report cycle, but listed here for completeness)

- [ ] **Scale clean survivor set, not epochs.** Single-knob change vs `configs/experiment_fast.yaml`: raise `top_k: 32 → 96` and keep `apply_contamination_filter: true`, `epochs: 1`, everything else identical.
  - Predicted post-filter survivor count: ~25–30 (vs ~10 in current `fast_filt_e1`).
  - Predicted gradient updates: ~25–30 (vs ~10).
  - Estimated wall-clock: ~10–12 min/target on CPU (more candidates to generate + score).
  - Configuration: [configs/experiment_fast_filtered_topk96.yaml](../configs/experiment_fast_filtered_topk96.yaml) (created). A100 / Colab Enterprise runbook: [notebooks/colab_enterprise_a100_runbook.md](../notebooks/colab_enterprise_a100_runbook.md). Note: the config also raises `num_conditions: 16 -> 24` and `candidates_per_condition: 4 -> 8` in lockstep with `top_k`, so the top-K cut remains a real top-50% selection (otherwise `top_k=96` would exceed the 64-candidate pool and degrade to "select everything"). All optimization-strength knobs unchanged.
  - Hypothesis to test: own − transfer asymmetry strengthens, distinct_2_post stays ≥ 0.45, neither failure mode appears.
  - Falsification: if own − transfer stays at ~+0.034 or decreases, the asymmetry is bounded by the structural difference between the two classifiers, not by clean-data scale.
  - Decide before running: do this *only* if the camera-ready review requests stronger evidence on the propagation mechanism. The three existing regimes already make the bounded-regime claim defensible on their own.

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
