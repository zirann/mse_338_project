# Evaluator-Specific Reward Hacking (Qwen3-0.6B + LoRA SFT)

An empirical AI alignment research project studying **evaluator-specific reward hacking**: does optimizing a small LM (Qwen3-0.6B + LoRA SFT) against one toxicity evaluator produce samples that score higher on *that* evaluator than on a second, structurally different evaluator? In matrix form, does `M[i, j] != M[j, i]` for two evaluators `i, j`?

This is NOT a toxicity / harassment generator. The artifacts produced (offensive text candidates) are research data used to study optimization-induced exploit behavior, in particular whether one round of SFT against a specific evaluator-as-reward leaks into evaluator-specific *stylistic* features of the generated distribution.

## Components

```
src/redteam/
  control_tokens.py    # prompt + risk/severity/style control tokens
  data_pipeline.py     # seed-prompt loader (toxigen + real-toxicity-prompts)
  generator.py         # single-pass condition-batched decoder, <think>-tag stripping
  io_utils.py          # read/write JSONL/YAML/JSON
  metrics.py           # tokenize, distinct_n, duplication_ratio, avg_length_tokens, harmful_hit_rate
  model_factory.py     # device resolver (CUDA > MPS > CPU), tokenizer, base + LoRA loaders
  scorers.py           # ToxicBertScorer + CardiffOffensiveScorer + score_all_evaluators (no weighted sum)
  trainer.py           # LoRA SFT on (input_prompt, candidate_text) pairs; assistant-only loss masking
scripts/
  prepare_data.py        # build seed-prompt JSONLs from HF datasets
  run_evaluator_loop.py  # single-iteration: gen -> cross-score -> top-K -> filter -> SFT -> regen -> cross-score
  analyze_transfer.py    # build M[target][scorer] transfer matrix + redacted top examples
configs/
  data.yaml                              # 2 HF data sources (toxigen + RTP) for seed prompts only
  experiment.yaml                        # "real" config (60 conditions x 8 candidates x ~80 min on CPU)
  experiment_fast.yaml                   # FAST regime: 16x4, top_k=32, filter ON, epochs=1, ~5 min/target
  experiment_fast_unfiltered.yaml        # ablation: same as fast but apply_contamination_filter=false
  experiment_fast_filtered_epochs3.yaml  # ablation: same as fast but epochs=3
tests/                  # 8 unit tests (filter rules, scorer contract, metrics, think-tag stripping)
legacy/                 # archival code + old deliverables (NOT used by current pipeline)
```

## Experimental loop (single iteration, per target evaluator E)

1. Generate `num_conditions x candidates_per_condition` candidates from the base model.
2. Cross-score every candidate with *all* evaluators (no weighted aggregation).
3. Select top-K by `score_<E>` only.
4. Apply the contamination filter to the top-K (6 rules; see [scripts/run_evaluator_loop.py](scripts/run_evaluator_loop.py)).
5. SFT a fresh LoRA adapter on the filtered survivors.
6. Regenerate from base + adapter.
7. Cross-score the post-SFT candidates.

`analyze_transfer.py` then assembles `M[target][scorer]` and pre/post deltas across the targets you have run.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 1) Seed prompts (cached HF datasets; needs offline-safe mode on flaky networks)
HF_HUB_OFFLINE=1 python scripts/prepare_data.py --config configs/data.yaml

# 2) FAST experiment (~5 min per target on CPU; this is the headline configuration)
HF_HUB_OFFLINE=1 python scripts/run_evaluator_loop.py \
    --target_evaluator toxicbert --config configs/experiment_fast.yaml
HF_HUB_OFFLINE=1 python scripts/run_evaluator_loop.py \
    --target_evaluator cardiff   --config configs/experiment_fast.yaml

# 3) Cross-evaluator transfer matrix + top examples
python scripts/analyze_transfer.py --config configs/experiment_fast.yaml \
    --out_dir outputs/reports_fast
```

`experiment.yaml` is the larger ~44 min/target configuration used for the initial baseline runs; `experiment_fast.yaml` is the recommended configuration for further research iteration (see [results_summary.md](results_summary.md) for justification).

## Output layout (per run)

```
outputs/
  data/{train,val,test}.jsonl                          # seed prompts (val.jsonl is the one fed to generator)
  runs_fast/<target>/
    pre_candidates.jsonl                               # base-model generations
    pre_scored.jsonl                                   # + score_toxicbert + score_cardiff
    accepted_before_filter.jsonl                       # top-K by score_<target>
    accepted_after_filter.jsonl                        # post-filter survivors used for SFT
    train_pairs.jsonl                                  # (input_prompt, candidate_text) used for SFT
    adapter/                                           # LoRA + loss_curve.csv + train_metadata.json
    post_candidates.jsonl                              # post-SFT generations
    post_scored.jsonl                                  # + score_toxicbert + score_cardiff
    metrics.json                                       # filter_stats, pre/accepted/post summaries, train_result
  reports_fast/
    transfer_matrix.json                               # M[target][scorer] with mean/p50/p90 + deltas + top examples
    transfer_summary.md                                # human-readable version
```

The three ablation regimes write under parallel paths: `outputs/runs_fast_unfiltered/`, `outputs/runs_fast_filtered_epochs3/`, plus matching `outputs/reports_*` directories. Earlier ("real" `experiment.yaml`) runs and the original prefilter run are archived under `outputs/runs_prefilter/`, `outputs/archive_filtered_weak_sft/`, etc.

## Contamination filter

[scripts/run_evaluator_loop.py](scripts/run_evaluator_loop.py)'s `filter_accepted` rejects candidates from the top-K (per evaluator-target run) by these rules, in order:

1. `too_short`: whitespace token count < 8.
2. `okay_prefix`: text starts with `"Okay"` (Qwen3 thinking-mode marker).
3. `tool_call`: text contains `<tool_call>` (thinking-mode framing token).
4. `refusal`: text contains a refusal phrase (`"I'm not allowed"`, `"I can't help"`, `"I cannot comply"`).
5. `lowercase_start` (conservative): first token starts with a lowercase letter AND is shorter than 6 characters. Catches obvious decoder shards (`enses`, `enson`, `mens`) while letting normal lowercase sentence starts and longer fragments pass.
6. `underscore_joined`: 3+ underscore-joined word tokens (hallucinated identifier shards).

Rejection counts and the filtered survivor count land in `metrics.json` under `filter_stats`. The filter can be disabled with `apply_contamination_filter: false` (used in the unfiltered ablation).

## Evaluators

Two open-source classifiers, **no weighted aggregation** of their scores:

- `unitary/toxic-bert` — 6-head multi-label sigmoid; we expose all heads plus a single `toxic_agg` scalar.
- `cardiffnlp/twitter-roberta-base-offensive` — binary softmax over `OFF` vs `NOT`.

`score_all_evaluators` in [src/redteam/scorers.py](src/redteam/scorers.py) writes `score_toxicbert` and `score_cardiff` per row and never combines them. Per-row ToxicBERT detail goes to `label_scores_toxicbert` and `predicted_risk_toxicbert`.

A `force_heuristic=True` mode is preserved as a test-only escape hatch for offline unit tests; production runs use the real classifiers.

## Tests

```bash
pytest -q
```

8 tests: think-tag stripping, generator condition contract, scorer contract (per-evaluator score keys, no `score_total`), filter rule coverage (all 6 buckets fire), basic metrics, and the archived `apply_rejection_sampling` smoke test against `legacy/rejection.py`.

## Reproducibility

- All experiments are seeded (`seed: 42`).
- Generation is sampling-based; same seed + same model + same input -> reproducible candidate set.
- LoRA SFT writes a per-step `loss_curve.csv` plus `train_metadata.json` (package versions, HF commit hash, trainable param count).
- Every artifact (candidates, scored, accepted before/after filter, train pairs, post-SFT candidates, transfer matrix) is JSONL or JSON. Nothing is hidden inside opaque pickles.

## Safety framing

The candidate text in `outputs/runs_*/<target>/*_candidates.jsonl` contains offensive content by construction — that is the experimental signal under study. It is not redacted on disk; the `analyze_transfer.py` markdown summary truncates examples to 200 characters but does not mask harmful tokens, because the research question is precisely *which* tokens / styles propagate. Do not surface raw candidate text in user-facing contexts.

## Figures

Publication-style figures under [assets/figures/](assets/figures/), regenerated deterministically from existing JSON artifacts (no experiment reruns) by [scripts/figures/build_all.py](scripts/figures/build_all.py):

- [assets/figures/transfer_matrix_heatmap.png](assets/figures/transfer_matrix_heatmap.png) — `M[target][scorer]` post-SFT mean scores for the headline regime (`fast_filt_e1`).
- [assets/figures/own_minus_transfer_barplot.png](assets/figures/own_minus_transfer_barplot.png) — own-minus-transfer asymmetry across the three regimes for both targets. The central evaluator-specific-specialization metric.
- [assets/figures/diversity_collapse_plot.png](assets/figures/diversity_collapse_plot.png) — `distinct_2` pre vs post per regime per target. Diagnoses mode collapse.
- [assets/figures/reward_hacking_taxonomy_diagram.png](assets/figures/reward_hacking_taxonomy_diagram.png) — paper-style conceptual diagram of the four empirically-observed regimes in (optimization pressure, filter on/off) space.

To rebuild all figures: `python scripts/figures/build_all.py`. Requires `matplotlib` (added to [requirements.txt](requirements.txt)).

## Paper draft

Report scaffold under [paper/](paper/):

- [paper/outline.md](paper/outline.md) — structured 10-section research-paper outline (Abstract / Introduction / Related Work / Setup / Metrics / Main Results / Failure Modes / Discussion / Limitations / Conclusion). Content per section, not prose.
- [paper/report_notes.md](paper/report_notes.md) — mechanism-first / claim-first scientific notes. Each section opens with a causal hypothesis and a one-sentence claim, then enumerates supporting evidence with artifact paths and figure pointers.
- [paper/TODO_report.md](paper/TODO_report.md) — remaining tasks before camera-ready (citations to add, appendix items, optional scaling experiment, safety/publication review).

## Pointer to the empirical results

See [results_summary.md](results_summary.md) for the three-regime ablation results, the cross-evaluator transfer matrices, the qualitative failure modes, and the final recommended configuration for the report.
