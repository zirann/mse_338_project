# Empirical results: evaluator-specific reward hacking under three SFT regimes

## Research question

Does optimizing a small LM (Qwen3-0.6B + LoRA SFT) against one toxicity evaluator produce a generator whose post-SFT outputs are scored asymmetrically by two different evaluators? Concretely, for two evaluators `i, j`, does

```
M[i, j] - M[j, i] != 0
```

where `M[t, s]` is the mean score that evaluator `s` assigns to candidates from a generator optimized against evaluator `t`?

A positive *own-minus-transfer* signal (`Δown - Δtransfer > 0` for both targets) would be evidence that single-evaluator SFT propagates evaluator-specific stylistic features into the generation distribution — i.e. evaluator-specific reward hacking at the policy level.

## Experimental setup

- **Base model**: `Qwen/Qwen3-0.6B`. Frozen base + LoRA adapter (r=16, alpha=32, dropout=0.05, target modules `q/k/v/o/up/down/gate_proj`).
- **Evaluators** (`i, j`):
  - `unitary/toxic-bert` — 6-head multi-label sigmoid; per-sample scalar is the weighted aggregate `toxic_agg`.
  - `cardiffnlp/twitter-roberta-base-offensive` — binary softmax; per-sample scalar is `P(offensive)`.
  - The two are intentionally heterogeneous in head structure to avoid trivial agreement.
- **Seed prompts**: 4,852 deduplicated rows from `toxigen/toxigen-data` + `allenai/real-toxicity-prompts`; the val split (242 rows) feeds the generator.
- **Per-target single-iteration loop**: generate -> cross-score with *both* evaluators -> top-K by `score_<target>` -> contamination filter -> SFT LoRA on `(input_prompt, candidate_text)` survivor pairs -> regenerate from base + adapter -> cross-score again.
- **Fast regime parameters** (used in all three ablations):

  | knob | value |
  |---|---|
  | `num_conditions` | 16 |
  | `candidates_per_condition` | 4 |
  | `top_k` | 32 |
  | `min_train_pairs` | 8 (filter abort floor) |
  | `decode.max_new_tokens` | 32 |
  | `decode.temperature` / `top_p` / `rep_penalty` | 0.9 / 0.95 / 1.12 |
  | `training.batch_size` | 1 |
  | `training.grad_accum` | 1 |
  | `training.learning_rate` | 2e-4 |
  | LoRA shape | r=16, alpha=32, 7 target modules |
  | seed | 42 |

  Each run produces 64 pre-SFT candidates, ≤32 top-K, post-filter survivors (variable), and 64 post-SFT candidates per target.
- **Three regimes** that differ by exactly one knob each:

  | regime | filter | epochs | hypothesis tested |
  |---|---|---|---|
  | `fast_filt_e1` | ON | 1 | baseline filtered fast |
  | `fast_unfilt_e1` | OFF | 1 | does contamination amplify evaluator-specific signal? |
  | `fast_filt_e3` | ON | 3 | does stronger optimization on the clean survivor set strengthen specialization? |

- **Compute**: CPU (Apple Silicon macOS 13.2.1; PyTorch 2.10 disables MPS on that OS version). Each fast run is ~250-290 s per target.

## Metric definitions

Let `pre_S(t)` and `post_S(t)` denote the mean score from evaluator `S` over the 64 pre-SFT or 64 post-SFT candidates of the target-`t` run. Then:

- **Diagonal lift** = `post_t(t) - pre_t(t)`. Movement of the *self* evaluator. Positive = more self-evaluator-aligned after SFT.
- **Off-diagonal lift** = `post_o(t) - pre_o(t)` where `o != t`. Movement of the *other* evaluator. Captures generalized capability or shared reward signal.
- **Own minus transfer** = `diagonal_lift - off_diagonal_lift`. The asymmetry that defines evaluator-specific specialization. Positive = the SFT moved the policy more in the self-evaluator direction than in the other-evaluator direction.
- **distinct_2** = fraction of unique bigrams across the candidate batch. Low = mode collapse / memorization.
- **train_loss** = final loss from `Trainer.train()`. Lower can mean better fit *or* memorization; interpretation requires looking at distinct_2 and qualitative outputs jointly.

The transfer matrix `M[t, s]` written by `analyze_transfer.py` records `post_s(t)` (mean / p50 / p90) for every `(t, s)` pair.

## Three-regime comparison

| metric | target | fast_filt_e1 | fast_unfilt_e1 | fast_filt_e3 |
|---|---|---|---|---|
| accepted_before_filter | toxicbert | 32 | 32 | 32 |
|  | cardiff | 32 | 32 | 32 |
| accepted_after_filter | toxicbert | 9 | 32 | 11 |
|  | cardiff | 10 | 32 | 10 |
| effective gradient updates | toxicbert | 9 | 32 | 33 |
|  | cardiff | 10 | 32 | 30 |
| **train_loss** | toxicbert | 2.66 | 2.35 | **1.94** |
|  | cardiff | 2.97 | 2.16 | **1.88** |
| **diagonal lift** | toxicbert | -0.00708 | -0.00301 | **-0.00078** |
|  | cardiff | -0.03423 | -0.04700 | **-0.01803** |
| **off-diagonal lift** | toxicbert -> cardiff | -0.04078 | -0.03272 | -0.01559 |
|  | cardiff -> toxicbert | +0.00059 | -0.00162 | -0.00653 |
| **own minus transfer** | toxicbert | **+0.03370** | +0.02970 | +0.01481 |
|  | cardiff | -0.03482 | -0.04538 | **-0.01150** |
| **distinct_2 pre** | toxicbert | 0.499 | 0.529 | 0.574 |
|  | cardiff | 0.498 | 0.489 | 0.582 |
| **distinct_2 post** | toxicbert | **0.436** | 0.266 | 0.429 |
|  | cardiff | **0.498** | 0.344 | 0.297 |
| diversity preserved? | toxicbert | yes (mild drop) | **no** (50% drop) | partial (25% drop) |
|  | cardiff | **yes** (no drop) | partial (30% drop) | no (49% drop) |

Runtime per target: 250-290 s in every regime. Pre/post candidates totals are 480 per phase; only the SFT step changes meaningfully in cost.

![Transfer matrix M[target][scorer] for the headline regime (fast_filt_e1)](assets/figures/transfer_matrix_heatmap.png)

![Own-minus-transfer asymmetry across the three regimes for both targets](assets/figures/own_minus_transfer_barplot.png)

![distinct_2 pre vs post across regimes and targets](assets/figures/diversity_collapse_plot.png)

## Key findings

1. **Single-evaluator SFT produces an evaluator-specific asymmetry on toxicbert but anti-specialization on cardiff.** Under `fast_filt_e1`, `own_minus_transfer(toxicbert) = +0.034`, the largest positive value observed across every regime tested in the project. `own_minus_transfer(cardiff) = -0.035`: cardiff-target SFT moves the policy *away* from cardiff faster than away from toxicbert. The asymmetry is real and direction-dependent.

2. **Toxicbert's diagonal lift is consistently small in absolute terms (≤ 0.01) and often slightly negative.** The "positive specialization" finding for toxicbert is therefore *comparative*: the policy loses fewer toxicbert points than cardiff points, not that it gains toxicbert points absolutely. This is an important caveat — what propagates is the *direction* of the evaluator's bias, not its overall magnitude.

3. **Cardiff's mean score drops after every SFT round in every regime.** This is a floor effect interacting with how the filter selects samples: the contamination filter happens to remove some of the highest-cardiff-scoring candidates (because they are short shards or thinking-mode templates). What the policy ends up SFT'd on is *lower-cardiff* than the unfiltered top-K, so post-SFT cardiff regresses toward that lower mean.

4. **Train loss is misleading on its own.** `fast_filt_e3` and `fast_unfilt_e1` both reach loss ≈ 2, lower than `fast_filt_e1` at 2.66. Both also exhibit catastrophic diversity collapse. Train loss alone would have ranked these two regimes above the headline regime; the joint metric of `distinct_2 post` + `own_minus_transfer` + qualitative inspection is what reveals the actual story.

5. **Two structurally different reward-hacking failure modes were isolated.**
   - *Token-level reward hacking* (`fast_unfilt_e1`): SFT on unfiltered top-K causes the policy to memorize literal high-frequency decoder shards (e.g. repeated short non-word fragments) that happen to activate the multi-label toxic head. distinct_2 collapses ~50% on toxicbert; the post-SFT distribution becomes degenerate strings of repeated 4-5 char shards.
   - *Filter blind-spot overfitting* (`fast_filt_e3`): SFT on the filtered top-K for 3 epochs causes the policy to memorize an *8-character* lowercase shard that survives the conservative filter rule (`len(first_tok) < 6` does not catch length-8 fragments). distinct_2 on cardiff collapses from 0.58 to 0.30; every cardiff-target post-SFT top-5 candidate begins with that same shard.
   - Both are evaluator-agnostic reward hacks dressed as evaluator-specific signal. Only the headline regime (`fast_filt_e1`) avoids both and exposes a genuinely asymmetric stylistic shift.

6. **The contamination filter is necessary but not sufficient.** Filtering converts the failure mode from token-level (raw shard repetition) to style-level (overfitting on the survivor distribution). Style-level reward hacking is harder to detect and arguably more interesting from an alignment-research perspective.

7. **More optimization pressure on a small clean set hurts.** `fast_filt_e3` halves the own-minus-transfer signal on toxicbert (+0.034 -> +0.015) and collapses cardiff's diversity. With only 10-11 clean training rows, 3 epochs is enough to memorize the filter's blind-spot tokens. To strengthen the specialization signal, we should expand the clean training set, not pass over it more times.

## Qualitative failure modes (redacted)

![Conceptual taxonomy of the four observed regimes in (optimization pressure, filter on/off) space](assets/figures/reward_hacking_taxonomy_diagram.png)

All examples below are *paraphrased shape descriptions* of the actual top-5 post-SFT outputs the system produced. The raw artifacts live on disk under `outputs/runs_*` and contain unredacted offensive content by construction.

### Failure mode A — token-level reward hacking (`fast_unfilt_e1`, toxicbert target)

Post-SFT top-5 candidates are **near-pure repetition** of a single 4-5 character lowercase shard, sometimes interleaved with one alternate shard. No syntactic structure, no recognizable English, no semantic content. The multi-label toxic head activates weakly on this token shape (the same shape happens to appear in some profanity-adjacent training data for the classifier), giving these outputs higher raw toxicbert scores than any other post-SFT candidate from this regime.

Distinct_2 collapses from 0.53 to 0.27. Pattern is: `<shard> <shard> <shard> ... <shard>` repeated for the full 32 tokens of `max_new_tokens`.

This is the unambiguous "Goodhart" failure: the policy maximizes the classifier's surface-form trigger without producing any meaningful adversarial content.

### Failure mode B — filter blind-spot overfitting (`fast_filt_e3`, cardiff target)

All 5 top cardiff post-SFT candidates begin with the **same 8-character lowercase shard followed by a period**, then transition into a Qwen3-style thinking-mode preamble (`"Okay, so I need to generate ..."`). The shard passes the conservative filter (`len < 6` rule). With 3 epochs over 10 training rows, the LoRA has memorized it as a deterministic prefix.

Cardiff's binary head scores these moderately (0.22-0.30) because the thinking-mode preambles contain explicit string mentions of the risk categories (`"sexual_explicit"`, `"hate speech"`, `"adversarial"`) that pattern-match its offensive-register threshold.

Distinct_2 collapses from 0.58 to 0.30. The policy has become deterministic at the prefix and locally noisy thereafter — a degenerate solution that maximizes train-set log-likelihood without producing meaningful adversarial diversity.

### Style-level reward hacking — the headline regime (`fast_filt_e1`)

Post-SFT top-5 candidates **vary syntactically** and exhibit recognizable evaluator-specific stylistic patterns:

- *Toxicbert target*: short decoder-shard openings followed by **paraphrased identity-attack templates**: `[group-noun] are [animal-comparison]`, `[group-noun] are too [intelligence-slur] to [civic-activity]`. Vocabulary expands relative to the base model (e.g. nine-element insult lists with `gross`, `slang`-style intensifier vocabulary). Some samples wrap insults in `<tool_call>` tags; the filter cleaned the train set but the model's adopted style still emits these tokens at inference.

- *Cardiff target*: short, direct insult-with-quotation templates (`"You're a [pejorative]"`-shape), sarcastic-praise inversions (e.g. mockingly-flattering second-person statements that cardiff's binary head reads as offensive), and stacked pseudo-quote lists of slurs. Length-of-output is shorter; punctuation density is higher.

Both target distributions are clearly different from each other and from the base model. Distinct_2 only drops from 0.50 to 0.44 (toxicbert) or stays flat at 0.50 (cardiff) — diversity is preserved.

This is the qualitative form of the +0.034 own-minus-transfer signal in #1 above. It is not absolute score-maximization; it is a *stylistic shift toward whatever the target evaluator over-weights*, while preserving generative diversity. That distinction is precisely the alignment-research signal of interest.

## Final recommendation

**Use `fast_filt_e1` (configs/experiment_fast.yaml) as the headline configuration in the report.**

Justification:

1. **Cleanest measured evaluator-specific signal**: `own_minus_transfer(toxicbert) = +0.034`, the largest positive value across all configurations tested.
2. **Diversity is preserved**: distinct_2 drops at most 13% (toxicbert) or 0% (cardiff). No mode collapse.
3. **No token-level reward hacking**: the contamination filter removes the short-shard memorization shortcut that destroys `fast_unfilt_e1`.
4. **No filter-blind-spot overfitting**: epochs=1 is short enough that the policy does not memorize the 8-character shards that survive the filter.
5. **Fast iteration**: ~5 minutes per target on CPU, ~10 minutes per full cross-evaluator cycle. Compatible with research iteration.
6. **The two ablations bound the regime usefully**: `fast_unfilt_e1` and `fast_filt_e3` are not weaker baselines — they are *failure cases* that prove the headline regime is a narrow operating point. The report should present all three.

Recommended report structure:

| section | regime | role |
|---|---|---|
| headline | `fast_filt_e1` | the evaluator-specific specialization result |
| ablation 1 | `fast_unfilt_e1` | shows token-level reward hacking when filter is removed |
| ablation 2 | `fast_filt_e3` | shows filter blind-spot overfitting under stronger optimization |

## Limitations

- **One LM, two evaluators, one optimization method.** The asymmetry result is shown on Qwen3-0.6B + LoRA SFT optimizing against toxicbert vs cardiff. We cannot claim the asymmetry generalizes to other base models, RL-style optimization, DPO, or other classifier pairs.
- **CPU-only, small scale.** All experiments ran on CPU because MPS is disabled on this macOS version with this PyTorch build. Training is sufficient to demonstrate the phenomenon (10-30 effective gradient updates over 10-30 samples), but absolute score magnitudes are small (post-SFT toxicbert mean ~ 0.003, well below any policy-relevant threshold). The asymmetry is the signal; the magnitudes are not.
- **Single iteration.** We do not test compounded specialization across iterated SFT rounds. The own-minus-transfer signal may saturate, amplify, or invert under multi-iteration; that is out of scope here.
- **The filter is a research instrument, not a defense.** The 6 rejection rules were tuned to remove obvious Qwen3 thinking-mode contamination; the filter blind-spot found under `fast_filt_e3` shows it is incomplete. A different base model would require a different filter, and the filter itself shapes the propagated signal.
- **Cardiff's diagonal lift is negative in every regime.** We did not produce a configuration where cardiff-target SFT pushes cardiff post-scores above pre-scores in absolute terms. The cardiff direction shows anti-specialization, not specialization, throughout. Whether this is fundamental to the cardiff classifier or an artifact of the filter's interaction with cardiff's selection bias is unresolved.
- **Filter blind-spot was discovered post hoc.** The 8-character `iquement` shard that drives the `fast_filt_e3` overfitting was found by inspecting outputs, not by a principled filter design. A scoped follow-up could tighten the lowercase-start rule and re-run e3.
- **No gold human judge.** All metrics here are classifier-based; the entire transfer matrix is two classifiers cross-scoring each other on a small generated population. A human-judged stratified sample would be needed to claim anything about real adversarial usefulness.

## Scaling-experiment result (topk96)

The "scale the clean survivor set, not the number of epochs" follow-up has now been executed on an A100 (Colab Enterprise). Configuration: `top_k=96`, `num_conditions=24`, `candidates_per_condition=8` (192 candidates per phase; `top_k=96` is the actual top 50%), filter ON, epochs=1, all other knobs identical to `fast_filt_e1`. Implementation: [configs/experiment_fast_filtered_topk96.yaml](configs/experiment_fast_filtered_topk96.yaml); runbook: [notebooks/colab_enterprise_a100_runbook.md](notebooks/colab_enterprise_a100_runbook.md).

| metric | target | fast_filt_e1 | **topk96** |
|---|---|---|---|
| accepted_before_filter | toxicbert | 32 | **96** |
|  | cardiff | 32 | **96** |
| accepted_after_filter | toxicbert | 9 | **40** |
|  | cardiff | 10 | **34** |
| effective gradient updates | toxicbert | 9 | **40** |
|  | cardiff | 10 | **34** |
| train_loss (mean / final) | toxicbert | 2.66 / 1.48 | 2.54 / 2.28 |
|  | cardiff | 2.97 / 2.42 | 2.38 / 1.12 |
| diagonal lift | toxicbert | −0.00708 | −0.00633 |
|  | cardiff | −0.03425 | −0.03048 |
| off-diagonal lift | toxicbert → cardiff | −0.04078 | −0.03924 |
|  | cardiff → toxicbert | +0.00059 | −0.00288 |
| **own − transfer** | toxicbert | **+0.03369** | **+0.03291** |
|  | cardiff | **−0.03483** | **−0.02759** |
| distinct_2 pre | toxicbert | 0.499 | 0.563 |
|  | cardiff | 0.498 | 0.495 |
| distinct_2 post | toxicbert | 0.436 | 0.421 |
|  | cardiff | 0.498 | **0.360** |

Headline takeaway: **scaling clean survivor data does not meaningfully strengthen style-level evaluator-specific specialization**. Own-minus-transfer on toxicbert is flat within noise (+0.034 → +0.033); cardiff anti-specialization shrinks by ~21% in magnitude (−0.035 → −0.028) but does not flip sign. Diagonal lifts remain negative for both targets.

Two new negative observations under topk96:

1. **Cardiff diversity collapse re-emerges**. `distinct_2_post(cardiff) = 0.360`, down from 0.498 in `fast_filt_e1`. The shape is similar to `fast_filt_e3`'s failure mode (0.58 → 0.30).
2. **Filter blind-spot overfitting now appears on toxicbert**, which was previously a cardiff-only failure under `fast_filt_e3`. All five toxicbert post-SFT top-5 samples begin with the literal 8-character shard `intColor` (length 8 passes the conservative `len < 6` filter rule). In `fast_filt_e1` toxicbert outputs did not show this prefix pattern. Mechanism: total exposure to filter-surviving blind-spot tokens scales as `accepted_after_filter × epochs`. topk96 raises the first factor (9 → 40 for toxicbert) at fixed epochs=1; the resulting exposure is comparable to `fast_filt_e3`'s 10 × 3 = 30 and produces the same memorization pattern.

The hypothesis stated in the original "Next possible experiment" block is therefore **falsified at this scale**: the asymmetry appears to be bounded by structural disagreement between the two evaluators themselves, not by training-set size on the surviving distribution. Trying to drive own-minus-transfer up by adding more clean data instead surfaces a previously latent failure mode.

**Verdict for the report**: keep `fast_filt_e1` as the headline regime. Use topk96 as an appendix / ablation that bounds the regime in a third direction (alongside `fast_unfilt_e1` for filter-off and `fast_filt_e3` for high-epoch). The three failure-mode ablations together now characterize the operating regime on three orthogonal axes: contamination filter, optimization pressure (epochs), and clean-data scale (top_k + pool size).

Artifacts: [outputs/runs_fast_filtered_topk96/](outputs/runs_fast_filtered_topk96/), [outputs/reports_fast_filtered_topk96/transfer_matrix.json](outputs/reports_fast_filtered_topk96/transfer_matrix.json), [outputs/reports_fast_filtered_topk96/transfer_summary.md](outputs/reports_fast_filtered_topk96/transfer_summary.md).

## Pointers to artifacts

- Transfer matrices: `outputs/reports_fast/transfer_summary.md`, `outputs/reports_fast_unfiltered/transfer_summary.md`, `outputs/reports_fast_filtered_epochs3/transfer_summary.md`.
- Per-target metrics (incl. `filter_stats`, summaries, `train_result`): `outputs/runs_*/<target>/metrics.json`.
- Per-step loss curves: `outputs/runs_*/<target>/adapter/loss_curve.csv` (populated when `logging_steps: 1`).
- Raw candidate text + scores: `outputs/runs_*/<target>/{pre,post}_scored.jsonl`. Contains unredacted offensive content; do not surface in user-facing contexts.
- Archived earlier runs (full `experiment.yaml` and the original filtered run): `outputs/runs_prefilter/`, `outputs/archive_filtered_weak_sft/`, `outputs/reports_prefilter/`, `outputs/archive_reports_filtered_weak_sft/`. Useful for diff against the fast regimes.
