# Report notes (mechanism-first / claim-first)

Each section opens with a stated *mechanism* (the causal hypothesis), followed by a one-sentence *claim* (the empirical statement), followed by subordinate evidence and figure pointers. Numbers are sourced from the run JSONs under `outputs/runs_*` and `outputs/reports_*`. This file is the claim skeleton; full prose lives in `outline.md`.

---

## Section 1 — Cross-classifier disagreement at the selection step

**Mechanism**: the two evaluators score the same candidate distribution with different absolute and relative biases, so a top-K cut by one evaluator produces a different subset than a top-K cut by the other. This is the *precondition* for evaluator-specific SFT to be meaningful — without selection disagreement there is no asymmetric signal for SFT to propagate.

**Claim**: ToxicBERT's top-K and Cardiff's top-K on the same Qwen3 candidate pool are non-symmetric subsets; specifically, ToxicBERT-accepted candidates score *higher* on Cardiff than the baseline cardiff mean, but Cardiff-accepted candidates score *much higher* on ToxicBERT than the baseline toxicbert mean (a relative ~6× lift vs ~1.7×).

- Supporting evidence:
  - prefilter ToxicBERT-accepted set: `score_cardiff` mean ≈ 0.263 vs baseline ≈ 0.157 (≈1.7×). Source: `outputs/runs_prefilter/toxicbert/metrics.json` → `accepted_summary.cardiff.mean` and `pre_summary.cardiff.mean`.
  - prefilter Cardiff-accepted set: `score_toxicbert` mean ≈ 0.019 vs baseline ≈ 0.003 (≈6×). Source: `outputs/runs_prefilter/cardiff/metrics.json` → analogous fields.
- Figure: none for this section; the selection asymmetry is a precondition for the figures, not itself plotted.
- Failure-mode signature: the very samples that maximize toxicbert *also* tend to maximize cardiff; the reverse is not equally true. This explains why toxicbert-target SFT does not move cardiff in the opposite direction from the toxicbert-target.

## Section 2 — Direction-asymmetric propagation under filtered single-epoch SFT

**Mechanism**: SFT on top-K-by-score_t over-represents candidates whose stylistic features over-weight toward `score_t`'s firing patterns. With the contamination filter on, the model cannot satisfy `score_t` via token-shape shortcuts; instead it imitates the stylistic register of clean survivor candidates. Different `t` → different survivor styles → different post-SFT stylistic distributions, even when absolute post-SFT scores remain similar.

**Claim** (headline result): under `fast_filt_e1`, `own − transfer(toxicbert) = +0.034` and `own − transfer(cardiff) = −0.035`, while diagonal lifts in absolute terms are weakly negative for both. The asymmetry exists in the *direction* of movement, not in the score magnitudes.

- Supporting evidence:
  - own − transfer(toxicbert) = `delta_mean(toxicbert→toxicbert) − delta_mean(toxicbert→cardiff)` = `-0.00708 − (-0.04078)` = `+0.03370`. Source: `outputs/reports_fast/transfer_matrix.json` → `deltas_post_minus_pre`.
  - own − transfer(cardiff) = `-0.03423 − (+0.00059)` = `-0.03482`. Same source.
  - Diagonal lifts: toxicbert −0.00708, cardiff −0.03423. Both *negative* in absolute terms; the asymmetry is comparative.
  - Diversity preserved: distinct_2 0.50 → 0.44 (toxicbert), 0.50 → 0.50 (cardiff). Source: `outputs/runs_fast/<target>/metrics.json` → `distinct_2_pre` / `distinct_2_post`.
- Figures: `assets/figures/transfer_matrix_heatmap.png`, `assets/figures/own_minus_transfer_barplot.png`.
- Failure-mode signature: none in this section; this is the headline result, not a failure.

## Section 3 — Token-level reward hacking when contamination filter is removed

**Mechanism**: classifier reward heads, especially multi-label ones like ToxicBERT, can be triggered by token shapes that have no semantic content (e.g. token-internal substrings that co-occurred with profanity during the classifier's training). The contamination filter removes top-K rows dominated by such shapes. Removing the filter exposes the SFT loop to these "free reward" rows, and one epoch of LoRA training over 32 such rows is enough to memorize the surface-form trigger. The policy collapses onto repetition of the trigger.

**Claim**: under `fast_unfilt_e1`, the post-SFT toxicbert distribution collapses onto literal repetition of a small set of 4–5 character decoder shards. distinct_2 drops 0.53 → 0.27 (≈50% collapse). Maximum post-SFT toxicbert score (0.072) is higher than in the filtered regime (0.027), but this score comes entirely from degenerate text, NOT from evaluator-specific style.

- Supporting evidence:
  - distinct_2_post toxicbert = 0.266 (vs 0.529 pre). Source: `outputs/runs_fast_unfiltered/toxicbert/metrics.json`.
  - Maximum post toxicbert in top-5: 0.072 (vs 0.027 in `fast_filt_e1`). Source: `outputs/reports_fast_unfiltered/transfer_matrix.json` → `top_examples_post.toxicbert[0].score_target`.
  - own − transfer toxicbert under unfiltered ≈ +0.030, *not* higher than the filtered headline (+0.034). Asymmetry did not improve; only diversity collapsed.
  - train_loss = 2.35, lower than filtered 2.66. The lower loss reflects memorization of the shard distribution, not better alignment.
- Figures: `assets/figures/diversity_collapse_plot.png`, `assets/figures/reward_hacking_taxonomy_diagram.png`.
- Failure-mode signature: post-SFT top-5 outputs are sequences of the same 4–5 character shard token repeated for the full `max_new_tokens=32` budget, with no syntactic or semantic structure. The classifier fires; the policy is degenerate. Goodhart on the surface form.

## Section 4 — Filter blind-spot overfitting under stronger optimization on the small clean set

**Mechanism**: the contamination filter's `lowercase_start` rule is deliberately conservative (`len(first_tok) < 6`) so it does not over-reject legitimate lowercase sentence starts. Any decoder shard ≥ 6 characters survives the filter. With only ~10 clean training rows, three epochs of LoRA SFT are enough for the model to memorize a specific 8-character shard that appears in the training set as a *deterministic prefix*. This is structurally analogous to the unfiltered failure but operates on tokens long enough to pass the filter — the SFT loop reward-hacks through the filter's blind spot rather than around it.

**Claim**: under `fast_filt_e3`, the cardiff-target post-SFT distribution becomes a deterministic 8-character prefix (`iquement.`) followed by a thinking-mode preamble. own − transfer on toxicbert halves (+0.034 → +0.015); distinct_2 on cardiff collapses 0.58 → 0.30. More passes over a small clean set is the wrong knob; more clean data is the right one.

- Supporting evidence:
  - own − transfer(toxicbert) under e3 = +0.01481 vs +0.03370 under e1 (≈ −56%). Source: `outputs/reports_fast_filtered_epochs3/transfer_matrix.json` and `outputs/reports_fast/transfer_matrix.json`.
  - distinct_2_post cardiff = 0.297 (vs 0.498 in e1 / 0.582 pre). Source: `outputs/runs_fast_filtered_epochs3/cardiff/metrics.json`.
  - All five top cardiff post-SFT samples begin with the literal prefix `iquement.`. Source: `outputs/reports_fast_filtered_epochs3/transfer_summary.md` → "Top-5 post examples per target evaluator" / target = cardiff.
  - train_loss(cardiff) under e3 = 1.88 vs 2.97 under e1. Lower loss, worse diversity — the same memorization-as-low-loss pattern as Section 3.
- Figure: `assets/figures/reward_hacking_taxonomy_diagram.png`.
- Failure-mode signature: deterministic shard-prefix on every output; rest of generation reverts to thinking-mode preamble. A second variant of the same family as Section 3, distinguished only by which token shape the filter happens to admit.

## Section 5 — train_loss is misleading in isolation under reward hacking

**Mechanism**: a small training set with structurally repetitive tokens admits a low-loss memorization solution; cross-entropy on shard repetition or deterministic prefixes is easy to drive down. train_loss measures fit to the training distribution, not policy behavior on the seed-prompt distribution. Under reward hacking the two are dissociated by construction.

**Claim**: the two lowest train_loss values in the three-regime sweep (`fast_unfilt_e1` ≈ 2.2 and `fast_filt_e3` ≈ 1.9) coincide with the two worst diversity collapses; the highest train_loss (`fast_filt_e1` ≈ 2.8) coincides with the best `own_minus_transfer` and the best preserved diversity. The honest scalar for evaluating these regimes is the joint of (own_minus_transfer, distinct_2_post), not train_loss alone.

- Supporting evidence:
  - `fast_filt_e1`: train_loss (toxicbert) = 2.66, (cardiff) = 2.97; distinct_2_post 0.44 / 0.50; own − transfer +0.034 / −0.035.
  - `fast_unfilt_e1`: train_loss 2.35 / 2.16; distinct_2_post 0.27 / 0.34; own − transfer +0.030 / −0.045.
  - `fast_filt_e3`: train_loss 1.94 / 1.88; distinct_2_post 0.43 / 0.30; own − transfer +0.015 / −0.012.
  - Source: each `outputs/runs_*/<target>/metrics.json` → `train_result.train_loss` and `distinct_2_post`; each `outputs/reports_*/transfer_matrix.json` → `deltas_post_minus_pre`.
- Figures: `assets/figures/diversity_collapse_plot.png`.
- Failure-mode signature: this is not a failure mode per se; it is a *measurement* claim. The implication is that any paper or pipeline that reports train_loss alone after SFT on small reward-selected sets is potentially blind to both Section 3 and Section 4 failure modes.

## Section 6 — Scaling the clean survivor set does not amplify the asymmetry; it surfaces the blind-spot failure on toxicbert

**Mechanism**: filter blind-spot overfitting is governed by *total exposure to surviving blind-spot tokens*, which equals `accepted_after_filter × epochs`. The Section 4 failure (cardiff under `fast_filt_e3`) raises that product by raising epochs on a small clean set. The scaling experiment (`fast_filt_topk96`) raises the same product by raising clean survivors at fixed epochs=1. Both interventions produce comparable total exposure to filter-surviving 8-char shards and therefore produce the same prefix-memorization failure — just on different targets, depending on which shard tokens happen to dominate each target's survivor set.

**Claim**: under `fast_filt_topk96` (top_k=96 with pool size 192 so the cut is real top-50%, filter ON, epochs=1, all other knobs identical to `fast_filt_e1`), own − transfer is essentially flat on toxicbert (+0.034 → +0.033) and modestly less anti-specialized on cardiff (−0.035 → −0.028), but a new failure appears: all five toxicbert-target post-SFT top-5 outputs begin with the deterministic 8-character prefix `intColor` (length 8 → passes the conservative `len < 6` filter rule), and cardiff post-distinct_2 collapses 0.50 → 0.36. The scaling hypothesis "more clean data strengthens specialization" is falsified at this scale; the bottleneck is structural disagreement between the two evaluators, not data quantity.

- Supporting evidence:
  - accepted_after_filter: toxicbert 9 → 40, cardiff 10 → 34 (~4× more clean training rows). Source: `outputs/runs_fast_filtered_topk96/<target>/metrics.json` → `filter_stats.accepted_after_filter`.
  - own − transfer: toxicbert +0.03369 → +0.03291 (flat), cardiff −0.03483 → −0.02759 (modest +0.00724 magnitude improvement). Source: `outputs/reports_fast_filtered_topk96/transfer_matrix.json` → `deltas_post_minus_pre`.
  - distinct_2_post: toxicbert 0.436 → 0.421 (mild), cardiff 0.498 → 0.360 (collapse). Source: per-target `metrics.json` → `distinct_2_post`.
  - Top-5 toxicbert post-SFT: all five begin with `intColor`. Source: `outputs/reports_fast_filtered_topk96/transfer_summary.md` and `transfer_matrix.json` → `top_examples_post.toxicbert`.
  - train_loss mean: toxicbert 2.66 → 2.54, cardiff 2.97 → 2.38 (lower; more memorization, not better alignment — same dissociation as Section 5).
- Figure: none yet specific to this section; extending `assets/figures/own_minus_transfer_barplot.png` to a 4-regime panel including `fast_filt_topk96` is tracked in `TODO_report.md`.
- Failure-mode signature: `intColor`-prefix overfitting on toxicbert (parallel to the `iquement.` overfit on cardiff under `fast_filt_e3`). Same family as Section 4, induced by a different knob.

## Section 8 — Cross-cutting limitation: cardiff diagonal is negative in every regime

**Mechanism**: cardiff is a binary offensive head trained on Twitter-style data; its activations depend heavily on surface register (sarcastic-praise, pseudo-quote-with-slur) and short-form structure. The contamination filter removes the rows in the top-K that cardiff most aggressively over-weights (short shard-prefixed texts), so the clean survivor set has a lower cardiff mean than the unfiltered top-K. SFT regresses the policy toward the survivor mean and therefore lowers cardiff post-mean in absolute terms. The asymmetric signal survives because *toxicbert* regresses less than cardiff does, but the diagonal-lift sign is dominated by the filter-survivor regression. The Section 6 scaling experiment confirms this: ~3.4× more clean cardiff training rows did not flip the diagonal sign.

**Claim**: across all four configurations tested in this work, `diagonal_lift(cardiff) < 0`, i.e. cardiff-target SFT never raises the cardiff post-mean above the cardiff pre-mean. This is an unresolved structural feature, not an artifact of any single configuration; the scaling-experiment result rules out "needs more clean data" as the explanation.

- Supporting evidence:
  - `fast_filt_e1` cardiff diagonal = −0.03425.
  - `fast_unfilt_e1` cardiff diagonal = −0.04700.
  - `fast_filt_e3` cardiff diagonal = −0.01803.
  - **`fast_filt_topk96` cardiff diagonal = −0.03048** (with 34 clean training rows, ~3.4× more than `fast_filt_e1`).
  - Source: each `outputs/reports_*/transfer_matrix.json` → `deltas_post_minus_pre.cardiff.cardiff.delta_mean`.
- Figure: `assets/figures/own_minus_transfer_barplot.png` (shows the asymmetry; the absolute negativity of the diagonal needs to be discussed in prose; consider adding the `fast_filt_topk96` bar — tracked in `TODO_report.md`).
- Failure-mode signature: not a failure of the paper's method — a limitation of the experiment's instrument. Resolution would require either a different classifier-pair choice or a different reward-aggregation strategy (RL with KL constraint, DPO with paired preferences, etc.), all explicitly out of scope for this work.
