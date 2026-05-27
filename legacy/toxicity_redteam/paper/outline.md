# Paper outline

Working title: *Evaluator-specific reward hacking in single-iteration LoRA-SFT against open-source toxicity classifiers*.

Each section below lists the content to write, not the prose. Cite figures by filename only; the assets live under `../assets/figures/`.

---

## 1. Abstract

- One sentence framing: small LMs SFT'd against a single reward classifier can develop classifier-specific stylistic biases that do not transfer to a structurally different reward classifier.
- One sentence on setup: Qwen3-0.6B + LoRA SFT on candidates selected by either `unitary/toxic-bert` or `cardiffnlp/twitter-roberta-base-offensive`.
- One sentence on the headline result: under contamination-filtered single-epoch SFT, `own − transfer` is +0.034 for toxicbert target vs −0.035 for cardiff target, with preserved generation diversity.
- One sentence on the two ablation failure modes: dropping the filter triggers token-level reward hacking (mode collapse); raising epochs on the small clean set triggers filter blind-spot overfitting.
- One sentence on the takeaway: what propagates through SFT is the *direction* of evaluator bias, not its absolute magnitude — visible only in a narrow operating regime that the two ablations bracket.

## 2. Introduction

- Why evaluator-as-reward matters in modern alignment: many pipelines optimize a base policy against a learned classifier, RM, or rubric. Anything systematic about the *classifier* leaks into the policy.
- What evaluator-specific reward hacking is, distinguished from generic Goodhart: not "the score goes up but capability is bad" but "the score moves direction-asymmetrically across two equally legitimate classifiers".
- What this paper does: a controlled three-regime ablation on a single 0.6B-parameter LM, single iteration of SFT, no RL, no DPO. Goal is to establish the phenomenon at minimal scale and characterize the regime boundaries.
- What we explicitly do NOT claim: generalization across base models, RL methods, classifier pairs, multi-iteration optimization, or large-scale settings. Section 9 collects all such limitations.
- Contributions, listed: (1) empirical evidence of direction-asymmetric SFT propagation on Qwen3-0.6B, (2) characterization of two failure modes that bound the operating regime, (3) a small open-source experimental harness.

## 3. Related work

- Reward modeling and SFT-as-RL: Christiano 2017, Stiennon 2020, Ouyang 2022 (instruct-tuning), Bai 2022 (constitutional AI as a related reward proxy).
- Goodhart's Law / reward hacking framings: Goodhart 1975, Manheim 2018, Skalse 2022 (categorizing reward hacking failure modes), Krakovna 2020 (specification gaming).
- Classifier-based safety/reward heads: HuggingFace `unitary/toxic-bert`, `cardiffnlp/twitter-roberta-base-offensive` model cards; toxic / offensive classifier benchmarks (Borkan 2019 civil-comments, Hartvigsen 2022 toxigen).
- Evaluator brittleness in NLG: Welleck 2020 (mauve), classifier sensitivity to surface form, recent work on classifier-as-judge reliability.
- Adjacent reward-hacking-as-evidence-of-reward-mismatch papers if available — placeholder for citation pass.
- Note in the text: this paper is *not* in the RL line; the deliberate scope is SFT-on-top-K to isolate the propagation mechanism without compounding optimization-method-specific artifacts.

## 4. Experimental setup

- **Base model**: `Qwen/Qwen3-0.6B`, frozen base, LoRA adapter r=16 / alpha=32 / dropout=0.05 / target modules `q/k/v/o/up/down/gate_proj`.
- **Evaluators (paired)**: `unitary/toxic-bert` (multi-label sigmoid; per-row scalar = weighted aggregate `toxic_agg`) and `cardiffnlp/twitter-roberta-base-offensive` (binary; per-row scalar = `P(OFF)`). Justification: structurally different (multi-label vs binary), different training distributions, both standard.
- **Seed prompts**: 4,852 dedup'd rows from `toxigen/toxigen-data` + `allenai/real-toxicity-prompts`. The val.jsonl slice (242 prompts) is fed into the generator.
- **Per-target single-iteration loop**: generate `num_conditions × candidates_per_condition` → cross-score with *both* evaluators → top-K by `score_<target>` → contamination filter → SFT LoRA on `(input_prompt, candidate_text)` pairs → regenerate from base + adapter → cross-score again.
- **Three regimes**:
  - `fast_filt_e1` (headline): filter ON, epochs=1, top_k=32 → ~10 effective gradient updates per target.
  - `fast_unfilt_e1` (ablation A): filter OFF, epochs=1, top_k=32 → ~32 updates.
  - `fast_filt_e3` (ablation B): filter ON, epochs=3, top_k=32 → ~30 updates on the same small clean set.
- **Compute**: CPU only (Apple Silicon; macOS 13.2.1 + PyTorch 2.10 = MPS unavailable). ~5 min per target. The diagnostic prints in `scripts/run_evaluator_loop.py` confirm the device.
- **Reproducibility**: every seed pinned (42); every artifact written as JSONL or JSON; per-step `loss_curve.csv` saved when `logging_steps: 1`.
- Pointer to artifacts: this section ends with a table mapping regime name → `outputs/runs_*` and `outputs/reports_*` paths.

## 5. Metrics

- `pre_S(t)` and `post_S(t)`: mean of evaluator `S`'s per-row score across the 64 pre/post candidates of target-`t` run.
- **Diagonal lift** = `post_t(t) − pre_t(t)` (self-evaluator movement).
- **Off-diagonal lift** = `post_o(t) − pre_o(t)` with `o ≠ t` (other-evaluator movement).
- **Own minus transfer** = `diagonal_lift − off_diagonal_lift`. This is the central asymmetry metric; positive → evaluator-specific propagation.
- **distinct_2** = unique bigram fraction across a candidate batch. Sanity check against mode collapse and memorization.
- **train_loss** = final `Trainer.train()` loss. Reported but never used alone for evaluation; always interpreted jointly with `distinct_2_post`.
- Why we report all four numbers together: each metric on its own admits a reward-hacking solution that the joint reading rules out.

## 6. Main results

- Figure: `transfer_matrix_heatmap.png`. The post-SFT M[target][scorer] for `fast_filt_e1`. Both rows have low toxicbert column values; both rows have similar cardiff column values. Asymmetry lives in the *deltas*, not in the matrix entries.
- Figure: `own_minus_transfer_barplot.png`. The headline regime achieves +0.034 own-minus-transfer on toxicbert and −0.035 on cardiff. Both ablations either weaken (e3) or marginally worsen (unfiltered) this asymmetry.
- Three-regime numerical table (reuse the table from `results_summary.md`; render as LaTeX in the camera-ready).
- The headline statement to write into the prose: in the narrow regime of contamination-filtered, single-epoch SFT, single-evaluator selection propagates a *directional* bias into the policy; this directionality is visible in `own_minus_transfer` and absent in absolute classifier scores.
- Important caveat to call out in the same paragraph as the headline: both diagonals are weakly negative in absolute terms. The signal is comparative, not absolute.

### 6.1 Scaling ablation (data-quantity probe)

- Subsection or appendix block, depending on space. Reports `fast_filt_topk96`: same as `fast_filt_e1` except `top_k: 32 → 96` plus the Phase A pool grown in lockstep (`num_conditions: 16 → 24`, `candidates_per_condition: 4 → 8`) so `top_k=96` is a real top-50% cut and not a degenerate "select everything".
- Result: own − transfer essentially flat on toxicbert (+0.034 → +0.033) and modestly less anti-specialized on cardiff (−0.035 → −0.028). Diagonal lifts stay negative.
- New negative observation to report: filter blind-spot prefix overfitting now appears on toxicbert (all five top-5 outputs begin with the literal 8-char shard `intColor`) and cardiff distinct_2 collapses 0.50 → 0.36. Interpretation: total exposure to filter-surviving blind-spot tokens equals `accepted_after_filter × epochs`; topk96 raises this product the same way that `fast_filt_e3` did, with the same memorization pathology.
- Bottom line in prose: the asymmetry is bounded by structural disagreement between the two classifiers, not by clean-data quantity. Scaling the clean survivor set is not a path to stronger specialization at this LM scale.

## 7. Failure modes

- Figure: `diversity_collapse_plot.png`. Quantitative diagnosis of the two failure modes.
- Figure: `reward_hacking_taxonomy_diagram.png`. Conceptual layout of the four observed regimes in (pressure × filter) space, with the headline regime marked.
- Subsection 7.1 — **Token-level reward hacking** (`fast_unfilt_e1`). Unfiltered top-K is dominated by short decoder shards that the multi-label toxic head incidentally fires on. SFT memorizes them; post-SFT outputs collapse onto literal shard repetition. distinct_2 toxicbert: 0.53 → 0.27. train_loss is misleadingly low (~2.3).
- Subsection 7.2 — **Filter blind-spot overfitting** (`fast_filt_e3`). With only ~10 clean rows, three epochs of SFT memorize an 8-character lowercase shard that survived the filter (the conservative `len(first_tok) < 6` rule). Every cardiff-target post-SFT top-5 begins with the same prefix. distinct_2 cardiff: 0.58 → 0.30.
- The structural point both subsections converge on: train_loss is a poor scalar for detecting reward hacking. It rewards memorization. The joint `(own_minus_transfer, distinct_2_post)` reading is the honest scalar.

## 8. Discussion

- What propagates is the *direction* of evaluator bias, not its magnitude. Mechanism: SFT on top-K by `score_<t>` over-represents candidates with high `score_t`; the policy learns the *stylistic features* that score_t over-weights, and those features are different per target.
- Why both diagonal lifts are weakly negative: the contamination filter removes the highest-self-scoring candidates (because the most-firing rows are also the most contaminated). The clean survivor mean is below the pre-SFT mean on the target's own evaluator. The policy regresses to that survivor mean. The asymmetry survives because the *other* evaluator regresses *further*.
- Why cardiff is anti-specialized: cardiff is more sensitive to surface register than toxicbert is to multi-label aggregation, so cardiff's diagonal is more vulnerable to the filter's regression-to-survivor-mean. The scaling ablation (Section 6.1) rules out "needs more clean data" as the explanation — ~3.4× more clean cardiff training rows did not flip the diagonal sign.
- Filter blind-spot overfitting is governed by *total exposure to surviving blind-spot tokens*, i.e. by `accepted_after_filter × epochs`, not by epochs alone. Two distinct knobs (more epochs on small clean set in `fast_filt_e3`, more clean rows on a single epoch in `fast_filt_topk96`) produce the same memorization failure, just on different targets depending on which surviving shard tokens dominate. This generalizes the failure-mode taxonomy from "epoch budget" to "total exposure".
- Classifier-cross-scoring as a research instrument vs. ground truth: every number in this paper is a classifier-vs-classifier measurement. It is appropriate for studying *propagation*, inappropriate for claiming *adversarial usefulness*. A gold human-judge appendix is listed in `TODO_report.md`.

## 9. Limitations

- One LM (Qwen3-0.6B), one optimization method (LoRA SFT), one iteration. We do not test multi-iteration optimization, RL, DPO, or larger bases.
- CPU-only; absolute score magnitudes are small. The asymmetry is the signal, not the magnitudes.
- Two classifiers chosen for structural difference but both English / both modern. We do not test typologically diverse evaluator pairs (e.g. classifier vs. rule-based vs. LM-as-judge).
- The filter is a research instrument. The blind-spot found under e3 (8-char lowercase shards) was discovered post-hoc; a different LM or different decoder tokenization would change the filter's required rules.
- Cardiff diagonal is negative in every regime; we did not produce a configuration where cardiff-target SFT pushes cardiff post-mean above pre-mean. Whether that is fundamental to cardiff or a filter-cardiff interaction is unresolved.
- No human-judged ground truth on the candidate distributions.

## 10. Conclusion

- Restate the headline: under a narrow operating regime, single-evaluator LoRA SFT on a 0.6B-parameter LM produces direction-asymmetric reward propagation between two structurally different toxicity classifiers.
- Restate the two failure modes that *bound* the operating regime: token-level reward hacking (filter off) and filter blind-spot overfitting (more epochs).
- Restate the proposed next experiment: scale clean survivor set (top_k 32 → 96) at filter ON / epochs=1. The hypothesis is that more clean data, not more passes, will amplify the asymmetry without inducing either failure mode.
- One sentence on alignment relevance: classifier-based reward proxies are common; this work shows that even *which* classifier you optimize against alters the policy in classifier-specific ways that simple absolute-score metrics will miss.
