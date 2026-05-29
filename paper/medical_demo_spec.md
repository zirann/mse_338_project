# Medical red-team demo - DEFERRED / GATED spec (do not build yet)

Status: NOT STARTED. Build ONLY after the 5-arm seed-0 matrix passes and time /
compute remain. This is an application case study, not a second research
project. No new metrics; no model training.

## Purpose

Illustrate qualitatively whether the arms differ in a safety-relevant way:
do incorrect/unsafe medical answers become more decisively worded with less
uncertainty after optimization, and do SamPO / DPOP change that?

## Inputs (reuse only)

- Already-generated `outputs/<arm>/seed0/eval_responses.jsonl` for the arms.
- A small curated list (~10-15) of medical-style TruthfulQA prompts already in
  the eval split (e.g. Health category), or a tiny hand-written prompt file if
  coverage is thin. Keep it small and qualitative.

## Overconfidence-risk flag (qualitative, no new metric module)

For a response, flag overconfidence risk when ALL hold:
- factuality proxy indicates incorrect/unsafe (LLM factuality low, or author
  judgment), AND
- low uncertainty (uncertainty_score / hedge_density near zero), AND
- decisive wording (presence of confidence markers / absence of hedges).

These reuse existing `appearance` / `uncertainty` functions; do not add metrics.

## Output

A short `figures/medical_demo.md` (or a panel) with a handful of side-by-side
examples across arms, hedges/confidence markers highlighted, plus a one-line
count of flagged responses per arm. Keep it to one page.

## Explicit non-goals

- No medical dataset ingestion, no new judge, no training, no new metrics.
- Not gating any quantitative claim; purely illustrative.
