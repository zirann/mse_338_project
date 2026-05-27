# Results summary — placeholder

The Optimization-Induced Epistemic Simulacra trajectory has not yet been run.
This file will be filled in during Phase 4 once `outputs/trajectory.json` and
the headline figure are produced.

The previous toxicity project's results summary is preserved at
`legacy/toxicity_redteam/results_summary.md` for reference.

## Expected content

Once Phase 3 finishes, this file will contain:

- The four-row per-round table of all six metrics (3 appearance + 2 substance + 1 composite).
- Reference to the headline figure at `assets/figures/headline.png`.
- The four-level success-criterion verdict (weak / moderate / strong / WOW).
- Side-by-side qualitative example transcripts on 3 evaluation prompts.

## Pointers

Per-round artifacts will be written to:

- `outputs/round_0/metrics.json` (base-model baseline; no DPO).
- `outputs/round_1/metrics.json` plus `adapter/`, `candidates.jsonl`, `preference_pairs.jsonl`, `eval_responses.jsonl`.
- `outputs/round_2/metrics.json` (with the same per-round artifacts).
- `outputs/round_3/metrics.json` (with the same per-round artifacts).
- `outputs/trajectory.json` (combined per-metric per-round table).
