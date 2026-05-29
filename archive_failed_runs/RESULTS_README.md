# Archived failed / superseded runs

These artifacts are retained for provenance only. Do NOT cite them as results.

## Why archived

The DPO runs here under-trained: `train_loss` sat at or above `ln2 = 0.693` and
`rewards_margins` were near zero or negative, so the policy barely moved. The
audited effects (hedge-density differences across arms) were not statistically
distinguishable from zero, the random-control pilot did not replicate at three
seeds, and the uncertainty-regularizer penalty was effectively inactive.

They are superseded by the Length-Controlled DPO + DPOP redesign (SamPO token
down-sampling + DPOP positive-preservation), which first establishes a stable
vanilla DPO baseline before running arms.

## Contents

- `outputs_old/` - earlier per-round / trajectory + calibration outputs and
  the old `_archive_trajectory_v1/` and `data/` splits.
- `results_bundle_1/` - the 7-arm x 3-seed rerun (baseline, judge_dpo, random,
  random_length_matched, judge_length_matched, mit_pairfilter, mit_uncertreg)
  whose audit showed the under-training above.

## Restore

Move a directory back out of `archive_failed_runs/` to restore it. Nothing was
deleted.
