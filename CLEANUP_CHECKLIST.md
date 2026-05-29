# Cleanup checklist

Principle: nothing is auto-deleted. Stale artifacts are MOVED (reversible). To
restore, move them back.

## Stage 0 archive for the Length-Controlled DPO + DPOP redesign (current)

The project is now anchored on SamPO length-bias correction + a DPOP extension.
The uncertainty-suppression rerun under-trained (DPO did not move the policy) and
is superseded. Moved (see `archive_failed_runs/RESULTS_README.md` and
`archive_configs/README.md`):

- `outputs_old/` -> `archive_failed_runs/outputs_old/`
- `results_bundle_1/` -> `archive_failed_runs/results_bundle_1/`
- `controls/` (random, random_length_matched, judge_length_matched) -> `archive_configs/controls/`
- `mitigations/` (pair_filter, uncertainty_reg) -> `archive_configs/mitigations/`
- `experiments/judge_dpo.yaml` -> `archive_configs/judge_dpo.yaml`

Kept active: `experiments/judge_dpo_strong.yaml` (stable knobs -> baseline defaults),
`src/complexity_theater/` (all modules importable; regularizer OFF by default),
`legacy/` (untouched).

Active run layout after redesign:
- `outputs/baseline/`                  base policy eval (no DPO)
- `outputs/vanilla_dpo/seed{N}/`       arm B
- `outputs/sampo_dpo/seed{N}/`         arm C (SamPO length debiasing)
- `outputs/dpop/seed{N}/`              arm D (DPOP)
- `outputs/sampo_dpop/seed{N}/`        arm E (SamPO + DPOP)
- `outputs/arms_summary.json`, `figures/*.png`

---

## Prior cleanup (uncertainty-suppression pivot) - historical

The earlier pivot moved the multi-round "Epistemic Simulacra" trajectory into
`outputs/_archive_trajectory_v1/` (now itself under `archive_failed_runs/outputs_old/`).
The notes below are retained for provenance.

## Archived (moved to outputs/_archive_trajectory_v1/)

These were computed under the old framing and are superseded by the new
arm-named runs. Retained for the appendix (trajectory) and provenance.

- `outputs/round_0/` -> `outputs/_archive_trajectory_v1/round_0/`
- `outputs/round_1/` -> `outputs/_archive_trajectory_v1/round_1/`
- `outputs/round_2/` -> `outputs/_archive_trajectory_v1/round_2/`
- `outputs/round_3/` -> `outputs/_archive_trajectory_v1/round_3/`
- `outputs/trajectory.json` -> `outputs/_archive_trajectory_v1/trajectory.json`
- `outputs/calibration_summary.json` -> `outputs/_archive_trajectory_v1/calibration_summary.json`
- `assets/figures/headline.png` -> `outputs/_archive_trajectory_v1/headline_oldframing.png`

## Kept in place (still useful)

- `outputs/data/train_prompts.jsonl`, `outputs/data/eval_prompts.jsonl`
  Reusable TruthfulQA splits. `scripts/prepare.py` can also regenerate them with
  a fixed seed, so they can be safely re-created if needed.
- `legacy/` (entire tree). The earlier toxicity project archive; untouched.

## New canonical run layout (written by the rerun-from-scratch pipeline)

- `outputs/baseline/`                          base policy eval (no DPO)
- `outputs/judge_dpo/seed{0,1,2}/`             PART I-A
- `outputs/random/seed{0,1,2}/`                PART I-B
- `outputs/random_length_matched/seed{0,1,2}/` PART I-C
- `outputs/judge_length_matched/seed{0,1,2}/`  optional appendix
- `outputs/mit_pairfilter/seed{0,1,2}/`        PART II-1
- `outputs/mit_uncertreg/seed{0,1,2}/`         PART II-2
- `outputs/manual_factuality.jsonl`            author correctness labels (seed 0)
- `outputs/arms_summary.json`                  analysis/aggregate_arms.py output
- `figures/*.png`                              analysis/make_figures.py output

These do not collide with the archive, so the rerun is non-destructive.

## Flagged for MANUAL deletion only (user confirms; not auto-deleted)

- `outputs/.DS_Store` (macOS Finder cruft; safe to delete).

## Restore instructions

To revert the archival and restore the old trajectory layout:

```bash
cd outputs/_archive_trajectory_v1
mv round_0 round_1 round_2 round_3 trajectory.json calibration_summary.json ..
mv headline_oldframing.png ../../assets/figures/headline.png
```
