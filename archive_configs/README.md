# Archived configs (de-emphasized arms)

Moved here during the Length-Controlled DPO + DPOP redesign. The corresponding
Python modules remain importable in `src/complexity_theater/`; only these arm
configs are archived so `experiments/` shows the active matrix.

- `controls/` - random-preference and length-matched control arms. The
  random-control / optimization-intrinsic-drift framing is no longer centered.
- `mitigations/pair_filter.yaml`, `mitigations/uncertainty_reg.yaml` - the
  uncertainty-preserving pair filter and uncertainty-token regularizer. The new
  mitigations are SamPO (length debiasing) and DPOP (positive preservation),
  both paper-anchored. The regularizer code stays in
  `src/complexity_theater/regularized_dpo.py` but is OFF (`reg_lambda=0`) for
  all active arms.
- `judge_dpo.yaml` - superseded by the stronger/stable knobs now in
  `configs/experiment.yaml` and the per-arm configs under `experiments/`.

Nothing was deleted; move a file back to reactivate it.
