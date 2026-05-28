# Colab Enterprise / A100 runbook (uncertainty-suppression + mitigation)

Rerun-from-scratch pipeline for the uncertainty-suppression project. Single
NVIDIA A100 (40 or 80 GB). Total budget ~3 to 3.5 hours for the core arms at
3 seeds, plus baseline and optional arms.

All experiments are single-round DPO from the base policy (reference = base),
evaluated against a shared baseline arm. Arms differ only by preference-pair
construction and mitigation, so every cross-arm difference is a matched
intervention.

The old multi-round trajectory runbook (`colab_enterprise_a100_runbook.md`) is
deprecated; the trajectory is now an appendix experiment.

## 1. Clone + install

```bash
cd /content
git clone <YOUR_REPO_URL> uncertainty_dpo
cd uncertainty_dpo
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Verify CUDA

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 3. Prepare data (once)

```bash
python scripts/prepare.py --config configs/experiment.yaml --limit 40
```

Expected: `outputs/data/train_prompts.jsonl` (20 rows) + `outputs/data/eval_prompts.jsonl` (20 rows). ~1 min.

## 4. Baseline (once, no DPO)

```bash
python scripts/evaluate.py --config configs/experiment.yaml --round 0 \
    --limit 40 --out_dir outputs/baseline
```

Expected: `outputs/baseline/{eval_responses.jsonl, metrics.json}`. ~5 min. This is the win-rate reference for every other arm.

## 5. Core arms x seeds

Each command trains one round + evaluates. `--baseline_dir outputs/baseline` makes the win-rate compare against the baseline arm. Run seeds 0,1,2. Approx 12 min per train+eval pair (15 for the regularized arm).

PART I-A judge DPO:

```bash
for S in 0 1 2; do
  python scripts/train_round.py --config experiments/judge_dpo.yaml --round 1 --limit 40 --seed $S
  python scripts/evaluate.py    --config experiments/judge_dpo.yaml --round 1 --limit 40 --seed $S \
      --out_dir outputs/judge_dpo/seed$S --baseline_dir outputs/baseline
done
```

PART I-B random:

```bash
for S in 0 1 2; do
  python scripts/train_round.py --config controls/random.yaml --round 1 --limit 40 --seed $S
  python scripts/evaluate.py    --config controls/random.yaml --round 1 --limit 40 --seed $S \
      --out_dir outputs/random/seed$S --baseline_dir outputs/baseline
done
```

PART I-C length-matched random:

```bash
for S in 0 1 2; do
  python scripts/train_round.py --config controls/random_length_matched.yaml --round 1 --limit 40 --seed $S
  python scripts/evaluate.py    --config controls/random_length_matched.yaml --round 1 --limit 40 --seed $S \
      --out_dir outputs/random_length_matched/seed$S --baseline_dir outputs/baseline
done
```

PART II-1 pair-filter mitigation (on random base):

```bash
for S in 0 1 2; do
  python scripts/train_round.py --config mitigations/pair_filter.yaml --round 1 --limit 40 --seed $S
  python scripts/evaluate.py    --config mitigations/pair_filter.yaml --round 1 --limit 40 --seed $S \
      --out_dir outputs/mit_pairfilter/seed$S --baseline_dir outputs/baseline
done
```

PART II-2 uncertainty regularizer (on random base):

```bash
for S in 0 1 2; do
  python scripts/train_round.py --config mitigations/uncertainty_reg.yaml --round 1 --limit 40 --seed $S
  python scripts/evaluate.py    --config mitigations/uncertainty_reg.yaml --round 1 --limit 40 --seed $S \
      --out_dir outputs/mit_uncertreg/seed$S --baseline_dir outputs/baseline
done
```

Note: the per-arm YAML's `arm.out_dir_template` already resolves to the same
`outputs/<arm>/seed{seed}` path, so `--out_dir` on the eval call is belt-and-suspenders.
The train command writes to the arm template automatically; pass `--out_dir`
explicitly on eval to be safe.

## 6. Optional arm (appendix)

```bash
for S in 0 1 2; do
  python scripts/train_round.py --config controls/judge_length_matched.yaml --round 1 --limit 40 --seed $S
  python scripts/evaluate.py    --config controls/judge_length_matched.yaml --round 1 --limit 40 --seed $S \
      --out_dir outputs/judge_length_matched/seed$S --baseline_dir outputs/baseline
done
```

## 7. Aggregate + figures (local or on A100; no GPU needed)

```bash
python analysis/aggregate_arms.py --config configs/experiment.yaml --seeds 0 1 2
python analysis/make_figures.py
```

Expected: `outputs/arms_summary.json` + `figures/{fig1_discovery,fig2_mitigation,fig3_correctness_conditioned}.png`.

## 8. Manual factuality annotation (author, no compute)

Label seed-0 responses of {baseline, random, random_length_matched, mit_pairfilter, mit_uncertreg} as CORRECT/PARTIAL/INCORRECT against the TruthfulQA references already in each `eval_responses.jsonl` row. Save to `outputs/manual_factuality.jsonl` with rows `{prompt_id, arm, llm_factuality, manual_label}`, then re-run step 7 to refresh the correctness-conditioned figure with manual labels.

## 9. Failure recovery

- A single arm/seed failed: just re-run that one train+eval pair; outputs are arm/seed-scoped and idempotent.
- Length/uncertainty filter sanity-abort ("only N pairs survive"): bump `--limit 80` for that arm (40 train prompts -> more candidate pairs survive the filter).
- OOM on the regularized arm: it loads policy + frozen reference (two 0.6B models). Reduce `dpo.per_device_train_batch_size` in `configs/experiment.yaml` to 2.
- TRL/transformers version drift on the unregularized arms: `dpo.py` already retries DPOConfig/DPOTrainer kwargs across versions; the regularized arm does not use TRL.

## 10. What to download locally

Per arm/seed (small): `metrics.json`, `preference_diagnostics.json`, `adapter/train_metadata.json`, `eval_responses.jsonl`, `winrate_pairs.jsonl`. Plus `outputs/arms_summary.json` and `figures/*.png`. Skip the LoRA adapter weight files unless you intend to resume training; the metrics + responses are sufficient for all analysis and the paper.
