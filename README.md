# Support Coupling in Combined Length-Debiased and Positive-Preserving Preference Optimization

MS&E 338 (Aligning Superintelligence), Spring 2026 — final project. Author: Ziran Zhou.

## Summary

Direct Preference Optimization (DPO) improves judge-rated quality but changes how a model presents epistemic uncertainty. On Qwen3-0.6B fine-tuned with LoRA against an LLM judge on TruthfulQA, vanilla DPO suppresses uncertainty markers (about a 53% relative drop) and raises overconfident errors from 11.2% to 30.4% with no
factuality gain. Two published mitigations — SamPO (length debiasing) and DPOP (positive preservation) — each reduce the effect, but their naive combination is sub-additive. We diagnose this as *support coupling*: the DPOP preservation anchor inherits SamPO's stochastic subsampled token support and so preserves only a random fragment of the chosen response. We propose Decoupled-Support SamPO+DPOP, which keeps SamPO's equal-token subset for the preference comparison but anchors DPOP on the full chosen completion. Decoupling improves over the naive combination
on uncertainty preservation (uncertainty score 0.851 → 1.004, paired p = 0.028) and overconfident-error reduction (22.5% → 18.8%). It does not beat standalone DPOP/SamPO overall, does not improve factuality, and carries a small judge
win-rate cost; the contribution is a diagnosis plus a minimal, faithfulness-restoring fix.

## Quick Start (Reproducing the Project)

The recommended workflow is the notebook **`mse338_project_runbook.ipynb`**. Open it in Google Colab with a GPU runtime and run the cells top to bottom; it clones the code repository, installs dependencies, and runs steps 1–6 below end to end, finishing by zipping the results for download. A TA can reproduce the entire project from the notebook without reading any source code.

The equivalent commands are listed here for reference (run from the cloned code repository, `https://github.com/zirann/mse_338_project`). A GPU is required for steps 3–4.

**1. Environment setup**
```
pip install -r requirements.txt
```

**2. Dataset preparation** — build the 80-train / 80-eval TruthfulQA split:
```
python scripts/prepare.py --config configs/experiment.yaml --limit 160
```

**3. Baseline evaluation** — evaluate the un-tuned policy:
```
python scripts/evaluate.py --config configs/experiment.yaml --round 0 \
    --limit 80 --out_dir outputs/baseline
```

**4. Running all experimental arms** — train + evaluate each arm for seeds 0, 1, 2:
```
for SEED in 0 1 2; do
  for ARM in vanilla_dpo sampo_dpo dpop sampo_dpop; do
    python scripts/train_round.py --config experiments/$ARM.yaml --round 1 \
        --limit 80 --seed $SEED
    python scripts/evaluate.py    --config experiments/$ARM.yaml --round 1 \
        --limit 80 --seed $SEED --out_dir outputs/$ARM/seed$SEED \
        --baseline_dir outputs/baseline
  done
done
# decoupled-support extension arm (all seeds + re-aggregation):
bash scripts/run_decoupled_extension.sh
```

**5. Regenerating figures** and **6. report statistics** — aggregate across arms and seeds, then render figures and `report_stats.json`:
```
python analysis/aggregate_arms.py --config configs/experiment.yaml --seeds 0 1 2
python analysis/make_figures.py
```

### Regenerating the paper figures and statistics without a GPU

The packaged `results/` are sufficient to recompute the paper's data figures and every cited number without any training (only `matplotlib` and `numpy` are required). From the repository root:
```
python analysis/make_report_figures.py --bundle results --out_dir paper/figures
```
This rewrites `paper/figures/fig_uncertainty_bars.png`, `fig_calibration_bars.png`, and `report_stats.json`. The two conceptual diagrams (`fig_pipeline_schematic.png`, `fig_support_coupling.png`) are static assets and are preserved, not overwritten.

## Repository Structure

This folder is a self-contained copy of the project repository: it contains all code, configs, and results needed to reproduce the experiments and regenerate the figures, plus the compiled paper and poster.

```
submission_ready/
  README.md                      this file
  requirements.txt               Python dependencies
  mse338_project_runbook.ipynb   primary reproducibility entry point (Colab/GPU)
  report.pdf                     compiled paper (5 pages)
  poster.pdf                     compiled conference poster
  src/complexity_theater/        package: metrics, judge, DPO loops (SamPO/DPOP)
  scripts/                       pipeline entry points
    prepare.py                   build the TruthfulQA train/eval split
    train_round.py               train one arm (vanilla/SamPO/DPOP/combinations)
    evaluate.py                  evaluate an arm vs the baseline
    run_decoupled_extension.sh   train+eval the decoupled arm, then aggregate
  analysis/
    aggregate_arms.py            aggregate metrics across arms and seeds
    make_figures.py              render the main result figures
    make_report_figures.py       standalone; recompute paper figures + stats
  configs/experiment.yaml        base config (model, data, DPO hyperparameters)
  experiments/                   the five per-arm configs (overrides on the base)
    vanilla_dpo.yaml sampo_dpo.yaml dpop.yaml sampo_dpop.yaml
    sampo_dpop_decoupled.yaml
  paper/figures/                 paper figures + report_stats.json
  results/                       finalized experimental results (no model weights)
    arms_summary.json            aggregated per-arm metrics (mean / SE over seeds)
    baseline/                    metrics.json + eval_responses.jsonl
    <arm>/seed{0,1,2}/           metrics.json, eval_responses.jsonl, train_metadata.json
    medical_redteam/             medical case-study summary + curated examples
```

- **`src/complexity_theater/`** — the package the scripts import: appearance and
  uncertainty metrics, the LLM judge, and the matched DPO loops that implement
  SamPO length debiasing and DPOP positive preservation.
- **`scripts/` and `analysis/`** — the pipeline used by the Quick Start and the
  notebook. Each arm differs only in the preference-loss components described in
  Section 3 of the paper.
- **`configs/experiment.yaml` + `experiments/`** — the base config and the five
  per-arm configs; the arm configs inherit the base via `base_config`.
- **`results/`** — the finalized outputs every number in the paper is computed
  from: per-arm/seed `metrics.json` and `eval_responses.jsonl`, the aggregated
  `arms_summary.json`, per-seed `train_metadata.json` (training dynamics), and the
  medical red-team artifacts. LoRA adapter weights are excluded to keep the
  package lightweight.
- **`mse338_project_runbook.ipynb`** — the end-to-end Colab notebook (clone →
  install → prepare → baseline → arms → aggregate → figures → download).

## Where the results live

`results/arms_summary.json` holds the aggregated per-arm metrics with standard errors. Per-response evidence for every arm and seed is in
`results/<arm>/seed*/eval_responses.jsonl`, and per-seed training diagnostics are in `train_metadata.json`. `paper/figures/report_stats.json` contains the exact numbers, the calibration cross-tab, and the paired p-values cited in the paper.

## What is excluded

- **LoRA adapter weights and tokenizer files** — large and unnecessary for reading
  the results or the compiled paper (`report.pdf`). Per-seed `train_metadata.json`
  is kept so the optimization dynamics remain inspectable.
- **Intermediate artifacts** (candidate generations, raw annotation CSVs) — the
  reported numbers are recomputed from `eval_responses.jsonl`.

## Limitations

- Three seeds per DPO arm; the baseline is a single evaluation of the un-tuned
  policy; a single dataset (TruthfulQA).
- Generations are capped near 200 tokens, which makes SamPO's equal-token subset
  approach the full sequence and compresses its effect.
- Factuality and calibration labels are AI-assisted (LLM-judged factuality plus a
  lexicon-based uncertainty score), so the calibration rates are exploratory.
- The medical red-team section is qualitative triage for human review, not a
  clinical evaluation; no medical recommendation should be inferred.
