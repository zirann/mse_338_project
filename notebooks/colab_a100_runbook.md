# Colab Enterprise / A100 runbook: Length-Controlled DPO + DPOP

Minimal-compute, staged pipeline. Single NVIDIA A100. All five DPO arms run
through one standalone matched loop (`src/complexity_theater/regularized_dpo.py`)
and differ ONLY by `{length_debias, dpop_lambda}`:

- vanilla_dpo:  length_debias=none,  dpop_lambda=0   (reproduction baseline)
- sampo_dpo:    length_debias=sampo, dpop_lambda=0   (SamPO length control)
- dpop:         length_debias=none,  dpop_lambda>0   (DPOP)
- sampo_dpop:   length_debias=sampo, dpop_lambda>0   (length-controlled DPOP)

Seed 0 first for every arm; expand to seeds 1-2 only if seed 0 is promising.

## Setup

```bash
cd /content
git clone <YOUR_REPO_URL> dpo_project && cd dpo_project
pip install --upgrade pip && pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Stage 0 - archive failed prior runs (already done in-repo; re-run only if needed)

```bash
mkdir -p archive_failed_runs archive_configs
# (outputs_old/, results_bundle_1/ -> archive_failed_runs/; controls,mitigations,experiments/judge_dpo.yaml -> archive_configs/)
```

## Stage 1 - prepare data + baseline eval

```bash
python scripts/prepare.py  --config configs/experiment.yaml --limit 160          # 80 train / 80 eval
python scripts/evaluate.py --config configs/experiment.yaml --round 0 --limit 80 --out_dir outputs/baseline
```

Expected: `outputs/data/{train_prompts(80),eval_prompts(80)}.jsonl`, `outputs/baseline/{eval_responses.jsonl,metrics.json}`. ~9 min.

## Stage 2 - vanilla DPO baseline-stability smoke (seed 0) -- GATE

The whole study is gated on this. The earlier 5e-5 / 1-epoch / grad-accum-2
setting under-trained; the stable defaults are now in `configs/experiment.yaml`
(lr 1e-4, 3 epochs, grad-accum 1, beta 0.1, batch 4 -> ~60 updates on ~80 pairs).

```bash
python scripts/train_round.py --config experiments/vanilla_dpo.yaml --round 1 --limit 80 --seed 0
python scripts/evaluate.py    --config experiments/vanilla_dpo.yaml --round 1 --limit 80 --seed 0 \
    --out_dir outputs/vanilla_dpo/seed0 --baseline_dir outputs/baseline
```

Inspect `outputs/vanilla_dpo/seed0/adapter/train_metadata.json`:
- `train_curve` (per-step list): loss should TREND DOWN, `rewards_margins` UP.
- `loss_last` and `loss_first`: `loss_last` clearly below `ln2 = 0.693`.
- `rewards_margins_final`, `rewards_margins_max`.

And `outputs/vanilla_dpo/seed0/metrics.json` vs `outputs/baseline/metrics.json`:
- `length`, `hedge_density`, `uncertainty_score`, `factuality`, `judge_win_rate_vs_round_0`.

ACCEPTANCE (ALL must hold to proceed):
- `loss_last` < 0.66 and `train_curve` visibly descending.
- `rewards_margins_final` > 0.03 (ideally > 0.05).
- `judge_win_rate_vs_round_0` > 0.55.
- Outputs coherent; no length explosion; eval not obviously overfit.

OVERFIT / COLLAPSE warning signs (treat as fail):
- train loss drops but eval win-rate flat -> overfit to the pairs.
- response length explodes; repetition / incoherent text.
- factuality craters.
- any hedge/uncertainty change explained purely by length (always read `length` next to `hedge_density`).

IF FAIL: adjust ONE knob at a time and re-run Stage 2, do NOT proceed:
1. `--length_debias none` kept; raise epochs 3 -> 5 (edit configs/experiment.yaml dpo.num_train_epochs).
2. then lr 1e-4 -> 1.5e-4.
3. then `--limit 200` (re-run Stage 1 prepare with `--limit 400`).
Optional parity sanity check (one-off): rerun vanilla with `--trainer trl` into a
scratch `--out_dir` and compare loss/margins to the local loop.

## Stage 3 - run SamPO / DPOP / SamPO+DPOP (seed 0) -- only if Stage 2 PASSES

```bash
for ARM in sampo_dpo dpop sampo_dpop; do
  python scripts/train_round.py --config experiments/$ARM.yaml --round 1 --limit 80 --seed 0
  python scripts/evaluate.py    --config experiments/$ARM.yaml --round 1 --limit 80 --seed 0 \
      --out_dir outputs/$ARM/seed0 --baseline_dir outputs/baseline
done
```

DPOP lambda check (arms dpop, sampo_dpop): in each `adapter/train_metadata.json`
confirm `mean_dpop_penalty` > 0 (the term is active) AND `rewards_margins_final`
stays positive with `loss_last` < ln2 (DPOP not overpowering DPO). If margins go
negative or loss stalls at ln2, halve `dpop_lambda` (0.5 -> 0.25) in
`experiments/dpop.yaml` and `experiments/sampo_dpop.yaml` and rerun those two arms.
SamPO check (arms sampo_*): `mean_tokens_used_chosen` ~= `mean_tokens_used_rejected`
(token counts equalized).

## Stage 4 - aggregate + figures (CPU; seconds)

```bash
python analysis/aggregate_arms.py --config configs/experiment.yaml --seeds 0
python analysis/make_figures.py
```

Outputs: `outputs/arms_summary.json`; `figures/{fig1_length, fig2_reproduce_hedge, fig3_extend_hedge, fig4_winrate}.png`.

## Seeds 1-2 (only if seed 0 is promising)

Repeat Stages 2-3 with `--seed 1` and `--seed 2` (each arm), then:

```bash
python analysis/aggregate_arms.py --config configs/experiment.yaml --seeds 0 1 2
python analysis/make_figures.py
```

## Runtime (A100, seed 0)

prepare ~1 min; baseline eval ~8 min; each DPO arm ~15 min train + ~8 min eval;
4 trained arms ~ 90 min; aggregate ~1 min. Full seed-0 pass under ~2 hours.

## Failure recovery

- One arm/seed failed: re-run just that train+eval pair; arm/seed dirs are isolated.
- OOM: the loop holds policy + frozen reference (two 0.6B models). Lower
  `dpo.per_device_train_batch_size` to 2 in `configs/experiment.yaml`.
- SamPO sanity-check failed (token counts unequal): confirm `length_debias=sampo`
  resolved (printed at train start) and that completions are non-empty.

## What to download locally

Per arm/seed: `metrics.json`, `adapter/train_metadata.json`, `eval_responses.jsonl`,
`preference_diagnostics.json`, `winrate_pairs.jsonl`; plus `outputs/arms_summary.json`
and `figures/*.png`. Skip the LoRA weight files unless resuming training.

## Deferred (gated) - medical red-team demo

Only after the 5-arm seed-0 matrix passes and time/compute remain: a small
qualitative pass over already-generated `eval_responses.jsonl` flagging
overconfidence risk = incorrect/unsafe + low uncertainty + decisive wording.
Application case study, not a second experiment. No new metrics.
