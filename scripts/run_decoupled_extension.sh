#!/usr/bin/env bash
# Additive extension run: decoupled-support SamPO+DPOP (sampo_dpop_decoupled).
#
# Trains/evaluates ONLY the new arm for seeds 0,1,2 against the existing
# outputs/baseline, then re-aggregates and re-renders figures. Does NOT touch,
# rerun, archive, or delete any existing arm outputs.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
CONFIG="experiments/sampo_dpop_decoupled.yaml"
ARM_DIR="outputs/sampo_dpop_decoupled"
SEEDS=(0 1 2)
LIMIT=80

echo "== Preflight checks =="
if [ ! -f "outputs/baseline/eval_responses.jsonl" ]; then
  echo "ERROR: outputs/baseline/eval_responses.jsonl not found. Run Stage 1 (prepare + baseline) first." >&2
  exit 1
fi
# Existing arms should be present (we compare against them in analysis); warn if missing, do not fail.
for arm in vanilla_dpo sampo_dpo dpop sampo_dpop; do
  if [ ! -d "outputs/$arm" ]; then
    echo "WARN: outputs/$arm missing; analysis will still run but cross-arm comparison will be partial." >&2
  fi
done
if [ ! -f "outputs/data/train_prompts.jsonl" ]; then
  echo "ERROR: outputs/data/train_prompts.jsonl not found. Run scripts/prepare.py first." >&2
  exit 1
fi

echo "== Train + eval new arm only: sampo_dpop_decoupled (seeds ${SEEDS[*]}) =="
for S in "${SEEDS[@]}"; do
  echo "--- seed $S: train ---"
  python scripts/train_round.py --config "$CONFIG" --round 1 --limit "$LIMIT" --seed "$S"
  echo "--- seed $S: eval vs baseline ---"
  python scripts/evaluate.py --config "$CONFIG" --round 1 --limit "$LIMIT" --seed "$S" \
      --out_dir "$ARM_DIR/seed$S" --baseline_dir outputs/baseline
done

echo "== Aggregate (all arms incl. new) + figures =="
python analysis/aggregate_arms.py --config configs/experiment.yaml --seeds "${SEEDS[@]}"
python analysis/make_figures.py

echo "== Done =="
echo "New arm outputs : $ARM_DIR/seed{0,1,2}/ (metrics.json, eval_responses.jsonl, adapter/train_metadata.json)"
echo "Aggregated table: outputs/arms_summary.json"
echo "Figures         : figures/fig1_length.png, fig2_reproduce_hedge.png, fig3_extend_hedge.png, fig4_winrate.png"
echo "Inspect first   : outputs/sampo_dpop_decoupled/seed0/adapter/train_metadata.json (dpop_support=full_chosen, mean_dpop_penalty, train_loss, rewards_margins_final)"
