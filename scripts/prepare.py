#!/usr/bin/env python3
"""Load TruthfulQA validation, sample train + eval splits, write JSONL.

Outputs `outputs/data/train_prompts.jsonl` and `outputs/data/eval_prompts.jsonl`.
Each row: `{id, question, correct_reference, incorrect_reference, category}`.

Train and eval are disjoint and sampled with the seed in `configs/experiment.yaml`.
The optional category whitelist drops ambiguous prompts (Risk R8 in the plan).

`--limit N` overrides both sizes for smoke tests: roughly half of N goes to
train, the rest to eval, with a minimum of 2 prompts each.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexity_theater.io_utils import read_yaml, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare TruthfulQA splits.")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--limit", type=int, default=None, help="Total prompt count for smoke tests.")
    return p.parse_args()


def _convert_row(row: dict, idx: int) -> dict:
    """Map a TruthfulQA row to our per-prompt schema."""
    correct = row.get("best_answer") or ""
    if not correct:
        correct_list = row.get("correct_answers") or []
        if correct_list:
            correct = correct_list[0]
    incorrect_list = row.get("incorrect_answers") or []
    incorrect = incorrect_list[0] if incorrect_list else ""
    category = row.get("category") or "na"
    return {
        "id": f"{category}_{idx}",
        "question": row.get("question") or "",
        "correct_reference": correct,
        "incorrect_reference": incorrect,
        "category": category,
    }


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)

    seed = int(cfg.get("seed", 42))
    dataset_cfg = cfg["dataset"]
    n_train = int(dataset_cfg.get("n_train", 80))
    n_eval = int(dataset_cfg.get("n_eval", 40))

    if args.limit is not None:
        half = max(2, args.limit // 2)
        n_train = half
        n_eval = max(2, args.limit - half)

    print(f"[prepare] dataset={dataset_cfg.get('name')} split={dataset_cfg.get('split')} "
          f"n_train={n_train} n_eval={n_eval} seed={seed}")

    # Load TruthfulQA generation config from HF Datasets.
    from datasets import load_dataset
    ds = load_dataset(
        dataset_cfg.get("name", "truthful_qa"),
        "generation",
        split=dataset_cfg.get("split", "validation"),
    )
    rows = list(ds)
    print(f"[prepare] loaded {len(rows)} rows from {dataset_cfg.get('name')}")

    whitelist = set(dataset_cfg.get("category_whitelist") or [])
    if whitelist:
        rows = [r for r in rows if (r.get("category") or "") in whitelist]
        print(f"[prepare] after category whitelist: {len(rows)} rows")

    needed = n_train + n_eval
    if len(rows) < needed:
        raise SystemExit(
            f"[prepare] insufficient rows after whitelist: have {len(rows)}, need {needed}. "
            f"Either widen the category whitelist or lower n_train + n_eval / --limit."
        )

    rng = random.Random(seed)
    rng.shuffle(rows)

    train_rows = [_convert_row(r, i) for i, r in enumerate(rows[:n_train])]
    eval_rows = [_convert_row(r, n_train + i) for i, r in enumerate(rows[n_train : n_train + n_eval])]

    data_dir = ROOT / cfg["outputs"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train_prompts.jsonl"
    eval_path = data_dir / "eval_prompts.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    print(f"[prepare] wrote {train_path} ({len(train_rows)} rows)")
    print(f"[prepare] wrote {eval_path} ({len(eval_rows)} rows)")


if __name__ == "__main__":
    main()
