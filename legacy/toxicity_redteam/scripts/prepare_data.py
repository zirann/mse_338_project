#!/usr/bin/env python3
"""Prepare seed prompts (toxigen + real-toxicity-prompts) for the generator.

Output: outputs/data/{train,val,test}.jsonl with rows `{id, source, prompt, meta}`.
These JSONLs are NOT used for SFT labels; they only provide realistic seed prompts
fed to `generate_candidates` via the `rows_for_prompts` argument.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from redteam.data_pipeline import DataPipeline
from redteam.io_utils import read_yaml, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare seed prompts")
    parser.add_argument("--config", default="configs/data.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = read_yaml(ROOT / args.config)

    output_dir = ROOT / data_cfg["output_dir"]
    pipeline = DataPipeline(data_cfg=data_cfg, output_dir=output_dir)
    result = pipeline.prepare()

    write_json(output_dir / "run_metadata_prepare.json", {"config": args.config, "result": result})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
