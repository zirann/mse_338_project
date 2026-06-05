from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge `overlay` onto a copy of `base`. Overlay wins on leaves;
    nested dicts merge key-by-key. Non-dict values are replaced wholesale."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def read_arm_config(path: str | Path, base_key: str = "base_config") -> dict[str, Any]:
    """Read an arm YAML, resolving an optional `base_config` pointer.

    A thin per-arm config (e.g. `experiments/judge_dpo.yaml`) carries
    `base_config: configs/experiment.yaml` plus an `arm` block. This loads the
    base, deep-merges the arm file on top, and drops the `base_config` key. The
    base path is resolved relative to the repo root (two levels up from this
    module: src/complexity_theater/io_utils.py -> repo root) when not absolute.
    """
    p = Path(path)
    cfg = read_yaml(p)
    if base_key in cfg:
        base_path = Path(cfg[base_key])
        if not base_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            base_path = repo_root / base_path
        base_cfg = read_yaml(base_path)
        merged = _deep_merge(base_cfg, cfg)
        merged.pop(base_key, None)
        return merged
    return cfg


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
