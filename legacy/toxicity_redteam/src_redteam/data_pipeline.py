from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .io_utils import write_json, write_jsonl


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")


class DataPipeline:
    """Minimal seed-prompt pipeline for the evaluator reward-hacking study.

    Two sources only (toxigen, real_toxicity_prompts). The new research loop does NOT
    SFT on public labels; we only need realistic seed prompts to feed `generate_candidates`.
    Each output row carries just `{id, source, prompt}` plus a small `meta` blob.
    """

    def __init__(
        self,
        data_cfg: dict[str, Any],
        output_dir: Path,
    ) -> None:
        self.data_cfg = data_cfg
        self.output_dir = output_dir
        self.seed = int(data_cfg.get("seed", 42))

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.replace("\n", " ").strip()
        text = " ".join(text.split())
        text = EMAIL_RE.sub("[EMAIL]", text)
        text = PHONE_RE.sub("[PHONE]", text)
        return text

    def _map_toxigen(self, row: dict, source_id: str, idx: int) -> dict | None:
        prompt = self._normalize_text(row.get("prompt", ""))
        if not prompt:
            return None
        return {
            "id": f"{source_id}:{idx}",
            "source": source_id,
            "prompt": prompt,
            "meta": {"group": row.get("group", "")},
        }

    def _map_rtp(self, row: dict, source_id: str, idx: int) -> dict | None:
        prompt_obj = row.get("prompt") or {}
        prompt = self._normalize_text(prompt_obj.get("text", ""))
        if not prompt:
            return None
        return {
            "id": f"{source_id}:{idx}",
            "source": source_id,
            "prompt": prompt,
            "meta": {
                "toxicity": float(prompt_obj.get("toxicity") or 0.0),
            },
        }

    def _map_row(self, source_id: str, row: dict, idx: int) -> dict | None:
        if source_id == "toxigen_train":
            return self._map_toxigen(row, source_id, idx)
        if source_id == "real_toxicity_prompts":
            return self._map_rtp(row, source_id, idx)
        return None

    def _load_rows(self, source_cfg: dict) -> tuple[list[dict], str | None]:
        name = source_cfg["hf_name"]
        config = source_cfg.get("config")
        split = source_cfg.get("split", "train")
        try:
            ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
            return list(ds), None
        except Exception as e:
            return [], f"{type(e).__name__}: {e}"

    def prepare(self) -> dict[str, Any]:
        rng = random.Random(self.seed)
        manifest: dict[str, Any] = {"seed": self.seed, "sources": []}
        rows: list[dict[str, Any]] = []
        max_samples = int(self.data_cfg.get("max_samples_per_source", 4000))

        for source_cfg in self.data_cfg.get("datasets", []):
            if not source_cfg.get("enabled", True):
                continue
            source_id = source_cfg["id"]
            loaded, err = self._load_rows(source_cfg)
            source_meta = {
                "id": source_id,
                "hf_name": source_cfg["hf_name"],
                "config": source_cfg.get("config"),
                "status": "ok" if err is None else "failed",
                "error": err,
                "loaded_rows": len(loaded),
                "used_rows": 0,
            }
            if err is not None:
                manifest["sources"].append(source_meta)
                continue
            for idx, row in enumerate(loaded[:max_samples]):
                mapped = self._map_row(source_id, row, idx)
                if mapped is not None and mapped.get("prompt"):
                    rows.append(mapped)
            source_meta["used_rows"] = len([r for r in rows if r["source"] == source_id])
            manifest["sources"].append(source_meta)

        if not rows:
            raise RuntimeError(
                "No seed prompts available from any configured source. "
                "Check HF dataset access or adjust configs/data.yaml."
            )

        dedup: dict[str, dict] = {}
        for row in rows:
            key = row["prompt"].lower()
            if key and key not in dedup:
                dedup[key] = row
        rows = list(dedup.values())

        rng.shuffle(rows)
        n = len(rows)
        split_cfg = self.data_cfg.get("split", {})
        n_train = int(n * float(split_cfg.get("train", 0.9)))
        n_val = int(n * float(split_cfg.get("val", 0.05)))
        train_rows = rows[:n_train]
        val_rows = rows[n_train : n_train + n_val]
        test_rows = rows[n_train + n_val :]

        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "train.jsonl", train_rows)
        write_jsonl(out_dir / "val.jsonl", val_rows)
        write_jsonl(out_dir / "test.jsonl", test_rows)

        source_dist = Counter(r["source"] for r in rows)
        stats = {
            "total_rows": n,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "source": dict(source_dist),
        }

        write_json(out_dir / "data_manifest.json", manifest)
        write_json(out_dir / "label_distribution.json", stats)

        return {
            "train_path": str(out_dir / "train.jsonl"),
            "val_path": str(out_dir / "val.jsonl"),
            "test_path": str(out_dir / "test.jsonl"),
            "manifest_path": str(out_dir / "data_manifest.json"),
            "label_distribution_path": str(out_dir / "label_distribution.json"),
            "stats": stats,
        }
