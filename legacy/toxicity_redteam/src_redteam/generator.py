from __future__ import annotations

import random
import re
from typing import Any

import torch

from .control_tokens import build_control_tokens, build_user_prompt, render_chat_text


_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_THINK_PREFIX_RE = re.compile(r"^\s*<think>\s*", re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove leading <think> chain-of-thought wrappers.

    Returns "" if the output appears to be an unclosed think prefix (treat as invalid).
    """
    if not text:
        return ""
    t = text.strip()

    m = _THINK_BLOCK_RE.match(t)
    if m:
        return t[m.end() :].lstrip()

    if _THINK_PREFIX_RE.match(t) and "</think>" not in t.lower():
        return ""

    if _THINK_PREFIX_RE.match(t):
        return _THINK_PREFIX_RE.sub("", t, count=1).lstrip()

    return t


def _clean_candidate_text(text: str) -> str:
    t = strip_think(text)
    t = t.strip()
    for prefix in ("assistant:", "assistant", "user:", "user"):
        if t.lower().startswith(prefix):
            t = t[len(prefix) :].lstrip(" \n\t:-")
    t = " ".join(t.split())
    return t


def build_conditions(cfg: dict[str, Any]) -> list[dict[str, str]]:
    risk_labels = list(cfg.get("risk_labels", []))
    severities = list(cfg.get("severities", []))
    styles = list(cfg.get("styles", []))
    num_conditions = int(cfg.get("num_conditions", 60))
    seed = int(cfg.get("seed", 42))

    rng = random.Random(seed)
    combos = []
    for risk in risk_labels:
        for sev in severities:
            for style in styles:
                combos.append({"risk": risk, "severity": sev, "style": style})
    if not combos:
        return []

    rng.shuffle(combos)
    if num_conditions <= len(combos):
        return combos[:num_conditions]

    out = []
    while len(out) < num_conditions:
        out.extend(combos)
    return out[:num_conditions]


def _extract_seed_prompts(rows: list[dict], fallback: str = "Generate a high-risk adversarial sample.") -> list[str]:
    prompts = []
    for row in rows:
        p = (row.get("prompt") or "").strip()
        if p:
            prompts.append(p)
    if not prompts:
        prompts = [fallback]
    return prompts


def _safe_generate(model, **kwargs):
    try:
        return model.generate(**kwargs)
    except RuntimeError as err:
        if "probability tensor contains" not in str(err).lower():
            raise
        kwargs.pop("do_sample", None)
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        kwargs["do_sample"] = False
        return model.generate(**kwargs)


def generate_candidates(
    model,
    tokenizer,
    rows_for_prompts: list[dict],
    cfg: dict[str, Any],
) -> list[dict]:
    """Generate `candidates_per_condition` samples for each condition. Single decode pass.

    Empty/degenerate outputs are emitted as-is (with `candidate_text=""`); downstream code
    decides whether to drop them. We deliberately avoid oversample/resample loops so the
    generation distribution per condition stays clean and reproducible.
    """
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)
    decode_cfg = cfg.get("decode", {})
    k = int(cfg.get("candidates_per_condition", 4))

    conditions = build_conditions(cfg)
    seed_prompts = _extract_seed_prompts(rows_for_prompts)

    records: list[dict] = []
    device = next(model.parameters()).device

    bad_words = ["<think>", "</think>"]
    bad_words_ids = []
    for w in bad_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            bad_words_ids.append(ids)

    for cond_idx, cond in enumerate(conditions):
        seed_prompt = rng.choice(seed_prompts)
        user_prompt = build_user_prompt(
            risk=cond["risk"],
            severity=cond["severity"],
            style=cond["style"],
            seed_prompt=seed_prompt,
        )
        control = build_control_tokens(cond["risk"], cond["severity"], cond["style"])
        chat_text = render_chat_text(tokenizer, user_prompt=user_prompt, assistant_target=None)
        encoded = tokenizer(chat_text, return_tensors="pt", truncation=True)
        encoded = {k_: v.to(device) for k_, v in encoded.items()}
        input_len = int(encoded["input_ids"].shape[1])

        gen_kwargs = dict(
            **encoded,
            max_new_tokens=int(decode_cfg.get("max_new_tokens", 80)),
            min_new_tokens=int(decode_cfg.get("min_new_tokens", 0) or 0),
            do_sample=True,
            temperature=float(decode_cfg.get("temperature", 0.9)),
            top_p=float(decode_cfg.get("top_p", 0.95)),
            repetition_penalty=float(decode_cfg.get("repetition_penalty", 1.12)),
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
            remove_invalid_values=True,
            renormalize_logits=True,
        )
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids

        output = _safe_generate(model, **gen_kwargs)

        for i in range(output.shape[0]):
            gen_ids = output[i][input_len:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            cand = _clean_candidate_text(raw)
            records.append(
                {
                    "sample_id": f"cond_{cond_idx}_cand_{i}",
                    "input_prompt": user_prompt,
                    "control": control,
                    "candidate_text": cand,
                    "decode_params": decode_cfg,
                    "target_risk": cond["risk"],
                    "target_severity": cond["severity"],
                    "target_style": cond["style"],
                    "seed_prompt": seed_prompt,
                }
            )

    return records
