from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
)

from .control_tokens import render_chat_text
from .io_utils import ensure_dir
from .model_factory import apply_lora, load_base_causal_lm, load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_training_pairs(rows: list[dict], tokenizer) -> list[dict[str, str]]:
    """Build (prompt_text, full_text) pairs from accepted-candidate rows.

    Each input row must carry both `input_prompt` (the user message already containing
    control tokens + seed context) and `candidate_text` (the assistant target). Loss is
    masked to the assistant span by the tokenize step downstream.
    """
    pairs: list[dict[str, str]] = []
    for row in rows:
        user_prompt = (row.get("input_prompt") or "").strip()
        target = (row.get("candidate_text") or "").strip()
        if not user_prompt or not target:
            continue
        prompt_text = render_chat_text(tokenizer, user_prompt=user_prompt, assistant_target=None)
        full_text = render_chat_text(tokenizer, user_prompt=user_prompt, assistant_target=target)
        pairs.append({"prompt_text": prompt_text, "full_text": full_text})
    return pairs


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = int(batch["input_ids"].shape[1])
        padded = []
        for l in labels:
            l = list(l)[:max_len]
            if len(l) < max_len:
                l = l + ([-100] * (max_len - len(l)))
            padded.append(l)
        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch


def train_lora_sft(
    rows: list[dict],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    model_name = cfg["model_name"]
    require_model_name = str(cfg.get("require_model_name", model_name))
    if model_name != require_model_name:
        raise RuntimeError(f"Config requires model_name={require_model_name} but got {model_name}")
    output_dir = Path(cfg["output_dir"])
    max_train_samples = int(cfg.get("max_train_samples", 12000))
    max_length = int(cfg.get("max_length", 256))
    min_prompt_tokens = int(cfg.get("min_prompt_tokens", 32))

    training_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    device = cfg.get("device", "auto")

    from .model_factory import resolve_device

    device = resolve_device(device)

    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    tokenizer = load_tokenizer(model_name)
    model = load_base_causal_lm(model_name, device=device)
    model = apply_lora(model, lora_cfg)

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    peft_enabled = hasattr(model, "peft_config") or trainable_params > 0
    commit_hash = getattr(getattr(model, "config", None), "_commit_hash", None)

    # If LoRA did not attach to any modules, Trainer will later crash with:
    # "element 0 of tensors does not require grad".
    if trainable_params == 0:
        linear_suffixes: list[str] = []
        try:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_suffixes.append(name.split(".")[-1])
        except Exception:
            linear_suffixes = []
        suffix_preview = ", ".join(sorted(set(linear_suffixes))[:40]) if linear_suffixes else "(unavailable)"
        raise RuntimeError(
            "LoRA attach produced 0 trainable parameters. This usually means your "
            "`lora.target_modules` names do not match the model's module names. "
            f"Linear module name suffixes seen in this model (preview): {suffix_preview}"
        )

    pairs = build_training_pairs(rows, tokenizer)
    pairs = pairs[:max_train_samples]
    if not pairs:
        raise RuntimeError("No (input_prompt, candidate_text) rows available for training")

    dataset = Dataset.from_list(pairs)

    def tokenize_fn(batch: dict) -> dict:
        prompt_texts = batch["prompt_text"]
        full_texts = batch["full_text"]
        input_ids = []
        attention_mask = []
        labels = []
        for ptxt, ftxt in zip(prompt_texts, full_texts):
            # Tokenize without truncation, then enforce max_length while preserving response tokens.
            p_ids = tokenizer(ptxt, truncation=False, padding=False)["input_ids"]
            f_ids = tokenizer(ftxt, truncation=False, padding=False)["input_ids"]

            p_len = len(p_ids)
            resp_ids = f_ids[p_len:]
            if not resp_ids:
                # Nothing to learn on (e.g., response got lost due to formatting); skip.
                continue

            # Ensure response doesn't completely consume the context window.
            if len(resp_ids) > max_length - min_prompt_tokens:
                resp_ids = resp_ids[: max(1, (max_length - min_prompt_tokens))]

            # Truncate prompt from the left if needed to fit.
            keep_prompt = max(0, max_length - len(resp_ids))
            if len(p_ids) > keep_prompt:
                p_ids = p_ids[-keep_prompt:]

            ids = list(p_ids) + list(resp_ids)
            am = [1] * len(ids)
            lab = ([-100] * len(p_ids)) + list(resp_ids)

            # Labels must align with input_ids; response labels are the response token ids in-place.
            # (i.e., labels[t] == input_ids[t] for response span)
            if len(lab) != len(ids):
                continue
            input_ids.append(ids)
            attention_mask.append(am)
            labels.append(lab)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt_text", "full_text"])

    # With PEFT (base weights frozen) + gradient checkpointing, embeddings outputs may not
    # require grad, which breaks checkpointing and can lead to loss having no grad_fn.
    if bool(training_cfg.get("gradient_checkpointing", False)) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=float(training_cfg.get("epochs", 1.0)),
        per_device_train_batch_size=int(training_cfg.get("batch_size", 4)),
        gradient_accumulation_steps=int(training_cfg.get("grad_accum", 4)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        save_strategy=str(training_cfg.get("save_strategy", "epoch")),
        fp16=bool(training_cfg.get("fp16", device == "cuda")),
        bf16=bool(training_cfg.get("bf16", False)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
        tf32=bool(training_cfg.get("tf32", device == "cuda")),
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=CausalLMCollator(tokenizer),
    )

    if device in {"mps", "cuda"}:
        trainer.model.to(device)
    print(
        f"[diag] training: model device = {next(trainer.model.parameters()).device}, "
        f"trainer.args.device = {trainer.args.device}",
        flush=True,
    )

    trainer.train()

    ensure_dir(output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Best-effort package/version provenance (must not block training).
    try:
        from importlib.metadata import version as pkg_version

        versions = {
            "torch": pkg_version("torch"),
            "transformers": pkg_version("transformers"),
            "peft": pkg_version("peft"),
            "datasets": pkg_version("datasets"),
            "accelerate": pkg_version("accelerate"),
        }
    except Exception:
        versions = {}

    metadata = {
        "model_name": model_name,
        "require_model_name": require_model_name,
        "device": device,
        "num_rows_input": len(rows),
        "num_rows_training_text": len(pairs),
        "max_length": max_length,
        "min_prompt_tokens": min_prompt_tokens,
        "seed": seed,
        "lora": lora_cfg,
        "training": training_cfg,
        "loss_masking": {"assistant_only": True},
        "peft_enabled": bool(peft_enabled),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": (trainable_params / total_params) if total_params else 0.0,
        "hf_commit_hash": commit_hash,
        "versions": versions,
    }
    with (output_dir / "train_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    loss_curve_path = output_dir / "loss_curve.csv"
    with loss_curve_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "learning_rate", "epoch"])
        writer.writeheader()
        for item in trainer.state.log_history:
            if "loss" in item:
                writer.writerow(
                    {
                        "step": item.get("step"),
                        "loss": item.get("loss"),
                        "learning_rate": item.get("learning_rate"),
                        "epoch": item.get("epoch"),
                    }
                )

    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "train_rows": len(pairs),
        "loss_curve": str(loss_curve_path),
    }
