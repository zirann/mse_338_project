"""Thin wrapper around `trl.DPOTrainer` for one preference-optimization round.

LoRA on Qwen3-0.6B, beta=0.1, 1 epoch, ~80 preference pairs per round. The
reference policy for round N is the round-(N-1) adapter (iterated DPO);
round 1's reference is the base model.

Two modes:

- `mock=True`: skip real training; write a placeholder
  `<output_adapter_dir>/train_metadata.json` so subsequent scripts can detect
  that the round "happened". Used by local CPU/MPS smoke tests where TRL +
  Qwen3 + LoRA is too heavy to exercise meaningfully.
- `mock=False`: real DPO via `trl.DPOTrainer`. Targets `trl >= 0.11`; falls
  back to attribute checks where the TRL API shifts between minor versions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_HPARAMS: dict[str, Any] = {
    "beta": 0.1,
    "learning_rate": 5e-5,
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "max_length": 512,
    "max_prompt_length": 256,
    "abort_kl_threshold": 5.0,
    # Saturation guard: a run that drives the implicit-reward margin this high
    # or the loss this low has over-optimized / memorized the pairs, which
    # breaks the matched-control identification (the optimization-intrinsic
    # drift is no longer common across arms). Such a run fails loud AFTER its
    # adapter + train_metadata.json are written, so the curve is inspectable.
    "saturation_rewards_margins_max": 3.0,
    "saturation_loss_min": 0.05,
}

DEFAULT_LORA: dict[str, Any] = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
}


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _lora_param_norm(model) -> float:
    """Sum-of-squares norm of LoRA A/B parameters in `model`.

    Fresh-init LoRA has B = 0 and A ~ Gaussian, so this is non-zero before
    training. The delta (final - initial) is the cleanest scalar indicator
    that the LoRA actually moved.
    """
    total_sq = 0.0
    for name, p in model.named_parameters():
        if "lora_" in name.lower():
            total_sq += float(p.detach().pow(2).sum().item())
    return total_sq ** 0.5


def _count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def run_one_dpo_round(
    base_model_name: str,
    reference_adapter_path: Path | str | None,
    preference_pairs_jsonl: Path | str,
    output_adapter_dir: Path | str,
    hparams: dict[str, Any] | None = None,
    lora_config: dict[str, Any] | None = None,
    seed: int = 42,
    mock: bool = False,
) -> dict[str, Any]:
    """Run one DPO round; save the new LoRA adapter; return metadata."""
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    lora = {**DEFAULT_LORA, **(lora_config or {})}
    pairs_path = Path(preference_pairs_jsonl)
    adapter_dir = Path(output_adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    num_pairs = _count_lines(pairs_path)

    if mock:
        metadata = {
            "status": "mock",
            "base_model_name": base_model_name,
            "reference_adapter_path": str(reference_adapter_path) if reference_adapter_path else None,
            "num_pairs": num_pairs,
            "hparams": hp,
            "lora": lora,
            "seed": seed,
        }
        with (adapter_dir / "train_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return {
            "adapter_path": str(adapter_dir),
            "train_loss": float("nan"),
            "final_kl": 0.0,
            "num_pairs": num_pairs,
            "hparams": hp,
            "mock": True,
        }

    # Real DPO path. Deferred imports keep mock-mode lightweight.
    import torch  # noqa: F401
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    # Tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model + reference adapter.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if reference_adapter_path is not None:
        ref_path = Path(reference_adapter_path)
        if ref_path.exists():
            model = PeftModel.from_pretrained(model, str(ref_path))
            # Merge the reference adapter so subsequent LoRA is fresh on top.
            try:
                model = model.merge_and_unload()
            except Exception:
                # Some PEFT versions return the merged model directly; some do not.
                pass

    # Apply a fresh LoRA on top for this round's update.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=list(lora["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    trainable_params, total_params = _count_trainable(model)
    lora_norm_initial = _lora_param_norm(model)
    print(
        f"[dpo] trainable={trainable_params:,} / total={total_params:,} "
        f"({100.0 * trainable_params / max(total_params, 1):.4f}%); "
        f"initial LoRA norm={lora_norm_initial:.4f}"
    )

    # Preference dataset.
    pairs = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pairs.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    if not pairs:
        raise RuntimeError(f"No preference pairs at {pairs_path}")
    dataset = Dataset.from_list(pairs)

    # DPOConfig. Field names differ slightly across trl 0.7 -> 0.13; we set
    # the common ones and rely on TRL to ignore unknown kwargs when applicable.
    dpo_kwargs = dict(
        output_dir=str(adapter_dir),
        beta=hp["beta"],
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        max_length=hp["max_length"],
        max_prompt_length=hp["max_prompt_length"],
        seed=seed,
        report_to=[],
        save_strategy="no",
        logging_steps=1,
        remove_unused_columns=False,
    )
    try:
        dpo_config = DPOConfig(**dpo_kwargs)
    except TypeError:
        # Drop kwargs that newer/older TRL releases reject; retry.
        for k in ("max_length", "max_prompt_length", "remove_unused_columns"):
            dpo_kwargs.pop(k, None)
        dpo_config = DPOConfig(**dpo_kwargs)

    # Trainer. The tokenizer kwarg was renamed `processing_class` in newer TRL.
    try:
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

    train_result = trainer.train()

    lora_norm_final = _lora_param_norm(model)
    lora_norm_delta = lora_norm_final - lora_norm_initial
    print(
        f"[dpo] final LoRA norm={lora_norm_final:.4f} "
        f"(delta={lora_norm_delta:+.4f})"
    )

    # Save adapter + tokenizer.
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Extract final-step metrics best-effort.
    train_loss = None
    final_kl = None
    last_rewards_chosen = None
    last_rewards_rejected = None
    last_rewards_margins = None
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if train_loss is None and "loss" in entry:
                train_loss = float(entry["loss"])
            for key in ("kl", "policy_kl", "reward_kl"):
                if key in entry:
                    final_kl = float(entry[key])
                    break
            if last_rewards_chosen is None and "rewards/chosen" in entry:
                last_rewards_chosen = float(entry["rewards/chosen"])
            if last_rewards_rejected is None and "rewards/rejected" in entry:
                last_rewards_rejected = float(entry["rewards/rejected"])
            if last_rewards_margins is None and "rewards/margins" in entry:
                last_rewards_margins = float(entry["rewards/margins"])
            done = (
                train_loss is not None
                and final_kl is not None
                and last_rewards_margins is not None
            )
            if done:
                break
    if hasattr(train_result, "training_loss") and train_loss is None:
        try:
            train_loss = float(train_result.training_loss)
        except Exception:
            pass

    # Per-step trajectory so we can confirm the policy actually moved (loss
    # decreasing, margins rising) rather than only inspecting the final value.
    train_curve = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            train_curve.append(
                {
                    "step": entry.get("step"),
                    "loss": float(entry["loss"]),
                    "rewards_margins": (
                        float(entry["rewards/margins"]) if "rewards/margins" in entry else None
                    ),
                }
            )
    loss_first = train_curve[0]["loss"] if train_curve else None
    loss_last = train_curve[-1]["loss"] if train_curve else None
    _margin_vals = [c["rewards_margins"] for c in train_curve if c["rewards_margins"] is not None]
    rewards_margins_max = max(_margin_vals) if _margin_vals else None
    rewards_margins_final = _margin_vals[-1] if _margin_vals else None
    print(
        f"[dpo] curve: {len(train_curve)} logged steps; "
        f"loss {loss_first} -> {loss_last}; margins max={rewards_margins_max}"
    )

    # KL abort sanity check (only when TRL actually reported it).
    if final_kl is not None and math.isfinite(final_kl):
        if final_kl > float(hp.get("abort_kl_threshold", 5.0)):
            raise RuntimeError(
                f"DPO KL ({final_kl:.3f}) exceeded abort threshold "
                f"({hp.get('abort_kl_threshold', 5.0):.3f}); adapter saved but flagged."
            )

    metadata = {
        "status": "ok",
        "base_model_name": base_model_name,
        "reference_adapter_path": str(reference_adapter_path) if reference_adapter_path else None,
        "num_pairs": len(pairs),
        "hparams": hp,
        "lora": lora,
        "seed": seed,
        "train_loss": train_loss,
        "final_kl": final_kl,  # None when TRL did not report a KL key
        "rewards_chosen": last_rewards_chosen,
        "rewards_rejected": last_rewards_rejected,
        "rewards_margins": last_rewards_margins,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "lora_norm_initial": lora_norm_initial,
        "lora_norm_final": lora_norm_final,
        "lora_norm_delta": lora_norm_delta,
        "loss_first": loss_first,
        "loss_last": loss_last,
        "rewards_margins_max": rewards_margins_max,
        "rewards_margins_final": rewards_margins_final,
        "num_logged_steps": len(train_curve),
        "train_curve": train_curve,
    }
    with (adapter_dir / "train_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Fail-loud saturation guard. Checked AFTER metadata is written so the
    # train_curve survives for inspection. A saturated run is not a valid
    # member of the matched-control design (see identification note in
    # DEFAULT_HPARAMS).
    sat_margin = float(hp.get("saturation_rewards_margins_max", 3.0))
    sat_loss = float(hp.get("saturation_loss_min", 0.05))
    margin_sat = rewards_margins_final is not None and rewards_margins_final > sat_margin
    loss_sat = loss_last is not None and loss_last < sat_loss
    if margin_sat or loss_sat:
        raise RuntimeError(
            f"DPO saturation guard tripped (adapter + metadata saved at {adapter_dir}): "
            f"rewards_margins_final={rewards_margins_final} (max {sat_margin}), "
            f"loss_last={loss_last} (min {sat_loss}). This run over-optimized and is "
            f"not a valid matched-control arm; reduce epochs/lr."
        )

    return {
        "adapter_path": str(adapter_dir),
        "train_loss": train_loss,
        "final_kl": final_kl,
        "rewards_margins": last_rewards_margins,
        "num_pairs": len(pairs),
        "trainable_params": trainable_params,
        "lora_norm_delta": lora_norm_delta,
        "hparams": hp,
        "mock": False,
    }
