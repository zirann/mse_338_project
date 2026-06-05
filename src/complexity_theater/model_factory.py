from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device_arg: str = "auto") -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_causal_lm(model_name: str, device: str):
    dtype = None
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    return model


def apply_lora(model, lora_cfg: dict):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", [])),
        bias="none",
    )
    model = get_peft_model(model, config)
    return model


def load_model_with_optional_adapter(
    model_name: str,
    adapter_path: str | None,
    device: str,
):
    model = load_base_causal_lm(model_name, device=device)
    if adapter_path:
        ap = Path(adapter_path)
        if ap.exists():
            model = PeftModel.from_pretrained(model, str(ap))
    model.to(device)
    model.eval()
    return model
