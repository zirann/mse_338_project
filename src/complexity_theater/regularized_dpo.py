"""Standalone minimal DPO + uncertainty-preservation penalty (Mitigation 2).

This module implements DPO from scratch (no `trl.DPOTrainer`) so we have full
control over an auxiliary uncertainty-preservation term added to the loss:

    L = L_DPO + lambda * penalty

The pure-math functions (`dpo_loss`, `penalty_a_mass_floor`,
`penalty_b_chosen_hedge_logprob`, `penalty_c_entropy_floor`) operate on plain
tensors and are unit-tested without loading any model. The training entry point
`run_one_regularized_dpo_round(...)` mirrors the signature and metadata contract
of `complexity_theater.dpo.run_one_dpo_round` (including a `mock=True` path) so
it drops into the existing runner.

Penalty formulations (see paper Methods):

- A (default) reference-anchored one-sided uncertainty-token-mass floor:
    penalty = mean_t ReLU( m_ref(t) - m_theta(t) ),  m(t) = sum_{v in U} p(v | y_<t)
  Only penalizes the policy for emitting LESS uncertainty-token mass than the
  frozen reference; never forces extra hedging. Anchored to the base
  distribution we want to preserve.
- B chosen hedge-token log-prob preservation:
    penalty = - mean_{t in H} log pi_theta(y_t | y_<t),  H = chosen positions with token in U
- C predictive-entropy floor (ablation; generic, not uncertainty-specific):
    penalty = mean_t ReLU( H_ref(t) - H_theta(t) )

U is the set of uncertainty-token ids built from `uncertainty.UNCERTAINTY_LEXICON_PHRASES`
(bag-of-subword-ids approximation; documented in `build_uncertainty_token_ids`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dpo import DEFAULT_HPARAMS, DEFAULT_LORA, _count_lines


DEFAULT_REG: dict[str, Any] = {
    "reg_formulation": "A",   # A | B | C
    "reg_lambda": 0.2,
}


# ---------------------------------------------------------------------------
# Pure math (unit-tested without a model)
# ---------------------------------------------------------------------------

def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float,
):
    """Standard DPO loss + reward statistics on sequence log-probabilities.

    All inputs are 1-D tensors of shape [batch]. Returns
    (loss_scalar, stats_dict).
    """
    import torch
    import torch.nn.functional as F

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    stats = {
        "rewards_chosen": float(chosen_rewards.mean().item()),
        "rewards_rejected": float(rejected_rewards.mean().item()),
        "rewards_margins": float((chosen_rewards - rejected_rewards).mean().item()),
        "rewards_accuracy": float((chosen_rewards > rejected_rewards).float().mean().item()),
    }
    return loss, stats


def _shift(logits, input_ids, completion_mask):
    """Shift for next-token prediction. Returns (shift_logits, shift_labels,
    shift_mask) where shift_mask marks positions whose predicted token is part
    of the completion."""
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:]
    return shift_logits, shift_labels, shift_mask


def sequence_logps(logits, input_ids, completion_mask):
    """Sum of per-token log-probabilities over completion positions. [batch]."""
    import torch
    import torch.nn.functional as F

    shift_logits, shift_labels, shift_mask = _shift(logits, input_ids, completion_mask)
    logp = F.log_softmax(shift_logits, dim=-1)
    tok_logp = torch.gather(logp, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * shift_mask).sum(dim=-1)


def penalty_a_mass_floor(policy_logits, ref_logits, input_ids, completion_mask, uncertainty_ids):
    """Formulation A: mean over completion positions of ReLU(m_ref - m_theta),
    where m(t) is the next-token probability mass on the uncertainty-token set.
    One-sided (only penalizes suppression below the reference)."""
    import torch

    pol_shift, _, shift_mask = _shift(policy_logits, input_ids, completion_mask)
    ref_shift, _, _ = _shift(ref_logits, input_ids, completion_mask)
    p_theta = torch.softmax(pol_shift, dim=-1)
    p_ref = torch.softmax(ref_shift, dim=-1)
    u = uncertainty_ids.to(p_theta.device)
    m_theta = p_theta.index_select(-1, u).sum(dim=-1)
    m_ref = p_ref.index_select(-1, u).sum(dim=-1)
    floor = torch.relu(m_ref - m_theta) * shift_mask
    denom = shift_mask.sum().clamp(min=1.0)
    return floor.sum() / denom


def penalty_b_chosen_hedge_logprob(policy_logits, input_ids, completion_mask, uncertainty_ids):
    """Formulation B: -mean of policy token log-prob at completion positions
    whose target token is an uncertainty token. Preserves the probability of
    the hedge tokens that already appear in the chosen response."""
    import torch
    import torch.nn.functional as F

    shift_logits, shift_labels, shift_mask = _shift(policy_logits, input_ids, completion_mask)
    logp = F.log_softmax(shift_logits, dim=-1)
    tok_logp = torch.gather(logp, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    u = uncertainty_ids.to(shift_labels.device)
    is_unc = torch.isin(shift_labels, u).float() * shift_mask
    denom = is_unc.sum().clamp(min=1.0)
    return -(tok_logp * is_unc).sum() / denom


def penalty_c_entropy_floor(policy_logits, ref_logits, input_ids, completion_mask):
    """Formulation C (ablation): mean over completion positions of
    ReLU(H_ref - H_theta), predictive-entropy floor. Generic; not specific to
    verbalized uncertainty."""
    import torch
    import torch.nn.functional as F

    pol_shift, _, shift_mask = _shift(policy_logits, input_ids, completion_mask)
    ref_shift, _, _ = _shift(ref_logits, input_ids, completion_mask)
    p_theta = torch.softmax(pol_shift, dim=-1)
    lp_theta = F.log_softmax(pol_shift, dim=-1)
    p_ref = torch.softmax(ref_shift, dim=-1)
    lp_ref = F.log_softmax(ref_shift, dim=-1)
    h_theta = -(p_theta * lp_theta).sum(dim=-1)
    h_ref = -(p_ref * lp_ref).sum(dim=-1)
    floor = torch.relu(h_ref - h_theta) * shift_mask
    denom = shift_mask.sum().clamp(min=1.0)
    return floor.sum() / denom


def build_uncertainty_token_ids(tokenizer, phrases):
    """Bag-of-subword-ids approximation of the uncertainty lexicon.

    Each phrase is tokenized (no special tokens) and ALL resulting subword ids
    are unioned. This approximates phrase-level uncertainty at the token level:
    the penalty acts on the constituent subwords (e.g. the "however", "might",
    "possibly" tokens). Documented as an approximation in the paper.
    Returns a 1-D LongTensor of unique ids.
    """
    import torch

    ids: set[int] = set()
    for phrase in phrases:
        for tid in tokenizer.encode(phrase.strip(), add_special_tokens=False):
            ids.add(int(tid))
        # Also encode with a leading space (BPE space-prefixed variants).
        for tid in tokenizer.encode(" " + phrase.strip(), add_special_tokens=False):
            ids.add(int(tid))
    return torch.tensor(sorted(ids), dtype=torch.long)


def compute_penalty(
    formulation: str,
    policy_logits,
    ref_logits,
    input_ids,
    completion_mask,
    uncertainty_ids,
):
    """Dispatch to the selected penalty formulation."""
    if formulation == "A":
        return penalty_a_mass_floor(policy_logits, ref_logits, input_ids, completion_mask, uncertainty_ids)
    if formulation == "B":
        return penalty_b_chosen_hedge_logprob(policy_logits, input_ids, completion_mask, uncertainty_ids)
    if formulation == "C":
        return penalty_c_entropy_floor(policy_logits, ref_logits, input_ids, completion_mask)
    raise ValueError(f"unknown reg_formulation {formulation!r}; expected A, B, or C")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def _tokenize_pair_side(tokenizer, prompt: str, completion: str, max_length: int, max_prompt_length: int):
    """Tokenize prompt+completion; return (input_ids, completion_mask) lists.

    completion_mask is 1 for completion tokens, 0 for prompt tokens. The chat
    template is applied to the prompt so the policy sees the same formatting as
    at generation time.
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"][:max_prompt_length]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + completion_ids)[:max_length]
    completion_mask = ([0] * len(prompt_ids) + [1] * len(completion_ids))[:max_length]
    return input_ids, completion_mask


def run_one_regularized_dpo_round(
    base_model_name: str,
    reference_adapter_path: Path | str | None,
    preference_pairs_jsonl: Path | str,
    output_adapter_dir: Path | str,
    hparams: dict[str, Any] | None = None,
    lora_config: dict[str, Any] | None = None,
    reg_config: dict[str, Any] | None = None,
    seed: int = 42,
    mock: bool = False,
) -> dict[str, Any]:
    """Run one DPO round with an uncertainty-preservation penalty; save the LoRA
    adapter; return metadata mirroring `dpo.run_one_dpo_round`."""
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    lora = {**DEFAULT_LORA, **(lora_config or {})}
    reg = {**DEFAULT_REG, **(reg_config or {})}
    pairs_path = Path(preference_pairs_jsonl)
    adapter_dir = Path(output_adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    num_pairs = _count_lines(pairs_path)

    if mock:
        metadata = {
            "status": "mock",
            "trainer": "regularized_dpo",
            "base_model_name": base_model_name,
            "reference_adapter_path": str(reference_adapter_path) if reference_adapter_path else None,
            "num_pairs": num_pairs,
            "hparams": hp,
            "lora": lora,
            "reg": reg,
            "seed": seed,
        }
        with (adapter_dir / "train_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return {
            "adapter_path": str(adapter_dir),
            "train_loss": float("nan"),
            "rewards_margins": 0.0,
            "mean_penalty": 0.0,
            "num_pairs": num_pairs,
            "reg": reg,
            "mock": True,
        }

    # Real path. Deferred heavy imports.
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .model_factory import resolve_device
    from .uncertainty import UNCERTAINTY_LEXICON_PHRASES

    torch.manual_seed(seed)
    device = resolve_device(hp.get("device", "auto") if isinstance(hp.get("device"), str) else "auto")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _load_base():
        m = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto", trust_remote_code=True)
        if reference_adapter_path is not None and Path(reference_adapter_path).exists():
            m = PeftModel.from_pretrained(m, str(reference_adapter_path))
            try:
                m = m.merge_and_unload()
            except Exception:
                pass
        return m

    # Frozen reference (base, no new LoRA).
    reference = _load_base().to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    # Policy = fresh LoRA on top of the same base.
    policy_base = _load_base()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=list(lora["target_modules"]),
        bias="none",
    )
    policy = get_peft_model(policy_base, peft_config).to(device)
    policy.train()

    uncertainty_ids = build_uncertainty_token_ids(tokenizer, UNCERTAINTY_LEXICON_PHRASES).to(device)

    # Load preference pairs.
    pairs = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pairs.append((r["prompt"], r["chosen"], r["rejected"]))
    if not pairs:
        raise RuntimeError(f"No preference pairs at {pairs_path}")

    max_length = int(hp["max_length"])
    max_prompt_length = int(hp["max_prompt_length"])
    batch_size = int(hp["per_device_train_batch_size"])
    lr = float(hp["learning_rate"])
    beta = float(hp["beta"])
    epochs = float(hp["num_train_epochs"])
    reg_lambda = float(reg["reg_lambda"])
    formulation = str(reg["reg_formulation"])

    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=lr)
    pad_id = tokenizer.pad_token_id

    def _collate(side_rows):
        """Pad a list of (input_ids, completion_mask) to a batch tensor set."""
        maxT = max(len(ids) for ids, _ in side_rows)
        input_ids, attn, comp = [], [], []
        for ids, cmask in side_rows:
            pad = maxT - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            comp.append(cmask + [0] * pad)
        return (
            torch.tensor(input_ids, device=device),
            torch.tensor(attn, device=device),
            torch.tensor(comp, dtype=torch.float, device=device),
        )

    n_steps = max(1, int(round(epochs * ((len(pairs) + batch_size - 1) // batch_size))))
    losses, penalties, margins = [], [], []
    step = 0
    keep_training = True
    for _epoch in range(max(1, int(round(epochs)))):
        if not keep_training:
            break
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            chosen_rows = [
                _tokenize_pair_side(tokenizer, p, c, max_length, max_prompt_length) for p, c, _ in batch
            ]
            rejected_rows = [
                _tokenize_pair_side(tokenizer, p, r, max_length, max_prompt_length) for p, _, r in batch
            ]
            c_ids, c_attn, c_comp = _collate(chosen_rows)
            r_ids, r_attn, r_comp = _collate(rejected_rows)

            pol_c_logits = policy(input_ids=c_ids, attention_mask=c_attn).logits
            pol_r_logits = policy(input_ids=r_ids, attention_mask=r_attn).logits
            with torch.no_grad():
                ref_c_logits = reference(input_ids=c_ids, attention_mask=c_attn).logits
                ref_r_logits = reference(input_ids=r_ids, attention_mask=r_attn).logits

            pol_c_logp = sequence_logps(pol_c_logits, c_ids, c_comp)
            pol_r_logp = sequence_logps(pol_r_logits, r_ids, r_comp)
            ref_c_logp = sequence_logps(ref_c_logits, c_ids, c_comp)
            ref_r_logp = sequence_logps(ref_r_logits, r_ids, r_comp)

            loss_dpo, stats = dpo_loss(pol_c_logp, pol_r_logp, ref_c_logp, ref_r_logp, beta)
            penalty = compute_penalty(
                formulation, pol_c_logits, ref_c_logits, c_ids, c_comp, uncertainty_ids
            )
            loss = loss_dpo + reg_lambda * penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss_dpo.item()))
            penalties.append(float(penalty.item()))
            margins.append(stats["rewards_margins"])
            step += 1
            print(
                f"[reg_dpo] step {step}/{n_steps} formulation={formulation} "
                f"L_dpo={loss_dpo.item():.4f} penalty={penalty.item():.4f} "
                f"margin={stats['rewards_margins']:+.4f}"
            )
            if step >= n_steps:
                keep_training = False
                break

    policy.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    metadata = {
        "status": "ok",
        "trainer": "regularized_dpo",
        "base_model_name": base_model_name,
        "reference_adapter_path": str(reference_adapter_path) if reference_adapter_path else None,
        "num_pairs": len(pairs),
        "hparams": hp,
        "lora": lora,
        "reg": reg,
        "seed": seed,
        "train_loss": _mean(losses),
        "dpo_loss_component": _mean(losses),
        "penalty_component": _mean(penalties),
        "mean_penalty": _mean(penalties),
        "reg_formulation": formulation,
        "reg_lambda": reg_lambda,
        "rewards_margins": _mean(margins),
        "num_steps": step,
    }
    with (adapter_dir / "train_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "adapter_path": str(adapter_dir),
        "train_loss": metadata["train_loss"],
        "rewards_margins": metadata["rewards_margins"],
        "mean_penalty": metadata["mean_penalty"],
        "num_pairs": len(pairs),
        "reg": reg,
        "mock": False,
    }
