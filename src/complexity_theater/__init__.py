"""Optimization-Induced Epistemic Simulacra.

Empirical study: under iterative DPO against an LLM judge, do appearance signals
(length, markdown structure, epistemic markers) rise across rounds while substance
signals (factuality, information density) stay flat or degrade? This package
provides the minimal modules required to run that experiment.

Modules:

- io_utils: JSONL / YAML / JSON helpers (reused from the prior project).
- model_factory: device resolution + base + LoRA loaders (reused).
- appearance: appearance metrics (length, structural complexity, epistemic markers).
- substance: substance metrics (factuality, information density).
- judge: single-instance LLM-as-judge wrapper for ranking and pairwise win-rate.
- dpo: thin wrapper around `trl.DPOTrainer` for one preference-optimization round.
"""

__all__ = [
    "io_utils",
    "model_factory",
    "appearance",
    "substance",
    "judge",
    "dpo",
]
