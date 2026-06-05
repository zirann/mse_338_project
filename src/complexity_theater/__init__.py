"""Preference-optimization-induced uncertainty suppression: empirical study package.

We study whether small-scale DPO + LoRA fine-tuning intrinsically suppresses
explicit uncertainty signaling in language-model outputs, even when the
preference signal carries no information about response quality. The package
name is preserved for back-compat; the project's earlier framing
("epistemic simulacra under iterated LLM-judge DPO") was falsified by a
random-preference DPO control and has been retired from headline language.

Operationally, we compare matched control interventions:

- judge-driven vs uniformly random preference assignment, and
- length-matched vs unmatched preference-pair construction (Park et al. 2024
  dataset-level length filtering, not the SamPO objective-side intervention).

We do not claim a causal decomposition of judge-attributable vs
optimization-intrinsic effects; cross-arm differences are reported as
matched-intervention evidence at n=20 per arm.

Modules:

- io_utils: JSONL / YAML / JSON helpers.
- model_factory: device resolution + base + LoRA loaders.
- appearance: appearance metrics (length, structural complexity, hedge_density,
  confidence_marker_density, reasoning_narration_density, epistemic_marker_density).
- substance: substance metrics (factuality, information density).
- judge: single-instance LLM-as-judge wrapper for ranking, pairwise win-rate,
  and reference-grounded factuality.
- dpo: thin wrapper around `trl.DPOTrainer` for one preference-optimization round
  (used by all unregularized arms).
- uncertainty: uncertainty lexicon + score + Mitigation 1 (uncertainty-preserving
  pair filter).
- regularized_dpo: standalone minimal DPO loop + uncertainty-preservation penalty
  (Mitigation 2).
"""

__all__ = [
    "io_utils",
    "model_factory",
    "appearance",
    "substance",
    "judge",
    "dpo",
    "uncertainty",
    "regularized_dpo",
]
