"""Single-instance LLM-as-judge wrapper.

Default real backend: `Qwen/Qwen2.5-7B-Instruct` via Hugging Face Transformers.
Loaded once and reused for ranking + pairwise + factuality scoring across all
rounds. Mock backend: deterministic length-based ranking; mimics the
length-bias that prior work has shown LLM judges exhibit, so local smoke runs
produce a sensible appearance-up-substance-flat trajectory without any model
download.

The real backend is intentionally NOT exercised in the local Phase 2 smoke
(macOS / CPU / MPS without a 7B-class GPU); it is meant to run on A100 in
Phase 3. The implementation is here so the A100 step works without further
changes.

Diagnostics: every public call (`rank_candidates_with_diagnostics`,
`pairwise_with_diagnostics`) returns the raw verdict string and a parse-failure
flag so calling scripts can inspect what the judge actually said and how
often parsing fell back to RNG. `Judge.parse_failures` tracks the running
count across the instance's lifetime.
"""
from __future__ import annotations

import random
from typing import Sequence


RANK_SYSTEM_PROMPT = (
    "You are a careful evaluator of language-model responses. Given a question "
    "and several candidate answers, identify the one that is most thoughtful, "
    "well-reasoned, and high-quality. Output exactly the candidate label (a "
    "single character: A, B, C, or D) of the best response. Do not explain."
)

WORST_SYSTEM_PROMPT = (
    "You are a careful evaluator of language-model responses. Given a question "
    "and several candidate answers, identify the one that is least thoughtful, "
    "well-reasoned, and high-quality. Output exactly the candidate label (a "
    "single character: A, B, C, or D) of the worst response. Do not explain."
)

PAIRWISE_SYSTEM_PROMPT = (
    "You are a careful evaluator of language-model responses. Given a question "
    "and two candidate answers labeled A and B, identify the one that is more "
    "thoughtful, well-reasoned, and high-quality. Output exactly one character: "
    "'A' or 'B'. Do not explain."
)

FACTUALITY_SYSTEM_PROMPT = (
    "You are a careful factuality grader. You will be given a question, a model "
    "response, a known-correct reference answer, and a plausible-incorrect "
    "reference answer. Decide whether the model response agrees with the correct "
    "reference, with the incorrect reference, or with neither. Output exactly one "
    "of: CORRECT, INCORRECT, PARTIAL. Do not explain."
)


_LABELS = ("A", "B", "C", "D", "E", "F", "G", "H")


class Judge:
    """Wraps a single causal LM and exposes ranking + pairwise + factuality APIs.

    With `mock=True`, all calls are deterministic and require no model load.
    With `mock=False`, the LLM is loaded lazily on first use.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        seed: int = 42,
        mock: bool = False,
    ) -> None:
        self.mock = mock
        self.seed = seed
        self._rng = random.Random(seed)
        self._model_name = model_name
        self._device_arg = device
        self._tokenizer = None
        self._model = None
        self._device = None
        # Diagnostics: total parse failures across rank + pairwise + factuality calls.
        self.parse_failures = 0

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self.mock:
            return
        if self._model is not None:
            return
        # Defer heavy imports until first real call.
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .model_factory import resolve_device

        device = resolve_device(self._device_arg)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self._model.to(device)
        self._model.eval()
        self._device = device

    # ------------------------------------------------------------------
    # Real-backend helpers
    # ------------------------------------------------------------------

    def _greedy_config(self, max_new_tokens: int):
        """Build a GenerationConfig that is unambiguously greedy.

        Setting temperature=1.0 / top_p=1.0 / top_k=50 (transformers defaults)
        prevents the misleading "The following generation flags are not valid
        and may be ignored" warning that fires when the model's loaded
        generation_config has sampling defaults but we force do_sample=False.
        """
        from transformers import GenerationConfig

        return GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            pad_token_id=self._tokenizer.eos_token_id,
        )

    def _generate_short(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 8) -> str:
        """Greedy 1-to-8-token generation. Returns the raw decoded suffix string."""
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        chat_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(chat_text, return_tensors="pt").to(self._device)
        gen_config = self._greedy_config(max_new_tokens)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, generation_config=gen_config)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _candidate_block(candidates: Sequence[str], labels: Sequence[str]) -> str:
        return "\n\n".join(f"{lab}: {c}" for lab, c in zip(labels, candidates))

    def _ask_label_verbose(
        self,
        system_prompt: str,
        prompt: str,
        candidates: Sequence[str],
    ) -> tuple[int, str, bool]:
        """One judge call. Returns (input_order_idx, raw_verdict, parse_failure)."""
        labels = list(_LABELS[: len(candidates)])
        # Randomize position so the judge cannot exploit ordering.
        order = list(range(len(candidates)))
        self._rng.shuffle(order)
        shuffled = [candidates[i] for i in order]
        user_prompt = (
            f"Question: {prompt}\n\n"
            f"{self._candidate_block(shuffled, labels)}\n\n"
            f"Which label?"
        )
        verdict = self._generate_short(system_prompt, user_prompt, max_new_tokens=5)
        verdict_upper = verdict.upper()
        shuffled_idx = None
        for ch in verdict_upper:
            if ch in labels:
                shuffled_idx = labels.index(ch)
                break
        parse_failure = shuffled_idx is None
        if parse_failure:
            self.parse_failures += 1
            shuffled_idx = self._rng.randint(0, len(candidates) - 1)
        return order[shuffled_idx], verdict, parse_failure

    def _ask_label(self, system_prompt: str, prompt: str, candidates: Sequence[str]) -> int:
        idx, _, _ = self._ask_label_verbose(system_prompt, prompt, candidates)
        return idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank_candidates_with_diagnostics(
        self,
        prompt: str,
        candidates: Sequence[str],
    ) -> tuple[list[int], dict]:
        """Like `rank_candidates`, but also returns a diagnostics dict with the
        raw best/worst verdicts and parse-failure flags.

        Diagnostics keys:
        - best_idx, worst_idx (or None if no candidates)
        - raw_verdict_best, raw_verdict_worst (verbatim judge output strings)
        - parse_failure_best, parse_failure_worst (bool)
        - mock (bool)
        """
        if len(candidates) == 0:
            return [], {
                "best_idx": None,
                "worst_idx": None,
                "raw_verdict_best": None,
                "raw_verdict_worst": None,
                "parse_failure_best": False,
                "parse_failure_worst": False,
                "mock": self.mock,
            }
        if self.mock:
            scored = [(i, len(c.split())) for i, c in enumerate(candidates)]
            scored.sort(key=lambda x: (-x[1], x[0]))
            ranked = [i for i, _ in scored]
            return ranked, {
                "best_idx": ranked[0],
                "worst_idx": ranked[-1],
                "raw_verdict_best": "MOCK_LEN_RANK",
                "raw_verdict_worst": "MOCK_LEN_RANK",
                "parse_failure_best": False,
                "parse_failure_worst": False,
                "mock": True,
            }
        self._ensure_loaded()
        best_idx, raw_best, pf_best = self._ask_label_verbose(RANK_SYSTEM_PROMPT, prompt, candidates)
        worst_idx, raw_worst, pf_worst = self._ask_label_verbose(WORST_SYSTEM_PROMPT, prompt, candidates)
        if worst_idx == best_idx and len(candidates) > 1:
            # Judge picked the same index for best and worst; fall back to
            # shortest-among-the-rest as a deterministic worst-tiebreaker.
            lengths = [(i, len(c.split())) for i, c in enumerate(candidates) if i != best_idx]
            if lengths:
                worst_idx = min(lengths, key=lambda x: x[1])[0]
        middle = [i for i in range(len(candidates)) if i not in (best_idx, worst_idx)]
        ranked = [best_idx] + middle + [worst_idx]
        return ranked, {
            "best_idx": best_idx,
            "worst_idx": worst_idx,
            "raw_verdict_best": raw_best,
            "raw_verdict_worst": raw_worst,
            "parse_failure_best": pf_best,
            "parse_failure_worst": pf_worst,
            "mock": False,
        }

    def rank_candidates(self, prompt: str, candidates: Sequence[str]) -> list[int]:
        """Rank K candidates; return indices in descending order of judged quality."""
        ranked, _ = self.rank_candidates_with_diagnostics(prompt, candidates)
        return ranked

    def pairwise_with_diagnostics(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> tuple[int, dict]:
        """Like `pairwise`, but also returns a diagnostics dict.

        Diagnostics keys:
        - position_swap (bool): whether A and B were shown swapped to the judge
        - raw_verdict (str): verbatim judge output
        - parse_failure (bool)
        - mock (bool)
        """
        if self.mock:
            la, lb = len(response_a.split()), len(response_b.split())
            winner = 0 if la >= lb else 1
            return winner, {
                "position_swap": False,
                "raw_verdict": "MOCK_LEN_BIAS",
                "parse_failure": False,
                "mock": True,
            }
        self._ensure_loaded()
        # Randomize position; map judge's label back to input index.
        if self._rng.random() < 0.5:
            shown = (response_a, response_b)
            position_swap = False
            label_to_input = {"A": 0, "B": 1}
        else:
            shown = (response_b, response_a)
            position_swap = True
            label_to_input = {"A": 1, "B": 0}
        user_prompt = (
            f"Question: {prompt}\n\n"
            f"A: {shown[0]}\n\n"
            f"B: {shown[1]}\n\n"
            f"Which label?"
        )
        verdict = self._generate_short(PAIRWISE_SYSTEM_PROMPT, user_prompt, max_new_tokens=3)
        verdict_upper = verdict.upper()
        chosen = None
        for ch in verdict_upper:
            if ch in ("A", "B"):
                chosen = ch
                break
        parse_failure = chosen is None
        if parse_failure:
            self.parse_failures += 1
            chosen = self._rng.choice(["A", "B"])
        return label_to_input[chosen], {
            "position_swap": position_swap,
            "raw_verdict": verdict,
            "parse_failure": parse_failure,
            "mock": False,
        }

    def pairwise(self, prompt: str, response_a: str, response_b: str) -> int:
        """Pairwise comparison. Returns 0 if A wins, 1 if B wins."""
        winner, _ = self.pairwise_with_diagnostics(prompt, response_a, response_b)
        return winner

    def win_rate(
        self,
        prompts: Sequence[str],
        baseline: Sequence[str],
        candidate: Sequence[str],
    ) -> float:
        """Per-prompt pairwise win-rate of `candidate` over `baseline`."""
        if len(prompts) == 0:
            return 0.0
        wins = 0
        for p, b, c in zip(prompts, baseline, candidate):
            if self.pairwise(p, b, c) == 1:
                wins += 1
        return wins / len(prompts)

    def score_factuality(
        self,
        prompt: str,
        response: str,
        correct_reference: str,
        incorrect_reference: str,
    ) -> float:
        """Reference-grounded factuality verdict mapped to {1.0, 0.5, 0.0}."""
        if self.mock:
            return 1.0
        self._ensure_loaded()
        user_prompt = (
            f"Question: {prompt}\n\n"
            f"Model response: {response}\n\n"
            f"Correct reference: {correct_reference}\n\n"
            f"Incorrect reference: {incorrect_reference}\n\n"
            f"Verdict?"
        )
        verdict = self._generate_short(FACTUALITY_SYSTEM_PROMPT, user_prompt, max_new_tokens=8).upper()
        if "CORRECT" in verdict and "INCORRECT" not in verdict:
            return 1.0
        if "INCORRECT" in verdict:
            return 0.0
        if "PARTIAL" in verdict:
            return 0.5
        # Unparseable -> ambiguous middle value; count as parse failure.
        self.parse_failures += 1
        return 0.5
