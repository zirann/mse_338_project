# Colab Enterprise / A100 runbook

> DEPRECATED. This runbook documents the old multi-round "Epistemic Simulacra"
> trajectory framing, which has been superseded by the uncertainty-suppression +
> mitigation project. Use `notebooks/colab_a100_runbook.md` instead. This file
> is retained only as a historical reference for the trajectory (now an appendix
> experiment).

Optimization-Induced Epistemic Simulacra. Multi-round DPO trajectory experiment.

Target runtime: a single NVIDIA A100 (40GB or 80GB) on Google Colab Enterprise / Vertex AI Workbench. The full trajectory fits in approximately 90 minutes of A100 time end-to-end (~25 to 35 minutes per round across 3 rounds, plus dataset preparation and round-0 baseline eval).

**Phase status note.** This runbook documents the commands the project's four scripts will accept once Phase 2 implementation lands. The scripts under `scripts/` are currently Phase 1 stubs that print a "STUB" line and exit. The CUDA verification step (step 4) and the requirements install (step 3) work today. The smoke (step 5) and full-trajectory (step 6) commands will become live once Phase 2 implementation is approved and merged.

---

## 1. Clone the repo

Replace `<YOUR_REPO_URL>` with the GitHub or GCS Source Repos remote.

```bash
cd /content
git clone <YOUR_REPO_URL> complexity_theater
cd complexity_theater
```

If the repo is already cloned and you only need to sync:

```bash
cd /content/complexity_theater
git fetch origin
git checkout <branch>
git pull
```

Verify the new project layout (Phase 1 skeleton):

```bash
ls src/complexity_theater/        # io_utils, model_factory, appearance, substance, judge, dpo
ls scripts/                       # prepare.py, train_round.py, evaluate.py, analyze.py, figures/
ls configs/experiment.yaml        # single config
ls legacy/toxicity_redteam/       # archived prior project (read-only reference)
```

---

## 2. Install requirements

The Colab Enterprise A100 image ships with CUDA-enabled PyTorch. Install the rest:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Expected new install footprint vs the prior project: only `trl>=0.11.0` is new on top of the standard transformers + datasets + peft + accelerate stack. If `pip install` accidentally downgrades torch to a CPU build (some pip resolvers do this when transformers pulls in older torch metadata), reinstate the CUDA build explicitly:

```bash
pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121 torch
```

---

## 3. Verify CUDA and A100 availability

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output on an A100 instance:

```
True
NVIDIA A100-SXM4-40GB
```

If the first line is `False`, **stop here**. Reassign the Colab runtime to a GPU instance and re-run. Continuing on CPU will silently take orders of magnitude longer.

Optional deeper checks:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
```

---

## 4. (Optional) Hugging Face authentication

`Qwen3-0.6B` and `Qwen2.5-7B-Instruct` are public, but some accounts hit gating dialogs the first time. If `transformers.from_pretrained` fails with a 401:

```bash
huggingface-cli login   # paste token from https://huggingface.co/settings/tokens
```

You can also export the token non-interactively:

```bash
export HF_TOKEN=<your_token>
# or, equivalently for older transformers releases:
export HUGGING_FACE_HUB_TOKEN=<your_token>
```

To pre-warm the model cache so the experiment's first model load does not include a multi-GB download in the timing:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
  AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype='auto'); \
  AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='auto')"
```

Expect ~1.2 GB + ~15 GB downloads on first run. Cached in `~/.cache/huggingface/hub/` thereafter.

---

## 5. Phase 2 smoke (10-prompt subset)

Smoke purpose: verify that the dataset prep, generator, judge, DPO step, and metrics evaluation all wire up end-to-end before consuming a full A100 hour on the trajectory. Expected wall-clock: 5 to 10 minutes total.

```bash
python scripts/prepare.py --config configs/experiment.yaml --limit 10
python scripts/evaluate.py --config configs/experiment.yaml --round 0 --limit 10
python scripts/train_round.py --config configs/experiment.yaml --round 1 --limit 10
python scripts/evaluate.py --config configs/experiment.yaml --round 1 --limit 10
```

Smoke stopping criterion (from the plan):

- `outputs/round_0/metrics.json` exists and contains the appearance and substance metrics in sane numerical ranges (length 10 to 200, distinct ratios in (0, 1], factuality scores in `{0, 0.5, 1}`).
- `outputs/round_1/metrics.json` exists with the same metrics plus a `judge_win_rate_vs_round_0` value in `[0, 1]`.
- `outputs/round_1/adapter/` contains the saved LoRA adapter (~10 to 30 MB).
- Reference KL reported during the DPO step is finite and below the abort threshold (configured to 5 nats in `configs/experiment.yaml`).
- The 10 round-1 evaluation responses are coherent English (visual sanity check, not automated).

If any of these fail, see Troubleshooting.

---

## 6. Full 3-round trajectory

Run only after the smoke passes. Expected wall-clock: ~90 minutes on A100.

```bash
python scripts/prepare.py    --config configs/experiment.yaml
python scripts/evaluate.py   --config configs/experiment.yaml --round 0

python scripts/train_round.py --config configs/experiment.yaml --round 1
python scripts/evaluate.py    --config configs/experiment.yaml --round 1

python scripts/train_round.py --config configs/experiment.yaml --round 2
python scripts/evaluate.py    --config configs/experiment.yaml --round 2

python scripts/train_round.py --config configs/experiment.yaml --round 3
python scripts/evaluate.py    --config configs/experiment.yaml --round 3

python scripts/analyze.py    --config configs/experiment.yaml
```

Alternative one-liner that captures stdout for each step (recommended for repeatability and post-hoc inspection):

```bash
mkdir -p logs
python scripts/prepare.py --config configs/experiment.yaml 2>&1 | tee logs/prepare.log
python scripts/evaluate.py --config configs/experiment.yaml --round 0 2>&1 | tee logs/round_0_eval.log
for N in 1 2 3; do
  python scripts/train_round.py --config configs/experiment.yaml --round $N 2>&1 | tee logs/round_${N}_train.log
  python scripts/evaluate.py --config configs/experiment.yaml --round $N 2>&1 | tee logs/round_${N}_eval.log
done
python scripts/analyze.py --config configs/experiment.yaml 2>&1 | tee logs/analyze.log
```

Outputs produced (per the simplified plan):

```
outputs/data/{train_prompts,eval_prompts}.jsonl
outputs/round_0/{eval_responses.jsonl, metrics.json}
outputs/round_1/{candidates.jsonl, preference_pairs.jsonl, adapter/, eval_responses.jsonl, metrics.json}
outputs/round_2/{...}
outputs/round_3/{...}
outputs/trajectory.json
assets/figures/headline.png
assets/figures/qualitative.png
```

---

## 7. Save and download artifacts

Three patterns. Pick whichever matches your workflow.

### A) Commit to git

```bash
cd /content/complexity_theater
git add outputs assets/figures logs
git commit -m "trajectory experiment: outputs, figures, logs"
git push origin <branch>
```

Candidate JSONLs and per-round adapter shards can be many megabytes. If GitHub size limits bite, either configure `git-lfs` for the adapter `.safetensors` files, or add a `.gitignore` rule that keeps only `metrics.json` and `trajectory.json` while excluding `candidates.jsonl` and `preference_pairs.jsonl`.

### B) Copy to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copytree('/content/complexity_theater/outputs',
                '/content/drive/MyDrive/complexity_theater/outputs',
                dirs_exist_ok=True)
shutil.copytree('/content/complexity_theater/assets/figures',
                '/content/drive/MyDrive/complexity_theater/figures',
                dirs_exist_ok=True)
shutil.copytree('/content/complexity_theater/paper',
                '/content/drive/MyDrive/complexity_theater/paper',
                dirs_exist_ok=True)
```

### C) Copy to a GCS bucket (Colab Enterprise native)

```bash
gcloud storage cp -r outputs/ gs://YOUR_BUCKET/complexity_theater/outputs/
gcloud storage cp -r assets/figures/ gs://YOUR_BUCKET/complexity_theater/figures/
gcloud storage cp -r paper/ gs://YOUR_BUCKET/complexity_theater/paper/
gcloud storage cp -r logs/ gs://YOUR_BUCKET/complexity_theater/logs/
```

---

## 8. Troubleshooting

### CUDA not available

Symptom: step 3 prints `False`.

Cause: the Colab runtime is CPU-only (or assigned a non-A100 GPU like T4 / V100). Fix by reallocating the runtime to A100 from the runtime menu. If the runtime is already A100 but `is_available()` is False, the most common cause is that pip silently installed a CPU build of torch on top of the CUDA-enabled one — see the force-reinstall command in section 2.

### HF authentication failure (401 on `from_pretrained`)

Symptom: `OSError: Qwen/Qwen2.5-7B-Instruct is gated` or HTTP 401 traceback.

Fix:

```bash
huggingface-cli login
```

If your token has no read access to the model, regenerate from <https://huggingface.co/settings/tokens> with "Read" scope. For non-interactive runs, set `HF_TOKEN` and re-run.

### Missing model cache / slow first load

Symptom: the first call to `transformers.from_pretrained` hangs for several minutes during an A100-paid window.

Fix: pre-warm the cache before timing-critical work. Use the cache-warming command in section 4. The cache lives under `~/.cache/huggingface/hub/` and persists for the lifetime of the Colab instance. Mounting it on a persistent disk (Drive) saves redownload on session restart:

```bash
export HF_HOME=/content/drive/MyDrive/hf_cache  # if Drive is mounted
```

### TRL / DPOTrainer API mismatch

Symptom: `train_round.py` raises `TypeError: DPOConfig.__init__() got an unexpected keyword argument`, or `ImportError: cannot import name 'DPOTrainer' from 'trl'`.

Cause: `trl` is moving fast (0.7 → 0.11 → 0.13 in 12 months). Hyperparameter names and class locations drift between minor versions.

Fix:

```bash
pip show trl | grep Version
pip install --upgrade 'trl>=0.11.0,<0.14.0'
```

If the upgrade does not converge, pin to the version the Phase 2 implementation was authored against (will be specified in `requirements.txt` when Phase 2 lands). The current pin is `trl>=0.11.0`; a hard upper bound will be added if a breaking release ships before the trajectory run.

### Out-of-memory (CUDA OOM)

Symptom: `RuntimeError: CUDA out of memory` during candidate generation, judge ranking, or DPO step.

Most common cause: the judge model (`Qwen2.5-7B-Instruct`, ~14 GB in fp16) and the base model + LoRA (Qwen3-0.6B, ~1.2 GB) are both loaded; A100 40GB has ~26 GB free for activations. If both are loaded simultaneously plus a long batch, OOM can occur.

Fix (in order of preference):

1. *Lower per-device batch size.* Edit `configs/experiment.yaml`:
   ```yaml
   dpo:
     per_device_train_batch_size: 2        # was 4
     gradient_accumulation_steps: 4         # was 2 (keep effective batch ~8)
   ```
2. *Lower max sequence length.* Edit `configs/experiment.yaml`:
   ```yaml
   dpo:
     max_length: 384         # was 512
     max_prompt_length: 192  # was 256
   ```
3. *Free the judge between phases.* When transitioning from candidate generation to DPO training, explicitly `del judge_model; gc.collect(); torch.cuda.empty_cache()`. The Phase 2 implementation will do this by default; manual workaround if not.
4. *Use bfloat16 for the judge.* Edit `configs/experiment.yaml`:
   ```yaml
   judge:
     dtype: bfloat16    # A100 supports bf16 natively; halves judge memory
   ```

### Judge output parse failures

Symptom: `Judge.rank_candidates` raises `ValueError: could not parse judge verdict` or returns garbage rankings.

Cause: the LLM judge occasionally produces explanatory text rather than the single-letter verdict the prompt asks for. With `Qwen2.5-7B-Instruct` at greedy decoding this is rare but possible (less than 5 % per call).

Fixes built into the Phase 2 implementation:

1. *Retry once with stricter system prompt.* On parse failure, re-issue the call with the suffix "Output ONLY the single letter, no explanation." Phase 2 default.
2. *Fall back to random ranking.* If the second call also fails, log a warning and assign a uniform-random ranking. The aggregate metric is robust to a small number of random rankings.
3. *Lower judge temperature.* Already 0 (greedy). If failures persist with greedy decoding, the prompt template is at fault and needs rewriting; this would be a Phase 2 implementation bug, not a runtime issue.

If parse failure rate exceeds 5 % on the smoke set, abort the trajectory and fix the prompt template before the full run.

### DPO step diverges (KL > 5 nats)

Symptom: `train_round.py` reports a final-step KL above the abort threshold.

Built-in mitigation: `abort_kl_threshold: 5.0` in `configs/experiment.yaml` causes `dpo.run_one_dpo_round` to raise rather than save a degenerate adapter.

Manual fixes (in order of preference):

1. Reduce learning rate: `dpo.learning_rate: 2.0e-5` (was 5.0e-5).
2. Increase DPO beta: `dpo.beta: 0.3` (was 0.1) — stronger KL regularization.
3. Shorten training: `dpo.num_train_epochs: 0.5` (was 1.0).

After any of these changes, restart the affected round and continue the trajectory. The relevant adapters from completed rounds are unchanged.

### Generation degeneracy / mode collapse

Symptom: round-N eval responses are repetitive, ungrammatical, or all start with the same token.

Indicator: `metrics.json` shows `information_density` dropping below 0.05 or `length` drifting to the `max_new_tokens` ceiling with high `1 - distinct_2`.

Fix: same as the DPO-diverge mitigations above, plus optionally raising eval-time `temperature` from 0.7 to 0.9 to test whether the issue is in the policy (degenerate) or the decoding (over-deterministic). If raising temperature does not restore diversity, the policy is collapsed and that round's adapter should be retrained with smaller `learning_rate`.

### Trajectory saturation (no further drift after round 2)

Symptom: `metrics.json` at round 3 is numerically indistinguishable from round 2.

Not a failure. Document the saturation in `analyze.py`'s output; the headline figure tolerates a flat tail. If saturation occurs by round 1, the experiment is suspicious (the policy may not have moved at all); double-check that round-1 generation used the round-1 adapter, not the base model.

### A100 not assigned despite GPU runtime selected

Symptom: `nvidia-smi` shows `T4` or `V100` instead of `A100`.

Colab Enterprise sometimes assigns the wrong GPU type. Stop and re-allocate the runtime explicitly requesting A100. The experiment will technically run on T4 or V100 but will take ~3x as long and OOM thresholds are tighter.

### Wall-clock substantially above estimate

Indicator: a single round exceeds 45 minutes of A100 time.

Likely causes and fixes:

- Slow tokenizer (~3x slowdown vs fast tokenizer). Ensure `transformers.AutoTokenizer.from_pretrained(..., use_fast=True)` is in use (default in Phase 2 implementation).
- DataLoader bottleneck on small batch / large max_length. Profile with one round, then either raise batch or lower max_length.
- Disk I/O during candidate JSONL writes. Move `outputs/` to a tmpfs RAM disk for the duration of the run if disk is the bottleneck.

---

## Sanity-check checklist (before running the full trajectory)

- [ ] Section 3 prints `True` and `NVIDIA A100-SXM4-...`.
- [ ] `pip show trl` reports a version compatible with the Phase 2 implementation (the version range listed in `requirements.txt`).
- [ ] `huggingface-cli whoami` returns your username (if you needed to log in).
- [ ] Section 5 smoke completes with `outputs/round_{0,1}/metrics.json` populated and round-1 generations coherent.
- [ ] At least 30 GB of free disk in `/content/` for the model cache + outputs.
- [ ] You know how to interrupt and resume mid-trajectory: each `train_round.py` and `evaluate.py` call is idempotent on its own outputs, so re-running a specific round overwrites only that round's directory.

## Hand-back checklist (after the full trajectory)

When the trajectory finishes, copy or commit the following:

- `outputs/round_{0,1,2,3}/metrics.json` (the four trajectory measurements)
- `outputs/trajectory.json` (the combined per-metric per-round table)
- `assets/figures/headline.png` and `assets/figures/qualitative.png`
- `logs/` (the per-step stdout, for paper appendix evidence that the run executed on A100)

These four artifacts are sufficient for the Phase 4 paper draft. The per-round JSONLs and LoRA adapters are useful for reproducibility but are optional for the writeup.
