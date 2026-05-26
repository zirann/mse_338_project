# Colab Enterprise / A100 runbook: topk96 scaling experiment

This runbook executes the `configs/experiment_fast_filtered_topk96.yaml` ablation on a CUDA-enabled environment (Google Colab Enterprise / Vertex AI Workbench with an A100 runtime is the assumed target; any Linux box with a single NVIDIA GPU will work the same).

Single-knob research framing: scale the *clean survivor set* by raising `top_k` from 32 to 96, with the Phase A candidate pool grown in lockstep (`num_conditions: 16 -> 24`, `candidates_per_condition: 4 -> 8`) so `top_k=96` is a real top-50% cut. Filter, epochs, batch_size, grad_accum, LoRA, learning rate, decoding, evaluators all unchanged. Hypothesis and falsification criterion are documented in [paper/TODO_report.md](../paper/TODO_report.md).

No source code changes are required. The device priority in [src/redteam/model_factory.py](../src/redteam/model_factory.py) is already CUDA > MPS > CPU, and the trainer's MPS fallback env var is a no-op on CUDA.

---

## 1. Pre-flight

- Colab Enterprise runtime: select an NVIDIA A100 GPU (40GB or 80GB; the run uses well under 10GB).
- Reasonable disk: ~5 GB for the model + HF dataset caches + the run artifacts.
- Network: outbound to `huggingface.co` and `github.com` must be allowed. `HF_HUB_OFFLINE=0` in every command below (this is the default; we set it explicitly so the runbook reads cleanly).

## 2. Clone / pull the repo

Replace `<YOUR_REPO_URL>` with the actual remote (the repo is local-only at the time of writing; push it to GitHub or GCS Source Repos first if you have not already).

```bash
# fresh clone
cd /content
git clone <YOUR_REPO_URL> mini_redteam_poc
cd mini_redteam_poc

# OR update an existing clone
cd /content/mini_redteam_poc
git fetch origin
git checkout <branch_with_topk96_config>
git pull
```

Verify the new config and runbook are present:

```bash
ls configs/experiment_fast_filtered_topk96.yaml
ls notebooks/colab_enterprise_a100_runbook.md
```

## 3. Install requirements

The Colab Enterprise A100 image ships with PyTorch + CUDA pre-installed. Add the rest from [requirements.txt](../requirements.txt):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `torch` shadowing causes a re-install of CPU-only PyTorch, force-reinstall the CUDA build instead:

```bash
pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121 torch
```

## 4. Verify CUDA / A100 availability

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected output on an A100 instance:

```
True
NVIDIA A100-SXM4-40GB
```

If the first line is `False`, **stop here**. Reassign the Colab runtime to a GPU instance and re-run step 4. Continuing with `cuda=False` will silently fall back to CPU and take orders of magnitude longer.

Optional deeper check (memory, compute capability):

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

## 5. Prepare seed prompts

```bash
HF_HUB_OFFLINE=0 python scripts/prepare_data.py --config configs/data.yaml
```

This downloads `toxigen/toxigen-data` + `allenai/real-toxicity-prompts` (first run only; HF dataset cache is persisted in `~/.cache/huggingface/datasets`). Writes `outputs/data/{train,val,test}.jsonl`. The val.jsonl slice is what the loop feeds into the generator.

Wall-clock: ~30 seconds with warm network, ~3-5 minutes if both datasets need downloading.

## 6. Run toxicbert target

```bash
HF_HUB_OFFLINE=0 python scripts/run_evaluator_loop.py \
    --target_evaluator toxicbert \
    --config configs/experiment_fast_filtered_topk96.yaml
```

Expected `[diag]` prints near the top of stdout:

```
[diag] torch=<version>, mps_available=False, mps_built=False, cuda_available=True
[diag] selected device: cuda
[diag] generation phase A: model on cuda:0
[diag] training: model device = cuda:0, trainer.args.device = cuda:0
[diag] generation phase B: model on cuda:0
```

If any of those lines says `cpu`, the run is on CPU and you should abort with Ctrl-C, fix the CUDA environment, and re-run.

Expected wall-clock on A100: ~25-35 minutes (192 candidates × 32 new tokens per phase × 2 phases + ~25 SFT updates + scoring). For comparison: the same experiment with the smaller `experiment_fast.yaml` config on CPU took ~5 min per target.

To capture stdout for later analysis (recommended):

```bash
HF_HUB_OFFLINE=0 python scripts/run_evaluator_loop.py \
    --target_evaluator toxicbert \
    --config configs/experiment_fast_filtered_topk96.yaml \
    2>&1 | tee outputs/runs_topk96_toxicbert_stdout.log
```

## 7. Run cardiff target

```bash
HF_HUB_OFFLINE=0 python scripts/run_evaluator_loop.py \
    --target_evaluator cardiff \
    --config configs/experiment_fast_filtered_topk96.yaml \
    2>&1 | tee outputs/runs_topk96_cardiff_stdout.log
```

Same wall-clock estimate as step 6.

## 8. Build the transfer matrix and summary

```bash
python scripts/analyze_transfer.py \
    --config configs/experiment_fast_filtered_topk96.yaml \
    --out_dir outputs/reports_fast_filtered_topk96
```

Outputs (~1 second; both files are created by this step):

- outputs/reports_fast_filtered_topk96/transfer_matrix.json (machine-readable)
- outputs/reports_fast_filtered_topk96/transfer_summary.md (human-readable; includes redacted top-5 examples per target)

## 9. Persist results back to your environment

Pick one of these patterns based on where you want the artifacts to live.

### Option A: commit results to git

```bash
cd /content/mini_redteam_poc
git add outputs/runs_fast_filtered_topk96 outputs/reports_fast_filtered_topk96 \
        outputs/runs_topk96_*.log
git commit -m "topk96 scaling experiment outputs (A100)"
git push origin <branch>
```

Note: candidate JSONLs can be several MB each. If you hit GitHub size limits, either add the candidate JSONLs to `.gitignore` (keeping only `metrics.json` + `transfer_*` in the commit) or set up `git-lfs` for the JSONL paths.

### Option B: copy to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copytree('/content/mini_redteam_poc/outputs/runs_fast_filtered_topk96',
                '/content/drive/MyDrive/mini_redteam_results/runs_fast_filtered_topk96')
shutil.copytree('/content/mini_redteam_poc/outputs/reports_fast_filtered_topk96',
                '/content/drive/MyDrive/mini_redteam_results/reports_fast_filtered_topk96')
```

### Option C: copy to a GCS bucket (Colab Enterprise native)

```bash
gcloud storage cp -r \
    outputs/runs_fast_filtered_topk96 \
    outputs/reports_fast_filtered_topk96 \
    outputs/runs_topk96_*.log \
    gs://YOUR_BUCKET/mini_redteam_results/
```

## 10. Local sync + figure regeneration

Back on your laptop, after pulling / downloading the results so they land at the same paths under `outputs/`:

```bash
# verify the new metrics + transfer_matrix exist locally
ls outputs/runs_fast_filtered_topk96/{toxicbert,cardiff}/metrics.json
ls outputs/reports_fast_filtered_topk96/transfer_matrix.json

# (optional) regenerate the existing figures
python scripts/figures/build_all.py
```

The existing figure scripts under [scripts/figures/](../scripts/figures/) currently consume the three fast regimes only. Extending them to a 4-regime panel that includes `topk96` is a follow-up task that is already tracked in [paper/TODO_report.md](../paper/TODO_report.md).

## 11. Troubleshooting

- **CUDA out of memory**. The default config uses `gradient_checkpointing: true`; if you somehow OOM on A100 with a 0.6B model, drop batch_size to 1 (it already is) and re-check that `fp16=true` got picked up (it should auto-enable on CUDA via the trainer arg defaults).
- **HF cache miss / 403 on HEAD**. Set `HF_HOME` to a writable path before any HF call: `export HF_HOME=/content/hf_cache`. If you have already prepared data and just want to skip HEAD probes, run with `HF_HUB_OFFLINE=1`.
- **A100 not assigned despite GPU runtime selected**. Stop the runtime, re-allocate, re-run step 4 from scratch. Colab Enterprise sometimes assigns T4 / V100 by default; explicitly request A100.
- **Slow generation despite CUDA**. Confirm with `python -c "import torch; x = torch.randn(2048, 2048, device='cuda'); print((x @ x.T).device)"` that real CUDA tensors work. If they error out, the CUDA driver / PyTorch version are mismatched.
- **HuggingFace login required for Qwen3**. Qwen3-0.6B is gated on HF for some accounts; if `from_pretrained` fails with a 401, run `huggingface-cli login` and accept the model license.

## After the run

Hand the following back to the assistant in a follow-up turn (these files are created by the run; the paths below intentionally do not exist yet in the local repo) to produce the 12-item Part C comparison summary (topk96 vs the headline `fast_filt_e1`):

- outputs/runs_fast_filtered_topk96/toxicbert/metrics.json
- outputs/runs_fast_filtered_topk96/cardiff/metrics.json
- outputs/reports_fast_filtered_topk96/transfer_matrix.json
- outputs/reports_fast_filtered_topk96/transfer_summary.md
- The two stdout logs from steps 6-7 (or just the four `[diag]` lines + the `ELAPSED_SEC=` numbers).
