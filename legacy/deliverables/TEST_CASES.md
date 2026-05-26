## 1. Unit and contract tests

Command:

```bash
pytest -q
```

Expected:

- All tests pass.
- Coverage includes data mapping, scoring/rejection, metrics, and e2e smoke logic.

## 2. Data pipeline sanity check

Command:

```bash
python scripts/prepare_data.py --config configs/data.yaml --label_mapping configs/label_mapping.yaml
```

Expected:

- Writes `outputs/data/data_manifest.json` and `outputs/data/label_distribution.json`.
- `l2` distribution includes at least 6 categories (target gate: 6/8+).

## 3. Smoke integration (heuristic)

Command:

```bash
bash scripts/run_end_to_end.sh smoke
```

Expected artifacts:

- `outputs/reports/eval_report_smoke.json`
- `outputs/reports/baseline_report_smoke.json`
- `outputs/reports/comparison_report_smoke.json`
- `outputs/reports/manual_audit_sheet_smoke.csv`

## 4. Quick formal integration (dual real scorers)

Command:

```bash
bash scripts/run_end_to_end.sh quick
```

Expected artifacts:

- `outputs/reports/eval_report.json`
- `outputs/reports/baseline_report.json`
- `outputs/reports/comparison_report.json`
- `outputs/reports/manual_audit_sheet.csv`

## 4b. Full integration (required for final submission)

Command:

```bash
bash scripts/run_end_to_end.sh
```

Expected artifacts (in addition to data/model caches):

- `outputs/reports/eval_report.json`
- `outputs/reports/baseline_report.json`
- `outputs/reports/comparison_report.json`
- `outputs/reports/manual_audit_sheet.csv`

## 5. Schema check

Command:

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path('outputs/reports')
eval_report = json.loads((base / 'eval_report.json').read_text())
assert 'kpi_gate' in eval_report
for key in ['harmful_hit_rate','mean_risk_score','category_coverage','distinct_2']:
    assert key in eval_report
print('schema-ok')
PY
```

Expected:

- Prints `schema-ok`.

## 6. Manual audit workflow

Steps:

1. Open `outputs/reports/manual_audit_sheet.csv`.
2. By default, the sheet is **redacted** for safe sharing (`text` is `[REDACTED]`). For a local-only manual audit, set `manual_audit.redact: false` in `configs/eval.yaml` and re-run `python scripts/evaluate.py --config configs/eval.yaml` to regenerate the sheet with text included.
3. Fill `manual_label` with `pass` or `fail`.
4. Re-run:

```bash
python scripts/evaluate.py --config configs/eval_quick.yaml
```

Expected:

- `kpi_gate.status` transitions from `pending_manual_audit` to `pass`/`fail` depending on labels and thresholds.
