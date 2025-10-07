# EDA Copilot

Deterministic, config‑driven EDA that outputs a contract CSV, curated plots, and short memos. One command profiles a dataset; a built‑in **doctor** verifies artifacts for reproducibility.

## Why this is trustworthy
- **Deterministic outputs:** all figures and stats are computed from your run, not hand‑edited.
- **Contract CSV:** `outputs/data_quality_report.csv` is the single source of truth.
- **Run metadata:** `logs/run_metadata.json` records Python/pandas versions, data SHA‑256, and current Git SHA.
- **Doctor check:** `python -m eda doctor` validates headers, required metrics, memos, and plots against the contract.

## Quickstart

```bash
# 1) Create & activate a virtual env (example)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install
pip install -e .         # from pyproject.toml
# or: pip install -r requirements.txt

# 3) Prepare a config
cp config/examples/ent.example.yml    config/ent.yml
# edit config/ent.yml: set data.path, data.name, optional checks/keys/plots

# 4) Run
python -m eda run --config config/ent.yml

# 5) Verify
python -m eda doctor
```

**Artifacts after a run**
- `outputs/data_quality_report.csv`
- `plots/published/*.png` (curated figures only)
- `reports/findings_memo.md`, `reports/next_actions.md`
- `logs/run_metadata.json`

## Configuration

Example keys you can use in `config/*.yml`:

```yaml
data:
  name: ufc
  path: data/sample/ufc_sample.csv
  # sheet: Sheet1          # for Excel

schema:
  expected_columns: [col1, col2, ...]

checks:
  amount: {min: 0, max: 50000}
  method: {allowed: ["KO/TKO", "Decision", "Submission"]}

keys:
  primary: [fight_id]

plots:
  - {kind: missingness}
  - {kind: cardinality_topk, k: 10}
  - {kind: topn, column: method, n: 5}
  - {kind: numeric_hist, column: time_seconds, bins: 20}
  - {kind: numeric_box,  column: amount}
```

## Project layout

```
eda/                     # package (CLI lives here)
config/
  examples/*.example.yml # shareable templates
data/
  sample/                # tiny, non‑sensitive samples only
outputs/                 # data_quality_report.csv (recreated)
plots/
  published/             # curated PNGs for README/report
reports/
  findings_memo.md
  next_actions.md
streamlit/               # (optional) app code
tests/                   # minimal smoke/CLI tests
```

> Large/raw data, logs, and temp files are intentionally **not** tracked. See `.gitignore`.

## Using the Streamlit app (optional)

```bash
streamlit run streamlit/app.py
```

If your entry file is different, update the command or provide a `run_app.sh`.

## Reproducibility checklist

- Commit only `.example.yml` templates; keep real configs untracked.
- Keep `plots/published/` small (only figures referenced in docs).
- Never commit full datasets; use `data/sample/` + a data README.
- Run `python -m eda doctor` before pushing.

## Troubleshooting

- **“Unsupported file type”** → Use `.csv`, `.parquet`, or Excel; install `openpyxl` for `.xlsx`.
- **Doctor failures** → Re‑run the pipeline, then `python -m eda doctor` to see exactly which artifact is missing or mismatched.
- **Missing venv** → Create/activate `.venv` as shown above.

## License
Released under the **MIT License**. See [LICENSE](LICENSE) for details.  
© 2025 Jonah Schwartz. SPDX-License-Identifier: MIT