import os
import io
from pathlib import Path
import subprocess

import pandas as pd
import yaml
import streamlit as st

# Page/setup
st.set_page_config(page_title="EDA Copilot", layout="wide")
st.title("EDA Copilot — Upload & Run")

# Upload
uploaded = st.file_uploader("Upload a CSV, Parquet, or Excel", type=["csv", "parquet", "xlsx", "xls"])
name = st.text_input("Dataset name (used in filenames)", value="dataset")

# If Excel, offer sheet picker
excel_sheet = None
if uploaded and uploaded.name.lower().endswith((".xlsx", ".xls")):
    try:
        xls = pd.ExcelFile(io.BytesIO(uploaded.getvalue()))
        excel_sheet = st.selectbox("Excel sheet", xls.sheet_names, index=0)
    except Exception as e:
        st.warning(f"Could not read Excel sheet names: {e}")

# Keys / checks
pk_input = st.text_input("Primary key columns (comma-separated)", value="")
pk_cols = [c.strip() for c in pk_input.split(",") if c.strip()]

with st.expander("Optional checks (quick)", expanded=False):
    num_cols_input = st.text_input("Numeric columns for range checks (comma-separated)", value="")
    min_val = st.number_input("Min (applied to each numeric col)", value=0.0)
    max_val = st.number_input("Max (applied to each numeric col)", value=1e9)
    cat_col = st.text_input("Categorical column for allowed values", value="")
    allowed_vals_input = st.text_input("Allowed values (comma-separated)", value="")

# Plots
st.header("Plots to generate")
miss_chk = st.checkbox("Missingness", value=True)

topn_chk = st.checkbox("Top-N for a categorical column", value=False)
topn_col = st.text_input("Top-N column", value="", disabled=not topn_chk)

hist_chk = st.checkbox("Histogram for a numeric column", value=False)
hist_col = st.text_input("Histogram column", value="", disabled=not hist_chk)
hist_bins = st.number_input("Bins", value=20, step=1, disabled=not hist_chk)

box_chk = st.checkbox("Box plot for a numeric column", value=False)
box_col = st.text_input("Box column", value="", disabled=not box_chk)

card_chk = st.checkbox("Cardinality Top-K", value=True)
card_k = st.number_input("K for cardinality", value=10, step=1, disabled=not card_chk)

# Actions
col_run, col_doc = st.columns([1, 1])
with col_run:
    run = st.button("Run EDA", type="primary")
with col_doc:
    doctor = st.button("Run doctor")

# Helpers
def _save_uploaded_to(path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

def _build_cfg_dict():
    cfg = {"data": {"name": name}}
    data_path = f"data/uploads/{name}{Path(uploaded.name).suffix.lower()}"
    cfg["data"]["path"] = data_path
    if excel_sheet:
        cfg["data"]["sheet"] = excel_sheet

    if pk_cols:
        cfg["keys"] = {"primary": pk_cols}

    checks = {}
    num_cols = [c.strip() for c in num_cols_input.split(",") if c.strip()]
    for c in num_cols:
        checks[c] = {"min": float(min_val), "max": float(max_val)}
    if cat_col and allowed_vals_input:
        checks.setdefault(cat_col, {})
        checks[cat_col]["allowed"] = [v.strip() for v in allowed_vals_input.split(",") if v.strip()]
    if checks:
        cfg["checks"] = checks

    plots = []
    if miss_chk:
        plots.append({"kind": "missingness"})
    if topn_chk and topn_col:
        plots.append({"kind": "topn", "column": topn_col, "n": 5})
    if hist_chk and hist_col:
        plots.append({"kind": "numeric_hist", "column": hist_col, "bins": int(hist_bins)})
    if box_chk and box_col:
        plots.append({"kind": "numeric_box", "column": box_col})
    if card_chk:
        plots.append({"kind": "cardinality_topk", "k": int(card_k)})
    if plots:
        cfg["plots"] = plots

    return cfg

def _run_cli(cfg):
    cfg_path = f"config/uploads/{name}.yml"
    Path("config/uploads").mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # call your existing CLI
    r = subprocess.run(["python", "-m", "eda", "run", "--config", cfg_path], capture_output=True, text=True)
    return r, cfg_path

# Run flow
if run:
    if not uploaded:
        st.error("Please upload a file first.")
    else:
        cfg = _build_cfg_dict()
        st.subheader("Config to run")
        st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")

        _save_uploaded_to(cfg["data"]["path"])

        with st.status("Running engine...", expanded=True):
            r, cfg_path = _run_cli(cfg)
            st.write(r.stdout)
            if r.returncode != 0:
                st.error("EDA run failed.")
                if r.stderr:
                    st.code(r.stderr)
            else:
                st.success("EDA run completed.")

        # Show outputs
        if os.path.exists("outputs/data_quality_report.csv"):
            st.subheader("Contract CSV")
            st.dataframe(pd.read_csv("outputs/data_quality_report.csv"))

        # Show any plots generated for this dataset
        plot_files = sorted([p for p in os.listdir("plots") if p.startswith(f"{name}_") and p.endswith(".png")])
        if plot_files:
            st.subheader("Plots")
            for p in plot_files:
                st.image(os.path.join("plots", p), caption=p)

if doctor:
    r = subprocess.run(["python", "-m", "eda", "doctor"], capture_output=True, text=True)
    if r.returncode == 0:
        st.success("Doctor: all checks passed")
    else:
        st.error("Doctor found issues")
    st.code(r.stdout or r.stderr)