import os
import io
from pathlib import Path
import subprocess
import json
import re

import pandas as pd
import yaml
import streamlit as st
from eda.llm import build_summary_payload, call_llm

import requests

def _ollama_alive(base: str) -> bool:
    try:
        r = requests.get(base.rstrip("/") + "/api/tags", timeout=2.5)
        return r.ok
    except Exception:
        return False

def _secret_or_env(key: str, default: str):
    # Use Streamlit secrets if present; otherwise fall back to env; otherwise default
    try:
        # accessing st.secrets may raise if no secrets file; protect with try/except
        return st.secrets.get(key, os.getenv(key, default))  # type: ignore[attr-defined]
    except Exception:
        return os.getenv(key, default)

_default_provider = _secret_or_env("LLM_PROVIDER", "ollama")
_default_model    = _secret_or_env("LLM_MODEL", "llama3.2:3b")
_default_base     = _secret_or_env("OLLAMA_BASE_URL", "http://localhost:11434")

# Auto-switch to Basic if Ollama isn't reachable
if _default_provider.lower() == "ollama" and not _ollama_alive(_default_base):
    st.warning("Ollama is not reachable; falling back to Basic (no-LLM) summary.")
    os.environ["LLM_PROVIDER"] = "rule"
else:
    os.environ["LLM_PROVIDER"] = _default_provider

os.environ["LLM_MODEL"] = _default_model
os.environ["OLLAMA_BASE_URL"] = _default_base

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
            # If the run failed, stop so we don't try to render outputs/plots directories that may not exist
            if r.returncode != 0:
                st.stop()

        # Show outputs
        if os.path.exists("outputs/data_quality_report.csv"):
            st.subheader("Contract CSV")
            st.dataframe(pd.read_csv("outputs/data_quality_report.csv"))

        # Show any plots generated for this dataset
        plot_dir = "plots"
        if os.path.isdir(plot_dir):
            plot_files = sorted([p for p in os.listdir(plot_dir) if p.startswith(f"{name}_") and p.endswith(".png")])
            if plot_files:
                st.subheader("Plots")
                for p in plot_files:
                    st.image(os.path.join(plot_dir, p), caption=p)

if doctor:
    r = subprocess.run(["python", "-m", "eda", "doctor"], capture_output=True, text=True)
    if r.returncode == 0:
        st.success("Doctor: all checks passed")
    else:
        st.error("Doctor found issues")
    st.code(r.stdout or r.stderr)


# AI Summary (optional)

st.header("AI Summary (optional)")
inc_sample = st.checkbox(
    "Include up to 30 sample rows in the prompt",
    value=False,
    help="Turn OFF for sensitive data. The LLM can still summarize using the contract CSV.",
)
summ_btn = st.button("Generate AI Summary")

# Provider picker (so non‑technical users can stick to free/basic)
st.caption("Choose a provider: **Basic** works offline; **Ollama** uses a local model; **OpenAI** uses your own API key.")
prov = st.selectbox(
    "AI provider",
    ["Basic (free)", "Ollama (local)", "OpenAI (cloud)"],
    index=0,
    help="Basic = built-in rule-based summary. Ollama requires a local server. OpenAI requires an API key.",
)

model = None
api_key = None
api_base = None

if "Ollama" in prov:
    model = st.text_input("Ollama model", value=_default_model)
    api_base = st.text_input("Ollama base URL", value=_default_base)
elif "OpenAI" in prov:
    model = st.text_input("OpenAI model", value=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    api_base = st.text_input("OpenAI base URL (optional)", value=os.getenv("OPENAI_BASE_URL", ""))


def _hash_file(p: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_markdown_from_json(data: dict) -> str:
    """Render a clean Markdown summary from the model's structured JSON."""
    def _section(title: str, items):
        if not items:
            return ""
        lines = [f"## {title}"] + [f"- {str(x)}" for x in items]
        return "\n".join(lines) + "\n\n"

    title = data.get("title", "AI Summary")
    one = data.get("one_line")
    blurb = data.get("blurb")
    domain = data.get("domain_guess")
    md_parts = [f"# {title}\n"]
    if one:
        md_parts.append(f"> {one}\n\n")
    if blurb:
        md_parts.append(f"{blurb}\n\n")
    if domain:
        md_parts.append(f"**Domain guess:** {domain}\n\n")
    md_parts.append(_section("Key facts", data.get("key_facts", [])))
    md_parts.append(_section("Issues", data.get("issues", [])))
    md_parts.append(_section("Next actions", data.get("actions", [])))
    md_parts.append(_section("Open questions", data.get("questions", [])))
    return "".join([p for p in md_parts if p])


def _extract_json_from_text(text: str):
    """Try to parse JSON even if wrapped in ```json fences or with extra text."""
    if not text:
        return None
    # 1) Look for ```json ... ``` fenced block
    m = re.search(r"```(?:json|javascript)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = None
    if m:
        candidate = m.group(1)
    else:
        # 2) Brace-matching: find first {...} block
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        break
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # 3) Last resort: try raw text
    try:
        return json.loads(text)
    except Exception:
        return None


if summ_btn:
    dq_path = "outputs/data_quality_report.csv"
    if not os.path.exists(dq_path):
        st.error("Run EDA first so the contract CSV exists.")
    else:
        # Build a minimal cfg dict for payload (use current UI selections)
        cfg_for_llm = _build_cfg_dict() if uploaded else {"data": {"name": name, "path": f"data/uploads/{name}.csv"}}
        payload = build_summary_payload(cfg_for_llm)
        if not inc_sample:
            payload["sample_rows"] = None  # privacy by default

        # Map picker to provider id + safe fallbacks
        provider = "rule"  # default: free/basic
        if "Ollama" in prov:
            provider = "ollama"
            # If Ollama is selected but not reachable, hard-fallback to rule so the UI always works
            if not _ollama_alive(api_base or _default_base):
                st.warning("Ollama is not reachable; using Basic (free) summary instead.")
                provider = "rule"
        elif "OpenAI" in prov:
            if not api_key:
                st.warning("OpenAI API key is empty; falling back to Basic (free) summary.")
                provider = "rule"
            else:
                provider = "openai"

        # Call provider-agnostic LLM
        res = call_llm(
            payload,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
        )

        # Prefer structured JSON if present; otherwise try to extract JSON from text; else use text
        data = res.get("json")
        text = res.get("text")

        # Normalize: if provider put JSON in a string or code fences, parse it
        if isinstance(data, str):
            data = _extract_json_from_text(data)
        if not data and text:
            data = _extract_json_from_text(text)

        if data:
            md = _build_markdown_from_json(data)
            raw_payload = json.dumps(data, indent=2)
        else:
            md = text or "No summary content returned."
            raw_payload = text or ""

        # Cache + display strictly as Markdown by default
        cache_path = Path(f"reports/ai_summary_{name}_{_hash_file(dq_path)[:12]}.md")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(md, encoding="utf-8")

        st.markdown(md)
        st.caption(f"Provider: {provider} · Model: {model or 'auto'}")
        st.download_button(
            label="Download AI summary (Markdown)",
            data=md,
            file_name=cache_path.name,
            mime="text/markdown",
        )
        # Raw payload only in an expander for debugging
        with st.expander("Raw model response"):
            st.code(raw_payload or "<empty>")
        st.success(f"Saved to {cache_path}")