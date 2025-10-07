import os, json, hashlib
from typing import Any, Dict, List, Optional

"""
Unified LLM interface for EDA-Copilot.

Purposes
--------
- Provide a **free, deterministic** rule-based summary by default (no API keys).
- Optionally support **OpenAI** or **Ollama** if the user configures them.
- Return a consistent JSON schema with a short natural-language **blurb**.

Environment variables
---------------------
LLM_PROVIDER = rule | ollama | openai   (default: rule)
LLM_MODEL    = model name per provider  (e.g., 'llama3.1:8b-instruct' or 'gpt-4o-mini')
OPENAI_API_KEY, OPENAI_BASE_URL (optional for OpenAI)
OLLAMA_BASE_URL (default http://localhost:11434)

Return value (dict)
-------------------
{
  "ok": bool,
  "json": {
     "title": str,
     "one_line": str,
     "domain_guess": str,
     "key_facts": [str, ...],
     "issues": [str, ...],
     "actions": [str, ...],
     "questions": [str, ...],
     "blurb": str                # NEW: short human-friendly paragraph
  } | None,
  "text": str|None,              # raw text (when provider returns text)
  "error": str|None
}
"""


# Payload utilities


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_summary_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a compact context object for summarization.
    Tries to add: contract_head (first 50 rows), metadata, small data sample.
    Safe for UI/CLI, works without network.
    """
    # Lazy imports
    import pandas as pd
    import yaml

    dq_path = "outputs/data_quality_report.csv"
    meta_path = "logs/run_metadata.json"

    payload: Dict[str, Any] = {
        "dataset_name": (cfg.get("data") or {}).get("name"),
        "source_path": (cfg.get("data") or {}).get("path"),
        "config": cfg,
        "contract_sha256": None,
        "meta": None,
        "contract_head": None,
        "sample_rows": None,
        "highlights": None,
    }

    # Contract head + hash
    if os.path.exists(dq_path):
        try:
            dq = pd.read_csv(dq_path)
            payload["contract_head"] = dq.head(50).to_dict(orient="records")
            payload["contract_sha256"] = _hash_file(dq_path)
            payload["highlights"] = _compute_highlights_from_contract_df(dq)
        except Exception:
            pass

    # Metadata
    if os.path.exists(meta_path):
        try:
            payload["meta"] = yaml.safe_load(open(meta_path)) or {}
        except Exception:
            payload["meta"] = None

    # Small data sample (optional; helps semantic guesses)
    try:
        data_path = (cfg.get("data") or {}).get("path")
        if data_path:
            ext = os.path.splitext(data_path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(data_path, nrows=30)
            elif ext == ".parquet":
                df = pd.read_parquet(data_path).head(30)
            elif ext in (".xlsx", ".xls"):
                sheet = (cfg.get("data") or {}).get("sheet")
                df = pd.read_excel(data_path, sheet_name=(sheet if sheet is not None else 0)).head(30)
            else:
                df = None
            if df is not None:
                payload["sample_rows"] = df.astype(str).to_dict(orient="records")
    except Exception:
        pass

    return payload


def _compute_highlights_from_contract_df(dq) -> Dict[str, Any]:
    """
    Derive compact, typed highlights from the full contract CSV DataFrame.
    Returns:
      {
        "rows": int|None,
        "duplicate_row_pct": float|None,
        "duplicate_key_groups_count": int|None,
        "missing_top": List[Dict[column, missing_pct]],
        "violations": List[Dict[column, metric, value]],
        "unique_top": List[Dict[column, unique_count]],
      }
    """
    rows = None
    dup_pct = None
    dup_keys = None

    # table-level metrics
    try:
        rc = dq[(dq["column"] == "__table__") & (dq["metric"] == "row_count")]["value"]
        if not rc.empty:
            rows = int(float(rc.iloc[0]))
    except Exception:
        rows = None

    try:
        dp = dq[(dq["column"] == "__table__") & (dq["metric"] == "duplicate_row_pct")]["value"]
        if not dp.empty:
            dup_pct = float(dp.iloc[0])
    except Exception:
        dup_pct = None

    try:
        dk = dq[(dq["column"] == "__table__") & (dq["metric"] == "duplicate_key_groups_count")]["value"]
        if not dk.empty:
            dup_keys = int(float(dk.iloc[0]))
    except Exception:
        dup_keys = None

    # missingness top-k
    missing_rows = dq[(dq["metric"] == "missing_pct") & (dq["column"] != "__table__")][["column", "value"]].copy()
    miss_list: List[Dict[str, Any]] = []
    if not missing_rows.empty:
        try:
            missing_rows["value"] = missing_rows["value"].astype(float)
        except Exception:
            pass
        missing_rows = missing_rows.sort_values("value", ascending=False)
        for _, r in missing_rows.head(5).iterrows():
            try:
                miss_list.append({"column": str(r["column"]), "missing_pct": float(r["value"])})
            except Exception:
                pass

    # violations where value > 0
    viol_metrics = {"invalid_category_count", "violations_out_of_range_count", "duplicate_key_groups_count"}
    viol_rows = dq[dq["metric"].isin(viol_metrics)][["column", "metric", "value"]].copy()
    viols: List[Dict[str, Any]] = []
    if not viol_rows.empty:
        for _, r in viol_rows.iterrows():
            try:
                v = int(float(r["value"])) if r["value"] not in ("", None) else 0
            except Exception:
                v = 0
            if v > 0:
                viols.append({"column": str(r["column"]), "metric": str(r["metric"]), "value": v})

    # unique_count top-k
    uniq_rows = dq[dq["metric"] == "unique_count"][["column", "value"]].copy()
    uniq_list: List[Dict[str, Any]] = []
    if not uniq_rows.empty:
        try:
            uniq_rows["value"] = uniq_rows["value"].astype(float)
        except Exception:
            pass
        uniq_rows = uniq_rows.sort_values("value", ascending=False).head(10)
        for _, r in uniq_rows.iterrows():
            try:
                uniq_list.append({"column": str(r["column"]), "unique_count": int(float(r["value"]))})
            except Exception:
                pass

    return {
        "rows": rows,
        "duplicate_row_pct": dup_pct,
        "duplicate_key_groups_count": dup_keys,
        "missing_top": miss_list,
        "violations": viols,
        "unique_top": uniq_list,
    }

# Rule-based (free) summary


def _compose_blurb(name: str,
                   rows: Optional[int],
                   ncols: Optional[int],
                   miss_top: List[tuple],
                   viols: List[tuple]) -> str:
    """Create a short, friendly paragraph about the dataset."""
    parts = []
    base = f"'{name}' looks like a tabular dataset"
    if rows is not None and ncols is not None:
        base += f" with ~{rows} rows and {ncols} columns."
    elif rows is not None:
        base += f" with ~{rows} rows."
    elif ncols is not None:
        base += f" with {ncols} columns."
    else:
        base += "."

    parts.append(base)

    if miss_top:
        top3 = ", ".join([f"{c} ({pct:.1%})" for c, pct in miss_top[:3]])
        parts.append(f"Highest missingness: {top3}.")
    else:
        parts.append("No notable missingness in the contract metrics.")

    if viols:
        ex = "; ".join([f"{c} {m.replace('_count','').replace('_',' ')}={v}" for c, m, v in viols[:2]])
        parts.append(f"Some validation issues were detected (e.g., {ex}).")
    else:
        parts.append("No validation issues were detected given the current checks.")

    return " ".join(parts)


def _summarize_rule_based(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Free, deterministic summary based solely on the contract payload.
    Produces JSON in the agreed schema (title, one_line, key_facts, ... + blurb).
    """
    cfg = payload.get("config") or {}
    name = payload.get("dataset_name") or (cfg.get("data") or {}).get("name") or "dataset"
    hl = payload.get("highlights") or {}

    rows = hl.get("rows")
    if rows is None:
        # fallback to contract_head scan (legacy)
        contract = payload.get("contract_head") or {}
        for r in contract:
            if r.get("column") == "__table__" and r.get("metric") == "row_count":
                try:
                    rows = int(float(r.get("value")))
                except Exception:
                    rows = None
                break

    miss_top_pairs: List[tuple] = []
    if hl.get("missing_top"):
        for item in hl["missing_top"]:
            try:
                miss_top_pairs.append((item["column"], float(item["missing_pct"])))
            except Exception:
                pass
    else:
        # fallback to contract_head
        contract = payload.get("contract_head") or {}
        miss = []
        for r in contract:
            if r.get("metric") == "missing_pct" and r.get("column") != "__table__":
                try:
                    miss.append((r.get("column"), float(r.get("value"))))
                except Exception:
                    pass
        miss.sort(key=lambda x: x[1], reverse=True)
        miss_top_pairs = miss[:5]

    issues_raw: List[str] = []
    viols_pairs: List[tuple] = []
    if hl.get("violations"):
        for v in hl["violations"]:
            issues_raw.append(f"{v['column']} · {v['metric']} = {v['value']}")
            viols_pairs.append((v["column"], v["metric"], v["value"]))
    else:
        # fallback to contract_head
        contract = payload.get("contract_head") or {}
        for r in contract:
            m = str(r.get("metric"))
            if m in ("invalid_category_count", "violations_out_of_range_count", "duplicate_key_groups_count"):
                try:
                    v = int(float(r.get("value"))) if r.get("value") not in ("", None) else 0
                except Exception:
                    v = 0
                if v > 0:
                    issues_raw.append(f"{r.get('column')} · {m} = {v}")
                    viols_pairs.append((r.get("column"), m, v))

    dup_pct = hl.get("duplicate_row_pct")
    dup_keys = hl.get("duplicate_key_groups_count")
    key_facts: List[str] = []
    if rows is not None:
        key_facts.append(f"Rows: {rows}")
    if dup_pct is not None:
        key_facts.append(f"Duplicate row %: {dup_pct:.6f}")
    if dup_keys is not None:
        key_facts.append(f"Duplicate key groups: {dup_keys}")
    if miss_top_pairs:
        for col, pct in miss_top_pairs:
            key_facts.append(f"Missing {col}: {pct:.3f}")

    # Infer ncols from sample rows (if present)
    ncols = None
    sample = payload.get("sample_rows") or []
    if sample:
        cols = set()
        for row in sample:
            cols.update(row.keys())
        ncols = len(cols) if cols else None

    actions: List[str] = []
    if miss_top_pairs:
        actions.append(f"Impute or drop missing values starting with **{miss_top_pairs[0][0]}**.")
    else:
        actions.append("Confirm no missingness; focus on value distributions & range checks.")
    if viols_pairs:
        actions.append("Fix invalid/out-of-range values; tighten YAML allow-lists/ranges.")
    actions.append("Ensure primary key uniqueness (see duplicate_key_groups_count).")
    actions.append("Document assumptions; keep config under version control.")

    questions = [
        "Which columns are categorical vs numeric for deeper profiling?",
        "What ranges are considered valid for key metrics?",
        "Is there a target variable or downstream task (modeling, dashboarding)?",
    ]

    blurb = _compose_blurb(name, rows, ncols, miss_top_pairs, viols_pairs)

    result = {
        "title": f"EDA Summary — {name}",
        "one_line": f"Contract-reviewed snapshot for '{name}'" + (f" ({rows} rows)" if rows is not None else ""),
        "domain_guess": "generic",
        "key_facts": key_facts or ["No contract facts available."],
        "issues": issues_raw or ["No validation issues found in contract."],
        "actions": actions,
        "questions": questions,
        "blurb": blurb,
    }
    return {"ok": True, "json": result, "text": None, "error": None}



# OpenAI / Ollama providers


def _summarize_openai(payload: Dict[str, Any],
                      model: Optional[str],
                      api_key: Optional[str],
                      api_base: Optional[str]) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as e:
        return {"ok": False, "json": None, "text": None, "error": f"OpenAI SDK not installed: {e}"}

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"),
                    base_url=api_base or os.getenv("OPENAI_BASE_URL", None))

    system = "You are a careful data analyst. Use only the provided numbers; do not invent results. Return concise JSON."
    user_msg = {
        "instruction": (
            "Summarize the dataset and list 5–8 next actions. "
            "Return JSON with keys: title, one_line, domain_guess, key_facts[], issues[], actions[], questions[], blurb."
        ),
        "context": payload,
    }

    try:
        resp = client.chat.completions.create(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_msg)},
            ],
        )
        txt = resp.choices[0].message.content
        try:
            return {"ok": True, "json": json.loads(txt), "text": txt, "error": None}
        except Exception:
            return {"ok": True, "json": None, "text": txt, "error": None}
    except Exception as e:
        return {"ok": False, "json": None, "text": None, "error": str(e)}


def _summarize_ollama(payload: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    try:
        import requests
    except Exception as e:
        return {"ok": False, "json": None, "text": None, "error": f"requests not installed: {e}"}

    # Build URLs and model
    base = (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") or "").rstrip("/")
    chat_url = f"{base}/api/chat"
    gen_url = f"{base}/api/generate"
    mdl = model or os.getenv("LLM_MODEL", "llama3.2:3b")

    system = "You are a careful data analyst. Use only the provided numbers; do not invent results. Return concise JSON."
    user_msg = {
        "instruction": (
            "Summarize the dataset and list 5–8 next actions. "
            "Return JSON with keys: title, one_line, domain_guess, key_facts[], issues[], actions[], questions[], blurb."
        ),
        "context": payload,
    }

    # 1) Try the newer /api/chat endpoint
    try:
        r = requests.post(
            chat_url,
            json={
                "model": mdl,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_msg)},
                ],
                "options": {"temperature": 0.2},
                "stream": False,
            },
            timeout=120,
        )
        if r.status_code == 404:
            raise RuntimeError("no_chat_endpoint")
        r.raise_for_status()
        data = r.json()
        # Prefer message.content; fall back to 'response'; if still empty, dump the whole payload
        txt = (data.get("message") or {}).get("content") or data.get("response") or ""
        if not (isinstance(txt, str) and txt.strip()):
            txt = json.dumps(data)  # ensure UI has something to show

        try:
            return {"ok": True, "json": json.loads(txt), "text": txt, "error": None}
        except Exception:
            return {"ok": True, "json": None, "text": txt, "error": None}
    except Exception:
        # 2) Fallback to the older /api/generate endpoint
        try:
            prompt = f"{system}\n\nUSER:\n{json.dumps(user_msg)}"
            rg = requests.post(
                gen_url,
                json={
                    "model": mdl,
                    "prompt": prompt,
                    "options": {"temperature": 0.2},
                    "stream": False,
                },
                timeout=120,
            )
            rg.raise_for_status()
            data = rg.json()
            txt = data.get("response") or ""
            if not (isinstance(txt, str) and txt.strip()):
                txt = json.dumps(data)

            try:
                return {"ok": True, "json": json.loads(txt), "text": txt, "error": None}
            except Exception:
                return {"ok": True, "json": None, "text": txt, "error": None}
        except Exception as e2:
            return {"ok": False, "json": None, "text": None, "error": f"Ollama request failed: {e2}"}



# Public API

def call_llm(payload: Dict[str, Any],
             provider: Optional[str] = None,
             model: Optional[str] = None,
             api_key: Optional[str] = None,
             api_base: Optional[str] = None) -> Dict[str, Any]:
    """Dispatch to rule/openai/ollama with a stable return shape.

    Default provider is 'rule' so the app works for everyone out of the box.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "rule")).lower()

    # Slightly slim payload (avoid giant samples)
    slim = dict(payload)
    sr = slim.get("sample_rows")
    if isinstance(sr, list) and len(sr) > 30:
        slim["sample_rows"] = sr[:30]
    ch = slim.get("contract_head")
    if isinstance(ch, list) and len(ch) > 50:
        slim["contract_head"] = ch[:50]

    if provider in ("rule", "fake"):  # 'fake' kept as alias
        return _summarize_rule_based(slim)
    if provider == "openai":
        return _summarize_openai(slim, model, api_key, api_base)
    if provider == "ollama":
        return _summarize_ollama(slim, model)

    # Unknown provider -> rule-based fallback
    return _summarize_rule_based(slim)