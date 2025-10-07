# eda/llm_providers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import os, json

@dataclass
class LLMResult:
    content: str
    model: str
    tokens: int | None = None

# Rule-based (free) 
def summarize_rule_based(payload: Dict[str, Any]) -> LLMResult:
    cfg = payload.get("config", {}) or {}
    name = payload.get("dataset_name") or (cfg.get("data") or {}).get("name") or "dataset"
    contract = payload.get("contract_head") or []
    meta = payload.get("meta") or {}

    rc = None
    for r in contract:
        if r.get("column") == "__table__" and r.get("metric") == "row_count":
            try: rc = int(float(r.get("value")))
            except: pass
            break

    viols, miss = [], []
    for r in contract:
        m = str(r.get("metric"))
        if m in ("invalid_category_count","violations_out_of_range_count","duplicate_key_groups_count"):
            try: v = int(float(r.get("value"))) if r.get("value") not in ("", None) else 0
            except: v = 0
            if v > 0: viols.append((r.get("column"), m, v))
        if m == "missing_pct" and r.get("column") != "__table__":
            try: miss.append((r.get("column"), float(r.get("value"))))
            except: pass
    miss.sort(key=lambda x: x[1], reverse=True)
    miss_top = miss[:5]

    parts: List[str] = []
    parts.append(f"## Overview\nDataset **{name}**{' with ' + str(rc) + ' rows' if rc is not None else ''}.")
    if meta:
        py = (meta.get("python") or "").split()[0]
        parts.append(f"Environment captured (Python {py or 'n/a'}).")

    parts.append("\n## Data quality")
    if miss_top:
        parts.append("Top missingness:")
        for col, pct in miss_top:
            parts.append(f"- {col}: {pct:.3f}")
    else:
        parts.append("- No missingness metrics available or all zeros.")

    if viols:
        parts.append("\nValidation issues:")
        for col, m, v in viols:
            parts.append(f"- {col} · {m} = {v}")
    else:
        parts.append("\nNo validation issues found in the contract metrics.")

    parts.append("\n## Suggested next actions")
    if miss_top:
        parts.append(f"1) Address missing values in **{miss_top[0][0]}** first (impute or drop).")
    else:
        parts.append("1) Confirm no missingness and focus on value distributions.")
    if viols:
        parts.append("2) Fix invalid/out-of-range values; update YAML ranges/allow-lists if needed.")
    parts.append("3) Check primary key duplicates and ensure uniqueness.")
    parts.append("4) Document assumptions and update the config to prevent regressions.")
    parts.append("\n_Caveat: This brief is rule-based; it does not infer semantics beyond the contract CSV._")

    return LLMResult(content="\n".join(parts), model="rule-based", tokens=None)

# Ollama (local, free for user) 
def summarize_ollama(payload: Dict[str, Any], model: str = "llama3.1:8b-instruct") -> LLMResult:
    import requests
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
    slim = dict(payload)
    if slim.get("sample_rows") and len(slim["sample_rows"]) > 30:
        slim["sample_rows"] = slim["sample_rows"][:30]
    system = (
        "You are a senior data analyst. Given a data-quality contract and optional sample, "
        "write a crisp Markdown brief with: Overview, Data quality, Next actions (ordered), Caveats. "
        "Use counts/percents from the contract when possible. Keep it practical."
    )
    user = "JSON input:\n" + json.dumps(slim, ensure_ascii=False)
    try:
        resp = requests.post(
            url,
            json={"model": model, "messages":[{"role":"system","content":system},{"role":"user","content":user}], "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content") or ""
        return LLMResult(content=content, model=model, tokens=None)
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")