import argparse
import os
import yaml 
import pandas as pd 
import sys 
import json, hashlib, subprocess
import numpy as np 

from datetime import datetime, timezone
def _utc_now_iso_z() -> str:
    # timezone-aware UTC, formatted as RFC3339 "Z" (Zulu)
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def load_data(path:str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext}")



def ensure_dirs():
    for d in ["outputs", "plots", "reports", "logs"]:
        os.makedirs(d, exist_ok=True)

def compute_minimal_metrics(df: pd.DataFrame, table: str) -> pd.DataFrame:
    rows = []
    # table-level row count 
    rows.append([table, "__table__", "row_count", int(len(df)), ""])

    # duplicate row percentage
    dup_pct = float(round(df.duplicated().mean(), 6))
    rows.append([table, "__table__", "duplicate_row_pct", dup_pct, ""])

    # column-level missingness (0..1, rounded to 6 dp)
    miss = df.isna().mean()
    for col, pct in miss.items():
        rows.append([table, col, "missing_pct", float(round(pct, 6)), ""])

    # per-column cardinality (ignoring NaN)
    for col in df.columns:
        uniq = int(df[col].nunique(dropna=True))
        rows.append([table, col, "unique_count", uniq, ""])

    # numeric summaries for numeric columns only 
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        s = df[col].dropna()
        stats = {
            "count_non_null": int(s.size),
            "mean":   float(round(s.mean(), 6)) if s.size else None,
            "std":    float(round(s.std(ddof=1), 6)) if s.size > 1 else None,
            "min":    float(s.min()) if s.size else None,
            "q25":    float(s.quantile(0.25)) if s.size else None,
            "median": float(s.quantile(0.50)) if s.size else None,
            "q75":    float(s.quantile(0.75)) if s.size else None,
            "max":    float(s.max()) if s.size else None,
        }
        for metric, val in stats.items():
            rows.append([table, col, metric, (val if val is not None else ""), ""])

    # Top-N values per column (deterministic)
    for col in df.columns:
        top = _topn_values(df[col], n=5)
        for rank, (_, r) in enumerate(top.iterrows(), start=1):
            rows.append([table, col, f"top{rank}_value", str(r["value"]), ""])
            rows.append([table, col, f"top{rank}_count", int(r["count"]), ""])

    return pd.DataFrame(rows, columns=["table", "column", "metric", "value", "note"])

def _topn_values(s, n=5):
    # determinstic: count desc, value(as str) asc for tie-break 
    vc = s.value_counts(dropna=True)
    dfc = vc.reset_index()
    dfc.columns = ["value", "count"]
    dfc["value_str"] = dfc["value"].astype(str)
    dfc = dfc.sort_values(["count", "value_str"], ascending=[False, True])
    return dfc.head(n)

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
        
def _write_metadata(config_path: str, data_path: str) -> dict:
    meta = {
        "started_at": _utc_now_iso_z(),
        "ended_at": None,
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "config_path": config_path,
        "data_path": data_path,
        "data_sha256": _sha256(data_path),
        "git_sha": None,
    }
    try:
        meta["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    os.makedirs("logs", exist_ok=True)
    with open("logs/run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta

def _finish_metadata(meta: dict) -> None:
    meta["ended_at"] = _utc_now_iso_z()
    with open("logs/run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)




def cmd_run(args):
    ensure_dirs()
    cfg = yaml.safe_load(open(args.config))
    table = cfg["data"]["name"]
    data_path = cfg["data"]["path"]

    if not os.path.exists(data_path):
        sys.stderr.write(f"[ERROR] data file not found: {data_path}\n")
        sys.exit(2)

    meta = _write_metadata(args.config, data_path)

    df = load_data(data_path)
    dq = compute_minimal_metrics(df, table)
    dq.to_csv("outputs/data_quality_report.csv", index=False)
    print("wrote outputs/data_quality_report.csv")

    _finish_metadata(meta)

def build_parser():
    p = argparse.ArgumentParser(prog="eda", description="EDA Copilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run the pipeline")
    pr.add_argument("--config", required=True, help="Path to YAML config")
    pr.set_defaults(func=cmd_run)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
