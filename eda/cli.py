import argparse
import os
import yaml 
import pandas as pd 
import sys 
import json, hashlib, subprocess
import numpy as np 
import csv 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 


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

def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def _validate_csv_schema(path: str, expected_cols: list[str]) -> list[list]:
    bad_rows = 0
    samples = []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, [])

        # --- header names/order check ---
        hdr_row = None
        if header != expected_cols:
            note = f"expected={expected_cols} got={header}"
            table = os.path.splitext(os.path.basename(path))[0]
            hdr_row = [table, "__table__", "schema_header_mismatch", 1, note]

        # --- per-row field-count check ---
        for i, row in enumerate(rdr, start=2):  # 1-based lines; 2 = first data line
            if len(row) != len(expected_cols):
                bad_rows += 1
                if len(samples) < 5:
                    samples.append((i, row))

    os.makedirs("logs", exist_ok=True)
    if samples:
        with open("logs/schema_samples.txt", "w") as out:
            for ln, r in samples:
                out.write(f"line {ln}: {r}\n")

    table = os.path.splitext(os.path.basename(path))[0]
    rows = [[table, "__table__", "schema_bad_row_count", int(bad_rows), ""]]
    if hdr_row:
        rows.append(hdr_row)
    return rows

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

def _apply_checks(df, table, checks):
    rows = []
    if not checks:
        return rows
    for col, spec in checks.items():   # ← items(), not item()
        if col not in df.columns:
            continue
        s = df[col]

        # numeric range
        if "min" in spec or "max" in spec:
            s_num = pd.to_numeric(s, errors="coerce")
            lo = spec.get("min", float("-inf"))
            hi = spec.get("max", float("inf"))    # ← +inf by default
            too_low  = (s_num < lo).sum()
            too_high = (s_num > hi).sum()
            rows.append([table, col, "violations_out_of_range_count",
                         int(too_low + too_high), ""])

        # allowed categories
        if "allowed" in spec:
            allowed = {str(a).strip() for a in spec["allowed"]}
            s_norm = s.astype(str).str.strip()
            invalid = (~s.isna()) & (~s_norm.isin(allowed))  # ← use s_norm
            rows.append([table, col, "invalid_category_count", int(invalid.sum()), ""])
    return rows

def _check_primary_key(df: pd.DataFrame, table: str, keys: list[str]) -> list[list]:
    if not keys: 
        return []
    grp = df.groupby(keys, dropna=False).size()
    viol = int((grp > 1).sum())
    return [[table, "__table__", "duplicate_key_groups_count", viol, f"keys={keys}"]]


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

def _save_series_bar(s: pd.Series, title: str, png_path: str, csv_path: str, xlabel: str, ylabel: str):
    s.to_csv(csv_path, header=[ylabel])
    plt.figure()
    s.plot(kind="bar")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(png_path); plt.close()


def _save_histogram(s: pd.Series, title: str, png_path: str, csv_path: str, bins: int = 10):
    s = s.dropna().astype(float)
    counts, edges = np.histogram(s, bins=bins)
    pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts}).to_csv(csv_path, index=False)
    plt.figure()
    plt.hist(s, bins=edges)
    plt.title(title); plt.xlabel("value"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(png_path); plt.close()


def _save_boxplot(s: pd.Series, title: str, png_path: str, stats_path: str):
    s = s.dropna().astype(float)
    stats = {
        "min": float(s.min()),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    plt.figure()
    plt.boxplot(s, vert=True, whis=1.5)
    plt.title(title); plt.ylabel("value")
    plt.tight_layout(); plt.savefig(png_path); plt.close()

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

def _write_reports():
    os.makedirs("reports", exist_ok=True)
    dq = pd.read_csv("outputs/data_quality_report.csv")

    # Findings memo (reads only the contract CSV)
    with open("reports/findings_memo.md", "w") as f:
        f.write("# Findings (from outputs/data_quality_report.csv)\n\n")

        # Rows (table-level)
        row_q = dq[(dq["column"] == "__table__") & (dq["metric"] == "row_count")]
        if not row_q.empty:
            row_cnt = int(float(row_q["value"].iloc[0]))
            f.write(f"- **Rows:** {row_cnt}\n")

        # Top-3 missingness
        miss = dq[(dq["metric"] == "missing_pct") & (dq["column"] != "__table__")].copy()
        if not miss.empty:
            miss["value"] = miss["value"].astype(float)
            f.write("\n## Missingness (top 3)\n")
            for _, r in miss.sort_values("value", ascending=False).head(3).iterrows():
                f.write(f"- {r['column']}: {r['value']}\n")

        # Validation issues
        viol = dq[dq["metric"].isin([
            "violations_out_of_range_count",
            "invalid_category_count",
            "duplicate_key_groups_count",
        ])].copy()
        if not viol.empty:
            f.write("\n## Validation issues\n")
            for _, r in viol.iterrows():
                v = int(float(r["value"])) if r["value"] != "" else 0
                f.write(f"- {r['column']} · {r['metric']} = {v}\n")

        # Sources footer (cite only computed artifacts)
        f.write("\n## Sources\n")
        f.write("- outputs/data_quality_report.csv\n")
        f.write("- logs/run_metadata.json\n")

    # Next-actions memo
    with open("reports/next_actions.md", "w") as f:
        f.write("# Next Actions\n")
        f.write("- Address highest missingness first.\n")
        f.write("- Resolve duplicate keys/rows if present.\n")
        f.write("- Fix invalid categories or update allow-list in YAML.\n")
        f.write("- Review out-of-range numeric values for entry/ETL errors.\n")
        f.write("\n## Sources\n")
        f.write("- outputs/data_quality_report.csv\n")
        f.write("- logs/run_metadata.json\n")


def cmd_run(args):
    ensure_dirs()
    cfg = yaml.safe_load(open(args.config))
    table = cfg["data"]["name"]
    data_path = cfg["data"]["path"]

    if not os.path.exists(data_path):
        sys.stderr.write(f"[ERROR] data file not found: {data_path}\n")
        sys.exit(2)

    meta = _write_metadata(args.config, data_path)

    # schema check (CSV only)
    if data_path.lower().endswith(".csv"):
        schema_rows = _validate_csv_schema(data_path, ['fight_id', 'winner', 'method', 'round', 'time'])
    else:
        schema_rows = []

    df = load_data(data_path)
    df = _normalize_frame(df)
    dq = compute_minimal_metrics(df, table)

    if schema_rows:
        dq = pd.concat([dq, pd.DataFrame(schema_rows, columns=["table","column","metric","value","note"])],
                   ignore_index=True)

    # apply optional validation rules from YAML 
    checks = cfg.get("checks") or {}
    extra_rows = _apply_checks(df, table, checks)
    if extra_rows:
        dq = pd.concat(
            [dq, pd.DataFrame(extra_rows, columns=["table", "column", "metric", "value", "note"])], 
            ignore_index=True
        )

    # primary-key duplicate detection
    key_spec = (cfg.get("keys") or {}).get("primary") or []
    pk_rows = _check_primary_key(df, table, key_spec)
    if pk_rows:
        dq = pd.concat([dq, pd.DataFrame(pk_rows, columns=["table","column","metric","value","note"])],
                       ignore_index=True)



    dq.to_csv("outputs/data_quality_report.csv", index=False)
    print("wrote outputs/data_quality_report.csv")

    _write_reports()

    # 5 validated plots (each with a companion values/stats file)
    # 1) Missingness by column (bar)
    miss = df.isna().mean().sort_values(ascending=False)
    _save_series_bar(miss, "Missingness by Column",
                     "plots/missingness.png", "plots/missingness_values.csv",
                     "column", "missing_pct")

    # 2) Unique count by column (bar)
    uniq = df.nunique(dropna=True).sort_values(ascending=False)
    _save_series_bar(uniq, "Unique Count by Column",
                     "plots/unique_count.png", "plots/unique_count_values.csv",
                     "column", "unique_count")

    # 3) Histogram of time (if present)
    if "time" in df.columns:
        _save_histogram(df["time"], "Histogram: time",
                        "plots/hist_time.png", "plots/hist_time_values.csv")

        # 4) Boxplot of time
        _save_boxplot(df["time"], "Boxplot: time",
                      "plots/box_time.png", "plots/box_time_stats.csv")

    # 5) Top-5 method values (bar)
    if "method" in df.columns:
        top = _topn_values(df["method"], n=5)
        _save_series_bar(top.set_index("value")["count"], "Top-5: method",
                         "plots/top_method.png", "plots/top_method_values.csv",
                         "method", "count")

    _finish_metadata(meta)

def cmd_doctor(args):
    problems = []

    # 1) Contract CSV exists, has exact headers, and includes row_count
    csv_path = "outputs/data_quality_report.csv"
    expected = ["table", "column", "metric", "value", "note"]
    if not os.path.exists(csv_path):
        problems.append(f"missing {csv_path}")
        dq = None
    else:
        try:
            dq = pd.read_csv(csv_path)
        except Exception as e:
            problems.append(f"cannot read {csv_path}: {e}")
            dq = None
        else:
            if list(dq.columns) != expected:
                problems.append(f"bad headers in {csv_path}: {list(dq.columns)} != {expected}")
            # Use boolean indexing (safer than .query strings)
            has_row_count = not dq[(dq["column"] == "__table__") & (dq["metric"] == "row_count")].empty
            if not has_row_count:
                problems.append("row_count metric missing in contract CSV")

    # 2) Metadata exists + required keys + UTC timestamps
    meta_path = "logs/run_metadata.json"
    must = {"started_at", "ended_at", "python", "pandas", "numpy", "config_path", "data_path", "data_sha256"}
    if not os.path.exists(meta_path):
        problems.append(f"missing {meta_path}")
    else:
        try:
            meta = json.load(open(meta_path))
        except Exception as e:
            problems.append(f"cannot read {meta_path}: {e}")
        else:
            missing = sorted(list(must - set(meta.keys())))
            if missing:
                problems.append(f"metadata missing keys: {missing}")
            for k in ("started_at", "ended_at"):
                v = meta.get(k, "")
                if not (isinstance(v, str) and (v.endswith("Z") or v.endswith("+00:00"))):
                    problems.append(f"{k} not UTC/RFC3339: {v}")

    # 3) Plots: exactly 5 PNGs + companion *_values.csv or *_stats.csv for each
    import glob
    pngs = sorted(glob.glob("plots/*.png"))
    if len(pngs) != 5:
        problems.append(f"need 5 plots, found {len(pngs)}: {pngs}")
    for p in pngs:
        stem = os.path.splitext(os.path.basename(p))[0]
        candidate_values = os.path.join("plots", f"{stem}_values.csv")
        candidate_stats  = os.path.join("plots", f"{stem}_stats.csv")
        if not (os.path.exists(candidate_values) or os.path.exists(candidate_stats)):
            problems.append(f"missing companion values/stats for {p}")

    # 4) Memos exist and include Sources footer lines
    for rp in ["reports/findings_memo.md", "reports/next_actions.md"]:
        if not os.path.exists(rp):
            problems.append(f"missing {rp}")
        else:
            txt = open(rp, encoding="utf-8").read()
            for needle in ["## Sources", "- outputs/data_quality_report.csv", "- logs/run_metadata.json"]:
                if needle not in txt:
                    problems.append(f"missing '{needle}' in {rp}")

    if problems:
        print("❌ doctor found issues:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("✅ doctor: all checks passed")



def build_parser():
    p = argparse.ArgumentParser(prog="eda", description="EDA Copilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run the pipeline")
    pdx = sub.add_parser("doctor", help="Check output contracts and metadata")
    pdx.set_defaults(func=cmd_doctor)
    pr.add_argument("--config", required=True, help="Path to YAML config")
    pr.set_defaults(func=cmd_run)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
