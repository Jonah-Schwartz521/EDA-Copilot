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


# --- Begin inserted helpers for config-driven plot file naming and doctor checks ---
def _expected_plot_files(table: str, spec: dict) -> tuple[str, str, str]:
    """Return (kind, png_path, values_path_or_stats_path)."""
    k = spec.get("kind")
    if k == "missingness":
        return k, f"plots/{table}_missingness.png", f"plots/{table}_missingness_values.csv"
    if k == "topn":
        col = spec["column"]
        return k, f"plots/{table}_topn_{col}.png", f"plots/{table}_topn_{col}_values.csv"
    if k == "numeric_hist":
        col = spec["column"]
        return k, f"plots/{table}_hist_{col}.png", f"plots/{table}_hist_{col}_values.csv"
    if k == "numeric_box":
        col = spec["column"]
        return k, f"plots/{table}_box_{col}.png", f"plots/{table}_box_{col}_stats.csv"
    if k == "cardinality_topk":
        return k, f"plots/{table}_cardinality_topk.png", f"plots/{table}_cardinality_topk_values.csv"
    return k, None, None


def _doctor_check_plots(dq: pd.DataFrame, cfg: dict) -> list[str]:
    """Return list of human-readable problems; empty if all good."""
    problems = []
    data = cfg.get("data") or {}
    table = data.get("name", "table")
    plots = cfg.get("plots") or []

    # Normalize DQ for lookups
    dq = dq.copy()
    dq["metric"] = dq["metric"].astype(str)
    dq["column"] = dq["column"].astype(str)

    for spec in plots:
        kind, png_path, vals_path = _expected_plot_files(table, spec)
        if not kind:
            problems.append(f"unknown plot kind in config: {spec}")
            continue
        if not os.path.exists(png_path):
            problems.append(f"missing PNG for {kind}: {png_path}")
        if not os.path.exists(vals_path):
            problems.append(f"missing values/stats file for {kind}: {vals_path}")
            # if values missing, skip deeper comparison to avoid exceptions
            continue

        try:
            vals = pd.read_csv(vals_path)
        except Exception as e:
            problems.append(f"cannot read {vals_path}: {e}")
            continue

        # Kind-specific content checks
        if kind == "missingness":
            # Expect columns: column,missing_pct
            req_cols = {"column", "missing_pct"}
            if not req_cols.issubset(set(vals.columns)):
                problems.append(f"{vals_path} missing columns {req_cols}")
            else:
                miss_dq = dq[(dq["metric"] == "missing_pct") & (dq["column"] != "__table__")][["column","value"]].copy()
                miss_dq["value"] = miss_dq["value"].astype(float).round(6)
                vals["missing_pct"] = vals["missing_pct"].astype(float).round(6)
                merged = vals.merge(miss_dq, on="column", how="left", suffixes=("_vals","_dq"))
                bad = merged[merged["missing_pct"] != merged["value"]]
                if not bad.empty:
                    problems.append(f"{vals_path} does not match DQ missing_pct for columns: {bad['column'].tolist()}")

        elif kind == "topn":
            # Expect: value,count columns, compare to DQ topk pairs
            col = spec["column"]
            if not {"value","count"}.issubset(set(vals.columns)):
                problems.append(f"{vals_path} must have columns ['value','count']")
            else:
                # Build expected ordered pairs from DQ
                exp_pairs = []
                rank = 1
                while True:
                    v = dq.loc[(dq["column"]==col) & (dq["metric"]==f"top{rank}_value"), "value"]
                    c = dq.loc[(dq["column"]==col) & (dq["metric"]==f"top{rank}_count"), "value"]
                    if v.empty or c.empty:
                        break
                    # c can be stored as float in CSV; cast robustly
                    exp_pairs.append((str(v.iloc[0]), int(float(c.iloc[0]))))
                    rank += 1
                got_pairs = list(zip(vals["value"].astype(str).tolist(),
                                     vals["count"].astype(int).tolist()))
                if exp_pairs[:len(got_pairs)] != got_pairs:
                    problems.append(f"{vals_path} topN differs from DQ for column '{col}'")

        elif kind == "numeric_box":
            # Expect: single-row stats with min,q25,median,q75,max
            required = ["min","q25","median","q75","max"]
            missing = [c for c in required if c not in vals.columns]
            if missing:
                problems.append(f"{vals_path} missing columns {missing}")
            else:
                col = spec["column"]
                # DQ has per-metric rows
                def _dq_val(m):
                    s = dq.loc[(dq["column"]==col) & (dq["metric"]==m), "value"]
                    return None if s.empty else float(s.iloc[0])
                exp = {m: _dq_val(m) for m in required}
                got = {m: float(vals[m].iloc[0]) for m in required}
                # round to avoid tiny FP diffs
                exp = {k: (None if v is None else round(v, 6)) for k,v in exp.items()}
                got = {k: round(v, 6) for k,v in got.items()}
                diffs = [k for k in required if exp[k] is not None and exp[k] != got[k]]
                if diffs:
                    problems.append(f"{vals_path} stats differ from DQ for {col}: {diffs}")

        elif kind == "numeric_hist":
            # Expect: bin_left,bin_right,count; sum(count) == DQ count_non_null
            need = {"bin_left","bin_right","count"}
            if not need.issubset(set(vals.columns)):
                problems.append(f"{vals_path} missing columns {need}")
            else:
                col = spec["column"]
                cnt = dq.loc[(dq["column"]==col) & (dq["metric"]=="count_non_null"), "value"]
                if not cnt.empty:
                    expected = int(float(cnt.iloc[0]))
                    got = int(vals["count"].sum())
                    if expected != got:
                        problems.append(f"{vals_path} total count {got} != DQ count_non_null {expected} for {col}")

        elif kind == "cardinality_topk":
            # Expect: column,unique_count; check counts match DQ
            need = {"column","unique_count"}
            if not need.issubset(set(vals.columns)):
                problems.append(f"{vals_path} missing columns {need}")
            else:
                uq = dq[dq["metric"]=="unique_count"][["column","value"]].copy()
                uq["value"] = uq["value"].astype(float)
                merged = vals.merge(uq, on="column", how="left")
                bad = merged[ merged["unique_count"].astype(float) != merged["value"] ]
                if not bad.empty:
                    problems.append(f"{vals_path} unique_count mismatches for: {bad['column'].tolist()}")

    return problems


# --- End inserted helpers ---


# --- Begin inserted renderer for config-driven plots ---
def _render_plots(df: pd.DataFrame, table: str, plots_spec: list[dict]):
    os.makedirs("plots", exist_ok=True)
    for spec in plots_spec or []:
        kind = spec.get("kind")
        if kind == "missingness":
            miss = df.isna().mean().astype(float)
            vals = miss.reset_index()
            vals.columns = ["column","missing_pct"]
            vals = vals.sort_values("missing_pct", ascending=False)
            _, png, csvp = _expected_plot_files(table, spec)
            vals.to_csv(csvp, index=False)
            plt.figure(); plt.bar(vals["column"], vals["missing_pct"])
            plt.title("Missingness by Column"); plt.xlabel("column"); plt.ylabel("missing_pct")
            plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(png); plt.close()

        elif kind == "topn":
            col = spec["column"]
            n = int(spec.get("n", 5))
            top = _topn_values(df[col], n=n)
            vals = top[["value","count"]].copy()
            _, png, csvp = _expected_plot_files(table, spec)
            vals.to_csv(csvp, index=False)
            plt.figure(); plt.bar(vals["value"].astype(str), vals["count"].astype(int))
            plt.title(f"Top-{n}: {col}"); plt.xlabel(col); plt.ylabel("count")
            plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(png); plt.close()

        elif kind == "numeric_hist":
            col = spec["column"]
            bins = int(spec.get("bins", 10))
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            counts, edges = np.histogram(s, bins=bins)
            vals = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
            _, png, csvp = _expected_plot_files(table, spec)
            vals.to_csv(csvp, index=False)
            plt.figure(); plt.hist(s, bins=edges)
            plt.title(f"Histogram: {col}"); plt.xlabel("value"); plt.ylabel("count")
            plt.tight_layout(); plt.savefig(png); plt.close()

        elif kind == "numeric_box":
            col = spec["column"]
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            stats = {
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
            _, png, statsp = _expected_plot_files(table, spec)
            pd.DataFrame([stats]).to_csv(statsp, index=False)
            plt.figure(); plt.boxplot(s, vert=True, whis=1.5)
            plt.title(f"Boxplot: {col}"); plt.ylabel("value")
            plt.tight_layout(); plt.savefig(png); plt.close()

        elif kind == "cardinality_topk":
            k = int(spec.get("k", 10))
            uniq = df.nunique(dropna=True).sort_values(ascending=False).head(k)
            vals = uniq.reset_index()
            vals.columns = ["column","unique_count"]
            _, png, csvp = _expected_plot_files(table, spec)
            vals.to_csv(csvp, index=False)
            plt.figure(); plt.bar(vals["column"], vals["unique_count"])
            plt.title("Cardinality (top-k)"); plt.xlabel("column"); plt.ylabel("unique_count")
            plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(png); plt.close()
# --- End inserted renderer ---

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

  # schema check (CSV only) — now driven by YAML
    expected = (cfg.get("schema") or {}).get("expected_columns")
    if data_path.lower().endswith(".csv") and expected:
        schema_rows = _validate_csv_schema(data_path, expected)
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

    # Render plots declared in the profile (reproducible, with values/stats files)
    _render_plots(df, table, cfg.get("plots") or [])

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

    # 3) Verify plots against config & DQ
    cfg = {}
    cfg_path = None
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            cfg_path = meta.get("config_path")
        except Exception:
            cfg_path = None
    if cfg_path and os.path.exists(cfg_path):
        try:
            cfg = yaml.safe_load(open(cfg_path)) or {}
        except Exception as e:
            problems.append(f"cannot load config from metadata config_path={cfg_path}: {e}")
    else:
        problems.append("metadata missing valid config_path; cannot verify plots")

    if isinstance(dq, pd.DataFrame) and cfg:
        problems.extend(_doctor_check_plots(dq, cfg))

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
