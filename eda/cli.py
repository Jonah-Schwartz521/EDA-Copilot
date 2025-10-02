import argparse
import os
import yaml 
import pandas as pd 
import sys 

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
    # column-level missingness (0..1, rounded to 6 dp)
    miss = df.isna().mean()
    for col, pct in miss.items():
        rows.append([table, col, "missing_pct", float(round(pct, 6)), ""]) 
    return pd.DataFrame(rows, columns=["table", "column", "metric", "value", "note"])


def cmd_run(args):
    ensure_dirs()
    cfg = yaml.safe_load(open(args.config))
    table = cfg["data"]["name"]
    data_path = cfg["data"]["path"]
    if not os.path.exists(data_path):
        sys.stderr.write(f"[ERROR] data file not found: {data_path}\n")
        sys.exit(2)
    df = load_data(data_path)

    dq = compute_minimal_metrics(df, table)
    dq.to_csv("outputs/data_quality_report.csv", index=False)
    print("wrote outputs/data_quality_report.csv")
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
