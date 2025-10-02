import argparse
import os
import yaml 

def ensure_dirs():
    for d in ["outputs", "plots", "reports", "logs"]:
        os.makedirs(d, exist_ok=True)


def cmd_run(args):
    ensure_dirs()
    cfg = yaml.safe_load(open(args.config))
    print("ok: dirs exist; confid loaded for", cfg["data"]["name"])

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
