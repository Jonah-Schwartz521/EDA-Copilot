import argparse
import yaml 

def cmd_run(args):
    print("running EDA with config:", args.config)
    cfg = yaml.safe_load(open(args.config))
    print("config loaded:", cfg)

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
