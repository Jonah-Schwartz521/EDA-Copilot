import os, subprocess, yaml

def run_cli(cfg_path: str):
    """Run your CLI and fail fast if it errors."""
    subprocess.run(["python", "-m", "eda", "run", "--config", cfg_path], check=True)

def doctor_ok():
    """Run doctor; return True if exit code is 0."""
    r = subprocess.run(["python", "-m", "eda", "doctor"])
    return r.returncode == 0

def expected_plot_files_from_yaml(cfg_path: str):
    """Derive expected plot file names from YAML using your naming convention."""
    cfg = yaml.safe_load(open(cfg_path))
    table = cfg["data"]["name"]
    plots = cfg.get("plots") or []
    files = []
    for spec in plots:
        k = spec["kind"]
        if k == "missingness":
            files += [f"plots/{table}_missingness.png", f"plots/{table}_missingness_values.csv"]
        elif k == "topn":
            col = spec["column"]
            files += [f"plots/{table}_topn_{col}.png", f"plots/{table}_topn_{col}_values.csv"]
        elif k == "numeric_hist":
            col = spec["column"]
            files += [f"plots/{table}_hist_{col}.png", f"plots/{table}_hist_{col}_values.csv"]
        elif k == "numeric_box":
            col = spec["column"]
            files += [f"plots/{table}_box_{col}.png", f"plots/{table}_box_{col}_stats.csv"]
        elif k == "cardinality_topk":
            files += [f"plots/{table}_cardinality_topk.png", f"plots/{table}_cardinality_topk_values.csv"]
    return files
