# tests/test_ent_contract.py
import os, pandas as pd
from tests.helpers import run_cli, doctor_ok, expected_plot_files_from_yaml

CFG = "config/ent.yml"

def test_ent_contract_end_to_end():
    run_cli(CFG)

    # Contract CSV created
    assert os.path.exists("outputs/data_quality_report.csv")
    dq = pd.read_csv("outputs/data_quality_report.csv")
    # Coerce 'value' to numeric for safe comparisons; non-numeric -> NaN
    dq["value_num"] = pd.to_numeric(dq["value"], errors="coerce")

    # No validation issues on golden ENT (strict == 0 or empty)
    bad = dq.query("(metric in ['invalid_category_count','violations_out_of_range_count','duplicate_key_groups_count']) and value_num>0")
    assert bad.empty, f"ENT golden should be clean, but got:\n{bad}"

    # Plots exist as declared
    for p in expected_plot_files_from_yaml(CFG):
        assert os.path.exists(p), f"missing plot artifact: {p}"

    # Doctor passes
    assert doctor_ok()