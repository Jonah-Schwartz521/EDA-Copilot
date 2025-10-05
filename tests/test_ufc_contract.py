# tests/test_ufc_contract.py
import os, pandas as pd
from tests.helpers import run_cli, doctor_ok, expected_plot_files_from_yaml

CFG = "config/ufc.yml"

def test_ufc_contract_end_to_end():
    run_cli(CFG)

    # 1) Contract CSV exists and has the row_count metric
    assert os.path.exists("outputs/data_quality_report.csv")
    dq = pd.read_csv("outputs/data_quality_report.csv")
    rc = dq.loc[(dq["column"]=="__table__") & (dq["metric"]=="row_count"), "value"]
    assert not rc.empty
    assert int(float(rc.iloc[0])) == 4  # UFC golden has 4 rows

    # 2) All expected plot files exist
    for p in expected_plot_files_from_yaml(CFG):
        assert os.path.exists(p), f"missing plot artifact: {p}"

    # 3) Doctor should validate plots vs. contract
    assert doctor_ok()