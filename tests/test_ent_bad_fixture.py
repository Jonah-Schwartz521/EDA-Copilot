# tests/test_ent_contract.py
import os, yaml, pandas as pd, tempfile
from tests.helpers import run_cli

GOLD = "config/ent.yml"

def _write_temp_cfg_pointing_to_bad(original_cfg: str) -> str:
    cfg = yaml.safe_load(open(original_cfg))
    # point to the bad fixture
    cfg["data"]["path"] = "data/fixtures/ent_bad.csv"
    fd, temp_path = tempfile.mkstemp(prefix="ent_bad_", suffix=".yml")
    os.close(fd)
    with open(temp_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return temp_path

def test_ent_bad_has_violations():
    temp_cfg = _write_temp_cfg_pointing_to_bad(GOLD)
    try:
        run_cli(temp_cfg)
        dq = pd.read_csv("outputs/data_quality_report.csv")
        dq["value_num"] = pd.to_numeric(dq["value"], errors="coerce")

        # Expect at least one violation across these metrics
        bad = dq.query("(metric in ['invalid_category_count','violations_out_of_range_count','duplicate_key_groups_count']) and value_num > 0")
        assert not bad.empty, "Expected violations for ent_bad fixture, found none"
    finally:
        os.remove(temp_cfg)