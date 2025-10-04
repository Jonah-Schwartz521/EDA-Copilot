.PHONY: run run-bad doctor clean
run:
\tpython -m eda run --config config/ufc.yml
\tpython -m eda doctor

run-bad:
\tpython -m eda run --config config/ufc_bad.yml || true
\t@grep -E 'invalid_category_count|violations_out_of_range|duplicate_key_groups_count|duplicate_row_pct' outputs/data_quality_report.csv || echo "no issues found"

doctor:
\tpython -m eda doctor

clean:
\trm -f outputs/data_quality_report.csv
\trm -f logs/run_metadata.json
\trm -f plots/*.png plots/*_values.csv plots/*_stats.csv
\trm -f reports/*.md