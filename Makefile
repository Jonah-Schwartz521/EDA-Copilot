.PHONY: run run-bad doctor clean

run:
	python -m eda run --config config/ufc.yml
	python -m eda doctor

run-bad:
	python -m eda run --config config/ufc_bad.yml || true
	@grep -E 'invalid_category_count|violations_out_of_range|duplicate_key_groups_count|duplicate_row_pct' outputs/data_quality_report.csv || echo "no issues found"

doctor:
	python -m eda doctor

clean:
	rm -f outputs/data_quality_report.csv
	rm -f logs/run_metadata.json
	rm -f plots/*.png plots/*_values.csv plots/*_stats.csv || true
	rm -f reports/*.md || true
