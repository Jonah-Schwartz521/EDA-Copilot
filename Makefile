.PHONY: run run-bad run-ufc run-ufc-bad run-ent run-ent-bad doctor clean

run:
	python -m eda run --config config/ufc.yml
	python -m eda doctor

run-bad:
	python -m eda run --config config/ufc_bad.yml || true
	@grep -E 'invalid_category_count|violations_out_of_range|duplicate_key_groups_count|duplicate_row_pct' outputs/data_quality_report.csv || echo "no issues found"

# Aliases for UFC clarity
run-ufc:
	$(MAKE) run

run-ufc-bad:
	$(MAKE) run-bad

doctor:
	python -m eda doctor

clean:
	rm -f outputs/data_quality_report.csv
	rm -f logs/run_metadata.json
	rm -f plots/*.png plots/*_values.csv plots/*_stats.csv || true
	rm -f reports/*.md || true


# ENT dataset targets
run-ent:
	python -m eda run --config config/ent.yml
	python -m eda doctor

run-ent-bad:
	@sed -i '' 's|data/golden/ent.csv|data/fixtures/ent_bad.csv|' config/ent.yml
	- python -m eda run --config config/ent.yml
	@sed -i '' 's|data/fixtures/ent_bad.csv|data/golden/ent.csv|' config/ent.yml
	@grep -E 'invalid_category_count|violations_out_of_range|duplicate_key_groups_count|duplicate_row_pct' outputs/data_quality_report.csv || echo "no issues found"


run-sparcs:
	python -m eda run --config config/sparcs.yml
	python -m eda doctor

run-sparcs-bad:
	@sed -i '' 's|data/golden/sparcs.csv|data/fixtures/sparcs_bad.csv|' config/sparcs.yml
	- python -m eda run --config config/sparcs.yml
	@sed -i '' 's|data/fixtures/sparcs_bad.csv|data/golden/sparcs.csv|' config/sparcs.yml
	@grep -E 'invalid_category_count|violations_out_of_range|duplicate_key_groups_count|duplicate_row_pct' outputs/data_quality_report.csv || echo "no issues found"