.PHONY: install test smoke reproduce clean

PYTHON := python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Fast: unit tests (toy-data label construction, encoder/scaler invariants,
# model shapes) plus a real-data 2% smoke test that skips gracefully if
# data/processed/ isn't populated. Run this before every commit.
test:
	$(PYTHON) -m pytest tests/ -v

# Slower: actually executes notebooks 03-05 end to end on a 2% stratified
# subsample, to verify the src/kkbox refactor is behavior-preserving at the
# notebook level (not just the unit-test level). Requires data/processed/
# populated. --output-dir only redirects where the *executed notebook file*
# is saved - it does NOT change cwd-relative paths used inside cells - so
# KKBOX_MODELS_DIR/KKBOX_RESULTS_DIR are ALSO redirected to /tmp, otherwise
# this would silently overwrite the real committed checkpoints/results with
# degraded 2%-subsample versions (this happened once; see git history).
SMOKE_ENV := KKBOX_SUBSAMPLE_FRAC=0.02 KKBOX_MODELS_DIR=/tmp/kkbox_smoke_models KKBOX_RESULTS_DIR=/tmp/kkbox_smoke_results
smoke:
	mkdir -p /tmp/kkbox_smoke_models /tmp/kkbox_smoke_results
	$(SMOKE_ENV) jupyter nbconvert --to notebook --execute --output-dir /tmp --output 03_smoke.ipynb 03_Model_Architecture.ipynb
	$(SMOKE_ENV) jupyter nbconvert --to notebook --execute --output-dir /tmp --output 04_smoke.ipynb 04_Training_Baselines.ipynb
	$(SMOKE_ENV) jupyter nbconvert --to notebook --execute --output-dir /tmp --output 05_smoke.ipynb 05_MultiTask_Ablation.ipynb

# Full pipeline reproduction, in order. Ask before running this - notebooks
# 00/02 involve multi-hour/multi-GB processing of the raw KKBox archives,
# and 04/05 are real (unsubsampled) training runs.
reproduce:
	jupyter nbconvert --to notebook --execute --inplace 00_data_processing.ipynb
	jupyter nbconvert --to notebook --execute --inplace 01_EDA.ipynb
	jupyter nbconvert --to notebook --execute --inplace 02_Feature_Engineering.ipynb
	jupyter nbconvert --to notebook --execute --inplace 03_Model_Architecture.ipynb
	jupyter nbconvert --to notebook --execute --inplace 04_Training_Baselines.ipynb
	jupyter nbconvert --to notebook --execute --inplace 05_MultiTask_Ablation.ipynb
	jupyter nbconvert --to notebook --execute --inplace 06_Calibration_and_Business_Layer.ipynb
	jupyter nbconvert --to notebook --execute --inplace 07_Final_Evaluation_Summary.ipynb
	jupyter nbconvert --to notebook --execute --inplace 08_GBT_Baseline_Comparison.ipynb

clean:
	rm -rf catboost_info .pytest_cache **/__pycache__ src/kkbox.egg-info
