#!/usr/bin/env bash
set -euo pipefail

# Runs the "final iteration" pipeline in order: feature engineering (8 new features) ->
# Optuna hyperparameter search (CatBoost churn/fwd_rev/cox + ZILN NN) -> CatBoost/Cox
# training -> ZILN ensemble training + CatBoost-vs-ZILN decision -> calibration ->
# final evaluation. Each step is a real (non-subsampled) notebook execution.
#
# Total runtime on a full/forced run: roughly 2.5-3.5 hours (Optuna search ~90-120 min,
# CatBoost/Cox retrain ~15-20 min, ZILN 5-seed ensemble ~15-30 min, 04/05 a few minutes
# each). Safe to run unattended overnight - re-execs itself under `caffeinate` below so
# the Mac sleeping doesn't pause/kill the python kernel mid-run (a real risk on a laptop:
# a paused kernel during a 90-minute Optuna cell would silently corrupt that notebook's
# saved state). A plain re-run with nothing changed skips everything and finishes in
# seconds - see markers_for() below.
#
# Usage: ./run_pipeline.sh          - skip any notebook whose output is already present
#                                       and newer than its upstream dependencies' output
#        ./run_pipeline.sh --force  - ignore all of that, re-run every notebook
# Progress: tail -f the printed log file from another terminal to watch live progress.
# If a notebook fails, this script stops immediately (does not run later notebooks) and
# prints which one - re-run this script to retry from the top; completed notebooks will
# be skipped (see markers_for() below) so retrying only re-does the actual failure point.
#
# The notebooks themselves are NOT internally idempotent (each one unconditionally
# retrains/rebuilds from scratch when executed) - the skip logic lives here instead,
# at the orchestration level, keyed off each notebook's known final output file(s).

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
    export CAFFEINATED=1
    exec caffeinate -i "$0" "$@"
fi

cd "$(dirname "$0")"

LOG_FILE="pipeline_run_$(date +%Y%m%d_%H%M%S).log"
MIN_FREE_GB=5   # disk space has repeatedly dropped to 1-2GB free on this machine during
                # long training runs this project - abort early with a clear message
                # rather than failing deep into a multi-hour run with a cryptic I/O error

NOTEBOOKS=(
    "02_Feature_Engineering.ipynb"
    "03b_Hyperparameter_Tuning.ipynb"
    "03a_CatBoost_and_Cox_Models.ipynb"
    "03c_ZILN_ForwardRevenue.ipynb"
    "04_Calibration_and_Business_Layer.ipynb"
    "05_Final_Evaluation_Summary.ipynb"
)

FORCE_RERUN=0
if [ "${1:-}" = "--force" ]; then
    FORCE_RERUN=1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# The file(s) each notebook writes last on a successful run - used as a completion
# marker. Bash 3.2 (macOS system default) has no associative arrays, hence the case
# statement instead of a dict.
markers_for() {
    case "$1" in
        "02_Feature_Engineering.ipynb")
            echo "data/processed/model_dataset_test.parquet data/processed/feature_manifest.json" ;;
        "03b_Hyperparameter_Tuning.ipynb")
            echo "results/optuna_best_params.json" ;;
        "03a_CatBoost_and_Cox_Models.ipynb")
            echo "results/catboost_results.json results/cox_results.json" ;;
        "03c_ZILN_ForwardRevenue.ipynb")
            echo "results/fwd_rev_model_choice.json" ;;
        "04_Calibration_and_Business_Layer.ipynb")
            echo "results/retention_priority_scores.csv" ;;
        "05_Final_Evaluation_Summary.ipynb")
            echo "results/final_model_metrics.json results/final_model_evaluation.png" ;;
    esac
}

# True (0) if $1's markers are missing/empty, or older than any marker belonging to a
# notebook earlier in NOTEBOOKS (i.e. an upstream step re-ran since $1 last did) - in
# either case $1 needs to run. False (1) only when every marker is present, non-empty,
# and newer than every upstream marker, meaning $1's prior output is still valid.
needs_run() {
    local nb="$1"
    local markers
    markers=$(markers_for "$nb")
    local f
    for f in $markers; do
        [ -s "$f" ] || return 0
    done
    local prior_nb
    for prior_nb in "${NOTEBOOKS[@]}"; do
        [ "$prior_nb" = "$nb" ] && break
        local prior_marker
        for prior_marker in $(markers_for "$prior_nb"); do
            [ -s "$prior_marker" ] || continue
            for f in $markers; do
                if [ "$prior_marker" -nt "$f" ]; then
                    return 0
                fi
            done
        done
    done
    return 1
}

check_kernel() {
    # jupyter nbconvert silently falls back to whatever "python3" kernel is registered
    # (e.g. anaconda's base env) if this project's own kernel isn't found or its deps
    # are missing - that failure looks instant and confusing (ModuleNotFoundError two
    # seconds in), so check explicitly upfront instead.
    if ! jupyter kernelspec list 2>/dev/null | grep -q '^\s*kkbox\s'; then
        log "ABORT: jupyter kernel 'kkbox' is not registered."
        log "Fix: python3 -m ipykernel install --user --name kkbox --display-name \"Python 3.12 (kkbox)\""
        log "(run that with the SAME python3 that has this project's deps installed - check via: python3 -c \"import duckdb, catboost, torch, optuna\")"
        exit 1
    fi
    local kernel_python
    kernel_python=$(python3 -c "
import json
with open('$HOME/Library/Jupyter/kernels/kkbox/kernel.json') as f:
    print(json.load(f)['argv'][0])
" 2>/dev/null) || true
    if [ -z "$kernel_python" ] || ! "$kernel_python" -c "import duckdb, catboost, torch, optuna" 2>/dev/null; then
        log "ABORT: the 'kkbox' kernel's python (${kernel_python:-unknown}) is missing required packages."
        log "Fix: ${kernel_python:-python3} -m pip install -e . (from the project root, with pyproject.toml's deps)"
        exit 1
    fi
    log "kernel check OK: 'kkbox' -> $kernel_python"
}

check_disk_space() {
    local avail_gb
    avail_gb=$(df -g . | awk 'NR==2 {print $4}')
    if [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
        log "ABORT: only ${avail_gb}GB free (need >= ${MIN_FREE_GB}GB). Free up disk space and re-run."
        exit 1
    fi
    log "disk space OK: ${avail_gb}GB free"
}

run_notebook() {
    local nb="$1"
    log "=== starting $nb ==="
    check_disk_space
    local start_ts
    start_ts=$(date +%s)
    if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=kkbox \
        "$nb" >> "$LOG_FILE" 2>&1; then
        local elapsed=$(( $(date +%s) - start_ts ))
        log "=== finished $nb in ${elapsed}s ($((elapsed / 60)) min) ==="
    else
        log "!!! FAILED: $nb - see $LOG_FILE above for the traceback."
        log "!!! Pipeline stopped. Notebooks after $nb were NOT run."
        exit 1
    fi
}

log "Pipeline started. ${#NOTEBOOKS[@]} notebooks queued. Log: $LOG_FILE"
check_kernel
check_disk_space

PIPELINE_START=$(date +%s)
for nb in "${NOTEBOOKS[@]}"; do
    if [ "$FORCE_RERUN" -eq 0 ] && ! needs_run "$nb"; then
        log "=== skipping $nb (output already up to date) ==="
        for f in $(markers_for "$nb"); do
            log "    $f ($(du -h "$f" | cut -f1), modified $(date -r "$f" '+%Y-%m-%d %H:%M:%S'))"
        done
    else
        run_notebook "$nb"
    fi
done
PIPELINE_ELAPSED=$(( $(date +%s) - PIPELINE_START ))

log "=== PIPELINE COMPLETE in $((PIPELINE_ELAPSED / 60)) minutes ==="

for result_file in fwd_rev_model_choice.json final_model_metrics.json cox_results.json catboost_results.json ziln_results.json; do
    if [ -f "results/$result_file" ]; then
        log "--- results/$result_file ---"
        cat "results/$result_file" | tee -a "$LOG_FILE"
    fi
done

log "Done. Full log: $LOG_FILE"
