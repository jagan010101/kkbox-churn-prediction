# Multi-Task MLP: Churn Prediction + LTV Estimation

A single PyTorch model that jointly predicts subscriber churn and near-term revenue for KKBox (WSDM 2017 Kaggle challenge) subscribers, combining a Factorization Machine interaction layer, a shared MLP backbone, and dual task heads trained with BCE + MSE loss. Built from [`MultiTask_MLP_ChurnLTV_ProjectPlan_v2.pdf`](MultiTask_MLP_ChurnLTV_ProjectPlan_v2.pdf).

See [`PORTFOLIO.md`](PORTFOLIO.md) for CV bullets and interview Q&A drawn from this project.

## Results at a glance

| Metric | Value |
|---|---|
| Churn AUC-ROC (test) | **0.844** |
| Churn AUC-PR (test) | 0.358 (vs. 0.064 base rate) |
| LTV RMSE (test, raw TWD) | **59.1** |
| LTV R² (test, raw) | 0.491 |
| Calibration ECE, before → after isotonic regression | 0.265 → **0.001** |
| Retention budget simulation, model ranking vs. naive strategies | **+750% to +1,300%** expected revenue saved |

Final model: **Exp-6 (uncertainty weighting, Kendall et al. 2018)** + isotonic probability calibration. Selected because it matched or exceeded *both* single-task ceilings simultaneously — see `05_MultiTask_Ablation.ipynb`.

## Repo structure

```
00_data_processing.ipynb           7z -> parquet ETL for all 7 raw KKBox tables (~35GB raw, streamed via FIFO for the 30GB user_logs.csv)
01_EDA.ipynb                       Exploratory analysis via DuckDB (queries the 392M-row table directly, no full load)
02_Feature_Engineering.ipynb       Per-user feature table + leak-free forward-looking LTV target
03_Model_Architecture.ipynb        FM layer + MultiTaskFMNet + Dataset/DataLoader, forward/backward pass verified
04_Training_Baselines.ipynb        Training loop; Exp-1 (churn-only) / Exp-2 (LTV-only) single-task ceilings
05_MultiTask_Ablation.ipynb        Exp-3..5 (fixed loss weights), Exp-6 (uncertainty weighting), Exp-7 (PCGrad), Pareto frontier
06_Calibration_and_Business_Layer.ipynb   ECE + temperature scaling + isotonic regression; Retention Priority Score, budget allocation, sensitivity analysis
07_Final_Evaluation_Summary.ipynb  Consolidated ROC / PR / LTV-scatter for the selected final model

data/raw/           original Kaggle .7z archives (not committed — see note below)
data/processed/     parquet tables + model_dataset_{train,val,test}.parquet + encoders/scaler/feature_manifest.json
models/             trained checkpoints, one per experiment (exp1..exp7)
results/            per-experiment histories, metrics JSON/CSV, saved figures
```

Run the notebooks in order (`00` → `07`); each is idempotent (re-running skips work whose output already exists) and self-contained (redefines the `Dataset`/model classes rather than importing between notebooks, so any notebook can be run standalone once `data/processed/` is populated).

**Data**: download the KKBox WSDM 2017 files from Kaggle into `data/raw/` (not committed — `transactions.csv.7z` alone is 700MB, `user_logs.csv.7z` is 7GB). `00_data_processing.ipynb` handles extraction and conversion from there.

## Key engineering decisions & findings

A few things came up during the build that changed the plan's original approach — recorded here since they're as much a part of the result as the metrics:

- **LTV target leakage, found and fixed.** The first version of the LTV target (`ltv = sum(actual_amount_paid)` over full transaction history) was defined over the *same* window as two of its own input features (`num_transactions`, `avg_actual_amount_paid`), which satisfy `num_transactions × avg_actual_amount_paid ≈ ltv` almost exactly by construction. The tell was a suspiciously perfect R²=0.999 on the LTV-only baseline. Fix: a hard temporal split — `FEATURE_CUTOFF` (2016-12-31) bounds every input feature, and `ltv` is redefined as forward-looking revenue in the two months *after* the cutoff. Real R² dropped to a plausible 0.46-0.49. See `02_Feature_Engineering.ipynb`.
- **30GB `user_logs.csv` streamed without ever touching disk in full.** The raw file is too large to extract-then-process on a 48GB disk budget alongside its parquet output. `py7zr` decompresses directly into a named pipe (FIFO) read by pandas in chunks, so only a small buffer is ever materialized. See `00_data_processing.ipynb`.
- **Feature engineering at scale via DuckDB, not pandas.** Per-user aggregation of the 392M-row table (e.g. `count(distinct date)`) is done with SQL pushed down to DuckDB directly against parquet files — a naive exact `GROUP BY` on the full table didn't finish in reasonable time, and DuckDB's approximate `HyperLogLog` count was fast but visibly wrong (9% of users exceeded the theoretical max). Restricting the aggregation window made the exact computation cheap instead.
- **No meaningful gradient conflict between churn and LTV.** All five multi-task loss-weighting strategies (fixed 50/50, churn-dominant, LTV-dominant, uncertainty weighting, PCGrad) land within a tight band (AUC spread 0.0014, RMSE spread 0.009) and all match or beat both single-task ceilings. PCGrad's gradient-surgery mechanism has little to correct here — a real empirical answer to the question the plan raises in Section 4.6, not an assumption.
- **`pos_weight`-based class reweighting wrecks probability calibration.** Needed for good AUC/ranking under 6.4% class imbalance, but it inflates every predicted probability (mean predicted P(churn)=0.33 vs. true rate 0.064; ECE=0.265). Temperature scaling — a pure sharpness correction — only helps marginally (ECE→0.2545) because the miscalibration is a systematic bias, not a sharpness problem. Isotonic regression fixes it almost completely (ECE→0.0012).
- **Isotonic regression's step function creates ranking ties.** Only 204 distinct calibrated probabilities exist across 148,940 test users (the plan's business-layer ranking uses these). Documented as a real, known limitation of isotonic regression rather than hidden behind a "very confident model" narrative — see the sensitivity-analysis caveat in `06_Calibration_and_Business_Layer.ipynb`.
- **PCGrad implementation fixes a bug in the plan's own pseudocode.** The plan's `PCGrad.step()` snippet drops `None`-gradient parameters from a list comprehension, desyncing the projected-gradient list from the actual parameter list the moment any parameter is task-specific (e.g. the churn/LTV heads). Fixed in `05_MultiTask_Ablation.ipynb` by keeping a fixed-length gradient list per task.

## Deferred / not implemented

- Hyperparameter grid search (Section 6.1/9.2 of the plan — backbone width, FM latent dim `k`, dropout rate). Skipped: the ablation study already showed loss-weighting choice barely matters here, so the expected payoff of a further architecture sweep is low relative to its cost (~12 configs × 50 epochs).
- A calibration method that avoids isotonic regression's tie artifact while keeping its ECE quality (e.g. Platt/logistic scaling with both scale and intercept terms) — noted as a natural follow-up in `06`.
