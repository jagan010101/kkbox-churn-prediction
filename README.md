# Multi-Task MLP: Churn Prediction + LTV Estimation

A single PyTorch model that jointly predicts subscriber churn and near-term revenue for KKBox (WSDM 2017 Kaggle challenge) subscribers, combining a Factorization Machine interaction layer, a shared MLP backbone, and dual task heads trained with BCE + MSE loss. Built from [`MultiTask_MLP_ChurnLTV_ProjectPlan_v2.pdf`](MultiTask_MLP_ChurnLTV_ProjectPlan_v2.pdf).

**Methodology note**: the population and labels are built from a per-user reference date (each user's own last confirmable paid subscription cycle) rather than KKBox's official `train`/`train_v2` snapshot files, which only include users who happened to still be subscribers near one fixed date — a survivorship bias that excludes ~40% of everyone who ever paid KKBox. See `02_Feature_Engineering.ipynb` and `01_EDA.ipynb`'s "Survival analysis" / "Corrected churn rates" sections for the full discussion. This means churn in this project is **~74%**, not the ~9% the raw Kaggle snapshot suggests.

## Results at a glance

| Metric | Value |
|---|---|
| Churn AUC-ROC (test) | **0.932** |
| Churn AUC-PR (test) | 0.971 (vs. 0.736 base rate) |
| LTV RMSE (test, raw TWD) | **153.7** |
| LTV R² (test, raw) | 0.099 |
| Calibration ECE, before → after isotonic regression | 0.099 → **0.002** |
| Retention budget simulation, model ranking vs. naive strategies | **+678% to +708%** expected revenue saved |
| CatBoost baseline (churn / LTV), for comparison | AUC-ROC 0.936 / RMSE 155.4 — see `08_GBT_Baseline_Comparison.ipynb` |
| Cox survival model (CatBoost, full censoring-inclusive population) | Concordance index **0.941** |

Final model: **Exp-6 (uncertainty weighting, Kendall et al. 2018)** + isotonic probability calibration. Selected because it matched or exceeded *both* single-task ceilings simultaneously — see `05_MultiTask_Ablation.ipynb`. Note LTV R² is substantially lower than an earlier (pre-bias-fix) run — the corrected population is far more heterogeneous (full 2015-2017 history vs. one narrow snapshot window), making forward-revenue prediction a genuinely harder regression problem.

## Repo structure

```
00_data_processing.ipynb           7z -> parquet ETL for all 7 raw KKBox tables (~35GB raw, streamed via FIFO for the 30GB user_logs.csv)
01_EDA.ipynb                       Exploratory analysis via DuckDB (queries the 392M-row table directly, no full load)
02_Feature_Engineering.ipynb       Per-user feature table + leak-free forward-looking LTV target
03_Model_Architecture.ipynb        FM layer + MultiTaskFMNet + Dataset/DataLoader, forward/backward pass verified
04_Training_Baselines.ipynb        Training loop; Exp-1 (churn-only) / Exp-2 (LTV-only) single-task ceilings
05_MultiTask_Ablation.ipynb        Exp-3..5 (fixed loss weights), Exp-6 (uncertainty weighting), Exp-7 (PCGrad), Pareto frontier
06_Calibration_and_Business_Layer.ipynb   ECE + temperature scaling + isotonic regression; Retention Priority Score (percentile-rescaled), budget allocation, sensitivity analysis
07_Final_Evaluation_Summary.ipynb  Consolidated ROC / PR / LTV-scatter for the selected final model
08_GBT_Baseline_Comparison.ipynb   CatBoost churn/LTV baselines vs. the neural net, plus a CatBoost Cox proportional-hazards survival model on the full censoring-inclusive population

data/raw/           original Kaggle .7z archives (not committed — see note below)
data/processed/     parquet tables + model_dataset_{train,val,test}.parquet + encoders/scaler/feature_manifest.json
models/             trained checkpoints, one per experiment (exp1..exp7)
results/            per-experiment histories, metrics JSON/CSV, saved figures
```

Run the notebooks in order (`00` → `08`); each is idempotent (re-running skips work whose output already exists) and self-contained (redefines the `Dataset`/model classes rather than importing between notebooks, so any notebook can be run standalone once `data/processed/` is populated).

**Data**: download the KKBox WSDM 2017 files from Kaggle into `data/raw/` (not committed — `transactions.csv.7z` alone is 700MB, `user_logs.csv.7z` is 7GB). `00_data_processing.ipynb` handles extraction and conversion from there.
