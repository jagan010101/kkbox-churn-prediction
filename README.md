# Multi-Task MLP: Churn Prediction + LTV Estimation

A single PyTorch model that jointly predicts subscriber churn and near-term revenue for KKBox (WSDM 2017 Kaggle challenge) subscribers, combining a Factorization Machine interaction layer, a shared MLP backbone, and dual task heads trained with BCE + MSE loss. Built from [`MultiTask_MLP_ChurnLTV_ProjectPlan_v2.pdf`](MultiTask_MLP_ChurnLTV_ProjectPlan_v2.pdf).

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
