# KKBox Churn Prediction + Forward-Revenue Estimation: CatBoost + Cox Survival

Predicts subscriber churn and near-term revenue for KKBox (WSDM 2017 Cup) subscribers. The primary model is a **CatBoost churn classifier + forward-revenue regressor, plus a CatBoost Cox proportional-hazards survival model** (`03a_CatBoost_and_Cox_Models.ipynb`). A **Zero-Inflated LogNormal (ZILN) neural net** was built and directly compared against it as a dedicated forward-revenue challenger (`03c_ZILN_ForwardRevenue.ipynb`, 5-seed ensemble, Optuna-tuned, same feature set) rather than assuming CatBoost superior — **CatBoost won on every metric**; see the comparison below.

**Methodology note**: the population and labels are built from a per-user reference date (each user's own last confirmable paid subscription cycle) rather than KKBox's official `train`/`train_v2` snapshot files, which only include users who happened to still be subscribers near one fixed date — a survivorship bias that excludes ~40% of everyone who ever paid KKBox. See `02_Feature_Engineering.ipynb` and `01_EDA.ipynb`'s "Survival analysis" / "Corrected churn rates" sections for the full discussion. This means churn in this project is **~74%**, not the ~9% the raw official snapshot suggests.

## Results at a glance

| Metric | Value |
|---|---|
| Churn AUC-ROC (test) | **0.961** |
| Churn AUC-PR (test) | 0.984 (vs. 0.736 base rate) |
| Forward-revenue RMSE (test, raw TWD) | **127.9** |
| Forward-revenue R² (test, raw) | 0.385 |
| Calibration ECE, raw → after isotonic regression | 0.0043 → **0.0018** (CatBoost's raw probabilities are already close to well-calibrated) |
| Retention budget simulation, model ranking vs. random-selection baseline | **+393%** expected revenue saved |
| Cox survival model (CatBoost, full censoring-inclusive population) | Concordance index **0.965** |

**Forward-revenue model comparison, same 27-feature set, test set** — CatBoost wins:

| Model | Forward-revenue RMSE (TWD) | Forward-revenue R² |
|---|---|---|
| **CatBoost (production)** | **127.9** | **0.385** |
| ZILN neural net (5-seed ensemble, Optuna-tuned) | 131.5 | 0.349 |

ZILN uses a loss function purpose-built for zero-inflated revenue targets and came within ~3.5 RMSE of CatBoost, but still didn't win. Full history of the comparison, including the numerical-instability debugging that went into getting ZILN a fair shot, is in `project_report.md`.

Final model: **CatBoost churn classifier + Tweedie-loss forward-revenue regressor** (`03a_CatBoost_and_Cox_Models.ipynb`, 27 features + 1 auxiliary feature, Optuna-tuned hyperparameters) with isotonic probability calibration. Built up in three rounds over the original 13-feature/RMSE-loss/default-hyperparameter baseline:

1. **First feature pass** (13→19 features): added `is_cancel_rate`, `avg_discount_rate`, `num_distinct_payment_methods`, `avg_num_unq_songs`, `recent_engagement_ratio`, `txn_frequency`, grounded in previously-unused raw columns (`is_cancel`, `plan_list_price`, `num_unq`) plus a 7-day-vs-30-day engagement-trend signal.
2. **Modeling changes**: the forward-revenue regressor switched from `RMSE`-on-`log1p` to a **Tweedie loss trained directly on the raw, zero-inflated `fwd_rev_59d` target** (39% of users pay nothing in the 59-day window); a leakage-free, cross-fitted "P(any payment)" classifier output was added as an extra feature to that regressor (a naive two-stage `P(pay) x E[amount|pay]` hurdle model was tried first and did *not* beat the single-model approach — compounding independently-trained-model errors ate the gain, see `03a_CatBoost_and_Cox_Models.ipynb`'s auxiliary-classifier section); all models get Optuna-tuned hyperparameters (`03b_Hyperparameter_Tuning.ipynb`).
3. **Second feature pass** (19→27 features), grounded in the published WSDM 2018 KKBox Churn Challenge winning solutions rather than guesswork: **recency** (`days_since_last_login`, `days_since_last_cancel`) and **most-recent-transaction snapshot** (`most_recent_payment_plan_days`, `most_recent_actual_amount_paid`, `most_recent_is_auto_renew`, `most_recent_is_cancel`) beat the model's own lifetime-average features by a wide margin — `most_recent_is_cancel` alone is now the **single most important forward-revenue feature** (31% importance), ahead of the lifetime `is_cancel_rate` it's related to. Two magnitude-based listening-trend features (`secs_trend_recent_vs_prior`, `numunq_trend_recent_vs_prior`) round out the set. A **Zero-Inflated LogNormal (ZILN) neural net** was built as a dedicated forward-revenue challenger on this same feature set — architecturally well-motivated (single jointly-optimized model vs. the failed two-stage hurdle above) and grounded in Google's published LTV methodology, but it did not beat CatBoost even after Optuna tuning and 5-seed ensembling (RMSE 131.5 vs. 127.9, R² 0.349 vs. 0.385) — an honest negative result, not pursued further; see `03c_ZILN_ForwardRevenue.ipynb`.

Combined, these moved churn AUC-ROC from 0.936 to **0.961** and forward-revenue R² from 0.081 to **0.385** (RMSE 155.3 → 127.9 TWD) over the original baseline.

**A caveat worth reading, not just skimming**: the churn model's extra confidence has a side effect — probabilities are now so concentrated at the top that 903 of the top 1,000 budget-ranked users share the *exact same* calibrated `p_churn`. Ranking by `P(churn)` alone (ignoring forward revenue) collapses to near-zero expected value in this regime, because there's nothing left to break the tie with; the combined priority score doesn't have this failure mode since `E[forward revenue]` still fully differentiates within the tied group. See `04_Calibration_and_Business_Layer.ipynb`'s Section 8.3 caveat for the full explanation. Forward-revenue R² is better than it was but still leaves most variance unexplained — the post-bias-fix population is far more heterogeneous (full 2015-2017 history vs. one narrow snapshot window), making forward-revenue prediction a genuinely harder regression problem than the churn side. The Optuna search itself is limited to a ~30-minute-per-model budget (four studies now, including the ZILN net), so there's plausibly more headroom in a longer search — not pursued further here.

## Repo structure

```
00_data_processing.ipynb                7z -> parquet ETL for all 7 raw KKBox tables (~35GB raw, streamed via FIFO for the 30GB user_logs.csv)
01_EDA.ipynb                            Exploratory analysis via DuckDB, survivorship-bias discovery, Kaplan-Meier survival curves
02_Feature_Engineering.ipynb            Per-user feature table (27 features) + leak-free forward-looking revenue target (`fwd_rev_59d`)
03a_CatBoost_and_Cox_Models.ipynb       PRIMARY MODEL: CatBoost churn classifier + forward-revenue regressor, CatBoost Cox survival model
03b_Hyperparameter_Tuning.ipynb         Optuna search (offline, not part of the run-in-order sequence) - output consumed by 03a and 03c
03c_ZILN_ForwardRevenue.ipynb           ZILN neural-net forward-revenue challenger (5-seed ensemble) vs. CatBoost Tweedie - writes the winner decision 04/05 read
04_Calibration_and_Business_Layer.ipynb ECE + isotonic regression; Retention Priority Score (percentile-rescaled), budget allocation, sensitivity analysis
05_Final_Evaluation_Summary.ipynb       Consolidated ROC / PR / forward-revenue-scatter for the selected final model

data/raw/           original KKBox .7z archives (not committed — see note below)
data/processed/     parquet tables + model_dataset_{train,val,test}.parquet + encoders/scaler/feature_manifest.json
models/             catboost_{churn,fwd_rev,cox,pay_clf}.cbm (primary model + auxiliary payment classifier) + ziln_seed*.pt + ziln_ensemble_manifest.json (forward-revenue challenger)
results/            per-model histories, metrics JSON/CSV, saved figures, fwd_rev_model_choice.json (CatBoost-vs-ZILN decision)
run_pipeline.sh                         Runs 02→03b→03a→03c→04→05 in order, unattended (disk-space checks, caffeinate, stops on first failure)
```

Run the notebooks in order (`00` → `05`) via `run_pipeline.sh`, which is idempotent at the orchestration level: it skips any notebook whose output file(s) are already present and newer than its upstream dependencies', so a clean re-run after nothing has changed finishes in seconds instead of re-training everything (`./run_pipeline.sh --force` bypasses this and re-runs every notebook). The notebooks themselves aren't internally idempotent — each one unconditionally rebuilds/retrains when it does run. `03b` is not part of that sequence — it's an offline Optuna search (four studies: churn, forward-revenue, Cox, ZILN; ~2 hours total) whose output (`results/optuna_best_params.json`) `03a` and `03c` read at the top; neither runs without it present at least once. `03c` must run after `03a` (it compares against `03a`'s `catboost_results.json`) and before `04`/`05` (they read `03c`'s winner decision) — `run_pipeline.sh` handles this ordering automatically. `00`-`02`, `04`, `05` are fast; `03a`'s CatBoost/Cox training and `03c`'s ZILN ensemble are each ~30 min on full data.

**Data**: download the KKBox WSDM 2017 files from https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data into `data/raw/` (not committed — `transactions.csv.7z` alone is 700MB, `user_logs.csv.7z` is 7GB). `00_data_processing.ipynb` handles extraction and conversion from there.
