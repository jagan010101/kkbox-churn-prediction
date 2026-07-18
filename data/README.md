# Data

## Source

[KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) (WSDM 2017), via Kaggle.
Subject to Kaggle's and KKBox's own terms - **not redistributed in this repository**.
Download the raw `.7z` archives yourself into `data/raw/` before running `00_data_processing.ipynb`.

## Expected raw files (`data/raw/`)

| File | Notes |
|---|---|
| `members_v3.csv.7z` | user demographics |
| `train.csv.7z` | churn labels, snapshot 1 |
| `train_v2.csv.7z` | churn labels, snapshot 2 |
| `transactions.csv.7z` | full transaction history (~700MB compressed) |
| `transactions_v2.csv.7z` | later transaction snapshot |
| `user_logs.csv.7z` | daily listening logs, ~30GB uncompressed (~7GB compressed) |
| `user_logs_v2.csv.7z` | later listening-log snapshot |

## Integrity verification

No raw files were present in this repository's `data/raw/` at the time this file was written, so no hashes could be
computed - **do not treat any hash values elsewhere in this repo as verified until you've run the command below
yourself.** Once you've downloaded the archives, generate and check them with:

```bash
cd data/raw
shasum -a 256 *.7z > SHA256SUMS.txt
# to verify on a later run / another machine:
shasum -a 256 -c SHA256SUMS.txt
```

`SHA256SUMS.txt` is intentionally not committed (raw data itself isn't tracked - see `.gitignore`); regenerate it
locally after downloading, and treat any mismatch against a shared `SHA256SUMS.txt` as a sign the archive was
altered or re-exported by Kaggle.

## Processed data (`data/processed/`)

Populated by `00_data_processing.ipynb` (raw → typed parquet) and `02_Feature_Engineering.ipynb` (parquet →
model-ready splits). Not committed except `feature_manifest.json` (small, human-readable, documents the exact
feature/label construction parameters used for the committed model checkpoints in `models/`).
