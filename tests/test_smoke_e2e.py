"""End-to-end smoke test on a 2% stratified subsample of the real processed
data. Skips (rather than fails) when data/processed/ isn't populated, since
raw/processed data is intentionally not committed to this repo - see
data/README.md. Run `make smoke` after populating data/processed/ to
actually exercise this.
"""

import json
import os

import pandas as pd
import pytest
import torch
import torch.nn as nn

from kkbox import config as kkbox_config
from kkbox.data import KKBoxDataset, columns_from_manifest
from kkbox.determinism import seed_everything
from kkbox.models import build_model
from kkbox.train import run_epoch

CFG = kkbox_config.load_config()
PROCESSED_DIR = kkbox_config.abspath(CFG, CFG["paths"]["processed_dir"])
TRAIN_PARQUET = os.path.join(PROCESSED_DIR, "model_dataset_train.parquet")
VAL_PARQUET = os.path.join(PROCESSED_DIR, "model_dataset_val.parquet")
MANIFEST_PATH = os.path.join(PROCESSED_DIR, "feature_manifest.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(TRAIN_PARQUET) and os.path.exists(MANIFEST_PATH)),
    reason="data/processed/ not populated - run 00_data_processing.ipynb and "
           "02_Feature_Engineering.ipynb first, or see data/README.md",
)


def test_two_percent_subsample_trains_one_epoch_without_error():
    seed_everything(CFG["seeds"]["default"])

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    cat_cols, num_cols, cardinalities, embed_dims = columns_from_manifest(manifest)

    train_df = pd.read_parquet(TRAIN_PARQUET).sample(frac=0.02, random_state=0)
    val_df = pd.read_parquet(VAL_PARQUET).sample(frac=0.02, random_state=0)
    assert len(train_df) > 100, "2% subsample is implausibly small - is model_dataset_train.parquet truncated?"

    train_loader = torch.utils.data.DataLoader(
        KKBoxDataset(train_df, cat_cols, num_cols), batch_size=CFG["training"]["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        KKBoxDataset(val_df, cat_cols, num_cols), batch_size=CFG["training"]["batch_size"], shuffle=False
    )

    model = build_model(cardinalities, embed_dims, cat_cols, len(num_cols), CFG["model"])
    pos_weight = torch.tensor((train_df["is_churn"] == 0).sum() / max((train_df["is_churn"] == 1).sum(), 1))
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["training"]["lr"])

    train_loss, train_auc, train_rmse = run_epoch(model, train_loader, bce_fn, mse_fn, 1.0, 1.0, optimizer)
    val_loss, val_auc, val_rmse = run_epoch(model, val_loader, bce_fn, mse_fn, 1.0, 1.0)

    for value, name in [(train_loss, "train_loss"), (train_rmse, "train_rmse"),
                         (val_loss, "val_loss"), (val_rmse, "val_rmse")]:
        assert value == value, f"{name} is NaN"  # NaN != NaN
        assert value < float("inf"), f"{name} is inf"
    assert 0.0 <= train_auc <= 1.0
    assert 0.0 <= val_auc <= 1.0
