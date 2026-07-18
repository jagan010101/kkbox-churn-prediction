"""PyTorch Dataset/DataLoader helpers for model_dataset_{train,val,test}.parquet."""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from kkbox.determinism import make_generator, seed_worker


class KKBoxDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols):
        self.x_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.x_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
        self.y_churn = torch.tensor(df["is_churn"].values, dtype=torch.float32)
        self.y_ltv = torch.tensor(df["log1p_ltv"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y_churn)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y_churn[idx], self.y_ltv[idx]


def columns_from_manifest(manifest):
    """Returns (cat_cols, num_cols, cardinalities, embed_dims) from feature_manifest.json."""
    cat_cols = [v["column"] for v in manifest["categorical"].values()]
    num_cols = manifest["numerical_scaled"] + manifest["numerical_unscaled"]
    cardinalities = {v["column"]: v["cardinality"] for v in manifest["categorical"].values()}
    embed_dims = {v["column"]: v["embedding_dim"] for v in manifest["categorical"].values()}
    return cat_cols, num_cols, cardinalities, embed_dims


def load_splits(processed_dir, splits=("train", "val", "test")):
    """Loads model_dataset_{split}.parquet for each requested split name."""
    return {
        name: pd.read_parquet(os.path.join(processed_dir, f"model_dataset_{name}.parquet"))
        for name in splits
    }


def stratified_subsample(df, frac, stratify_col="is_churn", random_state=0):
    """Stratified subsample for smoke-testing on a fraction of real data.

    Uses GroupBy.sample rather than groupby(...).apply(lambda g: g.sample(...)) -
    the latter silently drops the grouping column in some pandas versions
    (reproduced: pandas 3.0.1), which would raise a confusing KeyError
    downstream in KKBoxDataset rather than at the sampling call site.
    """
    return df.groupby(stratify_col, group_keys=False).sample(frac=frac, random_state=random_state)


def make_loader(df, cat_cols, num_cols, batch_size, shuffle, seed=None):
    """Builds a DataLoader; pass seed for deterministic shuffling (train loaders)."""
    ds = KKBoxDataset(df, cat_cols, num_cols)
    kwargs = {}
    if shuffle and seed is not None:
        kwargs["worker_init_fn"] = seed_worker
        kwargs["generator"] = make_generator(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
