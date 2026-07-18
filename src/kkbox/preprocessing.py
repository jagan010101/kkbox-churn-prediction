"""Categorical encoding, numerical scaling, and imputation.

Extracted from 02_Feature_Engineering.ipynb. The leak-free invariant this
project relies on: every encoder/scaler/imputation statistic is fit on the
train split ONLY, then applied unchanged to val/test.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_encoder(train_col):
    """Label-encodes to integer indices starting at 0; a reserved
    '__unknown__' index absorbs missing/unseen categories at val/test/
    inference time (nn.Embedding-safe).
    """
    categories = sorted(train_col.dropna().unique().tolist())
    mapping = {cat: i for i, cat in enumerate(categories)}
    mapping["__unknown__"] = len(categories)
    return mapping


def apply_encoder(col, mapping):
    return col.map(mapping).fillna(mapping["__unknown__"]).astype(int)


def fit_encoders(train_df, categorical_cols):
    """Fits one encoder per categorical column on the train split only."""
    return {col: fit_encoder(train_df[col]) for col in categorical_cols}


def apply_encoders(df, encoders):
    """Returns df with a new f'{col}_enc' column per fitted encoder."""
    df = df.copy()
    for col, mapping in encoders.items():
        df[f"{col}_enc"] = apply_encoder(df[col], mapping)
    return df


def fit_scaler(train_df, scale_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[scale_cols])
    return scaler


def apply_scaler(df, scaler, scale_cols):
    df = df.copy()
    df[[f"{c}_scaled" for c in scale_cols]] = scaler.transform(df[scale_cols])
    return df


def fit_median_imputers(train_df, cols):
    """Median imputation statistics, fit on train only."""
    return {col: train_df[col].median() for col in cols}


def apply_median_imputers(df, medians):
    df = df.copy()
    for col, median in medians.items():
        df[col] = df[col].fillna(median)
    return df


def clean_age(bd_series):
    """bd (age) is self-reported and mostly garbage; only 1-100 is plausible."""
    return bd_series.where(bd_series.between(1, 100))


def song_completion(sum25, sum50, sum75, sum985, sum100):
    total = sum25 + sum50 + sum75 + sum985 + sum100
    weighted = 0.25 * sum25 + 0.50 * sum50 + 0.75 * sum75 + 0.985 * sum985 + 1.0 * sum100
    return weighted / (total + 1)


def registration_tenure_days(ref_date, registration_init_time):
    """ref_date - registration_init_time (int YYYYMMDD), in days."""
    import pandas as pd

    reg_date = pd.to_datetime(registration_init_time.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return (pd.to_datetime(ref_date) - reg_date).dt.days


def log1p_safe(series):
    return np.log1p(series)
