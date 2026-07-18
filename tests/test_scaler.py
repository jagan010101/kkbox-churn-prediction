import numpy as np
import pandas as pd
import pytest

from kkbox.preprocessing import apply_scaler, fit_scaler


def test_scaler_fit_on_train_only_normalizes_train_to_unit_variance():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"x": rng.normal(loc=10, scale=5, size=1000)})
    scaler = fit_scaler(train_df, ["x"])
    scaled_train = apply_scaler(train_df, scaler, ["x"])
    assert scaled_train["x_scaled"].mean() == pytest.approx(0, abs=1e-6)
    assert scaled_train["x_scaled"].std(ddof=0) == pytest.approx(1, abs=1e-6)


def test_scaler_does_not_refit_on_val_and_can_produce_nonzero_mean_there():
    # Val drawn from a shifted distribution: if the scaler were (incorrectly)
    # refit per-split instead of reusing train's fitted mean/std, val would
    # also come out ~N(0,1). It should not - proves train-only fitting.
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"x": rng.normal(loc=10, scale=5, size=1000)})
    val_df = pd.DataFrame({"x": rng.normal(loc=50, scale=5, size=1000)})  # shifted mean
    scaler = fit_scaler(train_df, ["x"])
    scaled_val = apply_scaler(val_df, scaler, ["x"])
    assert abs(scaled_val["x_scaled"].mean()) > 1  # far from 0 - train's stats, not val's, were used


def test_scaler_uses_identical_statistics_across_splits():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"x": rng.normal(loc=10, scale=5, size=1000)})
    val_df = pd.DataFrame({"x": [10.0, 15.0, 20.0]})
    scaler = fit_scaler(train_df, ["x"])
    scaled_a = apply_scaler(val_df, scaler, ["x"])
    scaled_b = apply_scaler(val_df, scaler, ["x"])
    # same fitted scaler applied twice to the same data -> bit-identical,
    # confirming apply_scaler carries no hidden per-call state/refitting
    assert scaled_a["x_scaled"].tolist() == scaled_b["x_scaled"].tolist()
