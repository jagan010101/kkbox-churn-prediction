import numpy as np
import pandas as pd

from kkbox.preprocessing import apply_encoder, fit_encoder


def test_unknown_bucket_gets_reserved_index_past_train_categories():
    train_col = pd.Series(["a", "b", "c", "a", "b"])
    mapping = fit_encoder(train_col)
    assert mapping == {"a": 0, "b": 1, "c": 2, "__unknown__": 3}


def test_unseen_category_at_apply_time_maps_to_unknown():
    mapping = fit_encoder(pd.Series(["a", "b"]))
    encoded = apply_encoder(pd.Series(["a", "b", "z", "z"]), mapping)
    assert encoded.tolist() == [0, 1, 2, 2]


def test_missing_value_at_apply_time_maps_to_unknown():
    mapping = fit_encoder(pd.Series(["a", "b"]))
    encoded = apply_encoder(pd.Series(["a", np.nan, "b"]), mapping)
    assert encoded.tolist() == [0, 2, 1]


def test_nan_in_train_col_is_not_itself_a_category():
    # dropna() in fit_encoder means NaN never becomes its own learned
    # category - it always falls through to __unknown__ at apply time
    mapping = fit_encoder(pd.Series(["a", np.nan, "b"]))
    assert set(mapping.keys()) == {"a", "b", "__unknown__"}


def test_encoded_output_is_embedding_safe_int_dtype():
    mapping = fit_encoder(pd.Series(["a", "b"]))
    encoded = apply_encoder(pd.Series(["a", "z"]), mapping)
    assert encoded.dtype.kind in ("i", "u")  # nn.Embedding requires integer indices
