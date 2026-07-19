"""Loads whichever forward-revenue model won the final comparison in
03c_ZILN_ForwardRevenue.ipynb (CatBoost Tweedie + p_pay_feature, vs. the ZILN
neural-net ensemble), exposing a single predict_fwd_rev(df) -> np.ndarray
interface so 04/05 don't need to know or care which model is in production.
"""

import json
import os

import torch
from catboost import CatBoostClassifier, CatBoostRegressor

from kkbox.data import columns_from_manifest
from kkbox.ziln import ZILNNet, predict_ensemble


def load_fwd_rev_predictor(processed_dir, models_dir, results_dir, feature_cols):
    """Returns (predict_fn, winner_name). predict_fn(df) -> np.ndarray of raw-TWD
    forward-revenue predictions; callers just pass the working dataframe (which
    must contain feature_cols) and don't need to branch on the winner themselves.
    """
    choice_path = os.path.join(results_dir, "fwd_rev_model_choice.json")
    with open(choice_path) as f:
        choice = json.load(f)
    winner = choice["winner"]

    if winner == "catboost":
        fwd_rev_model = CatBoostRegressor()
        fwd_rev_model.load_model(os.path.join(models_dir, "catboost_fwd_rev.cbm"))
        pay_clf = CatBoostClassifier()
        pay_clf.load_model(os.path.join(models_dir, "catboost_pay_clf.cbm"))

        def predict_fn(df):
            p_pay = pay_clf.predict_proba(df[feature_cols])[:, 1]
            df_with_pay = df[feature_cols].assign(p_pay_feature=p_pay)
            return fwd_rev_model.predict(df_with_pay)

        return predict_fn, winner

    if winner == "ziln":
        with open(os.path.join(processed_dir, "feature_manifest.json")) as f:
            manifest = json.load(f)
        cat_cols, num_cols, cardinalities, embed_dims = columns_from_manifest(manifest)

        with open(os.path.join(models_dir, "ziln_ensemble_manifest.json")) as f:
            ensemble_manifest = json.load(f)

        models = []
        for seed_path in ensemble_manifest["seed_paths"]:
            ckpt = torch.load(os.path.join(models_dir, seed_path), map_location="cpu")
            m = ZILNNet(cat_cols, cardinalities, embed_dims, len(num_cols), **ckpt["config"])
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            models.append(m)

        def predict_fn(df):
            return predict_ensemble(models, df, cat_cols, num_cols)

        return predict_fn, winner

    raise ValueError(f"unknown fwd_rev_model_choice.json winner: {winner!r}")
