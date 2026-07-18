"""Full churn+LTV evaluation, extracted verbatim from 04_Training_Baselines.ipynb."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

from kkbox.train import gather_predictions


def evaluate_full(model, loader):
    """AUC-ROC/AUC-PR (churn) + RMSE/MAE/R2 in both log and raw-TWD scale (LTV)."""
    logits, churn_true, ltv_pred, ltv_true = gather_predictions(model, loader)
    logits, churn_true = logits.numpy(), churn_true.numpy()
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    ltv_pred_log, ltv_true_log = ltv_pred.numpy(), ltv_true.numpy()
    ltv_pred_raw, ltv_true_raw = np.expm1(ltv_pred_log), np.expm1(ltv_true_log)
    return {
        "churn_auc_roc": roc_auc_score(churn_true, probs),
        "churn_auc_pr": average_precision_score(churn_true, probs),
        "ltv_rmse_log": float(np.sqrt(np.mean((ltv_pred_log - ltv_true_log) ** 2))),
        "ltv_rmse_raw_twd": float(np.sqrt(np.mean((ltv_pred_raw - ltv_true_raw) ** 2))),
        "ltv_mae_raw_twd": float(np.mean(np.abs(ltv_pred_raw - ltv_true_raw))),
        "ltv_r2_raw": float(r2_score(ltv_true_raw, ltv_pred_raw)),
    }
