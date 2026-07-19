"""ECE computation and isotonic regression.

Extracted from 04_Calibration_and_Business_Layer.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(probs)
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            bin_stats.append((lo, hi, 0, np.nan, np.nan))
            continue
        conf, acc = probs[mask].mean(), labels[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
        bin_stats.append((lo, hi, mask.sum(), conf, acc))
    return ece, pd.DataFrame(bin_stats, columns=["bin_lo", "bin_hi", "n", "avg_confidence", "avg_accuracy"])


def reliability_diagram(ax, bin_stats, ece, title):
    valid = bin_stats.dropna()
    ax.bar((valid["bin_lo"] + valid["bin_hi"]) / 2, valid["avg_accuracy"], width=0.08,
           alpha=0.7, edgecolor="black", label="observed accuracy")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.scatter(valid["avg_confidence"], valid["avg_accuracy"], color="red", zorder=5, label="bin (conf, acc)")
    ax.set_xlabel("predicted probability (confidence)")
    ax.set_ylabel("observed churn rate (accuracy)")
    ax.set_title(f"{title}\nECE={ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)


def fit_isotonic(val_probs_raw, val_true):
    """Non-parametric monotone calibration map, fit on validation predicted-probability -> outcome pairs."""
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(val_probs_raw, val_true)
    return iso_reg
