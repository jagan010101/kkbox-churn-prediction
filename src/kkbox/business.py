"""Retention Priority Score, budget allocation, and rank-sensitivity analysis.

Extracted from 06_Calibration_and_Business_Layer.ipynb, including this
session's percentile-rescaling fix: since the survivorship-bias fix, churn
is the majority outcome (~74%), so raw calibrated P(churn) clusters too
tightly to discriminate ranking on its own - p_churn used throughout is
each user's *percentile rank* of the raw calibrated probability, not the
probability itself (kept alongside as p_churn_raw).
"""

import numpy as np
import pandas as pd


def build_priority_scores(msno, is_churn, p_churn_calibrated, ltv_pred_log):
    """One row per user: p_churn_raw (calibrated probability), p_churn (its
    percentile rank - used for the priority score), e_ltv (raw-TWD LTV
    prediction), priority_score = p_churn * e_ltv. Sorted descending by
    priority_score with a 1-indexed rank column.
    """
    results = pd.DataFrame({
        "msno": msno,
        "is_churn": is_churn,
        "p_churn_raw": p_churn_calibrated,
        "e_ltv": np.expm1(ltv_pred_log),
    })
    results["p_churn"] = results["p_churn_raw"].rank(pct=True)
    results["priority_score"] = results["p_churn"] * results["e_ltv"]
    results = results.sort_values("priority_score", ascending=False).reset_index(drop=True)
    results["rank"] = np.arange(1, len(results) + 1)
    return results


def five_segment_illustration(results, segments=None):
    """Picks a real user closest to each segment's median priority score,
    using the 33rd/67th percentiles of p_churn and e_ltv as low/med/high
    thresholds.
    """
    segments = segments or [
        ("high", "high", "Immediate: personal outreach, premium discount"),
        ("high", "low", "Low priority: automated email only"),
        ("low", "high", "Monitor: no immediate action needed"),
        ("medium", "medium", "Queue for weekly retention campaign"),
        ("low", "low", "No action: below cost-of-retention threshold"),
    ]
    p_lo, p_hi = results["p_churn"].quantile([0.33, 0.67])
    v_lo, v_hi = results["e_ltv"].quantile([0.33, 0.67])

    def pick_example(risk, value):
        p_mask = (results["p_churn"] < p_lo if risk == "low"
                  else results["p_churn"] > p_hi if risk == "high"
                  else results["p_churn"].between(p_lo, p_hi))
        v_mask = (results["e_ltv"] < v_lo if value == "low"
                  else results["e_ltv"] > v_hi if value == "high"
                  else results["e_ltv"].between(v_lo, v_hi))
        subset = results[p_mask & v_mask]
        return subset.iloc[(subset["priority_score"] - subset["priority_score"].median()).abs().argmin()]

    rows = []
    for risk, value, action in segments:
        row = pick_example(risk, value)
        rows.append({
            "segment": f"{risk}-risk, {value}-value", "p_churn_percentile": row["p_churn"],
            "p_churn_raw": row["p_churn_raw"], "e_ltv_twd": row["e_ltv"],
            "priority_score": row["priority_score"], "recommended_action": action,
        })
    return pd.DataFrame(rows)


def expected_revenue_saved(results, selected_idx, retention_success_rate):
    return (results.loc[selected_idx, "priority_score"] * retention_success_rate).sum()


def budget_allocation_comparison(results, budget_twd, voucher_cost_twd, retention_success_rate, seed=42):
    """Greedy allocation by priority score, vs. random selection and
    churn-probability-only ranking, all under the same expected-value
    formula so the comparison isolates the ranking strategy.
    """
    n_interventions = budget_twd // voucher_cost_twd
    n_actual_churners = int(results["is_churn"].sum())

    model_selected = results.index[:n_interventions]
    rng = np.random.default_rng(seed)
    random_selected = rng.choice(results.index, size=n_interventions, replace=False)
    churn_only_selected = results.sort_values("p_churn", ascending=False).index[:n_interventions]

    revenue_model = expected_revenue_saved(results, model_selected, retention_success_rate)
    revenue_random = expected_revenue_saved(results, random_selected, retention_success_rate)
    revenue_churn_only = expected_revenue_saved(results, churn_only_selected, retention_success_rate)

    return {
        "n_interventions": int(n_interventions),
        "n_test_users": len(results),
        "n_actual_churners": n_actual_churners,
        "coverage_of_test_set_pct": n_interventions / len(results) * 100,
        "coverage_of_at_risk_pct": n_interventions / n_actual_churners * 100,
        "revenue_model_twd": float(revenue_model),
        "revenue_random_twd": float(revenue_random),
        "revenue_churn_only_twd": float(revenue_churn_only),
        "model_vs_random_pct": (revenue_model / revenue_random - 1) * 100,
        "model_vs_churn_only_pct": (revenue_model / revenue_churn_only - 1) * 100,
    }


def rank_sensitivity(results, n_interventions, perturbation=0.10):
    """Perturbs p_churn by +/-perturbation and recomputes rank; users whose
    rank barely moves are robustly prioritized. Returns (results_with_deltas,
    top_k_summary_dict).
    """
    results = results.copy()
    results["score_upper"] = (results["p_churn"] * (1 + perturbation)).clip(0, 1) * results["e_ltv"]
    results["score_lower"] = (results["p_churn"] * (1 - perturbation)).clip(0, 1) * results["e_ltv"]
    results["rank_upper"] = results["score_upper"].rank(ascending=False)
    results["rank_lower"] = results["score_lower"].rank(ascending=False)
    results["rank_delta"] = (results["rank_upper"] - results["rank_lower"]).abs()

    top_k = results.iloc[:n_interventions]
    summary = {
        "median_rank_delta": float(top_k["rank_delta"].median()),
        "n_borderline": int((top_k["rank_delta"] > n_interventions).sum()),
        "pct_borderline": float((top_k["rank_delta"] > n_interventions).mean() * 100),
        "n_distinct_p_churn_raw_top_k": int(top_k["p_churn_raw"].nunique()),
        "largest_tie_group_top_k": int(top_k["p_churn_raw"].value_counts().max()),
    }
    return results, summary
