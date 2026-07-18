"""Label-construction correctness on the hand-built toy transactions table
in conftest.py. Every expected value here is hand-computed in that file's
comments - see there for the reasoning behind each case.
"""

import pandas as pd

from kkbox import labels
from conftest import CHURN_GRACE_DAYS, CUTOFF, LTV_WINDOW_DAYS


def _build(con, toy_transactions_path, ref_date_max_cutoff=CUTOFF, guard_corrupted_expiry=True):
    txn_table = labels.cast_transaction_dates(con, toy_transactions_path, require_paid=True)
    ref_dates = labels.build_ref_dates(con, txn_table, ref_date_max_cutoff, guard_corrupted_expiry)
    churn = labels.build_churn_labels(con, txn_table, ref_dates, CHURN_GRACE_DAYS)
    ltv = labels.build_ltv_targets(con, txn_table, ref_dates, LTV_WINDOW_DAYS)
    ref_df = con.execute(f"select * from {ref_dates}").df().set_index("msno")
    churn_df = con.execute(f"select * from {churn}").df().set_index("msno")
    ltv_df = con.execute(f"select * from {ltv}").df().set_index("msno")
    return ref_df, churn_df, ltv_df


def test_free_trial_excluded_from_ref_dates(con, toy_transactions_path):
    ref_df, _, _ = _build(con, toy_transactions_path)
    assert "user_free_trial_only" not in ref_df.index


def test_clean_churner_ref_date_and_churn_and_zero_ltv(con, toy_transactions_path):
    ref_df, churn_df, ltv_df = _build(con, toy_transactions_path)
    assert ref_df.loc["user_churn_clean", "ref_date"] == pd.Timestamp("2016-12-01")
    assert churn_df.loc["user_churn_clean", "is_churn"] == 1
    # no transaction falls in the LTV window, so this user has no row at all
    # in the raw ltv_targets table (coalesced to 0 only in the full merge)
    assert "user_churn_clean" not in ltv_df.index


def test_clean_renewal_not_churned_and_ltv_summed(con, toy_transactions_path):
    ref_df, churn_df, ltv_df = _build(con, toy_transactions_path)
    assert ref_df.loc["user_renew_clean", "ref_date"] == pd.Timestamp("2016-12-01")
    assert churn_df.loc["user_renew_clean", "is_churn"] == 0
    assert ltv_df.loc["user_renew_clean", "ltv"] == 150


def test_churn_grace_window_is_inclusive_at_exactly_30_days(con, toy_transactions_path):
    _, churn_df, _ = _build(con, toy_transactions_path)
    assert churn_df.loc["user_grace_boundary_exact", "is_churn"] == 0


def test_churn_grace_window_excludes_31_days(con, toy_transactions_path):
    _, churn_df, _ = _build(con, toy_transactions_path)
    assert churn_df.loc["user_grace_boundary_over", "is_churn"] == 1


def test_ltv_window_is_inclusive_at_exactly_59_days_and_excludes_60(con, toy_transactions_path):
    ref_df, churn_df, ltv_df = _build(con, toy_transactions_path)
    # this user churns per the 30-day rule despite the later win-back purchase -
    # churn and LTV are independent computations, not redundant with each other
    assert churn_df.loc["user_ltv_boundary", "is_churn"] == 1
    assert ltv_df.loc["user_ltv_boundary", "ltv"] == 200


def test_corrupted_expiry_guard_changes_selected_ref_date(con, toy_transactions_path):
    guarded, _, _ = _build(con, toy_transactions_path, guard_corrupted_expiry=True)
    unguarded, _, _ = _build(con, toy_transactions_path, guard_corrupted_expiry=False)
    assert guarded.loc["user_corrupted_masks_true_cycle", "ref_date"] == pd.Timestamp("2016-11-01")
    assert unguarded.loc["user_corrupted_masks_true_cycle", "ref_date"] == pd.Timestamp("2016-12-10")


def test_ref_date_max_cutoff_excludes_unresolvable_cycles(con, toy_transactions_path):
    capped, _, _ = _build(con, toy_transactions_path, ref_date_max_cutoff=CUTOFF)
    uncapped, _, _ = _build(con, toy_transactions_path, ref_date_max_cutoff=None)
    assert "user_beyond_cutoff" not in capped.index
    assert uncapped.loc["user_beyond_cutoff", "ref_date"] == pd.Timestamp("2017-01-15")
