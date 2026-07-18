import duckdb
import pandas as pd
import pytest

# Shared constants for the toy-transactions label tests. Match config.yaml's
# defaults so the hand-computed expected values below are meaningful checks
# against production behavior, not an arbitrary parallel convention.
CUTOFF = pd.Timestamp("2016-12-31")
CHURN_GRACE_DAYS = 30
LTV_WINDOW_DAYS = 59


@pytest.fixture
def con():
    conn = duckdb.connect()
    yield conn
    conn.close()


@pytest.fixture
def toy_transactions_path(tmp_path):
    """Hand-built transactions table covering: paid vs free-trial ref_date
    eligibility, churn-within-grace-window logic (both boundary directions),
    LTV window summation (both boundary directions), the corrupted
    expire_dt < txn_dt guard, and ref_date_max_cutoff exclusion. See
    test_labels.py for the hand-computed expected value of every row.
    """
    cols = [
        "msno", "payment_method_id", "payment_plan_days", "plan_list_price",
        "actual_amount_paid", "is_auto_renew", "transaction_date",
        "membership_expire_date", "is_cancel",
    ]
    rows = [
        # user_churn_clean: one paid cycle, no renewal -> churned, zero LTV
        ["user_churn_clean", 1, 30, 100, 100, 1, 20161101, 20161201, 0],

        # user_renew_clean: renews 4 days after expiry (well within 30-day
        # grace) and within the 59-day LTV window -> not churned, ltv=150.
        # The renewal's own expiry (2017-01-05) is past the cutoff so it
        # isn't itself a ref_date candidate - ref_date stays at 2016-12-01.
        ["user_renew_clean", 1, 30, 100, 100, 1, 20161101, 20161201, 0],
        ["user_renew_clean", 1, 30, 150, 150, 1, 20161205, 20170105, 0],

        # user_free_trial_only: $0 transaction never establishes a ref_date
        ["user_free_trial_only", 1, 30, 100, 0, 1, 20161101, 20161201, 0],

        # user_corrupted_masks_true_cycle: two real cycles (last valid one
        # expires 2016-11-01), then a cancel row whose membership_expire_date
        # (2016-12-10) precedes its OWN transaction_date (2016-12-15) - a
        # real KKBox data-quality pattern. Without the guard this corrupted,
        # larger-but-invalid date would be wrongly selected as ref_date
        # instead of the true 2016-11-01 cycle.
        ["user_corrupted_masks_true_cycle", 1, 30, 100, 100, 1, 20160901, 20161001, 0],
        ["user_corrupted_masks_true_cycle", 1, 30, 100, 100, 1, 20161001, 20161101, 0],
        ["user_corrupted_masks_true_cycle", 1, 30, 100, 100, 1, 20161215, 20161210, 1],

        # user_grace_boundary_exact: renewal transaction lands exactly 30
        # days after ref_date (2016-12-31) -> the grace window is inclusive,
        # so this must NOT count as churn. Its own expiry (2017-03-01) is
        # past the cutoff so it doesn't shift ref_date itself.
        ["user_grace_boundary_exact", 1, 30, 100, 100, 1, 20161101, 20161201, 0],
        ["user_grace_boundary_exact", 1, 30, 100, 100, 1, 20161231, 20170301, 0],

        # user_grace_boundary_over: renewal lands 31 days after ref_date -
        # one day past the grace window -> must count as churn.
        ["user_grace_boundary_over", 1, 30, 100, 100, 1, 20161101, 20161201, 0],
        ["user_grace_boundary_over", 1, 30, 100, 100, 1, 20170101, 20170201, 0],

        # user_ltv_boundary: churns per the 30-day rule (win-back transaction
        # arrives well after the grace window), but the win-back purchase at
        # exactly ref_date+59 must still count toward LTV (inclusive upper
        # bound), while a second purchase at ref_date+60 must not.
        ["user_ltv_boundary", 1, 30, 100, 100, 1, 20161101, 20161201, 0],
        ["user_ltv_boundary", 1, 30, 200, 200, 1, 20170129, 20170228, 0],
        ["user_ltv_boundary", 1, 30, 300, 300, 1, 20170130, 20170301, 0],

        # user_beyond_cutoff: only cycle expires after the Dec-2016 cutoff -
        # excluded when ref_date_max_cutoff is set, included when uncapped
        # (the survival-analysis / Cox population).
        ["user_beyond_cutoff", 1, 30, 100, 100, 1, 20170101, 20170115, 0],
    ]
    df = pd.DataFrame(rows, columns=cols)
    path = tmp_path / "toy_transactions.parquet"
    df.to_parquet(path)
    return str(path)
