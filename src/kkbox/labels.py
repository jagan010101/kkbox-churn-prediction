"""Per-user event-based reference date, churn label, and forward-revenue target construction.

This is the fix for the survivorship bias in KKBox's official train/train_v2
snapshot labels (see 02_Feature_Engineering.ipynb and 01_EDA.ipynb for the
full discussion): each user's label is computed relative to their OWN last
subscription cycle, not one shared global snapshot date.

Naming note: the forward-looking revenue target (`fwd_rev_59d`) is
deliberately NOT called "LTV" - a 59-day window is two billing cycles of
observed revenue, not a customer's lifetime value. See config.yaml's
`labels.fwd_rev_window_days` and project_report.md section 3.2 for the
two-billing-cycle justification.

The SQL here is a direct, parameterized extraction of the *label/target*
construction validated in 02_Feature_Engineering.ipynb (ref_date, is_churn,
fwd_rev_59d) and 01_EDA.ipynb / 03a_CatBoost_and_Cox_Models.ipynb (the Cox
survival model, which uses the uncapped population since handling
right-censored users properly is the entire reason to use a survival model).
tests/test_labels.py covers this and is the contract - do not change
build_ref_dates/build_churn_labels/build_fwd_rev_targets without
re-validating against it.

This module used to also have feature-column builders (build_txn_features,
build_latest_txn, build_engagement_features, build_feature_table) mirroring
02's per-user feature table, but they'd drifted out of sync with 02 (stuck
at the original 13-column version while 02 grew to 27) and were untested -
removed rather than left as misleading dead code. Only the label/target
construction lives here now.
"""

import pandas as pd


def cast_transaction_dates(con, transactions_path, table_name="txn", require_paid=True):
    """Creates a temp table with txn_dt/expire_dt DATE columns cast from the
    raw YYYYMMDD int columns. require_paid=True excludes $0 free-trial
    transactions, matching the survivorship-bias-fix convention: a trial
    that never converts isn't "customer churn" in the revenue sense.
    """
    paid_filter = "where actual_amount_paid > 0" if require_paid else ""
    con.execute(f"""
        create or replace temp table {table_name} as
        select *,
               strptime(cast(transaction_date as varchar), '%Y%m%d')::date as txn_dt,
               strptime(cast(membership_expire_date as varchar), '%Y%m%d')::date as expire_dt
        from '{transactions_path}'
        {paid_filter}
    """)
    return table_name


def build_ref_dates(con, txn_table, ref_date_max_cutoff=None, guard_corrupted_expiry=True, table_name="ref_dates"):
    """Each user's own last (optionally cutoff-capped) paid subscription cycle.

    ref_date_max_cutoff: pd.Timestamp, or None to use each user's true
    last-ever cycle regardless of how recent (needed for survival models,
    which handle right-censoring rather than requiring a fully-resolvable
    forward window).

    guard_corrupted_expiry: ~0.68% of transactions (mostly is_cancel=1 rows)
    carry a corrupted/reset membership_expire_date that precedes their own
    transaction_date (including literal 1970-01-01 sentinel values in some
    cases). When True, `expire_dt >= txn_dt` excludes these from being
    treated as a real cycle boundary - this is the historically correct
    behavior, applied in 01_EDA.ipynb's survival analysis and
    03a_CatBoost_and_Cox_Models.ipynb's Cox model.

    IMPORTANT: 02_Feature_Engineering.ipynb's currently-deployed pipeline
    (the one behind the committed models/*.pt checkpoints) predates this
    guard and does NOT apply it - impact was checked and is small (0.38%
    of that population), so it was left as a documented known issue rather
    than triggering a full retrain at the time. Pass
    guard_corrupted_expiry=False to exactly reproduce that pipeline's
    current behavior; this project's Phase 1 multi-seed retraining is a
    natural point to switch it to True everywhere and accept the retrain.
    """
    valid_expiry_source = (
        f"select msno, expire_dt from {txn_table} where expire_dt >= txn_dt"
        if guard_corrupted_expiry
        else f"select msno, expire_dt from {txn_table}"
    )
    cutoff_filter = ""
    if ref_date_max_cutoff is not None:
        cutoff_date = pd.Timestamp(ref_date_max_cutoff).date()
        cutoff_filter = f"and v.expire_dt <= date '{cutoff_date}'"

    con.execute(f"""
        create or replace temp table {table_name} as
        with valid_expiry as (
            {valid_expiry_source}
        )
        select t.msno, min(t.txn_dt) as start_dt, max(v.expire_dt) as ref_date
        from {txn_table} t
        join valid_expiry v using (msno)
        where 1=1 {cutoff_filter}
        group by t.msno
    """)
    return table_name


def build_churn_labels(con, txn_table, ref_dates_table, churn_grace_days, table_name="churn_labels"):
    """is_churn = 1 if no renewal transaction within churn_grace_days of ref_date.

    This is KKBox's own published definition ("no new valid service
    subscription within N days after the current membership expires"),
    generalized from a single shared snapshot date to each user's own
    ref_date.
    """
    con.execute(f"""
        create or replace temp table {table_name} as
        select r.msno,
               case when exists (
                   select 1 from {txn_table} t
                   where t.msno = r.msno and t.txn_dt > r.ref_date
                     and t.txn_dt <= r.ref_date + interval {churn_grace_days} day
               ) then 0 else 1 end as is_churn
        from {ref_dates_table} r
    """)
    return table_name


def build_fwd_rev_targets(con, txn_table, ref_dates_table, fwd_rev_window_days, table_name="fwd_rev_targets"):
    """Forward-looking revenue: sum(actual_amount_paid) in (ref_date, ref_date + fwd_rev_window_days].

    Named fwd_rev_59d downstream, not "LTV" - this is observed revenue over
    a fixed short window (two billing cycles at the default 59 days), not a
    customer's lifetime value.
    """
    con.execute(f"""
        create or replace temp table {table_name} as
        select t.msno, sum(t.actual_amount_paid) as fwd_rev_59d
        from {txn_table} t join {ref_dates_table} r using (msno)
        where t.txn_dt > r.ref_date and t.txn_dt <= r.ref_date + interval {fwd_rev_window_days} day
        group by t.msno
    """)
    return table_name
