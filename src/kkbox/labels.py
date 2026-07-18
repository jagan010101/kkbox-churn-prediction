"""Per-user event-based reference date, churn label, and LTV target construction.

This is the fix for the survivorship bias in KKBox's official train/train_v2
snapshot labels (see 02_Feature_Engineering.ipynb and 01_EDA.ipynb for the
full discussion): each user's label is computed relative to their OWN last
subscription cycle, not one shared global snapshot date.

The SQL here is a direct, parameterized extraction of what's already been
validated in 02_Feature_Engineering.ipynb (model_dataset_*.parquet, ref_date
capped at ref_date_max_cutoff so a full ltv_window_days of forward data is
always observable) and 01_EDA.ipynb / 08_GBT_Baseline_Comparison.ipynb (the
Cox survival model, which uses the uncapped population since handling
right-censored users properly is the entire reason to use a survival model).
Do not change this logic without re-validating against tests/test_labels.py.
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
    08_GBT_Baseline_Comparison.ipynb's Cox model.

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


def build_ltv_targets(con, txn_table, ref_dates_table, ltv_window_days, table_name="ltv_targets"):
    """Forward-looking revenue: sum(actual_amount_paid) in (ref_date, ref_date + ltv_window_days]."""
    con.execute(f"""
        create or replace temp table {table_name} as
        select t.msno, sum(t.actual_amount_paid) as ltv
        from {txn_table} t join {ref_dates_table} r using (msno)
        where t.txn_dt > r.ref_date and t.txn_dt <= r.ref_date + interval {ltv_window_days} day
        group by t.msno
    """)
    return table_name


def build_txn_features(con, txn_table, ref_dates_table, table_name="txn_agg"):
    """Pre-ref_date transaction aggregates: num_transactions, avg plan days/price, auto-renew rate."""
    con.execute(f"""
        create or replace temp table {table_name} as
        select t.msno,
               count(*) as num_transactions,
               avg(t.payment_plan_days) as avg_payment_plan_days,
               avg(t.actual_amount_paid) as avg_actual_amount_paid,
               avg(t.is_auto_renew) as is_auto_renew_rate
        from {txn_table} t join {ref_dates_table} r using (msno)
        where t.txn_dt <= r.ref_date
        group by t.msno
    """)
    return table_name


def build_latest_txn(con, txn_table, ref_dates_table, table_name="latest_txn"):
    """Most recent payment_method_id (and other latest-txn fields) as of each user's ref_date."""
    con.execute(f"""
        create or replace temp table {table_name} as
        select msno, payment_method_id, is_auto_renew, is_cancel, payment_plan_days from (
            select t.msno, t.payment_method_id, t.is_auto_renew, t.is_cancel, t.payment_plan_days,
                   row_number() over (partition by t.msno order by t.txn_dt desc) rn
            from {txn_table} t join {ref_dates_table} r using (msno)
            where t.txn_dt <= r.ref_date
        ) where rn = 1
    """)
    return table_name


def build_engagement_features(con, user_logs_path, ref_dates_table, engagement_window_days, table_name="logs_agg"):
    """Trailing listening-activity aggregates over [ref_date - (window-1), ref_date].

    total_secs is clipped to [0, 86400] - a small fraction of user_logs rows
    have corrupted extreme values (see 01_EDA.ipynb, "Scanning the full
    user_logs history" section).
    """
    con.execute(f"""
        create or replace temp table {table_name} as
        select l.msno,
               count(distinct l.log_dt) as daily_active_days,
               sum(greatest(least(l.total_secs, 86400), 0)) as total_secs_sum,
               sum(l.num_25) as sum25, sum(l.num_50) as sum50, sum(l.num_75) as sum75,
               sum(l.num_985) as sum985, sum(l.num_100) as sum100
        from (
            select *, strptime(cast(date as varchar), '%Y%m%d')::date as log_dt
            from '{user_logs_path}'
        ) l
        join {ref_dates_table} r using (msno)
        where l.log_dt >= r.ref_date - interval {engagement_window_days - 1} day
          and l.log_dt <= r.ref_date
        group by l.msno
    """)
    return table_name


def build_feature_table(con, transactions_path, members_path, user_logs_path, label_cfg,
                         ref_date_max_cutoff=None, guard_corrupted_expiry=True, include_ltv=True):
    """One row per user: msno, ref_date, is_churn, ltv (if include_ltv), members
    demographics, and pre-ref_date transaction/engagement aggregates.

    This reproduces 02_Feature_Engineering.ipynb's merge_query exactly (when
    called with the same ref_date_max_cutoff/guard_corrupted_expiry it uses)
    and 08_GBT_Baseline_Comparison.ipynb's Cox survival feature table (when
    called with ref_date_max_cutoff=None, include_ltv=False).

    label_cfg: the 'labels' section of config.yaml (churn_grace_days,
    ltv_window_days, engagement_window_days, ...).
    """
    txn_table = cast_transaction_dates(con, transactions_path, require_paid=label_cfg["require_paid_ref_cycle"])
    ref_dates = build_ref_dates(con, txn_table, ref_date_max_cutoff, guard_corrupted_expiry)
    churn = build_churn_labels(con, txn_table, ref_dates, label_cfg["churn_grace_days"])
    txn_agg = build_txn_features(con, txn_table, ref_dates)
    latest_txn = build_latest_txn(con, txn_table, ref_dates)
    logs_agg = build_engagement_features(con, user_logs_path, ref_dates, label_cfg["engagement_window_days"])

    ltv_select, ltv_join = "", ""
    if include_ltv:
        ltv = build_ltv_targets(con, txn_table, ref_dates, label_cfg["ltv_window_days"])
        ltv_select = "coalesce(lt.ltv, 0) as ltv,"
        ltv_join = f"left join {ltv} lt using (msno)"

    return con.execute(f"""
        select
            r.msno, r.ref_date, cf.is_churn,
            m.city, m.bd, m.gender, m.registered_via, m.registration_init_time,
            {ltv_select}
            coalesce(txn_agg.num_transactions, 0) as num_transactions,
            txn_agg.avg_payment_plan_days,
            txn_agg.avg_actual_amount_paid,
            coalesce(txn_agg.is_auto_renew_rate, 0) as is_auto_renew_rate,
            latest_txn.payment_method_id,
            coalesce(logs_agg.daily_active_days, 0) as daily_active_days,
            coalesce(logs_agg.total_secs_sum, 0) as total_secs_sum,
            coalesce(logs_agg.sum25, 0) as sum25, coalesce(logs_agg.sum50, 0) as sum50,
            coalesce(logs_agg.sum75, 0) as sum75, coalesce(logs_agg.sum985, 0) as sum985,
            coalesce(logs_agg.sum100, 0) as sum100
        from {ref_dates} r
        left join {churn} cf using (msno)
        left join '{members_path}' m using (msno)
        left join {txn_agg} txn_agg using (msno)
        left join {latest_txn} latest_txn using (msno)
        left join {logs_agg} logs_agg using (msno)
        {ltv_join}
    """).df()
