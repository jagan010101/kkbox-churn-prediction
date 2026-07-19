# Project Report: KKBox Churn & Forward-Revenue Prediction

## 1. Problem statement

KKBox is a subscription music-streaming service (the WSDM 2017/2018 Churn
Prediction Challenge sponsor). The business questions this project
answers:

1. **Which subscribers are about to churn?** (`is_churn`, binary)
2. **How much revenue is a subscriber likely to bring in over the next ~2
   months?** (`fwd_rev_59d`, raw TWD)
3. **Given a fixed retention budget, who should it be spent on?** (a joint
   ranking that combines both of the above)

A fourth question — full time-to-churn, not just a point-in-time
probability — is answered separately by a Cox proportional-hazards survival
model, since a large share of the population is right-censored (still an
active subscriber with no resolved outcome yet).

The project deliberately does **not** predict customer lifetime value (LTV)
in the traditional sense. See §3 for why, and §9 for the naming discussion.

## 2. The central methodology problem: survivorship bias

KKBox's own official release ships pre-built label files (`train.csv`,
`train_v2.csv`) built from a **single fixed snapshot window** (subscriptions
expiring in Jan/Feb 2017). Naively training on those files looks
straightforward, but it has a serious flaw: it only includes users who were
*still subscribers* near that one window. Anyone who churned earlier in the
2015–2016 history and never came back is invisible to that snapshot — they
never had a subscription expiring in the labeled window at all.

`01_EDA.ipynb` quantifies the effect directly:

- **Official snapshot (`train_v2`) churn rate: ~9%.**
- **Corrected population churn rate: ~74%.**

These are not two measurements of the same thing with different noise —
they're answering different questions on different populations. The 9%
figure is a survivors-only conditional rate; it excludes roughly 40% of
everyone who ever paid KKBox. The 74% figure is computed over *every* paying
user's own last confirmable subscription cycle, not one shared calendar
window.

More importantly, the bias doesn't just shift the headline number — it
**reverses signs** on real business signals. Under the biased snapshot,
`registered_via` channel 7 looked like the *best*-retaining acquisition
channel (4.5% churn). Under the corrected population, channel 7 is the
*worst* (86.0% churn). A business that optimized acquisition spend toward
channel 7 based on the biased number would have been optimizing for exactly
the wrong thing. The same reversal happens for missing-gender users: they
looked like the *best*-retained segment under the snapshot and are actually
the *worst*-retained once the bias is corrected. `01_EDA.ipynb`'s
"Corrected churn rates" section walks through both reversals in detail.

**The fix**, used throughout this project from `02_Feature_Engineering.ipynb`
onward: every user's reference date (`ref_date`) is *their own* last
observable paid subscription cycle, not a shared global date. A user who
churned for good in mid-2015 is included on equal footing with someone still
active in late 2016, each scored relative to their own timeline. Free-trial
($0) transactions never establish a `ref_date` — a trial that never converts
isn't "customer churn" in the revenue sense this project cares about.

## 3. Targets

### 3.1 `is_churn`

KKBox's own published definition, generalized from one shared snapshot date
to each user's own `ref_date`: **churn = no renewal transaction within 30
days of the subscription's expiry.**

### 3.2 `fwd_rev_59d` — "forward revenue," deliberately not "LTV"

Earlier iterations of this project called this target `ltv`. That was
overclaiming. The target is `sum(actual_amount_paid)` over a **fixed 59-day
forward window** from each user's `ref_date` — not an estimate of a
customer's total future value. A user who pays for three more years after
that window closes, and one who churns immediately after it, both get scored
on the same 59 days of observed revenue. Calling that "LTV" implies a
horizon the model was never asked to predict, so the project renamed it
**forward revenue** everywhere.

**Why 59 days specifically** (not just "it fits the dataset's span," which
is a coincidence, not the reason): KKBox's dominant payment plan is monthly
(~30 days). A single 30-day window is fragile as a target — a user who
renews on day 32 instead of day 28 is still a retained, revenue-generating
customer, but a 30-day window would score them as `fwd_rev_30d = 0` purely
from calendar noise. Two consecutive billing cycles (59 days, using KKBox's
own 30-day grace convention doubled) gives enough runway that one delayed
renewal doesn't zero out the target, while staying short enough to be a
genuinely *forward*, not *lifetime*, measurement.

**`fwd_rev_59d` is not redundant with `is_churn`.** `has_fwd_rev = (fwd_rev_59d
> 0)` disagrees with `is_churn` for **35% of users** — e.g. someone can miss
the 30-day churn-grace renewal (counted as `is_churn=1`) and still make one
more payment before day 59. This is why the business layer (§7) combines
both signals rather than ranking on churn probability alone.

### 3.3 Population construction

Base population: every user with at least one real (`actual_amount_paid >
0`) transaction whose `membership_expire_date` is on or before
`2016-12-31` — the latest date that still leaves a full 59 days of
observable forward data before the raw data ends (`2017-02-28`). That
transaction's expiry becomes the user's `ref_date`. Result: **1,610,171
users**, split 70/15/15 into train/val/test (1,127,119 / 241,526 / 241,526),
stratified by `is_churn`.

The Cox survival model (§6.3) uses a different, **uncapped** population —
every user's true last-ever cycle, however recent — since handling
right-censored users properly is the entire point of a survival model.

## 4. Data pipeline

| Notebook | Role |
|---|---|
| `00_data_processing.ipynb` | `.7z` → typed parquet ETL for all 7 raw KKBox tables (~35GB raw). `user_logs.csv` (30GB uncompressed) is streamed via a FIFO rather than loaded into memory. |
| `01_EDA.ipynb` | Exploratory analysis: raw data dictionary, the survivorship-bias discovery (§2), Kaplan-Meier survival curves, feature-distribution audit on the engineered table. |
| `02_Feature_Engineering.ipynb` | Builds the population/labels (§3.3), engineers all model features (§5), splits and encodes/scales the data. |

**Known data-quality issues, handled explicitly rather than ignored:**

- `bd` (self-reported age): only ~33% of values fall in a plausible 1–100
  range (min/max seen: -7168 and 2016). Invalid values are treated as
  missing and median-imputed on the training split only.
- `total_secs` in `user_logs`: a small fraction of rows (~0.05%) have
  corrupted extreme values (int64 sentinel/overflow artifacts). Clipped to
  `[0, 86400]` (one day in seconds) before aggregation.
- ~0.68% of transactions (mostly `is_cancel=1` rows) carry a corrupted or
  reset `membership_expire_date` that precedes their own `transaction_date`
  — including literal `1970-01-01` sentinel values in some cases. Excluded
  from ever being selected as a `ref_date` candidate via an `expire_dt >=
  txn_dt` guard, everywhere `ref_date` is computed (feature population, Cox
  population, and the Kaplan-Meier survival analysis).

## 5. Feature engineering

The feature set grew in two separate, independently-validated passes. Every
addition was tested (baseline-vs-expanded comparison, same splits, reduced
CatBoost iterations for speed) before being adopted — nothing here is
speculative.

### 5.1 Original baseline: 13 features (4 categorical + 9 numerical)

Demographics (`city`, `bd_clean`, `gender`, `registered_via`), account
tenure, transaction aggregates (`avg_payment_plan_days`,
`avg_actual_amount_paid`, `num_transactions`, `is_auto_renew_rate`), and
30-day trailing engagement (`total_secs_log`, `daily_active_days`,
`avg_song_completion`).

### 5.2 First expansion: 13 → 19 features

Added `is_cancel_rate`, `avg_discount_rate`, `num_distinct_payment_methods`,
`avg_num_unq_songs`, `recent_engagement_ratio` (7-day vs. 30-day active-day
rate — a coarse trend signal), and `txn_frequency`. Grounded in previously
unused raw columns (`is_cancel`, `plan_list_price`, `num_unq`).
`is_cancel_rate` came out as the #1 most important feature for
forward-revenue at this stage.

### 5.3 Second expansion: 19 → 27 features

Motivated by top submissions, the following features were added.

- **Recency** — "how long since X happened," not just "how much happened in
  a fixed window." Added `days_since_last_login` (unbounded lookback, not
  capped at the existing 30-day engagement window) and
  `days_since_last_cancel`.
- **Most-recent-transaction snapshot** — the state *right now*, not a
  lifetime average. Added `most_recent_payment_plan_days`,
  `most_recent_actual_amount_paid`, `most_recent_is_auto_renew`,
  `most_recent_is_cancel`.
- **Magnitude-based listening trend** — `secs_trend_recent_vs_prior` and
  `numunq_trend_recent_vs_prior` (last-7-days vs. prior-23-days listening
  *intensity*, not just active-day presence like
  `recent_engagement_ratio`).

Validated with a combined baseline-vs-expanded CatBoost test: churn AUC
0.9455 → 0.9540, forward-revenue R² 0.3287 → 0.3605 (both at matched reduced
iterations). **`most_recent_is_cancel` came out as the single most important
forward-revenue feature by a wide margin (31% importance)** — more than
double `num_transactions` and over 20× the lifetime `is_cancel_rate` it's
related to. Recency of a cancellation event matters far more than how often
a user has historically canceled. Two candidates from the same round —
`std_dev_daily_secs_30d` (a volatility feature) and explicit
`never_logged_in`/`never_canceled` boolean flags — were tested and found to
add no value beyond what the sentinel-encoded recency features already
capture, and were **not** included.

Missing-value handling for the new features follows the project's existing
convention: `days_since_last_login`/`days_since_last_cancel` use a sentinel
value (larger than any real observed gap — "never happened" sorts as
furthest away for a tree split) rather than a fake numeric value, since 13.4%
of users never logged in and 70.6% never canceled before their own
`ref_date`, and both are legitimate states, not missing data.

**Final feature set: 4 categorical + 23 numerical = 27 columns.**

## 6. Models

### 6.1 Primary: CatBoost churn classifier

Standard binary classification (`Logloss`, `eval_metric="AUC"`), categorical
columns passed as native CatBoost categoricals (not label-encoded integers
treated as ordinal — CatBoost fits its own ordered target-statistic encoding
internally). Hyperparameters (`depth`, `learning_rate`, `l2_leaf_reg`,
`bagging_temperature`, `random_strength`) tuned via Optuna (§6.4).

### 6.2 Primary: CatBoost forward-revenue regressor (Tweedie loss)

`fwd_rev_59d` is zero-inflated (39.1% of users pay nothing in the 59-day
window) and right-skewed for the rest — exactly the compound Poisson-Gamma
shape a **Tweedie loss** is built for, unlike a plain RMSE/Gaussian loss on
a `log1p`-transformed target. Tested against a plain `RMSE`-on-`log1p`
baseline (151.1 TWD RMSE, R²=0.163, original 13-feature set) and won
clearly.

**Auxiliary payment-occurrence classifier (`p_pay_feature`).** A diagnostic
split of the regressor's errors by "did this user pay anything at all"
found R²=0.087 among payers only, vs. 0.57 for a regressor trained on payers
in isolation — most of the single model's apparent skill was coming from
separating payers from non-payers, not from predicting the paid amount.
Two ways of exploiting that were tried:

- **Two-stage hurdle model** (`P(pay) × E[amount | pay]`, both trained
  independently) — did **not** beat the single Tweedie model. Compounding
  independently-trained-model errors ate the gain (129.48 vs. 132.01 TWD
  RMSE on matched validation splits — the hurdle model was *worse*).
- **Soft feature injection** — feed a leakage-free, 2-fold cross-fitted
  `P(pay)` in as an *extra feature* to the same single Tweedie regressor.
  This *did* help (RMSE 132.8→131.4, R² 0.329→0.342 at reduced iterations),
  and is what shipped. The leakage guard: training-split rows are scored by
  a classifier fit on the *other* fold (never their own), so no row's
  feature ever comes from a model that saw that row's label.

A **hard-threshold gate** (`predict 0 if P(pay) < threshold else E[amount]`)
was also tried across 13 threshold values — best case (threshold=0.55) gave
R²=0.240, clearly worse than both production and the soft-multiplication
approach (R²=0.352). A discontinuous decision boundary amplifies classifier
misclassifications instead of hedging against them; soft blending doesn't
have that failure mode.

### 6.3 Primary: Cox proportional-hazards survival model (CatBoost)

CatBoost's native `loss_function="Cox"` — gradient-boosted trees optimizing
the Cox partial likelihood, giving a nonlinear/interaction-aware hazard
model instead of a linear combination of covariates. Uses the full
censoring-inclusive population (§3.3), not `model_dataset`'s restricted one
— **62.5% of paying users have no confirmed outcome yet** (still active,
last cycle too close to the data boundary to know if they'll renew), and
handling that properly is the entire reason to use a survival model instead
of a point-in-time classifier.

Duration/event construction mirrors `01_EDA.ipynb`'s Kaplan-Meier section:
`duration_days` = first paid transaction to last-ever
`membership_expire_date`; `event_churned=1` if that expiry is confirmable
(30+ days of runway before the data boundary), else censored. Target is
*signed* duration (positive = observed event, negative = censored), the
same convention CatBoost/XGBoost's Cox loss expects.

### 6.4 Hyperparameter tuning (Optuna)

`03b_Hyperparameter_Tuning.ipynb` — four independent studies (churn,
forward-revenue, Cox, ZILN), each with reduced-cost trials (CatBoost:
`iterations=300`; ZILN: `max_epochs=15`) so per-trial cost stays low; full
production budget is only spent once, on the winning configuration. Each
study runs for a fixed ~30-minute wall-clock timeout rather than a fixed
trial count. This is an offline search notebook — not part of the
run-in-order `00`→`05` sequence — whose output (`results/optuna_best_params.json`)
`03a` and `03c` read at the top.

### 6.5 Challenger: ZILN neural net (forward-revenue)

Grounded in Google's published Zero-Inflated LogNormal methodology (Wang et
al. 2019) for exactly this class of problem (zero-inflated, heavy-tailed
revenue targets). Unlike the rejected hurdle model (§6.2), ZILN is **a
single jointly-optimized network** predicting `(p_logit, mu, log_sigma)`
from one backbone, trained with a combined BCE + LogNormal-NLL loss:

```
E[y] = sigmoid(p_logit) * exp(mu + sigma^2 / 2)
```

**This required real numerical-stability debugging before it produced a
trustworthy number at all.** An unclamped `mu`/`log_sigma` let `exp(mu +
sigma^2/2)` overflow for some examples (`ValueError: Input contains
infinity...`). A first attempt at clamping (`log_sigma` max=3.0, permitting
σ up to ~20) eliminated the crash but produced a *worse* failure: R² =
**−42,682**, RMSE = 33,478. The fix wasn't the clamping *idea* — it was
using the actual empirical distribution of `log(fwd_rev_59d)` among payers
(mean=5.18, **std=0.546**) to set a realistic clamp (`log_sigma ∈ [-3, 0.7]`,
σ up to ~2.0) instead of an arbitrary permissive one. After that fix,
training was stable and the point-estimate formula produced sane
predictions (range $0–$885 on the validation set).

Final production version: Optuna-tuned architecture (§6.4), trained as a
**5-seed ensemble with averaged predictions** — a standard ZILN robustness
technique, since a single model's `sigma` estimate can be noisy per-seed.
`p_pay_feature` is deliberately **not** fed into ZILN — its own `p_logit`
head already jointly models payment probability, so an externally-computed
P(pay) would be redundant for this architecture.

**Verdict: ZILN lost to CatBoost Tweedie, even after full tuning and
ensembling.**

| | Test RMSE (TWD) | Test R² |
|---|---|---|
| CatBoost Tweedie (production) | **127.87** | **0.385** |
| ZILN 5-seed ensemble | 131.52 | 0.349 |

The joint-optimization idea is architecturally sound — it doesn't have the
hurdle model's compounding-error problem — but it isn't extracting more
signal than CatBoost from the same 27 features. `results/fwd_rev_model_choice.json`
records the decision (`"winner": "catboost"`) that `04`/`05` read
automatically via `kkbox.fwd_rev.load_fwd_rev_predictor`.

*(An alternative not pursued: Gamma-Gamma/BG-NBD "Buy Till You Die" models,
the classical actuarial approach to subscription LTV. Considered and set
aside — it's built around pure RFM summary statistics with no natural way
to use this project's engineered feature set, which is precisely what's
been shown to drive the gains here.)*

## 7. Calibration & business decision layer

### 7.1 Probability calibration

CatBoost's raw churn probabilities are already close to well-calibrated:
**ECE = 0.0043** (comfortably under the 0.05 "well-calibrated" threshold),
mean predicted P(churn) (0.7360) essentially identical to the true rate
(0.7360). Isotonic regression (a non-parametric monotone map, fit on
validation predicted-probability → outcome pairs) tightens this further to
**ECE = 0.0018** and is the calibration method used downstream — a
non-parametric map can correct arbitrary miscalibration shapes, including a
systematic shift, which a simpler single-scalar rescaling could not.

### 7.2 Retention Priority Score

```
priority_score = p_churn_percentile × E[forward_revenue]
```

`E[forward_revenue]` is whichever model won §6.5's comparison (currently
CatBoost), predicting directly in raw TWD.

**Why percentile rank, not the raw calibrated probability.** Since the
survivorship-bias fix, churn is the *majority* outcome (~74%), so most
calibrated probabilities cluster in a similarly high range — raw `P(churn)`
no longer cleanly separates a small at-risk minority the way it would under
a low base-rate framing. Converting to each user's percentile rank restores
a well-spread [0, 1] scale that still meaningfully discriminates between
users.

**A caveat worth taking seriously, not glossing over**: the churn model's
confidence has a side effect. Isotonic regression fits a piecewise-*constant*
map (280 distinct output values across 241,526 test users). Among the
top 1,000 budget-ranked users, only **23** distinct `p_churn` values occur,
and the largest tie group alone has **903** of those 1,000 users. This
directly explains why a churn-probability-only ranking strategy collapses to
**~20 TWD** of expected value in the budget simulation below — once 903 of
the top 1,000 share the same score, sorting by churn probability alone has
almost nothing left to break ties with, and effectively picks an arbitrary
subset that happens to include near-zero-revenue users. The combined
priority score doesn't have this failure mode, because `E[forward_revenue]`
still fully differentiates *within* the tied group — which is exactly why
the score multiplies the two signals instead of using churn probability
alone.

### 7.3 Budget allocation simulation

Greedy allocation: rank all users by priority score, fund interventions
(voucher cost 50 TWD, assumed 30% retention success rate) down the ranked
list until a fixed 50,000 TWD budget runs out (1,000 interventions, 0.41%
of the test set — 0.56% of the 177,757 users who actually churn).

| Ranking strategy | Expected revenue saved | vs. model |
|---|---|---|
| **Combined priority score (model)** | **65,567 TWD** | — |
| Random selection | 13,310 TWD | +392.6% (model wins) |
| Churn-probability only | 20 TWD | +320,809.9% (model wins) |

## 8. Final results

| Metric | Value |
|---|---|
| Churn AUC-ROC (test) | 0.961 |
| Churn AUC-PR (test) | 0.984 |
| Forward-revenue RMSE (test, raw TWD) | 127.9 |
| Forward-revenue R² (test) | 0.385 |
| Forward-revenue model | CatBoost Tweedie (beat ZILN: 131.5 RMSE / 0.349 R²) |
| Cox concordance (test) | 0.965 |
| Calibration ECE (raw → isotonic) | 0.0043 → 0.0018 |
| Retention budget simulation | +392.6% vs. random selection |

Full run-to-run history, for context: original 13-feature/RMSE-loss/default-
hyperparameter baseline (churn AUC 0.936, forward-revenue RMSE 155.3, R²
0.081) → first feature pass + Tweedie loss (RMSE 134.8, R² 0.334) → + Optuna
tuning on 19 features (churn 0.952, RMSE 132.8, R² 0.353, Cox 0.951) →
second feature pass + fresh Optuna tuning on 27 features, current state
(churn 0.961, RMSE 127.9, R² 0.385, Cox 0.965).

## 9. Repository structure

```
00_data_processing.ipynb                7z -> parquet ETL
01_EDA.ipynb                            Survivorship-bias discovery, Kaplan-Meier survival curves
02_Feature_Engineering.ipynb            27-feature table + leak-free targets
03a_CatBoost_and_Cox_Models.ipynb       PRIMARY: CatBoost churn + forward-revenue, CatBoost Cox
03b_Hyperparameter_Tuning.ipynb         Optuna search (offline) - output consumed by 03a and 03c
03c_ZILN_ForwardRevenue.ipynb           ZILN challenger vs. CatBoost Tweedie - writes the winner decision
04_Calibration_and_Business_Layer.ipynb ECE, isotonic regression, Retention Priority Score, budget allocation
05_Final_Evaluation_Summary.ipynb       Consolidated headline metrics/plots for the shipped model

src/kkbox/          Shared code: business logic, calibration, ZILN model, CatBoost-vs-ZILN predictor
                     dispatch, config loading, label/target construction, encoder/scaler reference
                     implementation
tests/               16 tests: label-construction correctness (toy data), encoder/scaler leak-safety
data/processed/      Model-ready parquet splits + feature_manifest.json (only committed artifact)
models/              Trained model checkpoints (CatBoost .cbm, ZILN ensemble .pt)
results/             Metrics JSON/CSV, saved figures, the CatBoost-vs-ZILN decision record
run_pipeline.sh      Unattended runner: 02->03b->03a->03c->04->05, disk-space checks, stops on first failure
```

Reproduction: `./run_pipeline.sh` runs the `02`–`05` portion end to end
(disk-space checks, resumable failure point, ~2.5–3.5 hours - the Optuna
search in `03b` alone is ~2 hours). `00_data_processing.ipynb` and
`01_EDA.ipynb` aren't part of that script and are run directly (`jupyter
nbconvert --to notebook --execute --inplace <notebook>` or interactively) -
they involve multi-hour/multi-GB raw-data processing.

## 10. Known limitations and honest caveats

- **Forward-revenue R² (0.385) leaves most variance unexplained.** An
  oracle-ceiling diagnostic (substituting true labels for predicted ones)
  found the combination-strategy ceiling is R²=0.741 given a perfect
  classifier and the current payers-only regressor — meaning there's real
  headroom, but it's bounded by how hard "predict the exact paid amount
  conditional on paying" is from these features, not by an obviously wrong
  combination strategy.
- **The Optuna search is time-boxed** (~30 min/model), not exhaustive.
  Longer searches would plausibly find better configurations, especially
  for the ZILN net's first tuning pass.
- **`src/kkbox/labels.py`'s feature-builder functions are not kept in sync**
  with `02_Feature_Engineering.ipynb`'s current 27-column feature set — they
  reflect an earlier, smaller version. The *tested* part of that module
  (`build_ref_dates`/`build_churn_labels`/`build_fwd_rev_targets` — the
  core survivorship-bias-fix logic) is still accurate and covered by
  `tests/test_labels.py`; the feature columns are documentation/reference
  only, not a live mirror of the pipeline.
- **Cross-machine bit-reproducibility is not guaranteed.** `kkbox.determinism.seed_everything`
  seeds Python/NumPy/PyTorch and enables deterministic algorithms
  (`warn_only=True`, since a few ops used here have no deterministic
  implementation on all backends), but BatchNorm backward passes and DuckDB's
  multi-threaded query execution mean only same-machine, same-backend runs
  are claimed to match to many decimal places — not exact bit-for-bit
  reproduction across different hardware.
