# EDA Report: Warehouse Route Shipment Volume Forecasting

## 1. Task Summary

**Goal**: Predict `target_1h` (volume of goods shipped per route in the last hour) for each `(route_id, timestamp)` pair in the test set.

**Metric**: `WAPE + |Relative Bias|`, where:
- WAPE = sum(|y_pred - y_true|) / sum(y_true) — penalizes absolute errors proportional to total volume
- Relative Bias = |sum(y_pred) / sum(y_true) - 1| — penalizes systematic over/under-prediction

**Key implication of the metric**: An unbiased model (sum of predictions = sum of actuals) would have Relative Bias = 0, reducing the problem to minimizing WAPE alone. This means post-processing to correct aggregate bias is important.

**Data**: Tabular time series with 30-minute frequency, 1000 routes, status features describing goods flow through warehouse stages.

**Test**: 8 timestamps (4 hours) on 2025-11-01 11:00-14:30 for all 1000 routes, immediately after the last training observation.

---

## 2. Key Data Facts

| Property | Train | Test |
|---|---|---|
| Shape | 4,630,000 x 9 | 8,000 x 3 |
| Unique routes | 1,000 | 1,000 |
| Route intersection | 100% | 100% |
| Timestamps per route | 4,630 | 8 |
| Time frequency | 30 min | 30 min |
| Time range | 2025-07-28 00:00 — 2025-11-01 10:30 | 2025-11-01 11:00 — 2025-11-01 14:30 |
| Time span | 96.4 days | 3.5 hours |
| Missing values | 0 | 0 |
| Duplicates | 0 | 0 |

**Perfect data grid**: every route has exactly the same 4,630 timestamps. No gaps, no missing values, no duplicates.

### Target (`target_1h`)
- Integer, non-negative
- Mean: 261,242; Median: 235,769; Std: 169,083
- Min: 0; Max: 10,831,340
- Zeros: 69,841 (1.5%)
- Right-skewed with heavy tail

### Status columns
- **status_1, status_2, status_3** (items processed at *current* warehouse): small values, mean ~23, max ~500-800
- **status_4, status_5** (items processed at *previous* warehouse): large values, mean ~119K and ~702K, highly correlated (r=0.88)
- **status_6** (items at previous warehouse): mean ~3,345, high variance

### Route-level scale differences
- Route mean target ranges from 50,752 to 981,152 (19.3x ratio)
- Route 702 is an extreme outlier: mean 981K, max 6.66M
- Bottom routes: ~50-70K mean

---

## 3. Key Observations and Anomalies

### 3.1 Upward Trend
- Target mean increased ~27% from first week (214K) to last week (272K)
- Steady growth through weeks 31-44
- The last 4 weeks show stabilization around 273-278K (weeks 41-44)
- **Hypothesis**: seasonal ramp-up in shipment volume (possibly pre-holiday season)

### 3.2 Weak but Present Temporal Patterns
- **Day of week**: Monday lowest (252K), Wednesday-Thursday highest (268K), ~6% swing
- **Hour of day**: trough at 21:00 (229K), peak at 12:00 (279K), ~22% swing
- **Half-hour slots**: similar pattern, trough 20:30-21:30, peak 11:00-12:30
- **Test window** (Sat 11:00-14:30) falls in the midday peak zone

### 3.3 Low Target Autocorrelation
- Lag-1 (30 min): 0.45 — moderate
- Lag-2 (1 hour): -0.065 — nearly zero, drops sharply
- Lag-24h: 0.15 — weak daily periodicity
- Lag-7d: 0.12 — weak weekly periodicity
- **Implication**: target is fairly noisy, not easily predictable from recent values alone

### 3.4 Status-Target Correlations (Global)
| Feature | Correlation with target |
|---|---|
| status_2 | 0.507 |
| status_1 | 0.488 |
| status_3 | 0.451 |
| status_4 | 0.431 |
| status_5 | 0.370 |
| status_6 | 0.104 |

- Per-route correlation for status_4 is weaker (mean 0.13), suggesting the global correlation is partly driven by scale differences across routes.
- status_3 has the strongest lag-1 correlation (0.41), suggesting it is a leading indicator of near-future shipments.

### 3.5 Test Window Specifics
- Test is a single Saturday (2025-11-01), 11:00 to 14:30
- Recent similar windows (same DOW + hours, last 28 days): mean target ~287K
- This is a very short prediction horizon — just 8 time steps per route
- **No status features available in test** — must predict from route identity and time features alone, or use historical patterns

---

## 4. Hypotheses About Target Generation

1. **target_1h reflects actual hourly shipment volume**, aggregated over 1-hour windows with 30-min reporting frequency. The 30-min offset means consecutive observations overlap.
2. **status_1-3 are concurrent operational metrics** at the current warehouse — they measure processing activity *in the same time window* as the target. Their correlation with target is informative but **cannot be used for test prediction** since they're absent in test.
3. **status_4-6 measure upstream activity** that may lead to downstream shipments with some time delay.
4. **The upward trend** likely reflects real business growth or seasonal patterns, not data artifacts.
5. **Route 702's extreme outlier behavior** (mean ~4x the median route) suggests it may be a major hub or aggregation route.

---

## 5. Leakage Risks and Critical Errors to Avoid

### CRITICAL: Status features are absent in test
The test set contains only `id`, `route_id`, `timestamp`. Status features are NOT available for prediction. Any model that relies on status features at prediction time will fail.

**Safe to use**:
- Route-level historical aggregates (mean, median, quantiles of target per route)
- Temporal features (hour, day of week, week number, trend)
- Rolling/lag features of target computed from **train data only** (but only up to the last known timestamp)
- Historical status aggregates per route (mean status patterns)

**Risky / requires careful handling**:
- Rolling features that inadvertently include future data
- Features computed on the full train set without respecting temporal order

**Cannot use at test time**:
- Current status_1..6 values (not in test)
- Any feature that requires concurrent or future information

### Feature engineering for test
Since test timestamps are 2025-11-01 11:00-14:30 and train ends at 2025-11-01 10:30:
- We can use ALL train data as history
- Rolling averages up to the last train timestamp are valid
- Route-level seasonal patterns (same hour, same DOW) are valid

---

## 6. Local Validation Strategy

### Recommended: Time-based holdout mimicking test structure

**Setup**: Hold out the last 8 timestamps (4 hours) of training data for each route as the validation set.

Specifically:
- **Validation set**: 2025-11-01 07:00 to 2025-11-01 10:30 (8 timestamps x 1000 routes = 8,000 rows) — mirrors test structure exactly
- **Training for validation**: Everything before 2025-11-01 07:00

**Why this works**:
1. Same structure as test (8 timestamps per route, same 4-hour window format)
2. Time-forward split prevents data leakage
3. All routes are in both train and validation (same as test)
4. Same time-of-day range would be ideal, but the test hours (11-14:30) are the first unseen; using last known hours is the closest proxy

**Alternative splits** for robustness:
- Multiple time-forward folds: hold out last 8 timestamps of each of several different days
- Expanding window: train on weeks 31-N, validate on N+1's equivalent window

**What to avoid**:
- Random splits (ignore temporal structure)
- Route-based splits (all routes are in test)

---

## 7. Baseline Roadmap

### Level 1: Naive Baseline (Global Mean)
- **Approach**: Predict global mean of target (261,242) for all test rows
- **Complexity**: Trivial
- **Expected quality**: Poor WAPE, but Relative Bias = 0 by construction (if test has similar mean)
- **Risk**: Ignores route-level and temporal variation

### Level 2: Route Mean Baseline
- **Approach**: Predict per-route historical mean
- **Complexity**: Trivial
- **Expected quality**: Better than global mean, captures route-level scale
- **Risk**: Ignores temporal trends and daily patterns

### Level 3: Route + Time Features Baseline
- **Approach**: Per-route mean, adjusted by hour-of-day and day-of-week multipliers
- **Complexity**: Low
- **Expected quality**: Good improvement, captures both route scale and temporal patterns
- **Risk**: May not capture the upward trend well

### Level 4: Statistical Baseline (Recent Route Average)
- **Approach**: Average of last N days' same-DOW same-hour target per route
- **Complexity**: Low-Medium
- **Expected quality**: Captures trend + seasonality + route scale
- **Risk**: Small sample for specific DOW+hour combinations

### Level 5: Tabular ML Baseline (LightGBM/CatBoost)
- **Approach**: Train on tabular features (route embeddings, time features, historical aggregates, lag features)
- **Complexity**: Medium
- **Expected quality**: Strong, can capture non-linear interactions
- **Risk**: Must be careful about feature leakage from status columns; requires proper temporal cross-validation

### Level 6: Strong Baseline (Ensembled tabular models)
- **Approach**: LightGBM + CatBoost ensemble with rich feature engineering
- **Complexity**: Medium-High
- **Expected quality**: Near-optimal for this type of problem
- **Risk**: Overfitting if not validated carefully

### Metric-specific considerations for all baselines:
- **Bias correction**: After fitting, check if aggregate predictions match aggregate actuals; apply multiplicative correction if needed
- **WAPE sensitivity**: Errors on high-volume routes contribute more; consider weighting by route volume
- **Zero handling**: 1.5% of targets are 0; clipping predictions at 0 is essential

---

## 8. Priority Feature List

### Must-have features (available for test):
1. `route_id` (categorical / embedding)
2. `hour` (0-23)
3. `minute` (0 or 30)
4. `day_of_week` (0-6)
5. `is_weekend` (binary)
6. Route historical mean target
7. Route historical median target
8. Route historical std target
9. Route mean target for same hour
10. Route mean target for same DOW
11. Route mean target for same DOW + hour
12. Route mean target over last 7 days
13. Route mean target over last 14 days
14. Global trend feature (days since start / week number)

### Nice-to-have features:
15. Route target quantiles (25th, 75th percentile)
16. Route target coefficient of variation
17. Exponentially weighted moving average of route target
18. Route target at same time yesterday (lag-48)
19. Route target at same time last week (lag-336)
20. Historical mean of status_4/status_5 per route (proxy for upstream volume)
21. Route "activity ratio" (non-zero fraction)
22. Interaction: route_mean * hour_effect

---

## 9. Model Priority Order

1. **Route mean baseline** — immediate sanity check
2. **Route + DOW + hour statistical model** — fast, interpretable
3. **LightGBM** with features #1-14 — strong tabular learner, fast training
4. **CatBoost** with same features — handles categoricals natively, good for comparison
5. **LightGBM + CatBoost blend** — typically gives 2-5% improvement
6. **Quantile regression** variants — if bias is an issue

---

## 10. Action Plan: Next 3 Iterations

### Iteration 1: Foundation
- [x] Complete EDA and document findings
- [ ] Set up validation framework (time-based holdout)
- [ ] Implement WAPE + |Relative Bias| metric calculator
- [ ] Build route mean baseline, evaluate on validation
- [ ] Build route + time statistical baseline, evaluate

### Iteration 2: First ML Model
- [ ] Engineer features #1-14 from the list above
- [ ] Train LightGBM with proper temporal validation
- [ ] Evaluate, analyze error patterns by route and time
- [ ] Apply bias correction post-processing
- [ ] Generate first submission

### Iteration 3: Refinement
- [ ] Add features #15-22
- [ ] Train CatBoost, compare with LightGBM
- [ ] Build ensemble (simple average or weighted)
- [ ] Hyperparameter tuning on validation
- [ ] Analyze worst-predicted routes, add route-specific adjustments
- [ ] Final submission

---

## Appendix: Why Tabular + Temporal Features > Classical Time Series

This problem is better solved as **tabular prediction with temporal features** rather than classical time series forecasting because:

1. **No status features in test**: The most informative features (status_1-6) are unavailable at prediction time, eliminating the main advantage of sequential models
2. **Short prediction horizon**: Only 8 timesteps ahead — simple extrapolation from historical patterns works well
3. **1000 independent routes**: Each route has its own scale but similar temporal patterns; a single model with route features can learn shared patterns efficiently
4. **Low autocorrelation**: Target values don't carry much information from one step to the next
5. **The key signal is route identity + time of day/week**, not recent sequence dynamics

---

## Proposed `src/` Pipeline Structure

```
src/
  eda/
    explore.py          # EDA utilities, data loading
    visualize.py        # Plotting functions
  features/
    build_features.py   # Feature engineering pipeline
    time_features.py    # Temporal feature extractors
    route_features.py   # Route-level aggregation features
  validation/
    split.py            # Time-based validation splits
    metrics.py          # WAPE + Relative Bias calculator
  train/
    train_lgbm.py       # LightGBM training
    train_catboost.py   # CatBoost training
    ensemble.py         # Model blending
  infer/
    predict.py          # Inference pipeline
    postprocess.py      # Bias correction, clipping
  submission/
    format_submission.py # Generate submission CSV
```
