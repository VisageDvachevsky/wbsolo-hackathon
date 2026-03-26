# Deep Analytics Report: Validated EDA Findings and Forecasting Signal Analysis

This report builds on the initial EDA (`reports/eda_report.md`) and provides second-level analytical findings with verified hypotheses, quantitative evidence, and actionable conclusions for building the strongest possible baseline under `WAPE + |Relative Bias|`.

All results produced by scripts `experiments/04_target_process.py` through `experiments/12_validation_realism.py`.

---

## Executive Summary

1. **target_1h is a moderately smooth, noisy operational volume process** with high overlap-induced lag-1 autocorrelation (0.45) but near-zero true lag-2 autocorrelation (−0.08). True predictive signal from lags is weak.
2. **Route identity is the strongest single predictor**: route-mean explains much of the variance. Low-volume routes have CV ≈ 0.66, high-volume routes CV ≈ 0.41 — different regimes exist.
3. **Intraday hourly profiles are stable** (84.5% of routes have stability > 0.8), making `route × hour` a reliable feature. Day-of-week effect is weak (CV ≈ 0.03).
4. **Saturday midday (the test window) is very close to weekday midday**: ratio 0.985. No dramatic weekend shift. Recent Saturdays run ~6% above historical Saturday mean due to the upward trend.
5. **The best simple baseline is `profile × scale`** (route_mean_14d × global_hour_effect × global_dow_effect), achieving mean WAPE+RBias = 0.381 across 5 validation folds. A close second is the blend of route_mean_14d and route_dow_hour at 0.382.
6. **Bias correction is valuable**: post-hoc multiplicative correction improved route_mean_14d from 0.427 to 0.366 on the last holdout.
7. **status_3 is a true leading indicator** (within-route lead-1 correlation 0.41 vs concurrent 0.24), but status columns are unavailable at test time. Their main value is as route descriptors: route-level mean status_1/2/3 correlate ~0.96 with route-level mean target.
8. **Validation should use matching Saturday midday windows** (13 available in training) for most realistic proxy, supplemented by generic last-8-timestamp holdouts.

**What to build first**: `profile × scale` baseline (route's recent 14-day mean, scaled by global hourly and DOW profiles), with multiplicative bias correction post-processing.

---

## 1. Anatomy of target_1h as a Process

### 1.1 Stationarity and Trend

- **Global trend**: +27% increase over the training period (week 31 to week 44), slope ≈ 4,746/week.
- **Last 4 weeks stabilized**: weeks 41–44 range 273K–278K, suggesting the trend has plateaued.
- **Per-route trends**: 73.2% of routes show positive trend. Mean slope is positive but varies widely (std ≈ slope magnitude).
- **Conclusion**: There IS a meaningful upward trend. Recency-weighted features capture this better than full-history means.

### 1.2 Distribution Shape

| Statistic | Value |
|---|---|
| Mean | 261,242 |
| Median | 235,769 |
| Std | 169,083 |
| CV | 0.647 |
| Skewness | 2.279 |
| Kurtosis | 43.055 |
| Zeros | 1.51% |
| IQR | 193,012 |

**Interpretation**: Right-skewed, heavy-tailed. Not normally distributed. Large outlier events exist but are rare (1% above p99). The heavy tail means WAPE will be disproportionately affected by high-volume predictions.

### 1.3 Autocorrelation Structure

| Lag | Steps | Mean AC | Std | Interpretation |
|---|---|---|---|---|
| 30 min | 1 | 0.445 | 0.046 | **Mechanically inflated** by overlapping 1h windows |
| 1 hour | 2 | −0.076 | 0.081 | True signal: near zero, slightly negative |
| 2 hours | 4 | 0.027 | 0.069 | Negligible |
| 6 hours | 12 | 0.066 | 0.063 | Weak |
| 24 hours | 48 | 0.148 | 0.057 | Modest daily periodicity |
| 7 days | 336 | 0.117 | 0.053 | Weak weekly periodicity |

**Critical finding**: The lag-1 autocorrelation of 0.45 is **almost entirely due to overlapping 1-hour windows** (30-min reporting frequency means consecutive windows share half their data). The overlap-induced component is 0.52, meaning the true underlying process has essentially zero lag-1 autocorrelation.

**Smoothness ratio** (diff-variance / variance) = 1.10, confirming this is a **noisy operational volume**, not a smooth process.

**Verdict**: Lag-heavy approaches (ARIMA-style) are NOT recommended. The target is too noisy for sequential prediction. Route identity + temporal features >> lags.

---

## 2. Route Segmentation

### 2.1 Volume Segments

| Segment | Count | Mean Target | Mean CV | Mean Zero Frac |
|---|---|---|---|---|
| Low volume | 250 | 131,159 | 0.660 | 0.024 |
| Mid volume | 500 | 256,337 | 0.487 | 0.014 |
| High volume | 250 | 401,134 | 0.411 | 0.008 |

**Key observation**: Low-volume routes are fundamentally noisier (CV 0.66) than high-volume routes (CV 0.41). This means:
- High-volume routes are more predictable
- Low-volume routes will contribute disproportionate *relative* error per route
- But WAPE is volume-weighted, so high-volume errors matter more in absolute terms

### 2.2 Intraday Profile Stability

- **84.5% of routes have highly stable intraday profiles** (stability > 0.8)
- Only **0.0% are truly unstable** (stability < 0.5)
- **Conclusion**: Using route-specific hourly profiles is safe and reliable. The intraday shape is a reusable template.

### 2.3 Weekday Pattern

- DOW CV across routes is only **0.031** — weekday patterns are very weak
- This means DOW matters at the global level but individual routes don't deviate much from their own average across days

### 2.4 Recent vs Historical Stability

| Period | Corr with All-History | Mean Ratio | Abs Drift |
|---|---|---|---|
| Last 7d | 0.936 | 1.040 | 12.4% |
| Last 14d | 0.944 | 1.050 | 11.4% |
| Last 28d | 0.957 | 1.052 | 10.3% |

Route rankings are very stable (corr > 0.93), but recent means are ~4–5% higher than historical due to the upward trend. Using recent means captures the trend better.

### 2.5 Do Different Segments Need Different Strategies?

Using route-mean instead of global-mean improves error by:
- **Low-volume routes: −52.9%** (huge improvement)
- **Mid-volume routes: −6.5%**
- **High-volume routes: −24.3%**

**Conclusion**: Route-specific prediction is essential, especially for low-volume routes. A single global mean is catastrophically bad for diverse routes. However, a single model with route features (rather than separate per-segment models) is likely sufficient given the continuous nature of the variation.

### 2.6 WAPE Concentration

Under global-mean prediction:
- Top 10 routes contribute only **2.5%** of WAPE error
- Top 50 routes: **9.3%**
- Top 100 routes: **15.3%**

Error is relatively evenly distributed. No single route dominates. Optimizing specifically for a few routes is NOT the path to winning — broad accuracy matters.

---

## 3. Test-Window Centric Analysis

### 3.1 Saturday Midday Characteristics

| Metric | Saturday Midday | Weekday Midday | Ratio |
|---|---|---|---|
| Mean target | 271,075 | 275,113 | 0.985 |
| Median target | 244,764 | 248,549 | — |
| Std target | 171,413 | 170,651 | — |

**Saturday midday is almost identical to weekday midday** (ratio 0.985). There is no dramatic weekend drop. This means:
- DOW-specific modeling gives marginal benefit for this particular test window
- Route-mean with hourly adjustment is likely sufficient

### 3.2 Recent Saturday Trend

| Window | Mean Target | Ratio to All Sat |
|---|---|---|
| All Saturdays | 271,075 | 1.000 |
| Last 1 week | 289,587 | 1.068 |
| Last 2 weeks | 287,052 | 1.059 |
| Last 4 weeks | 286,789 | 1.058 |
| Last 8 weeks | 289,834 | 1.069 |

Recent Saturdays are 6–7% above the full-history Saturday mean, reflecting the upward trend. Using recent history (last 2–4 weeks) better approximates the test regime.

### 3.3 Per-Route Saturday/Weekday Ratio

- Mean ratio: 0.986 (very close to 1)
- Std: 0.061
- 42.9% of routes have higher Saturday than weekday values

Most routes behave similarly on Saturday vs weekdays. Route-specific DOW adjustment is low-value.

### 3.4 End-of-Train Drift

| Window | Mean Target | Ratio to Overall |
|---|---|---|
| Last 3 days | 274,635 | 1.051 |
| Last 14 days | 275,184 | 1.053 |
| Overall | 261,242 | 1.000 |

No abrupt drift at the end of training. The elevated level is consistent with the stabilized trend. **Safe to use last 14 days as the recent window**.

---

## 4. Deep Temporal Seasonality

### 4.1 Hour × DOW Interaction

- Cross-DOW hourly shape variation: **0.012** (very low)
- **All days of the week share essentially the same intraday shape**
- This means a single global hourly profile is sufficient; DOW-specific hourly profiles are unnecessary

### 4.2 Weekday vs Weekend

- Weekend mean: 257,661 vs weekday mean: 262,596 (ratio 0.981)
- Profile shape difference: **0.012** (negligible)
- Peak hour: 16 (weekday), 16 (weekend) — same
- Trough hour: 21 (both) — same

**Conclusion**: Weekend/weekday distinction has minimal practical value for this data.

### 4.3 Route × Hour Interaction

- Cross-route hourly shape std: **0.098**
- This is meaningfully higher than the cross-DOW variation (0.012)
- **Routes DO have somewhat different intraday shapes**, but 84.5% are stable

**Conclusion**: Route-specific hourly profiles are worth using — they capture real variation.

### 4.4 Peak/Trough Stability

- Peak hour mode: **16**, std: 5.77 — moderately stable
- Trough hour mode: **21**, std: 3.21 — stable

Troughs are more consistent than peaks.

### 4.5 Shape × Scale Model Viability

| Model | R² | WAPE | RBias | Total |
|---|---|---|---|---|
| Global shape × route scale × DOW | 0.405 | 0.370 | 0.002 | 0.372 |
| Route-specific hourly shape × DOW | 0.427 | 0.363 | 0.002 | 0.364 |

**This decomposition works well**: a multiplicative model of route scale × hourly shape × DOW effect explains ~40% of variance and achieves WAPE ~0.37. Route-specific hourly shapes add modest improvement (+2% R²).

---

## 5. Status Columns as Explanatory Features

### 5.1 Global vs Within-Route Correlations

| Status | Global Corr | Within-Route Corr | Scale-Driven Component |
|---|---|---|---|
| status_1 | 0.488 | 0.080 | 0.408 — mostly scale-driven |
| status_2 | 0.507 | 0.098 | 0.409 — mostly scale-driven |
| status_3 | 0.451 | 0.239 | 0.212 — partly genuine |
| status_4 | 0.431 | 0.143 | 0.288 — mostly scale-driven |
| status_5 | 0.370 | 0.157 | 0.213 — partly genuine |
| status_6 | 0.104 | 0.043 | 0.061 — partly genuine |

**Most global correlations are driven by route scale** (big routes have big statuses AND big targets). Only status_3 has a meaningful within-route correlation (0.24).

### 5.2 Leading Indicator Analysis

| Status | Concurrent (within-route) | Lead-1 step (within-route) | Is Leading? |
|---|---|---|---|
| status_1 | 0.080 | 0.107 | Yes |
| status_2 | 0.098 | 0.158 | Yes |
| status_3 | 0.239 | **0.412** | **Yes — strong** |
| status_4 | 0.143 | 0.145 | Marginal |
| status_5 | 0.157 | 0.161 | Marginal |
| status_6 | 0.043 | 0.043 | No |

**status_3 is a genuine leading indicator**: its lead-1 correlation (0.41) is much higher than concurrent (0.24). This suggests status_3 processing activity predicts next-period shipments. However, since status_3 is unavailable in test, this insight is analytical only.

### 5.3 Status as Route Descriptors

Route-level mean status → mean target correlation:
- **status_1: 0.955, status_2: 0.959, status_3: 0.959** — near-perfect proxies for route scale
- **status_4: 0.728, status_5: 0.614** — decent
- **status_6: 0.189** — weak

**Practical use**: Historical mean status_1/2/3 per route can serve as high-quality route descriptors in ML models, providing an alternative to raw route_id embeddings.

---

## 6. Outlier and Extreme-Volume Analysis

### 6.1 Volume Concentration

| Top N routes | Volume share |
|---|---|
| Top 1 (route 702) | 0.38% |
| Top 5 | 1.05% |
| Top 10 | 2.00% |
| Top 50 | 9.34% |
| Top 100 | 17.09% |
| Bottom 500 | 25.92% |

Volume is moderately concentrated but no single route dominates. The bottom 500 routes still contribute 26% of volume — they cannot be ignored.

### 6.2 Extreme Events

- p95: 564K, p99: 743K, p99.9: 1.03M
- Events above p99: 46,300 (1.0%), contributing **3.4%** of total volume
- Route 702 has 2,043 extreme events (most of any route)
- 678 out of 1,000 routes have at least one extreme event

### 6.3 Temporal Patterns in Extremes

- Peak hour for extremes: **12** (1.41× relative risk) — midday, which is the test window
- Peak DOW: **Wednesday** (1.13× relative risk)
- **The test window (Saturday midday) has elevated extreme event risk due to the midday effect**

### 6.4 Predictability of Large Volumes

For top routes, spike autocorrelation:
- Route 702: spike AC(30m)=0.65, AC(24h)=0.33, AC(7d)=0.26 — most predictable
- Other top routes: AC(30m)≈0.22, AC(24h)≈0.05 — low predictability

**Most extreme events are NOT strongly predictable from recent history** (except route 702). They are closer to random operational variance.

### 6.5 WAPE Sensitivity

Under route-mean prediction:
- Top 10 routes: error share 2.2%, volume share 2.2% — proportional
- Top 50 routes: error share 7.9%, volume share 9.3%

**Error is distributed proportionally to volume**. No disproportionate outlier effect. The strategy should focus on broad accuracy, not route-specific optimization.

---

## 7. Bias-Sensitive Analysis Under WAPE + |Relative Bias|

### 7.1 Baseline Bias Profiles (Last-8 Holdout)

| Strategy | WAPE | RBias | RBias Direction | Total |
|---|---|---|---|---|
| global_mean | 0.511 | 0.001 | over | 0.512 |
| route_mean_all | 0.381 | 0.001 | over | 0.382 |
| route_mean_7d | 0.369 | 0.045 | over | 0.414 |
| route_mean_14d | 0.372 | 0.055 | over | 0.427 |
| route_hour_mean | 0.371 | 0.029 | under | 0.401 |
| route_same_dow | 0.376 | 0.002 | under | 0.378 |
| route_dow_hour_mean | 0.378 | 0.031 | under | 0.409 |
| route_recent_dow_28d | 0.376 | 0.047 | over | 0.424 |
| blend_14d_dow | 0.373 | 0.026 | over | 0.399 |

**Key insight**: Recent-period means (7d, 14d) have LOWER WAPE but HIGHER BIAS than full-history means. The recent data is more accurate pointwise but systematically overpredicts (the trend has stabilized, so recent means overshoot).

### 7.2 Post-Hoc Bias Correction

| Strategy | Before | After Correction | Improvement |
|---|---|---|---|
| route_mean_all | 0.382 | 0.381 | +0.001 |
| route_mean_14d | 0.427 | **0.366** | **+0.061** |
| route_hour_mean | 0.401 | 0.373 | +0.028 |
| route_dow_hour_mean | 0.409 | 0.380 | +0.029 |

**Bias correction is most impactful for recent-period baselines** that systematically overshoot. After correction, route_mean_14d becomes the best strategy (0.366).

**Recommendation**: Use a recent-period baseline (14d or profile×scale) with multiplicative bias correction.

---

## 8. Baseline Strategy Comparison (Multi-Fold)

### 8.1 Results Across 5 Validation Folds

| Rank | Strategy | Mean Total | Std | Mean WAPE | Mean RBias |
|---|---|---|---|---|---|
| 1 | **profile_x_scale** | **0.381** | 0.013 | 0.363 | 0.018 |
| 2 | blend_route14d_dowhour | 0.382 | 0.008 | 0.361 | 0.021 |
| 3 | route_mean_14d | 0.393 | 0.027 | 0.367 | 0.027 |
| 4 | route_mean_7d | 0.395 | 0.020 | 0.365 | 0.030 |
| 5 | recency_weighted_route | 0.396 | 0.028 | 0.368 | 0.029 |
| 6 | route_same_dow | 0.397 | 0.014 | 0.376 | 0.021 |
| 7 | route_mean_all | 0.403 | 0.020 | 0.375 | 0.028 |
| 8 | route_recent_dow_28d | 0.419 | 0.021 | 0.373 | 0.046 |
| 9 | route_dow_hour_mean | 0.420 | 0.019 | 0.375 | 0.045 |
| 10 | route_hour_mean | 0.423 | 0.021 | 0.366 | 0.057 |
| 11 | global_mean | 0.526 | 0.018 | 0.498 | 0.028 |

### 8.2 Key Findings

1. **profile_x_scale is the best baseline**: route_mean_14d × global_hour_effect × global_dow_effect. It combines recency, route specificity, and temporal adjustment multiplicatively.

2. **Blending is effective**: 50/50 blend of route_mean_14d and route_dow_hour gives similar quality with lower variance (std 0.008 vs 0.013).

3. **Route-specific DOW/hour means have high bias** because of sparse samples: with only ~14 Saturdays in training, route-specific Saturday-hour means are noisy.

4. **Global mean is catastrophically bad** (0.526) — route identity is essential.

5. **The gap between best and worst non-trivial baseline is 0.04** — meaningful but not enormous. The biggest gains come from getting route scale right.

### 8.3 Where Each Strategy Breaks

| Strategy | Weakness |
|---|---|
| global_mean | Ignores route scale entirely |
| route_mean_all | Doesn't capture upward trend |
| route_mean_7d/14d | Overpredicts (biased) due to recent elevation |
| route_hour_mean | High bias from sparse hour buckets |
| route_same_dow | Sparse Saturday data per route |
| route_dow_hour_mean | Very sparse (14 Sat × specific hours) → noisy |
| profile_x_scale | Assumes multiplicative decomposition holds |
| blend | Averages out weaknesses — safe but not optimal |

---

## 9. Validation Realism Analysis

### 9.1 Matching Saturday Windows

Found **13 matching Saturday midday windows** (DOW=5, hours 11–14) in training data:
- Dates from 2025-08-02 to 2025-10-25
- Mean target ranges from 241K to 302K (reflecting the upward trend)

### 9.2 Saturday-Specific Cross-Validation

| Saturday | route_mean metric | route_dow_hour metric |
|---|---|---|
| 2025-08-16 | 0.421 | 0.440 |
| 2025-08-23 | 0.411 | 0.431 |
| 2025-08-30 | 0.502 | 0.515 |
| 2025-09-06 | 0.552 | 0.554 |
| 2025-10-11 | 0.436 | 0.408 |
| 2025-10-18 | 0.447 | 0.422 |
| 2025-10-25 | 0.455 | 0.431 |

**Significant variation across Saturdays** (0.41–0.55). Some Saturdays are inherently harder to predict. Route_dow_hour outperforms route_mean on later Saturdays (when more training data for DOW-specific patterns is available).

### 9.3 Validation Approach Comparison

| Approach | DOW | Hours | Mean Target |
|---|---|---|---|
| Last 8 timestamps | 5 (Sat) | 7–10 | 260,947 |
| Last Saturday match | 5 (Sat) | 11–14 | 289,587 |
| Test window | 5 (Sat) | 11–14 | Unknown |

**The last 8 timestamps happen to be Saturday** (lucky!), but cover hours 7–10, not 11–14. The last matching Saturday window (2025-10-25, hours 11–14) is a better proxy for the test regime.

### 9.4 Metric Stability

Metric std across various holdout windows: **0.018** — moderate. Results vary by ~0.05 between best and worst folds, mostly driven by DOW effects.

### 9.5 Recommendation

1. **Primary validation**: Last Saturday matching window (2025-10-25, hours 11–14) — same DOW + hour pattern as test
2. **Secondary**: Multiple historical Saturday midday windows (13 available) for robustness assessment
3. **Tertiary**: Generic last-8-timestamps for recency check
4. **Philosophy**: Report mean ± std across Saturday folds; trust the Saturday-specific results more than generic folds

---

## 10. Hypotheses: Confirmed vs Refuted

### Confirmed

| # | Hypothesis | Evidence |
|---|---|---|
| H1 | Hourly seasonality is useful | Intraday profiles are stable (84.5%), hourly adjustment improves WAPE |
| H2 | Route identity is the strongest feature | Route-mean reduces WAPE from 0.51 to 0.38 |
| H3 | Recent history is more relevant than full history | profile_x_scale (14d base) beats route_mean_all |
| H4 | The upward trend matters | 27% growth over training, recent means ~5% above overall |
| H5 | Bias correction improves total metric | route_mean_14d improves by 0.06 with multiplicative correction |
| H6 | status_3 is a leading indicator | Lead-1 corr = 0.41 vs concurrent = 0.24 |
| H7 | Different route segments have different noise levels | CV ranges from 0.41 (high-vol) to 0.66 (low-vol) |

### Refuted

| # | Hypothesis | Evidence |
|---|---|---|
| H8 | Saturday is dramatically different from weekdays | Ratio = 0.985, near-identical profiles |
| H9 | Lag features carry strong signal | True lag-2 AC = −0.08 (overlap explains lag-1 AC) |
| H10 | DOW-specific modeling is crucial | DOW CV = 0.03, hour×DOW interaction = 0.012 |
| H11 | A few top routes dominate WAPE | Error is proportional to volume; top 10 = 2.2% of error |
| H12 | Extreme events are predictable | Most routes show spike AC(24h) < 0.07 |

### Partially Confirmed

| # | Hypothesis | Evidence |
|---|---|---|
| H13 | Route-specific hourly profiles beat global | R² improves 0.405→0.427, WAPE improves 0.370→0.363 — modest gain |
| H14 | Different segments need different strategies | Route-mean helps low-vol most (53%), but a single model with route features should suffice |

---

## 11. Feature Shortlist (Evidence-Based)

### Tier 1: Must-Have (proven useful)

1. **route_id** — categorical feature, the single strongest predictor
2. **hour** — stable intraday effect, 22% swing
3. **Route mean target (last 14d)** — captures scale + trend
4. **Route mean target (all history)** — captures long-term scale
5. **Global hourly profile multiplier** — stable, reusable
6. **Global DOW profile multiplier** — small but consistent effect

### Tier 2: Recommended (supported by evidence)

7. **Route × hour mean** — routes have somewhat different shapes (std=0.098)
8. **Route CV** — proxy for predictability (correlates with error)
9. **Route zero fraction** — structural route characteristic
10. **Historical mean status_1/2/3 per route** — nearly perfect route scale descriptors
11. **days_since_start** — linear trend proxy
12. **is_weekend** — weak but non-zero signal

### Tier 3: Marginal (small or uncertain benefit)

13. **Route × DOW mean** — sparse, DOW effect is weak
14. **Recency-weighted route mean** — similar to 14d mean
15. **Route target quantiles** — might help ML models distinguish distributions

### Not Recommended

- **Lag features of target** — true autocorrelation is near zero
- **Route-specific DOW×hour means** — too sparse (14 Saturdays × 8 half-hours)
- **Status features at prediction time** — unavailable in test

---

## 12. Recommended Strongest Simple Baseline

### Profile × Scale Baseline

```
prediction = route_mean_14d[route_id] × hour_effect[hour] × dow_effect[dow]
```

Where:
- `route_mean_14d` = mean target per route over the last 14 days of training
- `hour_effect` = global mean target at this hour / global mean target
- `dow_effect` = global mean target on this DOW / global mean target
- Apply multiplicative bias correction post-hoc: scale all predictions by `expected_total / predicted_total`
- Clip at 0

**Expected performance**: WAPE+RBias ≈ 0.38 (multi-fold mean), with std ≈ 0.013.

### Why This Baseline

1. **Captures route scale** via recent 14-day mean
2. **Captures trend** via recency (14d > full history)
3. **Captures intraday seasonality** via global hourly profile
4. **Has low bias** (multiplicative structure naturally calibrates well)
5. **Is robust** (doesn't rely on sparse route×DOW×hour buckets)
6. **Easy to implement** and understand

### Next Steps After Baseline

1. Add bias correction post-processing
2. Train LightGBM with Tier 1+2 features
3. Use Saturday-matched validation for model selection
4. Consider ensemble of profile×scale + LightGBM

---

## Appendix: Analysis Scripts

| Script | Purpose |
|---|---|
| `experiments/04_target_process.py` | Target stationarity, trend, autocorrelation, smoothness |
| `experiments/05_route_segmentation.py` | Route segments, profiles, stability, WAPE contribution |
| `experiments/06_test_window.py` | Saturday midday vs weekday, drift, recent trends |
| `experiments/07_temporal_seasonality.py` | Hour×DOW interaction, shape+scale viability |
| `experiments/08_status_analysis.py` | Status correlations, leading indicators, route descriptors |
| `experiments/09_outlier_analysis.py` | Volume concentration, extreme events, predictability |
| `experiments/10_bias_analysis.py` | Bias profiles, correction, strategy comparison |
| `experiments/11_baseline_comparison.py` | Multi-fold baseline ranking |
| `experiments/12_validation_realism.py` | Saturday CV, validation approach comparison |

JSON results files are saved alongside scripts as `results_XX_*.json`.
