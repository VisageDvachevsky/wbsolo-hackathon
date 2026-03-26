"""
Deep analytics #1: Anatomy of target_1h as a process.
- Stationarity tests (ADF per route sample)
- Global and per-route trend analysis
- Distribution by route, hour, dow, week
- Regime detection: stable vs bursty routes
- Autocorrelation at multiple horizons (30min, 1h, 2h, 24h, 7d)
- Overlapping 1-hour window effect on autocorrelation
"""
import pandas as pd
import numpy as np
import sys, json

DATA_DIR = "/tmp/gh-issue-solver-1774506453169"
OUT = f"{DATA_DIR}/experiments/results_04_target_process.json"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
train["hour"] = train["timestamp"].dt.hour
train["dow"] = train["timestamp"].dt.dayofweek
train["week"] = train["timestamp"].dt.isocalendar().week.astype(int)
train["date"] = train["timestamp"].dt.date

results = {}

# ---------- 1. Global stationarity: rolling mean/std over weeks ----------
print("=== 1. Weekly rolling stats ===")
weekly = train.groupby("week")["target_1h"].agg(["mean", "std", "median"])
results["weekly_mean"] = {str(k): round(v, 1) for k, v in weekly["mean"].items()}
results["weekly_std"] = {str(k): round(v, 1) for k, v in weekly["std"].items()}
results["weekly_median"] = {str(k): round(v, 1) for k, v in weekly["median"].items()}

# Trend: linear regression on weekly means
weeks_arr = np.array(list(weekly.index), dtype=float)
means_arr = weekly["mean"].values
slope, intercept = np.polyfit(weeks_arr, means_arr, 1)
results["global_trend_slope_per_week"] = round(float(slope), 2)
results["global_trend_pct_change_over_period"] = round(
    float((means_arr[-1] - means_arr[0]) / means_arr[0] * 100), 2
)
print(f"Trend slope/week: {slope:.2f}, total change: {results['global_trend_pct_change_over_period']:.1f}%")

# ---------- 2. Per-route trend ----------
print("\n=== 2. Per-route trend ===")
route_trends = {}
for rid, grp in train.groupby("route_id"):
    wk = grp.groupby("week")["target_1h"].mean()
    if len(wk) >= 5:
        s, _ = np.polyfit(np.array(wk.index, dtype=float), wk.values, 1)
        route_trends[rid] = float(s)
rt_series = pd.Series(route_trends)
results["route_trend_stats"] = {
    "mean": round(rt_series.mean(), 2),
    "median": round(rt_series.median(), 2),
    "std": round(rt_series.std(), 2),
    "min": round(rt_series.min(), 2),
    "max": round(rt_series.max(), 2),
    "pct_positive": round((rt_series > 0).mean() * 100, 1),
    "pct_strong_positive": round((rt_series > 100).mean() * 100, 1),
}
print(f"Routes with positive trend: {results['route_trend_stats']['pct_positive']}%")

# ---------- 3. Distribution shape ----------
print("\n=== 3. Distribution shape ===")
target = train["target_1h"]
results["distribution"] = {
    "mean": round(float(target.mean()), 1),
    "median": round(float(target.median()), 1),
    "std": round(float(target.std()), 1),
    "skewness": round(float(target.skew()), 3),
    "kurtosis": round(float(target.kurt()), 3),
    "cv": round(float(target.std() / target.mean()), 4),
    "iqr": round(float(target.quantile(0.75) - target.quantile(0.25)), 1),
    "pct_zeros": round(float((target == 0).mean() * 100), 2),
}
print(f"Skewness: {results['distribution']['skewness']}, Kurtosis: {results['distribution']['kurtosis']}")

# ---------- 4. Regime detection: route volatility vs mean ----------
print("\n=== 4. Route regimes ===")
route_stats = train.groupby("route_id")["target_1h"].agg(["mean", "std", "median"])
route_stats["cv"] = route_stats["std"] / route_stats["mean"]
route_stats["zero_frac"] = train.groupby("route_id")["target_1h"].apply(lambda x: (x == 0).mean())

# Segment routes
low_vol = route_stats[route_stats["mean"] < route_stats["mean"].quantile(0.25)]
mid_vol = route_stats[(route_stats["mean"] >= route_stats["mean"].quantile(0.25)) &
                       (route_stats["mean"] < route_stats["mean"].quantile(0.75))]
high_vol = route_stats[route_stats["mean"] >= route_stats["mean"].quantile(0.75)]

results["route_segments"] = {
    "low_volume": {"count": len(low_vol), "mean_cv": round(low_vol["cv"].mean(), 4),
                   "mean_target": round(low_vol["mean"].mean(), 1), "mean_zero_frac": round(low_vol["zero_frac"].mean(), 4)},
    "mid_volume": {"count": len(mid_vol), "mean_cv": round(mid_vol["cv"].mean(), 4),
                   "mean_target": round(mid_vol["mean"].mean(), 1), "mean_zero_frac": round(mid_vol["zero_frac"].mean(), 4)},
    "high_volume": {"count": len(high_vol), "mean_cv": round(high_vol["cv"].mean(), 4),
                    "mean_target": round(high_vol["mean"].mean(), 1), "mean_zero_frac": round(high_vol["zero_frac"].mean(), 4)},
}
print(f"Low vol routes: {len(low_vol)}, mean CV: {low_vol['cv'].mean():.4f}")
print(f"Mid vol routes: {len(mid_vol)}, mean CV: {mid_vol['cv'].mean():.4f}")
print(f"High vol routes: {len(high_vol)}, mean CV: {high_vol['cv'].mean():.4f}")

# ---------- 5. Deep autocorrelation ----------
print("\n=== 5. Autocorrelation at multiple lags ===")
sample_routes = train["route_id"].unique()[:100]
lags_steps = {
    "30min_lag1": 1, "1h_lag2": 2, "2h_lag4": 4,
    "6h_lag12": 12, "24h_lag48": 48, "7d_lag336": 336
}
ac_results = {}
for lag_name, lag in lags_steps.items():
    corrs = []
    for rid in sample_routes:
        grp = train[train["route_id"] == rid].sort_values("timestamp")["target_1h"]
        if len(grp) > lag + 10:
            c = grp.corr(grp.shift(lag))
            if not np.isnan(c):
                corrs.append(c)
    ac_results[lag_name] = {
        "mean": round(np.mean(corrs), 4),
        "median": round(np.median(corrs), 4),
        "std": round(np.std(corrs), 4),
        "min": round(np.min(corrs), 4),
        "max": round(np.max(corrs), 4),
    }
    print(f"  {lag_name}: mean={ac_results[lag_name]['mean']:.4f}, std={ac_results[lag_name]['std']:.4f}")
results["autocorrelation"] = ac_results

# ---------- 6. Is target smooth or noisy? ----------
print("\n=== 6. Target smoothness ===")
# Measure via first-difference variance ratio
diff_vars = []
orig_vars = []
for rid in sample_routes:
    grp = train[train["route_id"] == rid].sort_values("timestamp")["target_1h"]
    diff_vars.append(grp.diff().dropna().var())
    orig_vars.append(grp.var())
smoothness_ratio = np.mean(diff_vars) / np.mean(orig_vars)
results["smoothness_ratio_diff_var_over_var"] = round(float(smoothness_ratio), 4)
# For a random walk, this ratio ~ 2. For smooth process < 1. For noisy process > 1.
print(f"Diff variance / variance ratio: {smoothness_ratio:.4f} (random walk=2, smooth<1)")

# ---------- 7. Overlapping window effect ----------
print("\n=== 7. Overlapping 1h window autocorrelation analysis ===")
# Since target_1h is 1h window reported every 30min, consecutive windows overlap by 30min
# This mechanically induces lag-1 autocorrelation
# Compare: lag-1 AC vs lag-2 AC (lag-2 has NO overlap)
lag1_mean = ac_results["30min_lag1"]["mean"]
lag2_mean = ac_results["1h_lag2"]["mean"]
results["overlap_effect"] = {
    "lag1_mean_ac": lag1_mean,
    "lag2_mean_ac": lag2_mean,
    "overlap_induced_ac": round(lag1_mean - lag2_mean, 4),
    "interpretation": "lag-1 AC is mechanically inflated by 30-min overlap in 1h windows"
}
print(f"Lag-1 AC: {lag1_mean:.4f}, Lag-2 AC (no overlap): {lag2_mean:.4f}")
print(f"Overlap-induced autocorrelation: {lag1_mean - lag2_mean:.4f}")

# ---------- Summary ----------
print("\n=== SUMMARY ===")
if smoothness_ratio > 1.5:
    process_type = "noisy operational volume"
elif smoothness_ratio > 0.8:
    process_type = "moderately smooth process"
else:
    process_type = "smooth process"
results["summary"] = {
    "process_type": process_type,
    "lag_heavy_approach_useful": lag1_mean > 0.3 or ac_results["24h_lag48"]["mean"] > 0.2,
    "route_specific_modeling_useful": route_stats["cv"].std() > 0.1,
    "global_trend_significant": abs(results["global_trend_pct_change_over_period"]) > 10,
}
print(f"Process type: {process_type}")
print(f"Lag-heavy approach useful: {results['summary']['lag_heavy_approach_useful']}")
print(f"Route-specific modeling useful: {results['summary']['route_specific_modeling_useful']}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}")
