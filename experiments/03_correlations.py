"""
Feature-target relationships: correlations, lags, route-level differences.
"""
import pandas as pd
import numpy as np

DATA_DIR = "/tmp/gh-issue-solver-1774505330257"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")

print("=" * 80)
print("BASIC CORRELATIONS WITH TARGET")
print("=" * 80)
status_cols = [c for c in train.columns if c.startswith("status_")]
corr = train[status_cols + ["target_1h"]].corr()["target_1h"].drop("target_1h")
print(corr)

print("\n" + "=" * 80)
print("PER-ROUTE CORRELATION: status_4 vs target_1h")
print("=" * 80)
route_corrs = {}
for rid, grp in train.groupby("route_id"):
    if grp["target_1h"].std() > 0 and grp["status_4"].std() > 0:
        route_corrs[rid] = grp[["status_4", "target_1h"]].corr().iloc[0, 1]
rc = pd.Series(route_corrs)
print(f"Mean per-route correlation: {rc.mean():.4f}")
print(f"Median: {rc.median():.4f}")
print(f"Min: {rc.min():.4f}")
print(f"Max: {rc.max():.4f}")
print(f"Std: {rc.std():.4f}")

print("\n" + "=" * 80)
print("LAG CORRELATIONS WITH TARGET")
print("=" * 80)
# For a sample of routes, compute lag correlations
sample_routes = train["route_id"].unique()[:50]
lag_corrs = {col: {} for col in status_cols + ["target_1h"]}

for lag in [1, 2, 3, 4, 6, 12, 24, 48]:
    for col in status_cols + ["target_1h"]:
        corrs = []
        for rid in sample_routes:
            grp = train[train["route_id"] == rid].sort_values("timestamp")
            lagged = grp[col].shift(lag)
            valid = pd.DataFrame({"target": grp["target_1h"], "lagged": lagged}).dropna()
            if valid["target"].std() > 0 and valid["lagged"].std() > 0:
                corrs.append(valid.corr().iloc[0, 1])
        lag_corrs[col][lag] = np.mean(corrs) if corrs else np.nan

print("\nLag correlations (mean over 50 routes, lag in 30-min steps):")
print(f"{'Lag (steps)':<12}", end="")
for col in status_cols + ["target_1h"]:
    print(f"{col:<12}", end="")
print()
for lag in [1, 2, 3, 4, 6, 12, 24, 48]:
    print(f"{lag:<12}", end="")
    for col in status_cols + ["target_1h"]:
        val = lag_corrs[col][lag]
        print(f"{val:<12.4f}" if not np.isnan(val) else f"{'NaN':<12}", end="")
    print()

print("\n" + "=" * 80)
print("ROUTE SCALE DIFFERENCES")
print("=" * 80)
route_stats = train.groupby("route_id")["target_1h"].agg(["mean", "median", "std", "min", "max"])
print(f"\nRoute-level target mean distribution:")
print(route_stats["mean"].describe())
print(f"\nTop 10 routes by mean target:")
print(route_stats.nlargest(10, "mean"))
print(f"\nBottom 10 routes by mean target:")
print(route_stats.nsmallest(10, "mean"))
print(f"\nRatio max_mean/min_mean: {route_stats['mean'].max() / route_stats['mean'].min():.1f}")

print("\n" + "=" * 80)
print("STATUS COLUMNS INTER-CORRELATION")
print("=" * 80)
print(train[status_cols].corr())

print("\n" + "=" * 80)
print("TARGET AUTOCORRELATION (SAME HALF-HOUR NEXT DAY)")
print("=" * 80)
# lag=48 means same time next day (48 * 30min = 24h)
auto_corrs = []
for rid in sample_routes:
    grp = train[train["route_id"] == rid].sort_values("timestamp")
    for lag_steps in [48]:  # 24h
        lagged = grp["target_1h"].shift(lag_steps)
        valid = pd.DataFrame({"t": grp["target_1h"], "l": lagged}).dropna()
        if valid["t"].std() > 0 and valid["l"].std() > 0:
            auto_corrs.append(valid.corr().iloc[0, 1])
print(f"Target autocorrelation at lag 24h (mean over 50 routes): {np.mean(auto_corrs):.4f}")

# Weekly autocorrelation (lag=336 = 7*48)
weekly_auto = []
for rid in sample_routes:
    grp = train[train["route_id"] == rid].sort_values("timestamp")
    lagged = grp["target_1h"].shift(336)
    valid = pd.DataFrame({"t": grp["target_1h"], "l": lagged}).dropna()
    if valid["t"].std() > 0 and valid["l"].std() > 0:
        weekly_auto.append(valid.corr().iloc[0, 1])
print(f"Target autocorrelation at lag 7d (mean over 50 routes): {np.mean(weekly_auto):.4f}")

print("\nDone with correlation analysis!")
