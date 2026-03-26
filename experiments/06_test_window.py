"""
Deep analytics #3: Test-window centric analysis.
- Test = 8 timestamps on 2025-11-01 (Saturday) 11:00-14:30
- How do Saturdays midday windows historically behave?
- Is Saturday midday different from weekdays?
- Recent weeks vs long-term for Saturday windows
- End-of-train drift detection
- Which historical subset best predicts this window?
"""
import pandas as pd
import numpy as np
import json

DATA_DIR = "/tmp/gh-issue-solver-1774506453169"
OUT = f"{DATA_DIR}/experiments/results_06_test_window.json"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
test = pd.read_parquet(f"{DATA_DIR}/test_solo_track.parquet")

train["hour"] = train["timestamp"].dt.hour
train["minute"] = train["timestamp"].dt.minute
train["dow"] = train["timestamp"].dt.dayofweek
train["date"] = train["timestamp"].dt.date
train["half_hour"] = train["hour"] + train["minute"] / 60
train["week"] = train["timestamp"].dt.isocalendar().week.astype(int)

results = {}

# ---------- 1. Test window details ----------
print("=== 1. Test window details ===")
test_timestamps = sorted(test["timestamp"].unique())
test_dow = pd.Timestamp(test_timestamps[0]).dayofweek
test_hours = [pd.Timestamp(t).hour for t in test_timestamps]
test_half_hours = [pd.Timestamp(t).hour + pd.Timestamp(t).minute / 60 for t in test_timestamps]
dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
results["test_window"] = {
    "timestamps": [str(t) for t in test_timestamps],
    "dow": dow_names[test_dow],
    "dow_num": int(test_dow),
    "hours": sorted(set(test_hours)),
    "half_hours": sorted(set(test_half_hours)),
}
print(f"Test: {dow_names[test_dow]}, hours {sorted(set(test_hours))}")

# ---------- 2. Historical Saturday midday windows ----------
print("\n=== 2. Historical Saturday midday comparison ===")
test_hh_set = set(test_half_hours)
sat_midday = train[(train["dow"] == test_dow) & (train["half_hour"].isin(test_hh_set))]
weekday_midday = train[(train["dow"] < 5) & (train["half_hour"].isin(test_hh_set))]
sun_midday = train[(train["dow"] == 6) & (train["half_hour"].isin(test_hh_set))]

results["saturday_midday_vs_weekday"] = {
    "saturday_mean": round(sat_midday["target_1h"].mean(), 1),
    "weekday_mean": round(weekday_midday["target_1h"].mean(), 1),
    "sunday_mean": round(sun_midday["target_1h"].mean(), 1),
    "sat_vs_weekday_ratio": round(sat_midday["target_1h"].mean() / weekday_midday["target_1h"].mean(), 4),
    "sat_vs_sun_ratio": round(sat_midday["target_1h"].mean() / sun_midday["target_1h"].mean(), 4) if sun_midday["target_1h"].mean() > 0 else None,
    "saturday_median": round(sat_midday["target_1h"].median(), 1),
    "weekday_median": round(weekday_midday["target_1h"].median(), 1),
    "saturday_std": round(sat_midday["target_1h"].std(), 1),
    "weekday_std": round(weekday_midday["target_1h"].std(), 1),
    "saturday_count": int(len(sat_midday)),
    "weekday_count": int(len(weekday_midday)),
}
print(f"Saturday midday mean: {sat_midday['target_1h'].mean():.0f}")
print(f"Weekday midday mean: {weekday_midday['target_1h'].mean():.0f}")
print(f"Ratio: {sat_midday['target_1h'].mean() / weekday_midday['target_1h'].mean():.3f}")

# ---------- 3. Recent Saturdays vs all Saturdays ----------
print("\n=== 3. Recent vs all Saturdays ===")
for n_weeks in [1, 2, 4, 8]:
    cutoff = train["timestamp"].max() - pd.Timedelta(weeks=n_weeks)
    recent_sat = sat_midday[sat_midday["timestamp"] > cutoff]
    if len(recent_sat) > 0:
        results[f"recent_{n_weeks}w_sat_midday"] = {
            "mean": round(recent_sat["target_1h"].mean(), 1),
            "count": int(len(recent_sat)),
            "ratio_to_all_sat": round(recent_sat["target_1h"].mean() / sat_midday["target_1h"].mean(), 4),
        }
        print(f"  Last {n_weeks}w Sat midday: mean={recent_sat['target_1h'].mean():.0f}, "
              f"ratio to all: {recent_sat['target_1h'].mean() / sat_midday['target_1h'].mean():.3f}")

# ---------- 4. Per-route Saturday vs weekday ----------
print("\n=== 4. Per-route Saturday vs weekday ratio ===")
route_sat = sat_midday.groupby("route_id")["target_1h"].mean()
route_wd = weekday_midday.groupby("route_id")["target_1h"].mean()
ratio_df = (route_sat / route_wd).dropna()
results["route_sat_weekday_ratio"] = {
    "mean": round(ratio_df.mean(), 4),
    "median": round(ratio_df.median(), 4),
    "std": round(ratio_df.std(), 4),
    "min": round(ratio_df.min(), 4),
    "max": round(ratio_df.max(), 4),
    "pct_sat_higher": round((ratio_df > 1).mean() * 100, 1),
}
print(f"Mean route Sat/weekday ratio: {ratio_df.mean():.3f}, std: {ratio_df.std():.3f}")

# ---------- 5. End-of-train drift detection ----------
print("\n=== 5. End-of-train drift ===")
# Compare last 3 days vs last 14 days
last_3d = train[train["timestamp"] > train["timestamp"].max() - pd.Timedelta(days=3)]
last_14d = train[train["timestamp"] > train["timestamp"].max() - pd.Timedelta(days=14)]
results["end_of_train_drift"] = {
    "last_3d_mean": round(last_3d["target_1h"].mean(), 1),
    "last_14d_mean": round(last_14d["target_1h"].mean(), 1),
    "overall_mean": round(train["target_1h"].mean(), 1),
    "last_3d_vs_14d_ratio": round(last_3d["target_1h"].mean() / last_14d["target_1h"].mean(), 4),
    "last_3d_vs_overall_ratio": round(last_3d["target_1h"].mean() / train["target_1h"].mean(), 4),
}
print(f"Last 3d mean: {last_3d['target_1h'].mean():.0f}")
print(f"Last 14d mean: {last_14d['target_1h'].mean():.0f}")
print(f"Overall mean: {train['target_1h'].mean():.0f}")

# ---------- 6. Which historical context is most relevant? ----------
print("\n=== 6. Historical context relevance (pseudo-validation) ===")
# Use second-to-last Saturday midday window as validation, predict from various strategies
# Get all Saturday dates
sat_dates = sorted(train[train["dow"] == test_dow]["date"].unique())
if len(sat_dates) >= 2:
    val_sat_date = sat_dates[-1]  # last Saturday in train
    pred_sat_dates = sat_dates[:-1]

    val_data = train[(train["date"] == val_sat_date) & (train["half_hour"].isin(test_hh_set))]
    train_excl = train[train["date"] != val_sat_date]

    if len(val_data) > 0:
        y_true = val_data.groupby("route_id")["target_1h"].mean()

        strategies = {}
        # 1. Global mean
        pred_global = train_excl["target_1h"].mean()
        # 2. Route mean (all history)
        pred_route_all = train_excl.groupby("route_id")["target_1h"].mean()
        # 3. Route mean (last 14d)
        cutoff_14d = pd.Timestamp(val_sat_date) - pd.Timedelta(days=14)
        pred_route_14d = train_excl[train_excl["timestamp"] > cutoff_14d].groupby("route_id")["target_1h"].mean()
        # 4. Route + same DOW mean
        pred_route_dow = train_excl[train_excl["dow"] == test_dow].groupby("route_id")["target_1h"].mean()
        # 5. Route + same DOW + same hours
        pred_route_dow_hr = train_excl[
            (train_excl["dow"] == test_dow) & (train_excl["half_hour"].isin(test_hh_set))
        ].groupby("route_id")["target_1h"].mean()
        # 6. Route + same DOW (last 4 weeks only)
        cutoff_4w = pd.Timestamp(val_sat_date) - pd.Timedelta(weeks=4)
        pred_route_dow_recent = train_excl[
            (train_excl["dow"] == test_dow) & (train_excl["timestamp"] > cutoff_4w)
        ].groupby("route_id")["target_1h"].mean()

        # Evaluate each
        def wape_rbias(y_true_s, y_pred_s):
            common = y_true_s.index.intersection(y_pred_s.index)
            yt = y_true_s[common].values
            yp = y_pred_s[common].values if hasattr(y_pred_s, 'index') else np.full(len(common), y_pred_s)
            if isinstance(y_pred_s, (int, float, np.floating)):
                yp = np.full(len(yt), y_pred_s)
            wape = np.abs(yp - yt).sum() / yt.sum()
            rbias = abs(yp.sum() / yt.sum() - 1)
            return round(wape + rbias, 4), round(wape, 4), round(rbias, 4)

        for name, pred in [
            ("global_mean", pred_global),
            ("route_mean_all", pred_route_all),
            ("route_mean_14d", pred_route_14d),
            ("route_same_dow", pred_route_dow),
            ("route_same_dow_hr", pred_route_dow_hr),
            ("route_same_dow_recent4w", pred_route_dow_recent),
        ]:
            total, wape, rbias = wape_rbias(y_true, pred)
            strategies[name] = {"total": total, "wape": wape, "rbias": rbias}
            print(f"  {name}: WAPE+RBias={total:.4f} (WAPE={wape:.4f}, RBias={rbias:.4f})")

        results["strategy_comparison_on_last_saturday"] = strategies

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}")
