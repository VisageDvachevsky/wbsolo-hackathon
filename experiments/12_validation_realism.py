"""
Deep analytics #9: Validation realism analysis.
- Is last-8-timestamps holdout the best proxy for test?
- Multiple historical windows with same DOW/hour pattern
- Stability of results across different holdout windows
- Validation philosophy recommendation
"""
import pandas as pd
import numpy as np
import json

DATA_DIR = "/tmp/gh-issue-solver-1774506453169"
OUT = f"{DATA_DIR}/experiments/results_12_validation_realism.json"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
train["hour"] = train["timestamp"].dt.hour
train["dow"] = train["timestamp"].dt.dayofweek
train["half_hour"] = train["hour"] + train["timestamp"].dt.minute / 60

results = {}

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    wape = np.abs(y_pred - y_true).sum() / y_true.sum()
    rbias = abs(y_pred.sum() / y_true.sum() - 1)
    return {"wape": round(float(wape), 6), "rbias": round(float(rbias), 6),
            "total": round(float(wape + rbias), 6)}

# ---------- 1. Test window characteristics ----------
print("=== 1. Test window matching ===")
test = pd.read_parquet(f"{DATA_DIR}/test_solo_track.parquet")
test_dow = test["timestamp"].dt.dayofweek.iloc[0]
test_hours = sorted(test["timestamp"].dt.hour.unique())
test_half_hours = sorted((test["timestamp"].dt.hour + test["timestamp"].dt.minute / 60).unique())
print(f"Test: DOW={test_dow} (Sat), hours={test_hours}")

# Find all historical windows matching test pattern (same DOW, same hours, 8 timestamps)
all_dates = sorted(train["timestamp"].dt.date.unique())
matching_windows = []
for date in all_dates:
    day_data = train[train["timestamp"].dt.date == date]
    if day_data["dow"].iloc[0] != test_dow:
        continue
    # Check if this Saturday has the matching hours
    avail_hh = set(day_data["half_hour"].unique())
    if set(test_half_hours).issubset(avail_hh):
        window_data = day_data[day_data["half_hour"].isin(test_half_hours)]
        matching_windows.append({
            "date": str(date),
            "n_rows": len(window_data),
            "mean_target": round(window_data["target_1h"].mean(), 1),
            "total_target": round(window_data["target_1h"].sum(), 1),
        })

results["matching_saturday_windows"] = matching_windows
print(f"Found {len(matching_windows)} matching Saturday midday windows")
for w in matching_windows:
    print(f"  {w['date']}: mean={w['mean_target']:.0f}, total={w['total_target']:.0f}")

# ---------- 2. Cross-validation on matching windows ----------
print("\n=== 2. Cross-validation on matching Saturday windows ===")
if len(matching_windows) >= 2:
    cv_results = []
    for i, window in enumerate(matching_windows):
        val_date = pd.Timestamp(window["date"]).date()
        val_data = train[
            (train["timestamp"].dt.date == val_date) &
            (train["half_hour"].isin(test_half_hours))
        ]
        tr_data = train[train["timestamp"].dt.date < val_date]

        if len(tr_data) < 100000 or len(val_data) == 0:
            continue

        y_true = val_data["target_1h"].values
        global_mean = tr_data["target_1h"].mean()
        route_mean = tr_data.groupby("route_id")["target_1h"].mean()

        # route_mean prediction
        pred_rm = val_data["route_id"].map(route_mean).fillna(global_mean).values
        m_rm = compute_metrics(y_true, pred_rm)

        # route + same DOW + hour
        rdh = tr_data[tr_data["dow"] == test_dow].groupby(["route_id", "hour"])["target_1h"].mean()
        rd = tr_data[tr_data["dow"] == test_dow].groupby("route_id")["target_1h"].mean()
        pred_rdh = val_data.apply(
            lambda r: rdh.get((r["route_id"], r["hour"]), rd.get(r["route_id"], global_mean)), axis=1).values
        m_rdh = compute_metrics(y_true, pred_rdh)

        cv_results.append({
            "date": window["date"],
            "val_mean_target": round(float(val_data["target_1h"].mean()), 1),
            "route_mean_metric": m_rm,
            "route_dow_hour_metric": m_rdh,
        })
        print(f"  {window['date']}: route_mean={m_rm['total']:.4f}, route_dow_hour={m_rdh['total']:.4f}")

    results["saturday_cv_results"] = cv_results

# ---------- 3. Last-8-timestamps vs Saturday-matched validation ----------
print("\n=== 3. Validation approach comparison ===")
all_ts = sorted(train["timestamp"].unique())

# Approach A: Last 8 timestamps (generic)
val_a_ts = all_ts[-8:]
val_a = train[train["timestamp"].isin(val_a_ts)]
tr_a = train[~train["timestamp"].isin(val_a_ts)]

# Approach B: Last Saturday matching window
last_sat_window = matching_windows[-1] if matching_windows else None
if last_sat_window:
    val_b_date = pd.Timestamp(last_sat_window["date"]).date()
    val_b = train[(train["timestamp"].dt.date == val_b_date) &
                  (train["half_hour"].isin(test_half_hours))]
    tr_b = train[train["timestamp"].dt.date < val_b_date]

    results["validation_comparison"] = {
        "approach_a_last8": {
            "val_timestamps": [str(t) for t in val_a_ts],
            "val_dow": int(val_a["dow"].iloc[0]),
            "val_hours": sorted(val_a["hour"].unique().tolist()),
            "val_mean_target": round(float(val_a["target_1h"].mean()), 1),
            "val_size": int(len(val_a)),
        },
        "approach_b_last_saturday": {
            "val_date": last_sat_window["date"],
            "val_dow": int(test_dow),
            "val_hours": test_hours,
            "val_mean_target": round(float(val_b["target_1h"].mean()), 1),
            "val_size": int(len(val_b)),
        },
    }
    print(f"Approach A (last 8 ts): DOW={val_a['dow'].iloc[0]}, hours={sorted(val_a['hour'].unique())}, "
          f"mean_target={val_a['target_1h'].mean():.0f}")
    print(f"Approach B (last Sat):  DOW={test_dow}, hours={test_hours}, "
          f"mean_target={val_b['target_1h'].mean():.0f}")

# ---------- 4. Stability of metrics across approaches ----------
print("\n=== 4. Metric stability analysis ===")
# Run route_mean baseline on multiple folds
fold_metrics = []
for offset_days in [0, 1, 2, 3, 7, 14]:
    cutoff = all_ts[-1] - pd.Timedelta(days=offset_days)
    # Find 8 consecutive timestamps just before cutoff
    ts_before = [t for t in all_ts if t <= cutoff]
    if len(ts_before) < 16:
        continue
    val_ts = ts_before[-8:]
    tr_ts = ts_before[:-8]

    val_fold = train[train["timestamp"].isin(val_ts)]
    tr_fold = train[train["timestamp"].isin(tr_ts)]

    y_true = val_fold["target_1h"].values
    route_mean = tr_fold.groupby("route_id")["target_1h"].mean()
    pred = val_fold["route_id"].map(route_mean).fillna(tr_fold["target_1h"].mean()).values
    m = compute_metrics(y_true, pred)

    fold_metrics.append({
        "offset_days": offset_days,
        "val_ts_start": str(val_ts[0]),
        "val_dow": int(val_fold["dow"].iloc[0]),
        "metric": m,
    })
    print(f"  Offset {offset_days}d: DOW={val_fold['dow'].iloc[0]}, total={m['total']:.4f}")

results["metric_stability"] = {
    "folds": fold_metrics,
    "total_std": round(np.std([f["metric"]["total"] for f in fold_metrics]), 6),
    "total_range": round(max(f["metric"]["total"] for f in fold_metrics) -
                         min(f["metric"]["total"] for f in fold_metrics), 6),
}
print(f"\nMetric std across folds: {results['metric_stability']['total_std']:.4f}")

# ---------- 5. Recommendation ----------
print("\n=== 5. Validation recommendation ===")
results["recommendation"] = {
    "primary_validation": "Last Saturday matching window (same DOW + hours as test)",
    "secondary_validation": "Multiple historical Saturday windows for robustness",
    "tertiary_validation": "Last 8 timestamps as additional check",
    "rationale": "Test is Saturday 11-14:30. Using matching DOW+hour windows gives the most realistic proxy. "
                 "Multiple Saturday folds increase confidence. Generic last-8 gives temporal recency but wrong DOW/hour context.",
}
print(results["recommendation"]["rationale"])

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}")
