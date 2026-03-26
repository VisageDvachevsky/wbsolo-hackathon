"""
Deep analytics #4: Deep temporal seasonality.
- Decomposition of temporal patterns: global and route-level
- Interactions: route x hour, route x dow, hour x dow
- Weekday vs weekend patterns
- Intraday peak/trough stability
- Normalized shape + scale approach viability
"""
import pandas as pd
import numpy as np
import json

DATA_DIR = "/tmp/gh-issue-solver-1774506453169"
OUT = f"{DATA_DIR}/experiments/results_07_temporal_seasonality.json"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
train["hour"] = train["timestamp"].dt.hour
train["minute"] = train["timestamp"].dt.minute
train["dow"] = train["timestamp"].dt.dayofweek
train["half_hour"] = train["hour"] + train["minute"] / 60
train["is_weekend"] = (train["dow"] >= 5).astype(int)

results = {}

# ---------- 1. Global hour x dow interaction ----------
print("=== 1. Hour x DOW interaction ===")
dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
hour_dow = train.groupby(["dow", "hour"])["target_1h"].mean().unstack(fill_value=0)
# Normalize each dow to see shape
hour_dow_norm = hour_dow.div(hour_dow.mean(axis=1), axis=0)
results["hour_dow_mean"] = {dow_names[d]: {str(h): round(v, 1) for h, v in row.items()}
                            for d, row in hour_dow.iterrows()}

# How different is the shape across DOWs?
shape_variation = hour_dow_norm.std(axis=0).mean()
results["hour_dow_shape_variation"] = round(float(shape_variation), 4)
print(f"Cross-DOW hourly shape std: {shape_variation:.4f}")

# ---------- 2. Weekday vs weekend ----------
print("\n=== 2. Weekday vs weekend ===")
wd = train[train["is_weekend"] == 0]
we = train[train["is_weekend"] == 1]
wd_hourly = wd.groupby("hour")["target_1h"].mean()
we_hourly = we.groupby("hour")["target_1h"].mean()

# Normalized profiles
wd_norm = wd_hourly / wd_hourly.mean()
we_norm = we_hourly / we_hourly.mean()
profile_diff = (wd_norm - we_norm).abs().mean()

results["weekday_vs_weekend"] = {
    "weekday_mean": round(wd["target_1h"].mean(), 1),
    "weekend_mean": round(we["target_1h"].mean(), 1),
    "ratio_we_wd": round(we["target_1h"].mean() / wd["target_1h"].mean(), 4),
    "profile_shape_diff": round(float(profile_diff), 4),
    "weekday_peak_hour": int(wd_hourly.idxmax()),
    "weekend_peak_hour": int(we_hourly.idxmax()),
    "weekday_trough_hour": int(wd_hourly.idxmin()),
    "weekend_trough_hour": int(we_hourly.idxmin()),
}
print(f"Weekday mean: {wd['target_1h'].mean():.0f}, Weekend mean: {we['target_1h'].mean():.0f}")
print(f"Ratio: {we['target_1h'].mean() / wd['target_1h'].mean():.3f}")
print(f"Profile shape difference: {profile_diff:.4f}")

# ---------- 3. Route x hour interaction ----------
print("\n=== 3. Route x hour interaction ===")
# Check if routes have different hourly shapes or if it's mostly global
sample_routes = train["route_id"].unique()[:100]
route_hourly_profiles = []
for rid in sample_routes:
    grp = train[train["route_id"] == rid]
    hp = grp.groupby("hour")["target_1h"].mean()
    if hp.mean() > 0:
        route_hourly_profiles.append(hp / hp.mean())

rhp = pd.DataFrame(route_hourly_profiles)
# Cross-route variation in hourly profiles
route_hour_std = rhp.std(axis=0).mean()
results["route_hour_interaction"] = {
    "cross_route_hourly_shape_std": round(float(route_hour_std), 4),
    "interpretation": "low = routes share similar hourly shapes, high = need route-specific hourly profiles"
}
print(f"Cross-route hourly shape std: {route_hour_std:.4f}")

# ---------- 4. Route x DOW interaction ----------
print("\n=== 4. Route x DOW interaction ===")
route_dow_profiles = []
for rid in sample_routes:
    grp = train[train["route_id"] == rid]
    dp = grp.groupby("dow")["target_1h"].mean()
    if dp.mean() > 0:
        route_dow_profiles.append(dp / dp.mean())

rdp = pd.DataFrame(route_dow_profiles)
route_dow_std = rdp.std(axis=0).mean()
results["route_dow_interaction"] = {
    "cross_route_dow_shape_std": round(float(route_dow_std), 4),
}
print(f"Cross-route DOW shape std: {route_dow_std:.4f}")

# ---------- 5. Intraday peak/trough stability across weeks ----------
print("\n=== 5. Intraday peak/trough stability ===")
train["week"] = train["timestamp"].dt.isocalendar().week.astype(int)
weekly_peaks = train.groupby(["week"]).apply(
    lambda g: g.groupby("hour")["target_1h"].mean().idxmax()
)
weekly_troughs = train.groupby(["week"]).apply(
    lambda g: g.groupby("hour")["target_1h"].mean().idxmin()
)
results["peak_trough_stability"] = {
    "peak_hour_mode": int(weekly_peaks.mode().iloc[0]) if len(weekly_peaks.mode()) > 0 else None,
    "peak_hour_std": round(float(weekly_peaks.std()), 2),
    "trough_hour_mode": int(weekly_troughs.mode().iloc[0]) if len(weekly_troughs.mode()) > 0 else None,
    "trough_hour_std": round(float(weekly_troughs.std()), 2),
    "peak_consistent": float(weekly_peaks.std()) < 2,
    "trough_consistent": float(weekly_troughs.std()) < 2,
}
print(f"Peak hour mode: {weekly_peaks.mode().iloc[0]}, std: {weekly_peaks.std():.2f}")
print(f"Trough hour mode: {weekly_troughs.mode().iloc[0]}, std: {weekly_troughs.std():.2f}")

# ---------- 6. Normalized shape + scale viability ----------
print("\n=== 6. Shape + scale approach ===")
# Can we describe target as: target ≈ route_scale * global_hourly_shape * dow_effect?
# Test by computing predicted values and measuring explained variance
global_mean = train["target_1h"].mean()
route_scale = train.groupby("route_id")["target_1h"].mean() / global_mean
hour_effect = train.groupby("hour")["target_1h"].mean() / global_mean
dow_effect = train.groupby("dow")["target_1h"].mean() / global_mean

# Predict for a sample
sample = train.sample(min(200000, len(train)), random_state=42)
pred = (global_mean
        * sample["route_id"].map(route_scale)
        * sample["hour"].map(hour_effect)
        * sample["dow"].map(dow_effect))
actual = sample["target_1h"]

ss_res = ((actual - pred) ** 2).sum()
ss_tot = ((actual - actual.mean()) ** 2).sum()
r_squared = 1 - ss_res / ss_tot

wape = np.abs(actual.values - pred.values).sum() / actual.values.sum()
rbias = abs(pred.values.sum() / actual.values.sum() - 1)

results["shape_scale_model"] = {
    "r_squared": round(float(r_squared), 4),
    "wape": round(float(wape), 4),
    "rbias": round(float(rbias), 4),
    "total_metric": round(float(wape + rbias), 4),
}
print(f"Shape+scale model: R²={r_squared:.4f}, WAPE={wape:.4f}, RBias={rbias:.4f}")

# ---------- 7. Same with route-specific hourly shape ----------
print("\n=== 7. Route-specific hourly shape ===")
route_hour_mean = train.groupby(["route_id", "hour"])["target_1h"].mean()
pred_rh = sample.apply(lambda r: route_hour_mean.get((r["route_id"], r["hour"]), global_mean), axis=1)
# Apply DOW adjustment
dow_multiplier = train.groupby("dow")["target_1h"].mean() / train["target_1h"].mean()
pred_rh_dow = pred_rh * sample["dow"].map(dow_multiplier)

wape_rh = np.abs(actual.values - pred_rh_dow.values).sum() / actual.values.sum()
rbias_rh = abs(pred_rh_dow.values.sum() / actual.values.sum() - 1)
ss_res_rh = ((actual - pred_rh_dow) ** 2).sum()
r_sq_rh = 1 - ss_res_rh / ss_tot

results["route_hour_dow_model"] = {
    "r_squared": round(float(r_sq_rh), 4),
    "wape": round(float(wape_rh), 4),
    "rbias": round(float(rbias_rh), 4),
    "total_metric": round(float(wape_rh + rbias_rh), 4),
}
print(f"Route-hour-DOW model: R²={r_sq_rh:.4f}, WAPE={wape_rh:.4f}, RBias={rbias_rh:.4f}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}")
