"""
Deep analytics #6: Outlier and extreme-volume analysis.
- Routes with largest target contribution
- Extreme volume dates/windows
- Regular patterns vs rare anomalies
- Predictability of large volumes
- Leaderboard sensitivity to top-volume routes
"""
import pandas as pd
import numpy as np
import json

DATA_DIR = "/tmp/gh-issue-solver-1774506453169"
OUT = f"{DATA_DIR}/experiments/results_09_outlier_analysis.json"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
train["hour"] = train["timestamp"].dt.hour
train["dow"] = train["timestamp"].dt.dayofweek
train["date"] = train["timestamp"].dt.date

results = {}

# ---------- 1. Route contribution to total volume ----------
print("=== 1. Route contribution to total volume ===")
route_total = train.groupby("route_id")["target_1h"].sum().sort_values(ascending=False)
total_volume = route_total.sum()
route_share = route_total / total_volume * 100

results["volume_concentration"] = {
    "top1_pct": round(float(route_share.iloc[0]), 2),
    "top5_pct": round(float(route_share.head(5).sum()), 2),
    "top10_pct": round(float(route_share.head(10).sum()), 2),
    "top20_pct": round(float(route_share.head(20).sum()), 2),
    "top50_pct": round(float(route_share.head(50).sum()), 2),
    "top100_pct": round(float(route_share.head(100).sum()), 2),
    "bottom_500_pct": round(float(route_share.tail(500).sum()), 2),
}
print(f"Top 1 route: {route_share.iloc[0]:.2f}% of volume")
print(f"Top 10 routes: {route_share.head(10).sum():.2f}%")
print(f"Top 50 routes: {route_share.head(50).sum():.2f}%")

# Top 10 routes details
top10_routes = route_total.head(10).index.tolist()
results["top10_routes"] = {}
for rid in top10_routes:
    rdata = train[train["route_id"] == rid]["target_1h"]
    results["top10_routes"][str(rid)] = {
        "mean": round(float(rdata.mean()), 1),
        "median": round(float(rdata.median()), 1),
        "max": round(float(rdata.max()), 1),
        "volume_share_pct": round(float(route_share[rid]), 2),
    }
    print(f"  Route {rid}: mean={rdata.mean():.0f}, share={route_share[rid]:.2f}%")

# ---------- 2. Extreme volume events ----------
print("\n=== 2. Extreme volume events ===")
p99 = train["target_1h"].quantile(0.99)
p999 = train["target_1h"].quantile(0.999)
p95 = train["target_1h"].quantile(0.95)
extreme = train[train["target_1h"] > p99]

results["extreme_events"] = {
    "p95": round(float(p95), 1),
    "p99": round(float(p99), 1),
    "p999": round(float(p999), 1),
    "above_p99_count": int(len(extreme)),
    "above_p99_pct": round(len(extreme) / len(train) * 100, 3),
    "above_p99_volume_share": round(float(extreme["target_1h"].sum() / total_volume * 100), 2),
}
print(f"p95={p95:.0f}, p99={p99:.0f}, p99.9={p999:.0f}")
print(f"Events above p99: {len(extreme)} ({len(extreme)/len(train)*100:.2f}%), "
      f"volume share: {extreme['target_1h'].sum()/total_volume*100:.2f}%")

# Routes with extreme events
extreme_routes = extreme.groupby("route_id").size().sort_values(ascending=False)
results["extreme_events_by_route"] = {
    "top5_routes": {str(k): int(v) for k, v in extreme_routes.head(5).items()},
    "n_routes_with_extremes": int(len(extreme_routes)),
}
print(f"Routes with extreme events: {len(extreme_routes)}")
print(f"Top 5 by count: {dict(extreme_routes.head(5))}")

# ---------- 3. Temporal patterns in extremes ----------
print("\n=== 3. Temporal patterns in extremes ===")
extreme_hourly = extreme.groupby("hour").size()
extreme_dow = extreme.groupby("dow").size()
dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

# Normalize by overall frequency to get relative risk
overall_hourly = train.groupby("hour").size()
overall_dow = train.groupby("dow").size()

hourly_risk = (extreme_hourly / overall_hourly).fillna(0)
dow_risk = (extreme_dow / overall_dow).fillna(0)

results["extreme_temporal_patterns"] = {
    "peak_hour": int(hourly_risk.idxmax()),
    "peak_hour_relative_risk": round(float(hourly_risk.max() / hourly_risk.mean()), 2),
    "peak_dow": dow_names[int(dow_risk.idxmax())],
    "peak_dow_relative_risk": round(float(dow_risk.max() / dow_risk.mean()), 2),
    "hourly_risk_cv": round(float(hourly_risk.std() / hourly_risk.mean()), 4),
    "dow_risk_cv": round(float(dow_risk.std() / dow_risk.mean()), 4),
}
print(f"Peak extreme hour: {hourly_risk.idxmax()}, relative risk: {hourly_risk.max()/hourly_risk.mean():.2f}x")
print(f"Peak extreme DOW: {dow_names[dow_risk.idxmax()]}, relative risk: {dow_risk.max()/dow_risk.mean():.2f}x")

# ---------- 4. Are large volumes predictable? ----------
print("\n=== 4. Predictability of large volumes ===")
# For top-volume routes, check if high-volume events are periodic or random
top_routes_data = train[train["route_id"].isin(top10_routes)]
for rid in top10_routes[:5]:
    rdata = train[train["route_id"] == rid].sort_values("timestamp")["target_1h"]
    # Check if spikes follow temporal patterns
    is_high = (rdata > rdata.quantile(0.9)).astype(int)
    # Autocorrelation of "is_high" binary indicator
    ac1 = is_high.corr(is_high.shift(1))
    ac48 = is_high.corr(is_high.shift(48))  # same time next day
    ac336 = is_high.corr(is_high.shift(336))  # same time next week
    results[f"high_volume_predictability_route_{rid}"] = {
        "ac_30min": round(float(ac1), 4) if not np.isnan(ac1) else None,
        "ac_24h": round(float(ac48), 4) if not np.isnan(ac48) else None,
        "ac_7d": round(float(ac336), 4) if not np.isnan(ac336) else None,
    }
    print(f"  Route {rid}: spike AC(30m)={ac1:.3f}, AC(24h)={ac48:.3f}, AC(7d)={ac336:.3f}")

# ---------- 5. WAPE sensitivity to top routes ----------
print("\n=== 5. WAPE sensitivity to top routes ===")
# If we get top routes right, how much does WAPE improve?
route_mean = train.groupby("route_id")["target_1h"].mean()
# Overall WAPE with route_mean prediction
overall_err = np.abs(train["target_1h"] - train["route_id"].map(route_mean)).sum()
overall_volume = train["target_1h"].sum()
overall_wape = overall_err / overall_volume

# WAPE contribution by route
for n_top in [10, 20, 50]:
    top_rids = route_total.head(n_top).index
    top_data = train[train["route_id"].isin(top_rids)]
    top_err = np.abs(top_data["target_1h"] - top_data["route_id"].map(route_mean)).sum()
    top_volume = top_data["target_1h"].sum()
    contribution = top_err / overall_volume  # their contribution to overall WAPE
    results[f"wape_contribution_top{n_top}"] = {
        "error_share_pct": round(float(top_err / overall_err * 100), 2),
        "volume_share_pct": round(float(top_volume / overall_volume * 100), 2),
        "wape_contribution": round(float(contribution), 4),
    }
    print(f"Top {n_top}: error share={top_err/overall_err*100:.1f}%, "
          f"volume share={top_volume/overall_volume*100:.1f}%")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}")
