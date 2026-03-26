"""
Deep analytics #2: Route segmentation.
- Scale: low/medium/high volume
- Volatility (CV)
- Zero fraction
- Intraday profile stability
- Weekday pattern stability
- Recent vs historical stability
- Outlier routes that dominate WAPE
"""
import pandas as pd
import numpy as np
import json

DATA_DIR = "/tmp/gh-issue-solver-1774506453169"
OUT = f"{DATA_DIR}/experiments/results_05_route_segmentation.json"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
train["hour"] = train["timestamp"].dt.hour
train["dow"] = train["timestamp"].dt.dayofweek
train["week"] = train["timestamp"].dt.isocalendar().week.astype(int)

results = {}

# ---------- 1. Basic route stats ----------
print("=== 1. Route-level statistics ===")
route_stats = train.groupby("route_id")["target_1h"].agg(
    ["mean", "median", "std", "min", "max", "count"]
)
route_stats["cv"] = route_stats["std"] / route_stats["mean"]
route_stats["zero_frac"] = train.groupby("route_id")["target_1h"].apply(lambda x: (x == 0).mean())
route_stats["iqr"] = train.groupby("route_id")["target_1h"].apply(lambda x: x.quantile(0.75) - x.quantile(0.25))

results["route_stats_summary"] = {
    "mean_of_means": round(route_stats["mean"].mean(), 1),
    "std_of_means": round(route_stats["mean"].std(), 1),
    "min_mean": round(route_stats["mean"].min(), 1),
    "max_mean": round(route_stats["mean"].max(), 1),
    "mean_cv": round(route_stats["cv"].mean(), 4),
    "std_cv": round(route_stats["cv"].std(), 4),
    "mean_zero_frac": round(route_stats["zero_frac"].mean(), 4),
}
print(f"Mean of route means: {route_stats['mean'].mean():.0f}")
print(f"CV range: {route_stats['cv'].min():.3f} - {route_stats['cv'].max():.3f}")

# ---------- 2. Volume-based segmentation ----------
print("\n=== 2. Volume segments ===")
q25, q75 = route_stats["mean"].quantile(0.25), route_stats["mean"].quantile(0.75)
route_stats["segment"] = pd.cut(route_stats["mean"],
    bins=[0, q25, q75, route_stats["mean"].max() + 1],
    labels=["low", "mid", "high"])

for seg in ["low", "mid", "high"]:
    subset = route_stats[route_stats["segment"] == seg]
    results[f"segment_{seg}"] = {
        "count": int(len(subset)),
        "mean_target": round(subset["mean"].mean(), 1),
        "mean_cv": round(subset["cv"].mean(), 4),
        "mean_zero_frac": round(subset["zero_frac"].mean(), 4),
        "target_range": f"{subset['mean'].min():.0f} - {subset['mean'].max():.0f}",
    }
    print(f"  {seg}: n={len(subset)}, mean_target={subset['mean'].mean():.0f}, CV={subset['cv'].mean():.3f}")

# ---------- 3. Intraday profile stability ----------
print("\n=== 3. Intraday profile stability ===")
# For each route, compute hourly profile and check how stable it is across weeks
profile_stabilities = []
sample_routes = train["route_id"].unique()[:200]
for rid in sample_routes:
    grp = train[train["route_id"] == rid]
    # Hourly profile: mean target by hour
    hourly_profile = grp.groupby("hour")["target_1h"].mean()
    # Normalized profile
    if hourly_profile.mean() > 0:
        norm_profile = hourly_profile / hourly_profile.mean()
    else:
        continue
    # Check stability: compute weekly profiles and measure variance
    weekly_profiles = []
    for wk, wk_grp in grp.groupby("week"):
        wp = wk_grp.groupby("hour")["target_1h"].mean()
        if wp.mean() > 0:
            weekly_profiles.append(wp / wp.mean())
    if len(weekly_profiles) >= 4:
        wp_df = pd.DataFrame(weekly_profiles)
        stability = 1 - wp_df.std().mean()  # higher = more stable
        profile_stabilities.append({"route_id": rid, "stability": stability})

ps = pd.DataFrame(profile_stabilities)
results["intraday_profile_stability"] = {
    "mean": round(ps["stability"].mean(), 4),
    "median": round(ps["stability"].median(), 4),
    "std": round(ps["stability"].std(), 4),
    "pct_highly_stable": round((ps["stability"] > 0.8).mean() * 100, 1),
    "pct_unstable": round((ps["stability"] < 0.5).mean() * 100, 1),
}
print(f"Mean intraday stability: {ps['stability'].mean():.3f}")
print(f"Highly stable routes (>0.8): {(ps['stability'] > 0.8).mean()*100:.1f}%")

# ---------- 4. Weekday pattern stability ----------
print("\n=== 4. Weekday pattern stability ===")
dow_stabilities = []
for rid in sample_routes:
    grp = train[train["route_id"] == rid]
    dow_profile = grp.groupby("dow")["target_1h"].mean()
    if dow_profile.mean() > 0:
        norm = dow_profile / dow_profile.mean()
        dow_stabilities.append({"route_id": rid, "dow_cv": norm.std()})

ds = pd.DataFrame(dow_stabilities)
results["weekday_pattern"] = {
    "mean_dow_cv": round(ds["dow_cv"].mean(), 4),
    "pct_strong_dow_pattern": round((ds["dow_cv"] > 0.1).mean() * 100, 1),
}
print(f"Mean DOW CV: {ds['dow_cv'].mean():.4f}")

# ---------- 5. Recent vs historical stability ----------
print("\n=== 5. Recent vs historical stability ===")
last_7d = train[train["timestamp"] > train["timestamp"].max() - pd.Timedelta(days=7)]
last_14d = train[train["timestamp"] > train["timestamp"].max() - pd.Timedelta(days=14)]
last_28d = train[train["timestamp"] > train["timestamp"].max() - pd.Timedelta(days=28)]
all_history = train

recent_means = {}
for label, subset in [("7d", last_7d), ("14d", last_14d), ("28d", last_28d), ("all", all_history)]:
    rm = subset.groupby("route_id")["target_1h"].mean()
    recent_means[label] = rm

# Compare recent to full history
for period in ["7d", "14d", "28d"]:
    merged = pd.DataFrame({
        "recent": recent_means[period],
        "all": recent_means["all"]
    }).dropna()
    corr = merged.corr().iloc[0, 1]
    ratio = (merged["recent"] / merged["all"]).mean()
    drift = ((merged["recent"] - merged["all"]) / merged["all"]).abs().mean()
    results[f"stability_{period}_vs_all"] = {
        "correlation": round(corr, 4),
        "mean_ratio": round(ratio, 4),
        "mean_abs_pct_drift": round(drift * 100, 2),
    }
    print(f"  {period} vs all: corr={corr:.4f}, ratio={ratio:.4f}, drift={drift*100:.2f}%")

# ---------- 6. WAPE contribution by route ----------
print("\n=== 6. WAPE contribution by route ===")
# Under global mean prediction, which routes contribute most to WAPE?
global_mean = train["target_1h"].mean()
route_wape_contrib = train.groupby("route_id").apply(
    lambda g: np.abs(g["target_1h"] - global_mean).sum()
).sort_values(ascending=False)
total_abs_err = route_wape_contrib.sum()
top10_routes = route_wape_contrib.head(10)
results["wape_contribution_top10"] = {
    str(rid): round(float(v / total_abs_err * 100), 2) for rid, v in top10_routes.items()
}
top10_share = top10_routes.sum() / total_abs_err * 100
top50_share = route_wape_contrib.head(50).sum() / total_abs_err * 100
results["wape_concentration"] = {
    "top10_pct": round(top10_share, 2),
    "top50_pct": round(top50_share, 2),
    "top100_pct": round(route_wape_contrib.head(100).sum() / total_abs_err * 100, 2),
}
print(f"Top 10 routes contribute {top10_share:.1f}% of total WAPE error (under global mean)")
print(f"Top 50 routes: {top50_share:.1f}%")

# ---------- 7. Do different segments need different strategies? ----------
print("\n=== 7. Segment-specific optimal strategy check ===")
# For each segment, compare route_mean vs global_mean error
for seg in ["low", "mid", "high"]:
    seg_routes = route_stats[route_stats["segment"] == seg].index
    seg_data = train[train["route_id"].isin(seg_routes)]
    seg_route_means = seg_data.groupby("route_id")["target_1h"].transform("mean")
    err_global = np.abs(seg_data["target_1h"] - global_mean).sum()
    err_route_mean = np.abs(seg_data["target_1h"] - seg_route_means).sum()
    improvement = (err_global - err_route_mean) / err_global * 100
    results[f"route_mean_improvement_{seg}"] = round(improvement, 2)
    print(f"  {seg}: route_mean improves over global_mean by {improvement:.1f}%")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}")
