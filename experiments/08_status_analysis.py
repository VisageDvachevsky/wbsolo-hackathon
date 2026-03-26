"""
Deep analytics #5: Status columns as explanatory (not predictive) features.
- Which statuses explain current target vs lead future target?
- Lagged relationships between statuses and future shipments
- Historical status summaries as route descriptors
- Global vs within-route correlations
- Scale-driven vs genuine correlations
"""
import pandas as pd
import numpy as np
import json

DATA_DIR = "/tmp/gh-issue-solver-1774506453169"
OUT = f"{DATA_DIR}/experiments/results_08_status_analysis.json"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
status_cols = [c for c in train.columns if c.startswith("status_")]

results = {}

# ---------- 1. Global vs within-route correlations ----------
print("=== 1. Global vs within-route correlations ===")
global_corr = train[status_cols + ["target_1h"]].corr()["target_1h"].drop("target_1h")
results["global_correlations"] = {col: round(float(v), 4) for col, v in global_corr.items()}
print("Global correlations:")
for col, v in global_corr.items():
    print(f"  {col}: {v:.4f}")

# Within-route correlations (remove scale effect)
print("\nWithin-route correlations (mean over all routes):")
within_route_corrs = {col: [] for col in status_cols}
for rid, grp in train.groupby("route_id"):
    for col in status_cols:
        if grp[col].std() > 0 and grp["target_1h"].std() > 0:
            within_route_corrs[col].append(grp[[col, "target_1h"]].corr().iloc[0, 1])

results["within_route_correlations"] = {}
for col in status_cols:
    if within_route_corrs[col]:
        mean_c = np.mean(within_route_corrs[col])
        median_c = np.median(within_route_corrs[col])
        results["within_route_correlations"][col] = {
            "mean": round(mean_c, 4),
            "median": round(median_c, 4),
            "std": round(np.std(within_route_corrs[col]), 4),
        }
        print(f"  {col}: mean={mean_c:.4f}, median={median_c:.4f}")

# ---------- 2. Scale-driven vs genuine ----------
print("\n=== 2. Scale-driven analysis ===")
# If global corr is high but within-route is low, it's scale-driven
results["scale_vs_genuine"] = {}
for col in status_cols:
    g = float(global_corr[col])
    w = results["within_route_correlations"].get(col, {}).get("mean", 0)
    scale_effect = round(g - w, 4)
    results["scale_vs_genuine"][col] = {
        "global": round(g, 4),
        "within_route": round(w, 4),
        "scale_driven_component": scale_effect,
        "interpretation": "mostly scale-driven" if scale_effect > 0.2 else "partly genuine" if scale_effect > 0.05 else "genuine"
    }
    print(f"  {col}: global={g:.4f}, within={w:.4f}, scale_effect={scale_effect:.4f}")

# ---------- 3. Lagged status-target relationships ----------
print("\n=== 3. Lagged status → future target ===")
sample_routes = train["route_id"].unique()[:100]
lag_results = {}
for col in status_cols:
    lag_results[col] = {}
    for lag in [1, 2, 4, 8, 16]:  # 30min to 8h ahead
        corrs = []
        for rid in sample_routes:
            grp = train[train["route_id"] == rid].sort_values("timestamp")
            # status now → target in the future (shift target back = status leads)
            future_target = grp["target_1h"].shift(-lag)
            valid = pd.DataFrame({"s": grp[col], "t": future_target}).dropna()
            if valid["s"].std() > 0 and valid["t"].std() > 0:
                corrs.append(valid.corr().iloc[0, 1])
        if corrs:
            lag_results[col][f"lead_{lag}_steps"] = round(np.mean(corrs), 4)

results["lagged_status_target"] = lag_results
print("Status leading target (mean within-route correlation):")
for col in status_cols:
    vals = lag_results[col]
    print(f"  {col}: " + ", ".join(f"{k}={v:.4f}" for k, v in vals.items()))

# ---------- 4. Historical status as route descriptors ----------
print("\n=== 4. Status as route descriptors ===")
route_status_means = train.groupby("route_id")[status_cols].mean()
route_target_mean = train.groupby("route_id")["target_1h"].mean()

# Cross-sectional correlation: does route's mean status predict route's mean target?
cross_corr = {}
for col in status_cols:
    c = route_status_means[col].corr(route_target_mean)
    cross_corr[col] = round(float(c), 4)
results["route_descriptor_correlations"] = cross_corr
print("Route-level: mean status → mean target correlation:")
for col, v in cross_corr.items():
    print(f"  {col}: {v:.4f}")

# ---------- 5. Status volatility as route descriptor ----------
print("\n=== 5. Status volatility as descriptor ===")
route_status_cv = train.groupby("route_id")[status_cols].std() / train.groupby("route_id")[status_cols].mean()
route_target_cv = train.groupby("route_id")["target_1h"].std() / train.groupby("route_id")["target_1h"].mean()
cv_corr = {}
for col in status_cols:
    c = route_status_cv[col].corr(route_target_cv)
    cv_corr[col] = round(float(c), 4)
results["route_cv_correlations"] = cv_corr
print("Route-level: status CV → target CV correlation:")
for col, v in cv_corr.items():
    print(f"  {col}: {v:.4f}")

# ---------- 6. Which statuses are truly leading indicators? ----------
print("\n=== 6. Leading indicator analysis ===")
# Compare: corr(status_t, target_t) vs corr(status_t, target_{t+1})
# If the lead corr is higher, status leads target
for col in status_cols:
    concurrent_corrs = []
    lead_corrs = []
    for rid in sample_routes[:50]:
        grp = train[train["route_id"] == rid].sort_values("timestamp")
        # Concurrent
        c = grp[[col, "target_1h"]].corr().iloc[0, 1]
        concurrent_corrs.append(c)
        # Status now → target next step
        future_t = grp["target_1h"].shift(-1)
        valid = pd.DataFrame({"s": grp[col], "t": future_t}).dropna()
        if valid["s"].std() > 0 and valid["t"].std() > 0:
            lead_corrs.append(valid.corr().iloc[0, 1])

    results[f"leading_indicator_{col}"] = {
        "concurrent_within_route": round(np.mean(concurrent_corrs), 4),
        "lead_1step_within_route": round(np.mean(lead_corrs), 4) if lead_corrs else None,
        "is_leading": round(np.mean(lead_corrs), 4) > round(np.mean(concurrent_corrs), 4) if lead_corrs else False,
    }
    print(f"  {col}: concurrent={np.mean(concurrent_corrs):.4f}, "
          f"lead-1={np.mean(lead_corrs):.4f}" if lead_corrs else "")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}")
