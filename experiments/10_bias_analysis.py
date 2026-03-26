"""
Deep analytics #7: Bias-sensitive analysis under WAPE + |Relative Bias| metric.
- Bias profiles of different baselines
- Systematic over/under-prediction patterns
- Aggregate volume estimation from recent history
- Trade-off between local accuracy and aggregate calibration
"""
import pandas as pd
import numpy as np
import json

DATA_DIR = "/tmp/gh-issue-solver-1774506453169"
OUT = f"{DATA_DIR}/experiments/results_10_bias_analysis.json"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
train["hour"] = train["timestamp"].dt.hour
train["dow"] = train["timestamp"].dt.dayofweek
train["week"] = train["timestamp"].dt.isocalendar().week.astype(int)

results = {}

def compute_metrics(y_true, y_pred):
    """WAPE + |Relative Bias| and components."""
    wape = np.abs(y_pred - y_true).sum() / y_true.sum()
    rbias_signed = y_pred.sum() / y_true.sum() - 1
    rbias = abs(rbias_signed)
    return {
        "wape": round(float(wape), 6),
        "rbias": round(float(rbias), 6),
        "rbias_signed": round(float(rbias_signed), 6),
        "total": round(float(wape + rbias), 6),
    }

# ---------- Use time-holdout validation ----------
# Hold out last 8 timestamps per route
print("=== Setting up validation ===")
last_ts = sorted(train["timestamp"].unique())[-8:]
val = train[train["timestamp"].isin(last_ts)]
tr = train[~train["timestamp"].isin(last_ts)]
print(f"Train: {len(tr)}, Val: {len(val)}")
print(f"Val timestamps: {list(last_ts)}")

y_true = val["target_1h"].values

# ---------- 1. Global mean ----------
print("\n=== 1. Baseline bias profiles ===")
strategies = {}

# Global mean
pred = np.full(len(val), tr["target_1h"].mean())
strategies["global_mean"] = compute_metrics(y_true, pred)

# Route mean (all history)
route_mean = tr.groupby("route_id")["target_1h"].mean()
pred = val["route_id"].map(route_mean).fillna(tr["target_1h"].mean()).values
strategies["route_mean_all"] = compute_metrics(y_true, pred)

# Route mean (last 7d)
last7d = tr[tr["timestamp"] > tr["timestamp"].max() - pd.Timedelta(days=7)]
route_mean_7d = last7d.groupby("route_id")["target_1h"].mean()
pred = val["route_id"].map(route_mean_7d).fillna(tr["target_1h"].mean()).values
strategies["route_mean_7d"] = compute_metrics(y_true, pred)

# Route mean (last 14d)
last14d = tr[tr["timestamp"] > tr["timestamp"].max() - pd.Timedelta(days=14)]
route_mean_14d = last14d.groupby("route_id")["target_1h"].mean()
pred = val["route_id"].map(route_mean_14d).fillna(tr["target_1h"].mean()).values
strategies["route_mean_14d"] = compute_metrics(y_true, pred)

# Route + same hour
val_with_feats = val.copy()
route_hour_mean = tr.groupby(["route_id", "hour"])["target_1h"].mean()
pred = val_with_feats.apply(
    lambda r: route_hour_mean.get((r["route_id"], r["hour"]), route_mean.get(r["route_id"], tr["target_1h"].mean())),
    axis=1
).values
strategies["route_hour_mean"] = compute_metrics(y_true, pred)

# Route + same DOW
val_dow = val["dow"].iloc[0]
route_dow_mean = tr[tr["dow"] == val_dow].groupby("route_id")["target_1h"].mean()
pred = val["route_id"].map(route_dow_mean).fillna(tr["target_1h"].mean()).values
strategies["route_same_dow"] = compute_metrics(y_true, pred)

# Route + same DOW + same hour
route_dow_hour_mean = tr[tr["dow"] == val_dow].groupby(["route_id", "hour"])["target_1h"].mean()
pred = val_with_feats.apply(
    lambda r: route_dow_hour_mean.get((r["route_id"], r["hour"]),
              route_dow_mean.get(r["route_id"], tr["target_1h"].mean())),
    axis=1
).values
strategies["route_dow_hour_mean"] = compute_metrics(y_true, pred)

# Route recent (14d) + same DOW
recent_dow = tr[(tr["dow"] == val_dow) &
               (tr["timestamp"] > tr["timestamp"].max() - pd.Timedelta(days=28))]
route_recent_dow = recent_dow.groupby("route_id")["target_1h"].mean()
pred = val["route_id"].map(route_recent_dow).fillna(tr["target_1h"].mean()).values
strategies["route_recent_dow_28d"] = compute_metrics(y_true, pred)

# Blend: 0.5 * route_mean_14d + 0.5 * route_same_dow
pred_blend = 0.5 * val["route_id"].map(route_mean_14d).fillna(tr["target_1h"].mean()).values + \
             0.5 * val["route_id"].map(route_dow_mean).fillna(tr["target_1h"].mean()).values
strategies["blend_14d_dow"] = compute_metrics(y_true, pred_blend)

results["strategy_metrics"] = strategies
print("\nStrategy comparison:")
print(f"{'Strategy':<30} {'Total':>8} {'WAPE':>8} {'RBias':>8} {'RBias_signed':>12}")
for name, m in sorted(strategies.items(), key=lambda x: x[1]["total"]):
    print(f"{name:<30} {m['total']:>8.4f} {m['wape']:>8.4f} {m['rbias']:>8.4f} {m['rbias_signed']:>12.4f}")

# ---------- 2. Bias direction analysis ----------
print("\n=== 2. Bias direction by strategy ===")
results["bias_direction"] = {}
for name, m in strategies.items():
    direction = "over" if m["rbias_signed"] > 0 else "under"
    results["bias_direction"][name] = {
        "direction": direction,
        "magnitude": m["rbias_signed"],
    }
    print(f"  {name}: {direction}-predicts by {abs(m['rbias_signed'])*100:.2f}%")

# ---------- 3. Aggregate volume estimation ----------
print("\n=== 3. Aggregate volume estimation ===")
actual_val_total = val["target_1h"].sum()
results["aggregate_volume"] = {
    "actual_val_total": round(float(actual_val_total), 1),
}
for name, m in strategies.items():
    ratio = 1 + m["rbias_signed"]
    results["aggregate_volume"][f"{name}_predicted_ratio"] = round(ratio, 4)

# ---------- 4. Post-hoc bias correction ----------
print("\n=== 4. Post-hoc bias correction effect ===")
results["bias_corrected"] = {}
for name in ["route_mean_all", "route_mean_14d", "route_hour_mean", "route_dow_hour_mean"]:
    m = strategies[name]
    # After bias correction, rbias becomes 0, only wape matters
    # But bias correction changes individual predictions
    if name == "route_mean_all":
        raw_pred = val["route_id"].map(route_mean).fillna(tr["target_1h"].mean()).values
    elif name == "route_mean_14d":
        raw_pred = val["route_id"].map(route_mean_14d).fillna(tr["target_1h"].mean()).values
    elif name == "route_hour_mean":
        raw_pred = val_with_feats.apply(
            lambda r: route_hour_mean.get((r["route_id"], r["hour"]), route_mean.get(r["route_id"], tr["target_1h"].mean())),
            axis=1).values
    elif name == "route_dow_hour_mean":
        raw_pred = val_with_feats.apply(
            lambda r: route_dow_hour_mean.get((r["route_id"], r["hour"]),
                      route_dow_mean.get(r["route_id"], tr["target_1h"].mean())),
            axis=1).values

    # Multiplicative correction
    correction = y_true.sum() / raw_pred.sum()
    corrected_pred = raw_pred * correction
    corrected_metrics = compute_metrics(y_true, corrected_pred)
    results["bias_corrected"][name] = {
        "original": strategies[name],
        "corrected": corrected_metrics,
        "correction_factor": round(float(correction), 4),
        "improvement": round(strategies[name]["total"] - corrected_metrics["total"], 4),
    }
    print(f"  {name}: {strategies[name]['total']:.4f} → {corrected_metrics['total']:.4f} "
          f"(correction={correction:.4f})")

# ---------- 5. Best strategy conclusion ----------
best = min(strategies.items(), key=lambda x: x[1]["total"])
results["best_strategy"] = {"name": best[0], "metrics": best[1]}
print(f"\nBest baseline: {best[0]} with total={best[1]['total']:.4f}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}")
