"""
Deep analytics #8: Baseline hypothesis comparison before modeling.
Multiple validation windows, comprehensive strategy evaluation.
"""
import pandas as pd
import numpy as np
import json

DATA_DIR = "/tmp/gh-issue-solver-1774506453169"
OUT = f"{DATA_DIR}/experiments/results_11_baseline_comparison.json"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
train["hour"] = train["timestamp"].dt.hour
train["dow"] = train["timestamp"].dt.dayofweek

results = {}

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    wape = np.abs(y_pred - y_true).sum() / y_true.sum()
    rbias = abs(y_pred.sum() / y_true.sum() - 1)
    return {"wape": round(float(wape), 6), "rbias": round(float(rbias), 6),
            "total": round(float(wape + rbias), 6)}

# ---------- Create multiple validation folds ----------
print("=== Setting up multi-fold validation ===")
all_ts = sorted(train["timestamp"].unique())
n_ts = len(all_ts)

# Create 5 folds, each holding out 8 consecutive timestamps
folds = []
for i in range(5):
    end_idx = n_ts - i * 48  # space folds by ~1 day (48 steps)
    if end_idx < 100:
        break
    val_ts = all_ts[end_idx - 8:end_idx]
    train_ts = all_ts[:end_idx - 8]
    folds.append({"val_ts": val_ts, "train_ts": train_ts, "fold_id": i})
    print(f"Fold {i}: val={val_ts[0]} to {val_ts[-1]}")

# ---------- Evaluate strategies across folds ----------
print("\n=== Evaluating strategies ===")
strategy_names = [
    "global_mean", "route_mean_all", "route_mean_7d", "route_mean_14d",
    "route_hour_mean", "route_same_dow", "route_dow_hour_mean",
    "route_recent_dow_28d", "blend_route14d_dowhour",
    "recency_weighted_route", "profile_x_scale"
]

all_fold_results = {s: [] for s in strategy_names}

for fold in folds:
    tr = train[train["timestamp"].isin(fold["train_ts"])]
    val = train[train["timestamp"].isin(fold["val_ts"])]
    y_true = val["target_1h"].values
    val_dow = val["dow"].iloc[0]

    global_mean = tr["target_1h"].mean()
    route_mean = tr.groupby("route_id")["target_1h"].mean()

    def get_pred(strategy):
        if strategy == "global_mean":
            return np.full(len(val), global_mean)
        elif strategy == "route_mean_all":
            return val["route_id"].map(route_mean).fillna(global_mean).values
        elif strategy == "route_mean_7d":
            rm = tr[tr["timestamp"] > tr["timestamp"].max() - pd.Timedelta(days=7)].groupby("route_id")["target_1h"].mean()
            return val["route_id"].map(rm).fillna(global_mean).values
        elif strategy == "route_mean_14d":
            rm = tr[tr["timestamp"] > tr["timestamp"].max() - pd.Timedelta(days=14)].groupby("route_id")["target_1h"].mean()
            return val["route_id"].map(rm).fillna(global_mean).values
        elif strategy == "route_hour_mean":
            rhm = tr.groupby(["route_id", "hour"])["target_1h"].mean()
            return val.apply(lambda r: rhm.get((r["route_id"], r["hour"]), route_mean.get(r["route_id"], global_mean)), axis=1).values
        elif strategy == "route_same_dow":
            rdm = tr[tr["dow"] == val_dow].groupby("route_id")["target_1h"].mean()
            return val["route_id"].map(rdm).fillna(global_mean).values
        elif strategy == "route_dow_hour_mean":
            rdhm = tr[tr["dow"] == val_dow].groupby(["route_id", "hour"])["target_1h"].mean()
            rdm = tr[tr["dow"] == val_dow].groupby("route_id")["target_1h"].mean()
            return val.apply(lambda r: rdhm.get((r["route_id"], r["hour"]), rdm.get(r["route_id"], global_mean)), axis=1).values
        elif strategy == "route_recent_dow_28d":
            recent = tr[(tr["dow"] == val_dow) & (tr["timestamp"] > tr["timestamp"].max() - pd.Timedelta(days=28))]
            rm = recent.groupby("route_id")["target_1h"].mean()
            return val["route_id"].map(rm).fillna(global_mean).values
        elif strategy == "blend_route14d_dowhour":
            rm14 = tr[tr["timestamp"] > tr["timestamp"].max() - pd.Timedelta(days=14)].groupby("route_id")["target_1h"].mean()
            rdhm = tr[tr["dow"] == val_dow].groupby(["route_id", "hour"])["target_1h"].mean()
            rdm = tr[tr["dow"] == val_dow].groupby("route_id")["target_1h"].mean()
            p1 = val["route_id"].map(rm14).fillna(global_mean).values
            p2 = val.apply(lambda r: rdhm.get((r["route_id"], r["hour"]), rdm.get(r["route_id"], global_mean)), axis=1).values
            return 0.5 * p1 + 0.5 * p2
        elif strategy == "recency_weighted_route":
            # Exponentially weighted: more weight to recent
            tr_sorted = tr.sort_values("timestamp")
            days_ago = (tr_sorted["timestamp"].max() - tr_sorted["timestamp"]).dt.total_seconds() / 86400
            weights = np.exp(-days_ago / 14)  # 14-day half-life
            tr_sorted["weighted_target"] = tr_sorted["target_1h"] * weights
            wm = tr_sorted.groupby("route_id")["weighted_target"].sum() / tr_sorted.groupby("route_id").apply(lambda g: weights[g.index].sum())
            return val["route_id"].map(wm).fillna(global_mean).values
        elif strategy == "profile_x_scale":
            # Route scale from last 14d * global hour profile * global DOW profile
            rm14 = tr[tr["timestamp"] > tr["timestamp"].max() - pd.Timedelta(days=14)].groupby("route_id")["target_1h"].mean()
            hour_effect = tr.groupby("hour")["target_1h"].mean() / global_mean
            dow_effect = tr.groupby("dow")["target_1h"].mean() / global_mean
            pred = val["route_id"].map(rm14).fillna(global_mean).values * \
                   val["hour"].map(hour_effect).values * \
                   val["dow"].map(dow_effect).values
            return pred
        return np.full(len(val), global_mean)

    for strategy in strategy_names:
        pred = get_pred(strategy)
        metrics = compute_metrics(y_true, pred)
        all_fold_results[strategy].append(metrics)

# ---------- Aggregate results ----------
print("\n=== Aggregate results across folds ===")
results["multi_fold_comparison"] = {}
print(f"\n{'Strategy':<30} {'Mean Total':>10} {'Std Total':>10} {'Mean WAPE':>10} {'Mean RBias':>10}")
ranked = []
for strategy in strategy_names:
    fold_metrics = all_fold_results[strategy]
    totals = [m["total"] for m in fold_metrics]
    wapes = [m["wape"] for m in fold_metrics]
    rbiases = [m["rbias"] for m in fold_metrics]

    agg = {
        "mean_total": round(np.mean(totals), 6),
        "std_total": round(np.std(totals), 6),
        "mean_wape": round(np.mean(wapes), 6),
        "mean_rbias": round(np.mean(rbiases), 6),
        "min_total": round(np.min(totals), 6),
        "max_total": round(np.max(totals), 6),
        "per_fold": fold_metrics,
    }
    results["multi_fold_comparison"][strategy] = agg
    ranked.append((strategy, np.mean(totals)))
    print(f"{strategy:<30} {np.mean(totals):>10.4f} {np.std(totals):>10.4f} {np.mean(wapes):>10.4f} {np.mean(rbiases):>10.4f}")

ranked.sort(key=lambda x: x[1])
results["ranking"] = [{"rank": i+1, "strategy": s, "mean_total": round(v, 6)} for i, (s, v) in enumerate(ranked)]
print(f"\nBest: {ranked[0][0]} (mean total={ranked[0][1]:.4f})")
print(f"2nd:  {ranked[1][0]} (mean total={ranked[1][1]:.4f})")
print(f"3rd:  {ranked[2][0]} (mean total={ranked[2][1]:.4f})")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}")
