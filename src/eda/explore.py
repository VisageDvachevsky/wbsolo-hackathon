"""
Comprehensive EDA: data structure, temporal patterns, feature-target relationships,
route analysis. Outputs findings to stdout in structured format.

Usage: python -m src.eda.explore
"""
import pandas as pd
import numpy as np
import sys
import os

# Allow running as module from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def load_data():
    train = pd.read_parquet(os.path.join(DATA_DIR, "train_solo_track.parquet"))
    test = pd.read_parquet(os.path.join(DATA_DIR, "test_solo_track.parquet"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "831419b5-3661-4980-94b3-081660040746.csv"))
    return train, test, sample_sub


def basic_structure(train, test, sample_sub):
    section("1. DATA STRUCTURE")
    print(f"Train: {train.shape[0]:,} rows x {train.shape[1]} cols")
    print(f"Test:  {test.shape[0]:,} rows x {test.shape[1]} cols")
    print(f"Sample submission: {sample_sub.shape}")
    print(f"\nTrain columns: {list(train.columns)}")
    print(f"Test columns:  {list(test.columns)}")
    print(f"\nTrain dtypes:\n{train.dtypes}")
    print(f"\nMissing values (train): {train.isnull().sum().sum()}")
    print(f"Missing values (test):  {test.isnull().sum().sum()}")
    print(f"Duplicates (train): {train.duplicated().sum()}")
    print(f"Key duplicates (route_id, timestamp): {train.duplicated(subset=['route_id', 'timestamp']).sum()}")
    print(f"\nUnique routes: train={train['route_id'].nunique()}, test={test['route_id'].nunique()}")
    print(f"Route intersection: {len(set(train['route_id']) & set(test['route_id']))}")
    print(f"\nTimestamps per route: train={train.groupby('route_id').size().unique()}, "
          f"test={test.groupby('route_id').size().unique()}")
    print(f"\nTime range: train [{train['timestamp'].min()} .. {train['timestamp'].max()}]")
    print(f"            test  [{test['timestamp'].min()} .. {test['timestamp'].max()}]")
    print(f"Time frequency: 30 min (verified: {train.groupby('route_id')['timestamp'].diff().dropna().unique()})")


def target_analysis(train):
    section("2. TARGET DISTRIBUTION")
    t = train["target_1h"]
    print(t.describe())
    print(f"\nZeros: {(t == 0).sum()} ({(t == 0).mean()*100:.1f}%)")
    print(f"Negatives: {(t < 0).sum()}")
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {t.quantile(p/100):,.0f}")

    section("2.1 ROUTE-LEVEL TARGET SCALE")
    route_means = train.groupby("route_id")["target_1h"].mean()
    print(f"Route mean target: min={route_means.min():,.0f}, max={route_means.max():,.0f}, "
          f"ratio={route_means.max()/route_means.min():.1f}x")
    print(f"\nTop 5 routes by mean target:")
    for rid, val in route_means.nlargest(5).items():
        print(f"  route {rid}: {val:,.0f}")
    print(f"\nBottom 5 routes by mean target:")
    for rid, val in route_means.nsmallest(5).items():
        print(f"  route {rid}: {val:,.0f}")


def temporal_analysis(train, test):
    section("3. TEMPORAL PATTERNS")

    # Trend
    first_week = train[train["timestamp"] < train["timestamp"].min() + pd.Timedelta(days=7)]
    last_week = train[train["timestamp"] > train["timestamp"].max() - pd.Timedelta(days=7)]
    print(f"Trend: first week mean={first_week['target_1h'].mean():,.0f}, "
          f"last week mean={last_week['target_1h'].mean():,.0f}, "
          f"ratio={last_week['target_1h'].mean()/first_week['target_1h'].mean():.3f}")

    # Day of week
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    dow = train.groupby(train["timestamp"].dt.dayofweek)["target_1h"].mean()
    print(f"\nDay of week pattern:")
    for d, val in dow.items():
        print(f"  {dow_names[d]}: {val:,.0f}")

    # Hour of day
    hourly = train.groupby(train["timestamp"].dt.hour)["target_1h"].mean()
    print(f"\nHour pattern: trough at {hourly.idxmin()}:00 ({hourly.min():,.0f}), "
          f"peak at {hourly.idxmax()}:00 ({hourly.max():,.0f})")

    # Test window
    test_dow = test["timestamp"].dt.dayofweek.iloc[0]
    test_hours = sorted(test["timestamp"].dt.hour.unique())
    print(f"\nTest window: {dow_names.get(test_dow)} hours {test_hours}")
    print(f"Test timestamps: {sorted(test['timestamp'].unique())}")


def correlation_analysis(train):
    section("4. FEATURE-TARGET CORRELATIONS")
    status_cols = [c for c in train.columns if c.startswith("status_")]
    corr = train[status_cols + ["target_1h"]].corr()["target_1h"].drop("target_1h")
    print("Global correlations with target_1h:")
    for col, val in corr.sort_values(ascending=False).items():
        print(f"  {col}: {val:.4f}")

    print("\nNote: status features are NOT available in test set!")
    print("These correlations are informational only.")

    section("4.1 TARGET AUTOCORRELATION")
    sample_routes = train["route_id"].unique()[:50]
    for lag_name, lag_steps in [("30min", 1), ("1h", 2), ("24h", 48), ("7d", 336)]:
        auto_corrs = []
        for rid in sample_routes:
            grp = train[train["route_id"] == rid].sort_values("timestamp")
            lagged = grp["target_1h"].shift(lag_steps)
            valid = pd.DataFrame({"t": grp["target_1h"], "l": lagged}).dropna()
            if valid["t"].std() > 0 and valid["l"].std() > 0:
                auto_corrs.append(valid.corr().iloc[0, 1])
        print(f"  Lag {lag_name}: {np.mean(auto_corrs):.4f}")


def leakage_assessment():
    section("5. LEAKAGE RISK ASSESSMENT")
    print("CRITICAL: test set has NO status features (only id, route_id, timestamp)")
    print()
    print("SAFE features for test prediction:")
    print("  - route_id (categorical)")
    print("  - Time features (hour, DOW, week, trend)")
    print("  - Historical route aggregates (mean, median, std)")
    print("  - Route + time-slot aggregates")
    print("  - Rolling/lag target features (up to last train timestamp)")
    print()
    print("CANNOT use at test time:")
    print("  - status_1..6 (not in test)")
    print("  - Any concurrent or future information")


def validation_recommendation():
    section("6. VALIDATION STRATEGY")
    print("Recommended: time-based holdout of last 8 timestamps per route")
    print("  - Validation: last 8 timestamps (4 hours) of training data")
    print("  - Mirrors test structure exactly (8 timestamps x 1000 routes)")
    print("  - Time-forward split prevents leakage")
    print("  - All routes present in both parts")


def main():
    print("Loading data...")
    train, test, sample_sub = load_data()
    basic_structure(train, test, sample_sub)
    target_analysis(train)
    temporal_analysis(train, test)
    correlation_analysis(train)
    leakage_assessment()
    validation_recommendation()
    print("\n" + "=" * 80)
    print("  EDA COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
