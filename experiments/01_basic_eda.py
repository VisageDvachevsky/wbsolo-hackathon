"""
Basic EDA script: data structure, columns, types, missing values, duplicates, time ranges.
"""
import pandas as pd
import numpy as np

DATA_DIR = "/tmp/gh-issue-solver-1774505330257"

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
test = pd.read_parquet(f"{DATA_DIR}/test_solo_track.parquet")
sample_sub = pd.read_csv(f"{DATA_DIR}/831419b5-3661-4980-94b3-081660040746.csv")

print(f"\nTrain shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

print("\n" + "=" * 80)
print("TRAIN INFO")
print("=" * 80)
print("\nColumns:", list(train.columns))
print("\nDtypes:")
print(train.dtypes)
print("\nHead:")
print(train.head(10))
print("\nDescribe:")
print(train.describe())

print("\n" + "=" * 80)
print("TEST INFO")
print("=" * 80)
print("\nColumns:", list(test.columns))
print("\nDtypes:")
print(test.dtypes)
print("\nHead:")
print(test.head(10))

print("\n" + "=" * 80)
print("SAMPLE SUBMISSION")
print("=" * 80)
print("\nColumns:", list(sample_sub.columns))
print("\nHead:")
print(sample_sub.head(10))

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
print("\nTrain missing:")
print(train.isnull().sum())
print(f"\nTotal train missing cells: {train.isnull().sum().sum()}")
print("\nTest missing:")
print(test.isnull().sum())

print("\n" + "=" * 80)
print("DUPLICATES")
print("=" * 80)
train_dups = train.duplicated().sum()
test_dups = test.duplicated().sum()
print(f"Train full duplicates: {train_dups}")
print(f"Test full duplicates: {test_dups}")
# Check route_id + timestamp duplicates
train_key_dups = train.duplicated(subset=["route_id", "timestamp"]).sum()
test_key_dups = test.duplicated(subset=["route_id", "timestamp"]).sum()
print(f"Train (route_id, timestamp) duplicates: {train_key_dups}")
print(f"Test (route_id, timestamp) duplicates: {test_key_dups}")

print("\n" + "=" * 80)
print("TIME RANGES")
print("=" * 80)
print(f"\nTrain timestamp min: {train['timestamp'].min()}")
print(f"Train timestamp max: {train['timestamp'].max()}")
print(f"Test timestamp min:  {test['timestamp'].min()}")
print(f"Test timestamp max:  {test['timestamp'].max()}")

# Check if test is after train
print(f"\nTest starts after train ends: {test['timestamp'].min() > train['timestamp'].max()}")
print(f"Test starts at same time as train end: {test['timestamp'].min() == train['timestamp'].max()}")

# Time span
train_span = train['timestamp'].max() - train['timestamp'].min()
test_span = test['timestamp'].max() - test['timestamp'].min()
print(f"\nTrain time span: {train_span}")
print(f"Test time span:  {test_span}")

print("\n" + "=" * 80)
print("ROUTE_ID ANALYSIS")
print("=" * 80)
train_routes = set(train["route_id"].unique())
test_routes = set(test["route_id"].unique())
print(f"Train unique route_id: {len(train_routes)}")
print(f"Test unique route_id:  {len(test_routes)}")
print(f"Intersection: {len(train_routes & test_routes)}")
print(f"Only in train: {len(train_routes - test_routes)}")
print(f"Only in test:  {len(test_routes - train_routes)}")

print("\n" + "=" * 80)
print("OBSERVATION FREQUENCY")
print("=" * 80)
# Check time differences within routes
sample_route = train["route_id"].value_counts().index[0]
route_data = train[train["route_id"] == sample_route].sort_values("timestamp")
time_diffs = route_data["timestamp"].diff().dropna()
print(f"\nSample route '{sample_route}' time diffs:")
print(time_diffs.value_counts().head(10))

# Global time diffs per route
all_diffs = []
for rid, grp in train.groupby("route_id"):
    grp = grp.sort_values("timestamp")
    diffs = grp["timestamp"].diff().dropna()
    all_diffs.append(diffs)
all_diffs = pd.concat(all_diffs)
print(f"\nAll routes time diff distribution:")
print(all_diffs.value_counts().head(10))
print(f"\nUnique time diffs: {all_diffs.nunique()}")

# Check if time grid is uniform within routes
print("\n" + "=" * 80)
print("TIME GRID UNIFORMITY CHECK")
print("=" * 80)
route_counts = train.groupby("route_id").size()
print(f"\nObservations per route:")
print(f"  Min: {route_counts.min()}")
print(f"  Max: {route_counts.max()}")
print(f"  Mean: {route_counts.mean():.1f}")
print(f"  Median: {route_counts.median()}")
print(f"  Std: {route_counts.std():.1f}")
print(f"\nRoutes with different counts: {route_counts.nunique()} unique values")
print(f"Value counts of observation counts:")
print(route_counts.value_counts().head(10))

# Same for test
test_route_counts = test.groupby("route_id").size()
print(f"\nTest observations per route:")
print(f"  Min: {test_route_counts.min()}")
print(f"  Max: {test_route_counts.max()}")
print(f"  Mean: {test_route_counts.mean():.1f}")
print(f"  Unique counts: {test_route_counts.nunique()}")

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)
print(f"\ntarget_1h stats:")
print(train["target_1h"].describe())
print(f"\nZeros: {(train['target_1h'] == 0).sum()} ({(train['target_1h'] == 0).mean()*100:.1f}%)")
print(f"Negatives: {(train['target_1h'] < 0).sum()}")
print(f"\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {train['target_1h'].quantile(p/100):.2f}")

print("\n" + "=" * 80)
print("STATUS COLUMNS STATS")
print("=" * 80)
status_cols = [c for c in train.columns if c.startswith("status_")]
for col in status_cols:
    print(f"\n{col}:")
    print(f"  mean={train[col].mean():.2f}, std={train[col].std():.2f}")
    print(f"  min={train[col].min()}, max={train[col].max()}")
    print(f"  zeros={( train[col] == 0).sum()} ({(train[col] == 0).mean()*100:.1f}%)")

print("\nDone with basic EDA!")
