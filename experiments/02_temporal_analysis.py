"""
Temporal analysis: seasonality, distribution shifts, test vs train tail.
"""
import pandas as pd
import numpy as np

DATA_DIR = "/tmp/gh-issue-solver-1774505330257"

train = pd.read_parquet(f"{DATA_DIR}/train_solo_track.parquet")
test = pd.read_parquet(f"{DATA_DIR}/test_solo_track.parquet")

train["hour"] = train["timestamp"].dt.hour
train["dow"] = train["timestamp"].dt.dayofweek  # 0=Mon
train["date"] = train["timestamp"].dt.date
train["half_hour"] = train["timestamp"].dt.hour + train["timestamp"].dt.minute / 60

print("=" * 80)
print("HOURLY PATTERNS")
print("=" * 80)
hourly = train.groupby("hour")["target_1h"].agg(["mean", "median", "std", "count"])
print(hourly)

print("\n" + "=" * 80)
print("DAY OF WEEK PATTERNS")
print("=" * 80)
dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
dow = train.groupby("dow")["target_1h"].agg(["mean", "median", "std", "count"])
dow.index = dow.index.map(dow_names)
print(dow)

print("\n" + "=" * 80)
print("DAILY AGGREGATE TREND")
print("=" * 80)
daily = train.groupby("date")["target_1h"].agg(["mean", "median", "count"])
print(f"First 5 days:")
print(daily.head(5))
print(f"\nLast 5 days:")
print(daily.tail(5))
print(f"\nTotal unique dates: {len(daily)}")

# Check if target mean drifts over time
first_week = train[train["timestamp"] < train["timestamp"].min() + pd.Timedelta(days=7)]
last_week = train[train["timestamp"] > train["timestamp"].max() - pd.Timedelta(days=7)]
print(f"\nFirst week target mean: {first_week['target_1h'].mean():.0f}")
print(f"Last week target mean:  {last_week['target_1h'].mean():.0f}")
print(f"Ratio last/first:       {last_week['target_1h'].mean() / first_week['target_1h'].mean():.3f}")

print("\n" + "=" * 80)
print("DISTRIBUTION SHIFT: FIRST vs LAST MONTH")
print("=" * 80)
train_sorted = train.sort_values("timestamp")
n = len(train_sorted)
first_quarter = train_sorted.iloc[:n//4]
last_quarter = train_sorted.iloc[-n//4:]
print(f"First quarter: {first_quarter['timestamp'].min()} to {first_quarter['timestamp'].max()}")
print(f"Last quarter:  {last_quarter['timestamp'].min()} to {last_quarter['timestamp'].max()}")
print(f"\nFirst quarter target stats:")
print(first_quarter["target_1h"].describe())
print(f"\nLast quarter target stats:")
print(last_quarter["target_1h"].describe())

print("\n" + "=" * 80)
print("TEST WINDOW ANALYSIS")
print("=" * 80)
test["hour"] = test["timestamp"].dt.hour
test["dow"] = test["timestamp"].dt.dayofweek
print(f"Test timestamps: {sorted(test['timestamp'].unique())}")
print(f"Test day of week: {dow_names.get(test['dow'].iloc[0], test['dow'].iloc[0])}")
print(f"Test hours: {sorted(test['hour'].unique())}")

# What does the same time window look like in recent train data?
test_hours = set(test["hour"].unique())
test_dow_val = test["dow"].iloc[0]
# Recent similar periods in train
similar_recent = train[
    (train["dow"] == test_dow_val) &
    (train["hour"].isin(test_hours)) &
    (train["timestamp"] > train["timestamp"].max() - pd.Timedelta(days=28))
]
print(f"\nSimilar recent windows (same dow + hours, last 28 days):")
print(f"Count: {len(similar_recent)}")
if len(similar_recent) > 0:
    print(f"Target mean: {similar_recent['target_1h'].mean():.0f}")
    print(f"Target median: {similar_recent['target_1h'].median():.0f}")

# Last day of train — same route, see how it looks
train_last_day = train[train["timestamp"].dt.date == train["timestamp"].max().date()]
print(f"\nLast day of train ({train['timestamp'].max().date()}):")
print(f"Timestamps: {sorted(train_last_day['timestamp'].unique())[:10]}... (total {train_last_day['timestamp'].nunique()})")
print(f"Target mean: {train_last_day['target_1h'].mean():.0f}")

print("\n" + "=" * 80)
print("WEEKLY PATTERN CHECK")
print("=" * 80)
# Weekly aggregated mean target
train["week"] = train["timestamp"].dt.isocalendar().week.astype(int)
weekly = train.groupby("week")["target_1h"].mean()
print("Weekly mean target:")
print(weekly)

print("\n" + "=" * 80)
print("HALF-HOUR SLOT PATTERN")
print("=" * 80)
hh = train.groupby("half_hour")["target_1h"].mean()
print(hh)

print("\nDone with temporal analysis!")
