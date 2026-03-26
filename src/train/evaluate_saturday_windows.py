"""Evaluate baselines on historical Saturday test-like windows."""

from __future__ import annotations

import pandas as pd

from src.features.time_features import add_time_features
from src.train.evaluate_baselines import (
    TARGET_COL,
    build_profile_x_scale,
    build_route_mean,
    load_train,
    metric_frame,
    prepare,
)


def saturday_windows(train_df: pd.DataFrame) -> list[pd.Timestamp]:
    df = add_time_features(train_df, timestamp_origin=train_df["timestamp"].min())
    midday = df.loc[
        (df["dow"] == 5)
        & (df["hour"].between(11, 14))
    ]
    dates = sorted(midday["timestamp"].dt.normalize().unique())
    return [pd.Timestamp(d) for d in dates]


def window_mask(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    return (
        (df["timestamp"].dt.normalize() == date)
        & (df["timestamp"].dt.hour.between(11, 14))
    )


def main() -> None:
    train_df = load_train()
    origin = train_df["timestamp"].min()
    windows = saturday_windows(train_df)
    print("Saturday 11:00-14:30 validation windows")
    print("-" * 100)
    for date in windows[-6:]:
        val_mask = window_mask(train_df, date)
        val_df = prepare(train_df.loc[val_mask].copy(), origin)
        hist_df = prepare(train_df.loc[train_df["timestamp"] < val_df["timestamp"].min()].copy(), origin)
        y_true = val_df[TARGET_COL].to_numpy(dtype=float)
        route_mean = build_route_mean(hist_df, val_df)
        profile = build_profile_x_scale(hist_df, val_df, days=14)
        route_parts = metric_frame(y_true, route_mean)
        profile_parts = metric_frame(y_true, profile)
        print(
            f"{date.date()} | "
            f"route_mean={route_parts['total']:.6f} "
            f"profile_x_scale={profile_parts['total']:.6f} "
            f"mean_target={val_df[TARGET_COL].mean():.1f}"
        )


if __name__ == "__main__":
    main()
