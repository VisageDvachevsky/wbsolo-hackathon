"""Route-level aggregate features computed from historical data."""

from __future__ import annotations

import pandas as pd


def build_route_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-route historical aggregates from training data.
    Returns a DataFrame indexed by route_id.
    """
    aggs = train_df.groupby("route_id")["target_1h"].agg(
        route_mean="mean",
        route_median="median",
        route_std="std",
        route_q25=lambda x: x.quantile(0.25),
        route_q75=lambda x: x.quantile(0.75),
        route_max="max",
        route_min="min",
        route_cv=lambda x: x.std() / x.mean() if x.mean() > 0 else 0,
        route_nonzero_frac=lambda x: (x > 0).mean(),
    )
    return aggs.reset_index()


def build_route_time_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-route + time-slot historical aggregates.
    Returns a DataFrame indexed by (route_id, hour, half_hour_flag).
    """
    df = train_df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["minute_30"] = (df["timestamp"].dt.minute >= 30).astype(int)

    aggs = df.groupby(["route_id", "hour", "minute_30"])["target_1h"].agg(
        route_hour_mean="mean",
        route_hour_median="median",
        route_hour_std="std",
    )
    return aggs.reset_index()


def build_route_dow_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-route + day-of-week aggregates.
    """
    df = train_df.copy()
    df["dow"] = df["timestamp"].dt.dayofweek

    aggs = df.groupby(["route_id", "dow"])["target_1h"].agg(
        route_dow_mean="mean",
        route_dow_median="median",
    )
    return aggs.reset_index()


def build_route_dow_hour_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-route + day-of-week + hour aggregates.
    """
    df = train_df.copy()
    df["dow"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour

    aggs = df.groupby(["route_id", "dow", "hour"])["target_1h"].agg(
        route_dow_hour_mean="mean",
        route_dow_hour_median="median",
    )
    return aggs.reset_index()


def build_route_recent_aggregates(train_df: pd.DataFrame, last_n_days: int = 7) -> pd.DataFrame:
    """
    Compute per-route aggregates from only the last N days of training data.
    """
    cutoff = train_df["timestamp"].max() - pd.Timedelta(days=last_n_days)
    recent = train_df[train_df["timestamp"] > cutoff]

    aggs = recent.groupby("route_id")["target_1h"].agg(
        **{f"route_mean_last{last_n_days}d": "mean",
           f"route_median_last{last_n_days}d": "median"}
    )
    return aggs.reset_index()
