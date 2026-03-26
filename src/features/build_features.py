"""Main feature engineering pipeline."""

from __future__ import annotations

import pandas as pd

from src.features.time_features import add_time_features
from src.features.route_features import (
    build_route_aggregates,
    build_route_time_aggregates,
    build_route_dow_aggregates,
    build_route_dow_hour_aggregates,
    build_route_recent_aggregates,
)


def build_feature_matrix(target_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for target_df using history_df as the source of historical aggregates.

    Parameters
    ----------
    target_df : DataFrame with (route_id, timestamp) to predict for
    history_df : DataFrame with full training history (must not overlap with target_df temporally)

    Returns
    -------
    DataFrame with all features, indexed same as target_df
    """
    # Time features
    df = add_time_features(target_df, timestamp_origin=history_df["timestamp"].min())

    # Route aggregates
    route_aggs = build_route_aggregates(history_df)
    df = df.merge(route_aggs, on="route_id", how="left")

    # Route + hour aggregates
    route_hour_aggs = build_route_time_aggregates(history_df)
    df = df.merge(
        route_hour_aggs,
        on=["route_id", "hour", "minute_30"],
        how="left",
    )

    # Route + DOW aggregates
    route_dow_aggs = build_route_dow_aggregates(history_df)
    df = df.merge(route_dow_aggs, on=["route_id", "dow"], how="left")

    # Route + DOW + hour aggregates
    route_dow_hour_aggs = build_route_dow_hour_aggregates(history_df)
    df = df.merge(route_dow_hour_aggs, on=["route_id", "dow", "hour"], how="left")

    # Recent aggregates (7d and 14d)
    for n_days in [7, 14]:
        recent_aggs = build_route_recent_aggregates(history_df, last_n_days=n_days)
        df = df.merge(recent_aggs, on="route_id", how="left")

    return df
