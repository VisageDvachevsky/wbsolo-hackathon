"""
Temporal feature extractors.
"""
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to a DataFrame with a timestamp column."""
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["minute_30"] = (out["timestamp"].dt.minute >= 30).astype(int)
    out["half_hour"] = out["hour"] + out["minute_30"] * 0.5
    out["dow"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["day_of_month"] = out["timestamp"].dt.day
    out["week_of_year"] = out["timestamp"].dt.isocalendar().week.astype(int)
    # Days since start of training (trend proxy)
    min_ts = out["timestamp"].min()
    out["days_since_start"] = (out["timestamp"] - min_ts).dt.total_seconds() / 86400
    return out
