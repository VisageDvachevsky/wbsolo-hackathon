"""Window-specific profile x scale baseline for Saturday-like test windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.time_features import add_time_features
from src.train.evaluate_baselines import TARGET_COL


WINDOW_HOURS = (11, 14)


@dataclass(frozen=True)
class WindowProfileConfig:
    scale_days: int = 21
    scale_mode: str = "median_day"
    shrink_k: float = 30.0
    calibration_saturdays: int = 4


def prepare_with_time(df: pd.DataFrame, timestamp_origin: pd.Timestamp) -> pd.DataFrame:
    return add_time_features(df.copy(), timestamp_origin=timestamp_origin)


def window_mask(df: pd.DataFrame) -> pd.Series:
    return df["hour"].between(WINDOW_HOURS[0], WINDOW_HOURS[1])


def saturday_midday_dates(history_df: pd.DataFrame) -> list[pd.Timestamp]:
    df = history_df.loc[(history_df["dow"] == 5) & window_mask(history_df)]
    return [pd.Timestamp(x) for x in sorted(df["timestamp"].dt.normalize().unique())]


def build_profiles(history_df: pd.DataFrame, shrink_k: float = 30.0) -> tuple[pd.Series, pd.Series]:
    hist = history_df.loc[window_mask(history_df)].copy()

    global_profile = hist.groupby(["hour", "minute_30"])[TARGET_COL].mean()
    global_profile = global_profile / global_profile.mean()

    route_slot = hist.groupby(["route_id", "hour", "minute_30"])[TARGET_COL].mean()
    route_window_mean = hist.groupby("route_id")[TARGET_COL].mean()
    route_profile = route_slot / route_slot.index.get_level_values("route_id").map(route_window_mean)

    counts = hist.groupby(["route_id", "hour", "minute_30"]).size()
    alpha = counts / (counts + shrink_k)

    global_mapped = route_profile.index.droplevel("route_id").map(global_profile)
    shrunk = alpha * route_profile + (1.0 - alpha) * global_mapped
    return shrunk.rename("route_window_profile"), global_profile.rename("global_window_profile")


def build_scale(history_df: pd.DataFrame, last_n_days: int = 21, mode: str = "median_day") -> pd.Series:
    hist = history_df.loc[window_mask(history_df)].copy()
    cutoff = hist["timestamp"].max() - pd.Timedelta(days=last_n_days)
    hist = hist.loc[hist["timestamp"] > cutoff].copy()
    hist["date"] = hist["timestamp"].dt.normalize()
    daily = hist.groupby(["route_id", "date"])[TARGET_COL].mean()

    if mode == "median_day":
        return daily.groupby("route_id").median().rename("route_window_scale")
    if mode == "mean_day":
        return daily.groupby("route_id").mean().rename("route_window_scale")
    if mode == "trimmed_mean":
        return daily.groupby("route_id").apply(lambda x: x.sort_values().iloc[max(0, int(len(x) * 0.1)): max(int(len(x) * 0.9), 1)].mean()).rename("route_window_scale")
    raise ValueError(f"Unknown scale mode: {mode}")


def predict_window_profile_scale(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: WindowProfileConfig,
) -> np.ndarray:
    profiles, global_profile = build_profiles(history_df, shrink_k=config.shrink_k)
    scale = build_scale(history_df, last_n_days=config.scale_days, mode=config.scale_mode)

    route_fallback = history_df.groupby("route_id")[TARGET_COL].mean()
    scale_fallback = float(route_fallback.mean())

    keys = list(zip(target_df["route_id"], target_df["hour"], target_df["minute_30"]))
    profile_vals = pd.Series([profiles.get(k, np.nan) for k in keys], index=target_df.index)
    global_keys = list(zip(target_df["hour"], target_df["minute_30"]))
    global_vals = pd.Series([global_profile.get(k, 1.0) for k in global_keys], index=target_df.index)
    profile_vals = profile_vals.fillna(global_vals)

    scale_vals = target_df["route_id"].map(scale).fillna(target_df["route_id"].map(route_fallback)).fillna(scale_fallback)
    pred = scale_vals.to_numpy(dtype=float) * profile_vals.to_numpy(dtype=float)
    return np.clip(pred, 0.0, None)


def calibrate_to_recent_saturday_total(
    pred: np.ndarray,
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    n_saturdays: int = 4,
) -> np.ndarray:
    hist = history_df.loc[(history_df["dow"] == 5) & window_mask(history_df)].copy()
    dates = saturday_midday_dates(hist)
    if not dates:
        return pred

    selected_dates = dates[-n_saturdays:]
    totals = []
    for date in selected_dates:
        mask = hist["timestamp"].dt.normalize() == date
        totals.append(float(hist.loc[mask, TARGET_COL].sum()))
    expected_total = float(np.mean(totals))
    pred_sum = float(pred.sum())
    if pred_sum <= 0:
        return pred
    scaled = pred * (expected_total / pred_sum)
    return np.clip(scaled, 0.0, None)
