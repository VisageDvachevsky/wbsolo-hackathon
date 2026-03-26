"""Same-day calibration for profile x scale forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SameDayCalibrationConfig:
    """Configuration for route-level same-day calibration."""

    obs_start_hour: int = 7
    obs_end_hour: int = 10
    shrink_k: float = 2_000_000.0
    clip_min: float = 0.70
    clip_max: float = 1.30


def mask_same_day_hours(
    df: pd.DataFrame,
    date: pd.Timestamp,
    start_hour: int,
    end_hour: int,
) -> pd.Series:
    """Mask rows belonging to the same calendar day and hour span."""
    return (df["timestamp"].dt.normalize() == date) & df["hour"].between(start_hour, end_hour)


def compute_route_multipliers_gamma_poisson(
    obs_df: pd.DataFrame,
    obs_base_pred: np.ndarray,
    config: SameDayCalibrationConfig,
) -> tuple[pd.Series, float]:
    """
    Compute route-specific multiplicative factors with empirical-Bayes shrinkage.

    The idea is to shrink noisy route-level factors toward the same-day global factor.
    """
    if len(obs_df) == 0:
        return pd.Series(dtype=float), 1.0

    tmp = obs_df[["route_id"]].copy()
    tmp["y_obs"] = obs_df["target_1h"].to_numpy(dtype=float)
    tmp["mu_obs"] = np.asarray(obs_base_pred, dtype=float)

    by_route = tmp.groupby("route_id", as_index=True).agg(y_sum=("y_obs", "sum"), mu_sum=("mu_obs", "sum"))

    y_total = float(by_route["y_sum"].sum())
    mu_total = float(by_route["mu_sum"].sum())
    global_theta = (y_total / mu_total) if mu_total > 0 else 1.0

    k = float(config.shrink_k)
    theta = (by_route["y_sum"] + global_theta * k) / (by_route["mu_sum"] + k)
    theta = theta.clip(lower=config.clip_min, upper=config.clip_max)
    return theta.astype(float), global_theta


def apply_same_day_calibration(
    base_pred: np.ndarray,
    target_df: pd.DataFrame,
    route_multipliers: pd.Series,
    default_multiplier: float = 1.0,
) -> np.ndarray:
    """Apply route-level multipliers to a base prediction vector."""
    multipliers = target_df["route_id"].map(route_multipliers).fillna(default_multiplier).to_numpy(dtype=float)
    pred = np.asarray(base_pred, dtype=float) * multipliers
    return np.clip(pred, 0.0, None)


def compute_global_multiplier(obs_df: pd.DataFrame, obs_base_pred: np.ndarray, config: SameDayCalibrationConfig) -> float:
    """Single same-day multiplier for the whole universe."""
    if len(obs_df) == 0:
        return 1.0
    y_total = float(obs_df["target_1h"].sum())
    mu_total = float(np.asarray(obs_base_pred, dtype=float).sum())
    if mu_total <= 0:
        return 1.0
    return float(np.clip(y_total / mu_total, config.clip_min, config.clip_max))


def assign_volume_segments(history_df: pd.DataFrame, n_segments: int = 3) -> pd.Series:
    """Assign each route to a volume segment based on historical mean target."""
    route_mean = history_df.groupby("route_id")["target_1h"].mean()
    ranks = pd.qcut(route_mean.rank(method="first"), q=n_segments, labels=False)
    return ranks.rename("volume_segment")


def compute_segment_multipliers(
    history_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    obs_base_pred: np.ndarray,
    config: SameDayCalibrationConfig,
    n_segments: int = 3,
) -> tuple[pd.Series, float]:
    """Compute segment-level same-day factors shrunk to the global factor."""
    if len(obs_df) == 0:
        return pd.Series(dtype=float), 1.0

    route_segments = assign_volume_segments(history_df, n_segments=n_segments)
    tmp = obs_df[["route_id"]].copy()
    tmp["segment"] = tmp["route_id"].map(route_segments)
    tmp["y_obs"] = obs_df["target_1h"].to_numpy(dtype=float)
    tmp["mu_obs"] = np.asarray(obs_base_pred, dtype=float)

    global_theta = compute_global_multiplier(obs_df, obs_base_pred, config)
    k = float(config.shrink_k)
    by_segment = tmp.groupby("segment", as_index=True).agg(y_sum=("y_obs", "sum"), mu_sum=("mu_obs", "sum"))
    theta = (by_segment["y_sum"] + global_theta * k) / (by_segment["mu_sum"] + k)
    theta = theta.clip(lower=config.clip_min, upper=config.clip_max)
    return theta.astype(float), global_theta


def apply_segment_calibration(
    base_pred: np.ndarray,
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    segment_multipliers: pd.Series,
    default_multiplier: float = 1.0,
    n_segments: int = 3,
) -> np.ndarray:
    """Apply segment-level multipliers determined by route volume buckets."""
    route_segments = assign_volume_segments(history_df, n_segments=n_segments)
    segments = target_df["route_id"].map(route_segments)
    multipliers = segments.map(segment_multipliers).fillna(default_multiplier).to_numpy(dtype=float)
    pred = np.asarray(base_pred, dtype=float) * multipliers
    return np.clip(pred, 0.0, None)
