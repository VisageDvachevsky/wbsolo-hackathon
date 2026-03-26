"""Evaluate simple forecasting baselines on a time holdout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.time_features import add_time_features
from src.validation.metrics import WapePlusRbias
from src.validation.split import time_holdout_split


TARGET_COL = "target_1h"
KEY_COLS = ["route_id", "timestamp"]


def load_train(path: str = "train_solo_track.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def prepare(df: pd.DataFrame, timestamp_origin: pd.Timestamp) -> pd.DataFrame:
    return add_time_features(df, timestamp_origin=timestamp_origin)


def metric_frame(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metric = WapePlusRbias()
    parts = metric.calculate_components(y_true=y_true, y_pred=y_pred)
    return {k: float(v) for k, v in parts.items()}


def merge_fill(
    target_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    on: list[str],
    value_col: str,
    fallback: np.ndarray | float,
) -> np.ndarray:
    merged = target_df.merge(feature_df, on=on, how="left")
    values = merged[value_col].to_numpy(dtype=float)
    if np.isscalar(fallback):
        return np.where(np.isnan(values), float(fallback), values)
    return np.where(np.isnan(values), fallback, values)


def build_global_mean(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    return np.full(len(val_df), float(train_df[TARGET_COL].mean()))


def build_route_mean(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    route_mean = (
        train_df.groupby("route_id", as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "pred"})
    )
    return merge_fill(val_df, route_mean, ["route_id"], "pred", train_df[TARGET_COL].mean())


def build_route_recent_mean(train_df: pd.DataFrame, val_df: pd.DataFrame, days: int) -> np.ndarray:
    cutoff = train_df["timestamp"].max() - pd.Timedelta(days=days)
    recent = train_df.loc[train_df["timestamp"] > cutoff]
    route_recent = (
        recent.groupby("route_id", as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "pred"})
    )
    fallback = build_route_mean(train_df, val_df)
    return merge_fill(val_df, route_recent, ["route_id"], "pred", fallback)


def build_profile_x_scale(train_df: pd.DataFrame, val_df: pd.DataFrame, days: int = 14) -> np.ndarray:
    global_mean = float(train_df[TARGET_COL].mean())
    route_recent = (
        train_df.loc[train_df["timestamp"] > train_df["timestamp"].max() - pd.Timedelta(days=days)]
        .groupby("route_id", as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "route_mean_recent"})
    )
    hour_effect = (
        train_df.groupby(["hour", "minute_30"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "hour_mean"})
    )
    dow_effect = (
        train_df.groupby("dow", as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "dow_mean"})
    )

    df = val_df.merge(route_recent, on="route_id", how="left")
    df = df.merge(hour_effect, on=["hour", "minute_30"], how="left")
    df = df.merge(dow_effect, on="dow", how="left")
    route_fallback = build_route_mean(train_df, val_df)
    route_scale = df["route_mean_recent"].to_numpy(dtype=float)
    route_scale = np.where(np.isnan(route_scale), route_fallback, route_scale)
    hour_mult = df["hour_mean"].to_numpy(dtype=float) / global_mean
    dow_mult = df["dow_mean"].to_numpy(dtype=float) / global_mean
    return route_scale * hour_mult * dow_mult


def build_route_slot_mean(train_df: pd.DataFrame, val_df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    slot_mean = (
        train_df.groupby(keys, as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "pred"})
    )
    fallback = build_route_mean(train_df, val_df)
    return merge_fill(val_df, slot_mean, keys, "pred", fallback)


def build_recent_same_dow_hour_mean(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    days: int,
) -> np.ndarray:
    cutoff = train_df["timestamp"].max() - pd.Timedelta(days=days)
    recent = train_df.loc[train_df["timestamp"] > cutoff]
    keys = ["route_id", "dow", "hour", "minute_30"]
    feature_df = (
        recent.groupby(keys, as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "pred"})
    )
    fallback = build_route_slot_mean(train_df, val_df, keys)
    return merge_fill(val_df, feature_df, keys, "pred", fallback)


def build_blend(*preds: np.ndarray, weights: list[float]) -> np.ndarray:
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    stacked = np.vstack(preds)
    return np.average(stacked, axis=0, weights=weights_arr)


@dataclass(frozen=True)
class BaselineResult:
    name: str
    total: float
    wape: float
    rbias: float
    rbias_signed: float


def evaluate(train_df: pd.DataFrame) -> list[BaselineResult]:
    train_part, val_part = time_holdout_split(train_df, n_test_timestamps=8)
    origin = train_df["timestamp"].min()
    train_part = prepare(train_part, origin)
    val_part = prepare(val_part, origin)

    preds: dict[str, np.ndarray] = {}
    preds["global_mean"] = build_global_mean(train_part, val_part)
    preds["route_mean"] = build_route_mean(train_part, val_part)
    preds["route_recent_7d"] = build_route_recent_mean(train_part, val_part, days=7)
    preds["route_recent_14d"] = build_route_recent_mean(train_part, val_part, days=14)
    preds["route_hour"] = build_route_slot_mean(train_part, val_part, ["route_id", "hour", "minute_30"])
    preds["route_dow"] = build_route_slot_mean(train_part, val_part, ["route_id", "dow"])
    preds["route_dow_hour"] = build_route_slot_mean(
        train_part,
        val_part,
        ["route_id", "dow", "hour", "minute_30"],
    )
    preds["recent_28d_route_dow_hour"] = build_recent_same_dow_hour_mean(train_part, val_part, days=28)
    preds["blend_recent14_route_dow_hour"] = build_blend(
        preds["route_recent_14d"],
        preds["route_dow_hour"],
        weights=[0.55, 0.45],
    )
    preds["blend_recent14_recent28slot"] = build_blend(
        preds["route_recent_14d"],
        preds["recent_28d_route_dow_hour"],
        weights=[0.6, 0.4],
    )
    preds["blend_mean_hour_005"] = build_blend(
        preds["route_mean"],
        preds["route_hour"],
        weights=[0.95, 0.05],
    )
    preds["blend_mean_dow_035"] = build_blend(
        preds["route_mean"],
        preds["route_dow"],
        weights=[0.65, 0.35],
    )
    preds["blend_mean_dowhour_005"] = build_blend(
        preds["route_mean"],
        preds["route_dow_hour"],
        weights=[0.95, 0.05],
    )
    preds["profile_x_scale_14d"] = build_profile_x_scale(train_part, val_part, days=14)

    y_true = val_part[TARGET_COL].to_numpy(dtype=float)
    results: list[BaselineResult] = []
    for name, y_pred in preds.items():
        y_pred = np.clip(y_pred, 0.0, None)
        parts = metric_frame(y_true=y_true, y_pred=y_pred)
        results.append(
            BaselineResult(
                name=name,
                total=parts["total"],
                wape=parts["wape"],
                rbias=parts["rbias"],
                rbias_signed=parts["rbias_signed"],
            )
        )
    return sorted(results, key=lambda x: x.total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="train_solo_track.parquet")
    args = parser.parse_args()

    train_df = load_train(args.train_path)
    results = evaluate(train_df)

    print("Baseline validation on the last 8 timestamps per route")
    print("-" * 80)
    for row in results:
        print(
            f"{row.name:<28} total={row.total:.6f} "
            f"wape={row.wape:.6f} rbias={row.rbias:.6f} rbias_signed={row.rbias_signed:+.6f}"
        )


if __name__ == "__main__":
    main()
