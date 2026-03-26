"""Evaluate recursive baselines and LightGBM variants."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.time_features import add_time_features
from src.validation.metrics import WapePlusRbias
from src.validation.split import time_holdout_split


TARGET_COL = "target_1h"
LAGS = [1, 2, 3, 4, 48, 336]
ROLLING_WINDOWS = [4, 8]


def load_train(path: str = "train_solo_track.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    metric = WapePlusRbias()
    parts = metric.calculate_components(y_true=y_true, y_pred=np.clip(y_pred, 0.0, None))
    parts["name"] = name
    return parts


def calibrate_to_recent_total(
    raw_pred: np.ndarray,
    train_df: pd.DataFrame,
    recent_days: int = 14,
) -> np.ndarray:
    recent = train_df.loc[train_df["timestamp"] > train_df["timestamp"].max() - pd.Timedelta(days=recent_days)]
    mean_per_timestamp = recent.groupby("timestamp")[TARGET_COL].sum().mean()
    expected_total = mean_per_timestamp * (len(raw_pred) / train_df["route_id"].nunique())
    pred_sum = float(raw_pred.sum())
    if pred_sum <= 0:
        return raw_pred
    return raw_pred * (expected_total / pred_sum)


def add_static_route_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    train_feat = add_time_features(train_df, timestamp_origin=train_df["timestamp"].min())
    target_feat = add_time_features(target_df, timestamp_origin=train_df["timestamp"].min())

    route_mean = (
        train_feat.groupby("route_id", as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "route_mean"})
    )
    route_recent_14d = (
        train_feat.loc[train_feat["timestamp"] > train_feat["timestamp"].max() - pd.Timedelta(days=14)]
        .groupby("route_id", as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "route_mean_14d"})
    )
    route_hour = (
        train_feat.groupby(["route_id", "hour", "minute_30"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "route_hour_mean"})
    )
    route_dow_hour = (
        train_feat.groupby(["route_id", "dow", "hour", "minute_30"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "route_dow_hour_mean"})
    )

    out = target_feat.merge(route_mean, on="route_id", how="left")
    out = out.merge(route_recent_14d, on="route_id", how="left")
    out = out.merge(route_hour, on=["route_id", "hour", "minute_30"], how="left")
    out = out.merge(route_dow_hour, on=["route_id", "dow", "hour", "minute_30"], how="left")
    out["route_mean_14d"] = out["route_mean_14d"].fillna(out["route_mean"])
    out["route_hour_mean"] = out["route_hour_mean"].fillna(out["route_mean"])
    out["route_dow_hour_mean"] = out["route_dow_hour_mean"].fillna(out["route_hour_mean"])
    return out


def build_training_matrix(train_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = add_static_route_features(train_df, train_df)
    grp = df.groupby("route_id", group_keys=False)
    for lag in LAGS:
        df[f"lag_{lag}"] = grp[TARGET_COL].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = grp[TARGET_COL].shift(1).rolling(window).mean().reset_index(level=0, drop=True)

    feature_cols = [
        "route_id",
        "hour",
        "minute_30",
        "dow",
        "is_weekend",
        "day_of_month",
        "week_of_year",
        "days_since_start",
        "route_mean",
        "route_mean_14d",
        "route_hour_mean",
        "route_dow_hour_mean",
        *[f"lag_{lag}" for lag in LAGS],
        *[f"roll_mean_{window}" for window in ROLLING_WINDOWS],
    ]
    model_df = df.dropna(subset=[f"lag_{lag}" for lag in LAGS]).copy()
    model_df["route_id"] = model_df["route_id"].astype("category")
    return model_df, feature_cols


def initial_histories(train_df: pd.DataFrame) -> dict[int, deque[float]]:
    histories: dict[int, deque[float]] = {}
    for route_id, grp in train_df.groupby("route_id"):
        histories[route_id] = deque(grp[TARGET_COL].tolist(), maxlen=400)
    return histories


def build_recursive_frame(
    static_df: pd.DataFrame,
    histories: dict[int, deque[float]],
    idx: pd.Index,
) -> pd.DataFrame:
    frame = static_df.loc[idx].copy()
    for lag in LAGS:
        frame[f"lag_{lag}"] = [
            hist[-lag] if len(hist) >= lag else np.nan
            for hist in (histories[rid] for rid in frame["route_id"])
        ]
    for window in ROLLING_WINDOWS:
        vals = []
        for rid in frame["route_id"]:
            hist = histories[rid]
            if len(hist) >= window:
                vals.append(float(np.mean(list(hist)[-window:])))
            else:
                vals.append(np.nan)
        frame[f"roll_mean_{window}"] = vals
    frame["route_id"] = frame["route_id"].astype("category")
    return frame


def recursive_predict(
    model: lgb.LGBMRegressor | None,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    strategy: str,
) -> np.ndarray:
    static_val = add_static_route_features(train_df, val_df)
    histories = initial_histories(train_df)
    preds = np.zeros(len(val_df), dtype=float)

    feature_cols = [
        "route_id",
        "hour",
        "minute_30",
        "dow",
        "is_weekend",
        "day_of_month",
        "week_of_year",
        "days_since_start",
        "route_mean",
        "route_mean_14d",
        "route_hour_mean",
        "route_dow_hour_mean",
        *[f"lag_{lag}" for lag in LAGS],
        *[f"roll_mean_{window}" for window in ROLLING_WINDOWS],
    ]

    ordered_timestamps = sorted(static_val["timestamp"].unique())
    for ts in ordered_timestamps:
        idx = static_val.index[static_val["timestamp"] == ts]
        step_df = build_recursive_frame(static_val, histories, idx)

        if strategy == "last_value":
            step_pred = step_df["lag_1"].to_numpy(dtype=float)
        elif strategy == "blend_last_slot":
            step_pred = (
                0.70 * step_df["lag_1"].to_numpy(dtype=float)
                + 0.20 * step_df["route_dow_hour_mean"].to_numpy(dtype=float)
                + 0.10 * step_df["route_mean_14d"].to_numpy(dtype=float)
            )
        elif strategy == "blend_last_roll":
            step_pred = (
                0.55 * step_df["lag_1"].to_numpy(dtype=float)
                + 0.25 * step_df["roll_mean_4"].fillna(step_df["lag_1"]).to_numpy(dtype=float)
                + 0.20 * step_df["route_hour_mean"].to_numpy(dtype=float)
            )
        elif strategy == "lgbm":
            assert model is not None
            step_pred = model.predict(step_df[feature_cols])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        step_pred = np.clip(step_pred, 0.0, None)
        preds[idx.to_numpy()] = step_pred
        for rid, pred in zip(step_df["route_id"].astype(int), step_pred):
            histories[rid].append(float(pred))

    return preds


def train_lgbm(train_df: pd.DataFrame) -> tuple[lgb.LGBMRegressor, list[str]]:
    model_df, feature_cols = build_training_matrix(train_df)
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        model_df[feature_cols],
        model_df[TARGET_COL],
        categorical_feature=["route_id"],
    )
    return model, feature_cols


def train_residual_lgbm(train_df: pd.DataFrame) -> tuple[lgb.LGBMRegressor, list[str]]:
    model_df, feature_cols = build_training_matrix(train_df)
    baseline = (
        model_df["route_mean_14d"].to_numpy(dtype=float)
        * (model_df["route_hour_mean"].to_numpy(dtype=float) / model_df["route_mean"].to_numpy(dtype=float))
    )
    baseline = np.where(
        np.isfinite(baseline),
        baseline,
        model_df["route_mean_14d"].to_numpy(dtype=float),
    )
    residual = model_df[TARGET_COL].to_numpy(dtype=float) - baseline

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        model_df[feature_cols],
        residual,
        categorical_feature=["route_id"],
    )
    return model, feature_cols


def evaluate(train_df: pd.DataFrame) -> list[dict[str, float | str]]:
    train_part, val_part = time_holdout_split(train_df, n_test_timestamps=8)
    y_true = val_part[TARGET_COL].to_numpy(dtype=float)

    rows: list[dict[str, float | str]] = []
    rows.append(metric_row("recursive_last_value", y_true, recursive_predict(None, train_part, val_part, "last_value")))
    rows.append(metric_row("recursive_blend_last_slot", y_true, recursive_predict(None, train_part, val_part, "blend_last_slot")))
    rows.append(metric_row("recursive_blend_last_roll", y_true, recursive_predict(None, train_part, val_part, "blend_last_roll")))

    model, _ = train_lgbm(train_part)
    raw_lgbm = recursive_predict(model, train_part, val_part, "lgbm")
    rows.append(metric_row("recursive_lgbm", y_true, raw_lgbm))
    rows.append(metric_row("recursive_lgbm_cal14d", y_true, calibrate_to_recent_total(raw_lgbm, train_part, recent_days=14)))

    residual_model, _ = train_residual_lgbm(train_part)
    residual_raw = recursive_predict(residual_model, train_part, val_part, "lgbm")
    static_val = add_static_route_features(train_part, val_part)
    profile_prior = (
        static_val["route_mean_14d"].to_numpy(dtype=float)
        * (static_val["route_hour_mean"].to_numpy(dtype=float) / static_val["route_mean"].to_numpy(dtype=float))
    )
    profile_prior = np.where(np.isfinite(profile_prior), profile_prior, static_val["route_mean_14d"].to_numpy(dtype=float))
    residual_pred = np.clip(profile_prior + residual_raw, 0.0, None)
    rows.append(metric_row("residual_lgbm_on_profile", y_true, residual_pred))
    rows.append(metric_row("residual_lgbm_cal14d", y_true, calibrate_to_recent_total(residual_pred, train_part, recent_days=14)))
    return sorted(rows, key=lambda x: float(x["total"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="train_solo_track.parquet")
    args = parser.parse_args()

    train_df = load_train(args.train_path)
    results = evaluate(train_df)
    print("Recursive validation on the last 8 timestamps per route")
    print("-" * 80)
    for row in results:
        print(
            f"{row['name']:<28} total={float(row['total']):.6f} "
            f"wape={float(row['wape']):.6f} rbias={float(row['rbias']):.6f} "
            f"rbias_signed={float(row['rbias_signed']):+.6f}"
        )


if __name__ == "__main__":
    main()
