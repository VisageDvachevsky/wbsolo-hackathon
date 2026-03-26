"""Evaluate LightGBM and residual LightGBM on Saturday test-like windows."""

from __future__ import annotations

import pandas as pd

from src.train.evaluate_baselines import TARGET_COL, build_profile_x_scale, load_train, prepare
from src.train.evaluate_recursive_models import (
    add_static_route_features,
    calibrate_to_recent_total,
    metric_row,
    recursive_predict,
    train_lgbm,
    train_residual_lgbm,
)


def saturday_windows(train_df: pd.DataFrame) -> list[pd.Timestamp]:
    mask = (
        (train_df["timestamp"].dt.dayofweek == 5)
        & (train_df["timestamp"].dt.hour.between(11, 14))
    )
    return [pd.Timestamp(d) for d in sorted(train_df.loc[mask, "timestamp"].dt.normalize().unique())]


def window_mask(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    return (
        (df["timestamp"].dt.normalize() == date)
        & (df["timestamp"].dt.hour.between(11, 14))
    )


def main() -> None:
    train_df = load_train()
    origin = train_df["timestamp"].min()
    windows = saturday_windows(train_df)
    print("Saturday 11:00-14:30 model comparison")
    print("-" * 120)
    for date in windows[-3:]:
        val_mask = window_mask(train_df, date)
        val_df = prepare(train_df.loc[val_mask].copy(), origin)
        hist_df = prepare(train_df.loc[train_df["timestamp"] < val_df["timestamp"].min()].copy(), origin)
        y_true = val_df[TARGET_COL].to_numpy(dtype=float)

        profile = build_profile_x_scale(hist_df, val_df, days=14)
        profile_total = float(metric_row("profile", y_true, profile)["total"])

        lgbm, _ = train_lgbm(hist_df)
        lgbm_pred = recursive_predict(lgbm, hist_df, val_df, "lgbm")
        lgbm_total = float(metric_row("lgbm", y_true, lgbm_pred)["total"])
        lgbm_cal_total = float(metric_row("lgbm_cal", y_true, calibrate_to_recent_total(lgbm_pred, hist_df, 14))["total"])

        residual_model, _ = train_residual_lgbm(hist_df)
        residual_component = recursive_predict(residual_model, hist_df, val_df, "lgbm")
        static_val = add_static_route_features(hist_df, val_df)
        prior = (
            static_val["route_mean_14d"].to_numpy(dtype=float)
            * (static_val["route_hour_mean"].to_numpy(dtype=float) / static_val["route_mean"].to_numpy(dtype=float))
        )
        prior = pd.Series(prior).replace([pd.NA, pd.NaT, float("inf"), float("-inf")], pd.NA).fillna(static_val["route_mean_14d"]).to_numpy(dtype=float)
        residual_pred = prior + residual_component
        residual_total = float(metric_row("residual", y_true, residual_pred)["total"])
        residual_cal_total = float(metric_row("residual_cal", y_true, calibrate_to_recent_total(residual_pred, hist_df, 14))["total"])

        print(
            f"{date.date()} | profile={profile_total:.6f} "
            f"lgbm={lgbm_total:.6f} lgbm_cal={lgbm_cal_total:.6f} "
            f"residual={residual_total:.6f} residual_cal={residual_cal_total:.6f}"
        )


if __name__ == "__main__":
    main()
