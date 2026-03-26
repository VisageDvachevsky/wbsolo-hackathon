"""Train and evaluate a profile-x-scale baseline with residual LightGBM correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.time_features import add_time_features
from src.train.evaluate_baselines import TARGET_COL, build_profile_x_scale, load_train
from src.validation.metrics import WapePlusRbias


FEATURE_COLS = [
    "route_id",
    "hour",
    "minute_30",
    "dow",
    "is_weekend",
    "day_of_month",
    "week_of_year",
    "days_since_start",
    "route_mean_all",
    "route_mean_14d",
    "route_hour_mean",
    "route_dow_mean",
    "route_dow_hour_mean",
    "route_cv",
    "route_zero_frac",
    "status1_route_mean",
    "status2_route_mean",
    "status3_route_mean",
]


def saturday_midday_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    mask = (df["timestamp"].dt.dayofweek == 5) & (df["timestamp"].dt.hour.between(11, 14))
    return [pd.Timestamp(x) for x in sorted(df.loc[mask, "timestamp"].dt.normalize().unique())]


def saturday_window_mask(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    return (df["timestamp"].dt.normalize() == date) & (df["timestamp"].dt.hour.between(11, 14))


def add_route_reference_features(history_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    origin = history_df["timestamp"].min()
    hist = add_time_features(history_df.copy(), timestamp_origin=origin)
    target = add_time_features(target_df.copy(), timestamp_origin=origin)

    route_all = hist.groupby("route_id", as_index=False)[TARGET_COL].mean().rename(columns={TARGET_COL: "route_mean_all"})
    route_14d = (
        hist.loc[hist["timestamp"] > hist["timestamp"].max() - pd.Timedelta(days=14)]
        .groupby("route_id", as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "route_mean_14d"})
    )
    route_stats = hist.groupby("route_id", as_index=False)[TARGET_COL].agg(
        route_std="std",
        route_nonzero_frac=lambda x: (x > 0).mean(),
    )
    route_stats = route_stats.merge(route_all, on="route_id", how="left")
    route_stats["route_cv"] = route_stats["route_std"] / route_stats["route_mean_all"]
    route_stats = route_stats.rename(columns={"route_nonzero_frac": "route_zero_frac"})
    route_stats["route_zero_frac"] = 1.0 - route_stats["route_zero_frac"]

    route_hour = hist.groupby(["route_id", "hour", "minute_30"], as_index=False)[TARGET_COL].mean().rename(columns={TARGET_COL: "route_hour_mean"})
    route_dow = hist.groupby(["route_id", "dow"], as_index=False)[TARGET_COL].mean().rename(columns={TARGET_COL: "route_dow_mean"})
    route_dow_hour = hist.groupby(["route_id", "dow", "hour", "minute_30"], as_index=False)[TARGET_COL].mean().rename(columns={TARGET_COL: "route_dow_hour_mean"})

    status_route = hist.groupby("route_id", as_index=False).agg(
        status1_route_mean=("status_1", "mean"),
        status2_route_mean=("status_2", "mean"),
        status3_route_mean=("status_3", "mean"),
    )

    out = target.merge(route_all, on="route_id", how="left")
    out = out.merge(route_14d, on="route_id", how="left")
    out = out.merge(route_stats[["route_id", "route_cv", "route_zero_frac"]], on="route_id", how="left")
    out = out.merge(route_hour, on=["route_id", "hour", "minute_30"], how="left")
    out = out.merge(route_dow, on=["route_id", "dow"], how="left")
    out = out.merge(route_dow_hour, on=["route_id", "dow", "hour", "minute_30"], how="left")
    out = out.merge(status_route, on="route_id", how="left")

    out["route_mean_14d"] = out["route_mean_14d"].fillna(out["route_mean_all"])
    out["route_hour_mean"] = out["route_hour_mean"].fillna(out["route_mean_all"])
    out["route_dow_mean"] = out["route_dow_mean"].fillna(out["route_mean_all"])
    out["route_dow_hour_mean"] = out["route_dow_hour_mean"].fillna(out["route_hour_mean"])
    return out


def build_training_frame(history_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = add_route_reference_features(history_df, history_df)
    feat_df["profile_baseline"] = build_profile_x_scale(feat_df, feat_df, days=14)
    feat_df["residual_target"] = feat_df[TARGET_COL] - feat_df["profile_baseline"]
    feat_df["route_id"] = feat_df["route_id"].astype("category")
    return feat_df


def train_residual_model(history_df: pd.DataFrame) -> lgb.LGBMRegressor:
    train_df = build_training_frame(history_df)
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train_df[FEATURE_COLS],
        train_df["residual_target"],
        categorical_feature=["route_id"],
    )
    return model


def calibrate_with_recent_scale(pred: np.ndarray, history_df: pd.DataFrame, horizon_steps: int) -> np.ndarray:
    recent = history_df.loc[history_df["timestamp"] > history_df["timestamp"].max() - pd.Timedelta(days=14)]
    expected_total = recent.groupby("timestamp")[TARGET_COL].sum().mean() * horizon_steps
    pred_sum = float(pred.sum())
    if pred_sum <= 0:
        return pred
    return pred * (expected_total / pred_sum)


def predict_with_model(model: lgb.LGBMRegressor, history_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hist_feat = add_time_features(history_df.copy(), history_df["timestamp"].min())
    feat_df = add_route_reference_features(history_df, target_df)
    feat_df["profile_baseline"] = build_profile_x_scale(hist_feat, feat_df, days=14)
    feat_df["route_id"] = feat_df["route_id"].astype("category")
    residual_pred = model.predict(feat_df[FEATURE_COLS])
    raw_pred = np.clip(feat_df["profile_baseline"].to_numpy(dtype=float) + residual_pred, 0.0, None)
    calibrated = calibrate_with_recent_scale(raw_pred, history_df, horizon_steps=feat_df["timestamp"].nunique())
    return raw_pred, calibrated


def evaluate_last_saturdays(train_df: pd.DataFrame, n_windows: int = 4) -> list[dict[str, float | str]]:
    dates = saturday_midday_dates(train_df)[-n_windows:]
    metric = WapePlusRbias()
    rows: list[dict[str, float | str]] = []
    for date in dates:
        val_mask = saturday_window_mask(train_df, date)
        val_df = train_df.loc[val_mask].copy()
        history_df = train_df.loc[train_df["timestamp"] < val_df["timestamp"].min()].copy()
        model = train_residual_model(history_df)
        profile_only = build_profile_x_scale(
            add_time_features(history_df.copy(), history_df["timestamp"].min()),
            add_time_features(val_df.copy(), history_df["timestamp"].min()),
            days=14,
        )
        raw_pred, cal_pred = predict_with_model(model, history_df, val_df)
        y_true = val_df[TARGET_COL].to_numpy(dtype=float)
        rows.append(
            {
                "date": str(date.date()),
                "profile_total": float(metric.calculate(y_true, np.clip(profile_only, 0, None))),
                "residual_total": float(metric.calculate(y_true, raw_pred)),
                "residual_cal_total": float(metric.calculate(y_true, cal_pred)),
            }
        )
    return rows


def make_submission(model: lgb.LGBMRegressor, history_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path) -> None:
    _, pred = predict_with_model(model, history_df, test_df)
    out = pd.DataFrame({"id": test_df["id"].to_numpy(), "y_pred": pred})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="train_solo_track.parquet")
    parser.add_argument("--test-path", default="test_solo_track.parquet")
    parser.add_argument("--submission-out", default="submissions/profile_residual_lgbm.csv")
    parser.add_argument("--report-out", default="reports/profile_residual_lgbm_eval.json")
    args = parser.parse_args()

    train_df = load_train(args.train_path)
    test_df = pd.read_parquet(args.test_path)
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    eval_rows = evaluate_last_saturdays(train_df, n_windows=4)
    model = train_residual_model(train_df)
    make_submission(model, train_df, test_df, Path(args.submission_out))

    report = {"evaluation": eval_rows, "submission_out": args.submission_out}
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
