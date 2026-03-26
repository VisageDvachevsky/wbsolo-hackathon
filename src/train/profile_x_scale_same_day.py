"""Evaluate and run same-day calibrated profile x scale forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.time_features import add_time_features
from src.train.evaluate_baselines import TARGET_COL, build_profile_x_scale, load_train, metric_frame, prepare
from src.train.same_day_calibration import (
    SameDayCalibrationConfig,
    apply_same_day_calibration,
    compute_route_multipliers_gamma_poisson,
    mask_same_day_hours,
)


WINDOW_HOURS = (11, 14)


def saturday_midday_dates(train_feat: pd.DataFrame) -> list[pd.Timestamp]:
    mask = (train_feat["dow"] == 5) & train_feat["hour"].between(WINDOW_HOURS[0], WINDOW_HOURS[1])
    return [pd.Timestamp(x) for x in sorted(train_feat.loc[mask, "timestamp"].dt.normalize().unique())]


def mask_midday_window(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    return (df["timestamp"].dt.normalize() == date) & df["hour"].between(WINDOW_HOURS[0], WINDOW_HOURS[1])


def evaluate_last_saturdays(
    train_df: pd.DataFrame,
    n_windows: int,
    cfg: SameDayCalibrationConfig,
) -> dict[str, object]:
    origin = train_df["timestamp"].min()
    train_feat = prepare(train_df.copy(), origin)
    dates = saturday_midday_dates(train_feat)[-n_windows:]
    rows: list[dict[str, float | str]] = []

    for date in dates:
        val_mask = mask_midday_window(train_feat, date)
        val_df = train_feat.loc[val_mask].copy()
        hist_df = train_feat.loc[train_feat["timestamp"] < val_df["timestamp"].min()].copy()

        base_pred_val = build_profile_x_scale(hist_df, val_df, days=14)

        obs_mask = mask_same_day_hours(
            hist_df,
            date=date,
            start_hour=cfg.obs_start_hour,
            end_hour=cfg.obs_end_hour,
        )
        obs_df = hist_df.loc[obs_mask].copy()
        obs_base_pred = build_profile_x_scale(hist_df, obs_df, days=14)
        route_mult, global_theta = compute_route_multipliers_gamma_poisson(
            obs_df=obs_df,
            obs_base_pred=obs_base_pred,
            config=cfg,
        )

        same_day_pred = apply_same_day_calibration(
            base_pred=base_pred_val,
            target_df=val_df,
            route_multipliers=route_mult,
            default_multiplier=float(np.clip(global_theta, cfg.clip_min, cfg.clip_max)),
        )

        y_true = val_df[TARGET_COL].to_numpy(dtype=float)
        base_parts = metric_frame(y_true=y_true, y_pred=base_pred_val)
        same_parts = metric_frame(y_true=y_true, y_pred=same_day_pred)

        rows.append(
            {
                "date": str(date.date()),
                "mean_target": float(val_df[TARGET_COL].mean()),
                "profile_total": float(base_parts["total"]),
                "profile_wape": float(base_parts["wape"]),
                "profile_rbias": float(base_parts["rbias"]),
                "profile_rbias_signed": float(base_parts["rbias_signed"]),
                "same_day_total": float(same_parts["total"]),
                "same_day_wape": float(same_parts["wape"]),
                "same_day_rbias": float(same_parts["rbias"]),
                "same_day_rbias_signed": float(same_parts["rbias_signed"]),
                "global_theta_obs": float(global_theta),
                "n_route_multipliers": int(route_mult.shape[0]),
            }
        )

    profile_mean = float(np.mean([row["profile_total"] for row in rows])) if rows else float("nan")
    same_day_mean = float(np.mean([row["same_day_total"] for row in rows])) if rows else float("nan")

    return {
        "config": {
            "obs_start_hour": cfg.obs_start_hour,
            "obs_end_hour": cfg.obs_end_hour,
            "shrink_k": cfg.shrink_k,
            "clip_min": cfg.clip_min,
            "clip_max": cfg.clip_max,
        },
        "summary": {
            "n_windows": int(n_windows),
            "profile_mean_total": profile_mean,
            "same_day_mean_total": same_day_mean,
            "delta_mean_total": float(profile_mean - same_day_mean),
        },
        "evaluation": rows,
    }


def make_submission(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: SameDayCalibrationConfig,
    output_path: Path,
) -> None:
    origin = train_df["timestamp"].min()
    train_feat = prepare(train_df.copy(), origin)
    test_feat = prepare(test_df.copy(), origin)

    base_pred_test = build_profile_x_scale(train_feat, test_feat, days=14)
    test_day = pd.Timestamp(test_feat["timestamp"].min()).normalize()

    obs_mask = mask_same_day_hours(
        train_feat,
        date=test_day,
        start_hour=cfg.obs_start_hour,
        end_hour=cfg.obs_end_hour,
    )
    obs_df = train_feat.loc[obs_mask].copy()
    obs_base_pred = build_profile_x_scale(train_feat, obs_df, days=14)
    route_mult, global_theta = compute_route_multipliers_gamma_poisson(
        obs_df=obs_df,
        obs_base_pred=obs_base_pred,
        config=cfg,
    )

    pred_test = apply_same_day_calibration(
        base_pred=base_pred_test,
        target_df=test_feat,
        route_multipliers=route_mult,
        default_multiplier=float(np.clip(global_theta, cfg.clip_min, cfg.clip_max)),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_df["id"].to_numpy(), "y_pred": pred_test}).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="train_solo_track.parquet")
    parser.add_argument("--test-path", default="test_solo_track.parquet")
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--obs-start-hour", type=int, default=7)
    parser.add_argument("--obs-end-hour", type=int, default=10)
    parser.add_argument("--shrink-k", type=float, default=2_000_000.0)
    parser.add_argument("--clip-min", type=float, default=0.70)
    parser.add_argument("--clip-max", type=float, default=1.30)
    parser.add_argument("--report-out", default="reports/profile_x_scale_same_day_eval.json")
    parser.add_argument("--submission-out", default="submissions/profile_x_scale_same_day.csv")
    args = parser.parse_args()

    train_df = load_train(args.train_path)
    test_df = pd.read_parquet(args.test_path)
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    cfg = SameDayCalibrationConfig(
        obs_start_hour=args.obs_start_hour,
        obs_end_hour=args.obs_end_hour,
        shrink_k=args.shrink_k,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    report = evaluate_last_saturdays(train_df, n_windows=args.n_windows, cfg=cfg)
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    make_submission(train_df, test_df, cfg=cfg, output_path=Path(args.submission_out))
    print(f"Saved submission to: {args.submission_out}")


if __name__ == "__main__":
    main()
