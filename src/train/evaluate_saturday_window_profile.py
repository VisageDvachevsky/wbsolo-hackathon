"""Evaluate window-profile-scale variants on Saturday test-like windows."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train.evaluate_baselines import build_profile_x_scale, load_train, metric_frame
from src.train.window_profile_scale import (
    WindowProfileConfig,
    calibrate_to_recent_saturday_total,
    predict_window_profile_scale,
    prepare_with_time,
    saturday_midday_dates,
    window_mask,
)


def evaluate_one_config(
    train_df: pd.DataFrame,
    config: WindowProfileConfig,
    n_windows: int = 6,
) -> dict[str, object]:
    origin = train_df["timestamp"].min()
    train_feat = prepare_with_time(train_df, origin)
    dates = saturday_midday_dates(train_feat)[-n_windows:]
    rows: list[dict[str, float | str]] = []

    for date in dates:
        val_mask = (train_feat["timestamp"].dt.normalize() == date) & window_mask(train_feat)
        val_df = train_feat.loc[val_mask].copy()
        history_df = train_feat.loc[train_feat["timestamp"] < val_df["timestamp"].min()].copy()
        y_true = val_df["target_1h"].to_numpy(dtype=float)

        base_profile = build_profile_x_scale(history_df, val_df, days=14)
        base_parts = metric_frame(y_true, base_profile)

        raw_pred = predict_window_profile_scale(history_df, val_df, config)
        raw_parts = metric_frame(y_true, raw_pred)

        cal_pred = calibrate_to_recent_saturday_total(
            raw_pred,
            history_df,
            val_df,
            n_saturdays=config.calibration_saturdays,
        )
        cal_parts = metric_frame(y_true, cal_pred)

        rows.append(
            {
                "date": str(date.date()),
                "profile_x_scale_total": round(float(base_parts["total"]), 6),
                "window_profile_total": round(float(raw_parts["total"]), 6),
                "window_profile_cal_total": round(float(cal_parts["total"]), 6),
                "window_profile_rbias": round(float(raw_parts["rbias"]), 6),
                "window_profile_cal_rbias": round(float(cal_parts["rbias"]), 6),
            }
        )

    profile_mean = float(np.mean([row["profile_x_scale_total"] for row in rows]))
    window_mean = float(np.mean([row["window_profile_total"] for row in rows]))
    calib_mean = float(np.mean([row["window_profile_cal_total"] for row in rows]))
    return {
        "config": {
            "scale_days": config.scale_days,
            "scale_mode": config.scale_mode,
            "shrink_k": config.shrink_k,
            "calibration_saturdays": config.calibration_saturdays,
        },
        "profile_x_scale_mean_total": round(profile_mean, 6),
        "window_profile_mean_total": round(window_mean, 6),
        "window_profile_cal_mean_total": round(calib_mean, 6),
        "rows": rows,
    }


def main() -> None:
    train_df = load_train()
    configs = [
        WindowProfileConfig(scale_days=days, scale_mode=mode, shrink_k=shrink, calibration_saturdays=4)
        for days, mode, shrink in itertools.product([14, 21, 28], ["median_day", "mean_day"], [15.0, 30.0, 60.0])
    ]

    results = [evaluate_one_config(train_df, cfg, n_windows=6) for cfg in configs]
    results.sort(key=lambda x: x["window_profile_cal_mean_total"])

    best = results[0]
    payload = {
        "best": best,
        "top5": results[:5],
    }
    out_path = Path("reports/window_profile_scale_eval.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
