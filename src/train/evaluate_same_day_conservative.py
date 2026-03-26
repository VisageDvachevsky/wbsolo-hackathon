"""Evaluate conservative same-day calibrations for profile x scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.train.evaluate_baselines import build_profile_x_scale, load_train, metric_frame, prepare
from src.train.profile_x_scale_same_day import mask_midday_window, saturday_midday_dates
from src.train.same_day_calibration import (
    SameDayCalibrationConfig,
    apply_same_day_calibration,
    apply_segment_calibration,
    compute_global_multiplier,
    compute_route_multipliers_gamma_poisson,
    compute_segment_multipliers,
    mask_same_day_hours,
)


def evaluate(n_windows: int, cfg: SameDayCalibrationConfig) -> dict[str, object]:
    train_df = load_train()
    origin = train_df["timestamp"].min()
    train_feat = prepare(train_df.copy(), origin)
    dates = saturday_midday_dates(train_feat)[-n_windows:]

    rows = []
    for date in dates:
        val_df = train_feat.loc[mask_midday_window(train_feat, date)].copy()
        hist_df = train_feat.loc[train_feat["timestamp"] < val_df["timestamp"].min()].copy()
        obs_df = hist_df.loc[
            mask_same_day_hours(hist_df, date=date, start_hour=cfg.obs_start_hour, end_hour=cfg.obs_end_hour)
        ].copy()

        y_true = val_df["target_1h"].to_numpy(dtype=float)
        base_pred = build_profile_x_scale(hist_df, val_df, days=14)
        obs_base_pred = build_profile_x_scale(hist_df, obs_df, days=14)

        global_mult = compute_global_multiplier(obs_df, obs_base_pred, cfg)
        seg_mult, seg_default = compute_segment_multipliers(hist_df, obs_df, obs_base_pred, cfg, n_segments=3)
        route_mult, route_default = compute_route_multipliers_gamma_poisson(obs_df, obs_base_pred, cfg)

        global_pred = np.clip(base_pred * global_mult, 0.0, None)
        seg_pred = apply_segment_calibration(base_pred, val_df, hist_df, seg_mult, default_multiplier=seg_default, n_segments=3)
        route_pred = apply_same_day_calibration(base_pred, val_df, route_mult, default_multiplier=route_default)

        rows.append(
            {
                "date": str(date.date()),
                "profile_total": round(metric_frame(y_true, base_pred)["total"], 6),
                "global_total": round(metric_frame(y_true, global_pred)["total"], 6),
                "segment_total": round(metric_frame(y_true, seg_pred)["total"], 6),
                "route_total": round(metric_frame(y_true, route_pred)["total"], 6),
                "global_mult": round(float(global_mult), 6),
                "segment_default": round(float(seg_default), 6),
                "route_default": round(float(route_default), 6),
            }
        )

    summary = {}
    for key in ["profile_total", "global_total", "segment_total", "route_total"]:
        summary[f"mean_{key}"] = round(float(np.mean([row[key] for row in rows])), 6)

    return {
        "config": {
            "obs_start_hour": cfg.obs_start_hour,
            "obs_end_hour": cfg.obs_end_hour,
            "shrink_k": cfg.shrink_k,
            "clip_min": cfg.clip_min,
            "clip_max": cfg.clip_max,
            "n_windows": n_windows,
        },
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--obs-start-hour", type=int, default=7)
    parser.add_argument("--obs-end-hour", type=int, default=10)
    parser.add_argument("--shrink-k", type=float, default=2_000_000.0)
    parser.add_argument("--clip-min", type=float, default=0.85)
    parser.add_argument("--clip-max", type=float, default=1.15)
    parser.add_argument("--report-out", default="reports/same_day_conservative_eval.json")
    args = parser.parse_args()

    cfg = SameDayCalibrationConfig(
        obs_start_hour=args.obs_start_hour,
        obs_end_hour=args.obs_end_hour,
        shrink_k=args.shrink_k,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    payload = evaluate(n_windows=args.n_windows, cfg=cfg)
    Path(args.report_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
