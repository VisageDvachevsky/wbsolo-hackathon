"""Blend route-level and global same-day calibration for profile_x_scale.

This experiment searches for a conservative blend between:
- route-level same-day multiplier
- global same-day multiplier

The base forecast stays the same: profile_x_scale(days=14).
Only the multiplicative same-day calibration is blended.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train.evaluate_baselines import build_profile_x_scale, load_train, metric_frame, prepare
from src.train.profile_x_scale_same_day import mask_midday_window, saturday_midday_dates
from src.train.same_day_calibration import (
    SameDayCalibrationConfig,
    apply_same_day_calibration,
    compute_global_multiplier,
    compute_route_multipliers_gamma_poisson,
    mask_same_day_hours,
)


def build_blended_prediction(
    base_pred: np.ndarray,
    route_pred: np.ndarray,
    global_pred: np.ndarray,
    mode: str,
    route_weight: float,
) -> np.ndarray:
    """Blend route and global same-day predictions."""
    global_weight = 1.0 - route_weight

    if mode == "linear":
        pred = route_weight * route_pred + global_weight * global_pred
        return np.clip(pred, 0.0, None)

    if mode == "geo":
        eps = 1e-9
        route_factor = np.clip(route_pred / np.clip(base_pred, eps, None), eps, None)
        global_factor = np.clip(global_pred / np.clip(base_pred, eps, None), eps, None)
        factor = np.exp(route_weight * np.log(route_factor) + global_weight * np.log(global_factor))
        return np.clip(base_pred * factor, 0.0, None)

    raise ValueError(f"unknown blend mode: {mode}")


def evaluate_grid(
    train_feat: pd.DataFrame,
    n_windows: int,
    cfg: SameDayCalibrationConfig,
    route_weights: list[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    dates = saturday_midday_dates(train_feat)[-n_windows:]
    windows: list[dict[str, object]] = []

    for date in dates:
        val_df = train_feat.loc[mask_midday_window(train_feat, date)].copy()
        hist_df = train_feat.loc[train_feat["timestamp"] < val_df["timestamp"].min()].copy()
        obs_df = hist_df.loc[
            mask_same_day_hours(hist_df, date=date, start_hour=cfg.obs_start_hour, end_hour=cfg.obs_end_hour)
        ].copy()

        y_true = val_df["target_1h"].to_numpy(dtype=float)
        base_pred = build_profile_x_scale(hist_df, val_df, days=14)
        obs_base_pred = build_profile_x_scale(hist_df, obs_df, days=14)

        route_mult, route_default = compute_route_multipliers_gamma_poisson(obs_df, obs_base_pred, cfg)
        global_mult = compute_global_multiplier(obs_df, obs_base_pred, cfg)

        route_pred = apply_same_day_calibration(base_pred, val_df, route_mult, default_multiplier=route_default)
        global_pred = np.clip(base_pred * global_mult, 0.0, None)
        base_parts = metric_frame(y_true, base_pred)
        route_parts = metric_frame(y_true, route_pred)
        global_parts = metric_frame(y_true, global_pred)

        windows.append(
            {
                "date": str(date.date()),
                "y_true": y_true,
                "base_pred": base_pred,
                "route_pred": route_pred,
                "global_pred": global_pred,
                "profile_total": float(base_parts["total"]),
                "route_total": float(route_parts["total"]),
                "global_total": float(global_parts["total"]),
                "global_multiplier": float(global_mult),
                "route_default": float(route_default),
            }
        )

    results: list[dict[str, object]] = []
    for mode in ["linear", "geo"]:
        for route_weight in route_weights:
            totals = []
            last2 = []
            per_date: list[dict[str, float | str]] = []
            for window in windows:
                pred = build_blended_prediction(
                    base_pred=window["base_pred"],
                    route_pred=window["route_pred"],
                    global_pred=window["global_pred"],
                    mode=mode,
                    route_weight=route_weight,
                )
                parts = metric_frame(window["y_true"], pred)
                totals.append(float(parts["total"]))
                if window["date"] in {"2025-10-18", "2025-10-25"}:
                    last2.append(float(parts["total"]))
                per_date.append(
                    {
                        "date": str(window["date"]),
                        "total": float(parts["total"]),
                        "wape": float(parts["wape"]),
                        "rbias": float(parts["rbias"]),
                        "rbias_signed": float(parts["rbias_signed"]),
                    }
                )

            results.append(
                {
                    "mode": mode,
                    "route_weight": float(route_weight),
                    "global_weight": float(1.0 - route_weight),
                    "mean_total": float(np.mean(totals)),
                    "last2_mean_total": float(np.mean(last2)),
                    "max_total": float(np.max(totals)),
                    "per_date": per_date,
                }
            )

    best_mean = min(results, key=lambda row: row["mean_total"])
    best_last2 = min(results, key=lambda row: row["last2_mean_total"])

    payload = {
        "config": {
            "n_windows": int(n_windows),
            "obs_start_hour": cfg.obs_start_hour,
            "obs_end_hour": cfg.obs_end_hour,
            "shrink_k": cfg.shrink_k,
            "clip_min": cfg.clip_min,
            "clip_max": cfg.clip_max,
            "route_weights": route_weights,
        },
        "baseline_summary": {
            "profile_mean_total": float(np.mean([window["profile_total"] for window in windows])),
            "route_mean_total": float(np.mean([window["route_total"] for window in windows])),
            "global_mean_total": float(np.mean([window["global_total"] for window in windows])),
            "profile_last2_mean_total": float(
                np.mean([window["profile_total"] for window in windows if window["date"] in {"2025-10-18", "2025-10-25"}])
            ),
            "route_last2_mean_total": float(
                np.mean([window["route_total"] for window in windows if window["date"] in {"2025-10-18", "2025-10-25"}])
            ),
            "global_last2_mean_total": float(
                np.mean([window["global_total"] for window in windows if window["date"] in {"2025-10-18", "2025-10-25"}])
            ),
        },
        "best_by_mean": best_mean,
        "best_by_last2": best_last2,
        "windows": [
            {
                "date": window["date"],
                "profile_total": window["profile_total"],
                "route_total": window["route_total"],
                "global_total": window["global_total"],
                "global_multiplier": window["global_multiplier"],
                "route_default": window["route_default"],
            }
            for window in windows
        ],
        "all_results": [{k: v for k, v in row.items() if k != "per_date"} for row in results],
    }
    return payload, windows


def make_submission(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    cfg: SameDayCalibrationConfig,
    mode: str,
    route_weight: float,
    out_path: Path,
) -> None:
    base_pred = build_profile_x_scale(train_feat, test_feat, days=14)
    test_day = pd.Timestamp(test_feat["timestamp"].min()).normalize()
    obs_df = train_feat.loc[
        mask_same_day_hours(train_feat, date=test_day, start_hour=cfg.obs_start_hour, end_hour=cfg.obs_end_hour)
    ].copy()
    obs_base_pred = build_profile_x_scale(train_feat, obs_df, days=14)
    route_mult, route_default = compute_route_multipliers_gamma_poisson(obs_df, obs_base_pred, cfg)
    global_mult = compute_global_multiplier(obs_df, obs_base_pred, cfg)

    route_pred = apply_same_day_calibration(base_pred, test_feat, route_mult, default_multiplier=route_default)
    global_pred = np.clip(base_pred * global_mult, 0.0, None)
    pred = build_blended_prediction(
        base_pred=base_pred,
        route_pred=route_pred,
        global_pred=global_pred,
        mode=mode,
        route_weight=route_weight,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_feat["id"].to_numpy(), "y_pred": pred}).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="train_solo_track.parquet")
    parser.add_argument("--test-path", default="test_solo_track.parquet")
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--obs-start-hour", type=int, default=8)
    parser.add_argument("--obs-end-hour", type=int, default=10)
    parser.add_argument("--shrink-k", type=float, default=2_000_000.0)
    parser.add_argument("--clip-min", type=float, default=0.95)
    parser.add_argument("--clip-max", type=float, default=1.05)
    parser.add_argument("--report-out", default="reports/exp_blended_same_day_eval.json")
    parser.add_argument("--submission-out", default="submissions/exp_blended_same_day.csv")
    args = parser.parse_args()

    train_df = load_train(args.train_path)
    origin = train_df["timestamp"].min()
    train_feat = prepare(train_df.copy(), origin)

    cfg = SameDayCalibrationConfig(
        obs_start_hour=args.obs_start_hour,
        obs_end_hour=args.obs_end_hour,
        shrink_k=args.shrink_k,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    route_weights = [round(weight, 2) for weight in np.arange(0.70, 1.001, 0.01)]
    payload, _ = evaluate_grid(train_feat, n_windows=args.n_windows, cfg=cfg, route_weights=route_weights)

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["baseline_summary"], indent=2))
    print(json.dumps({"best_by_mean": payload["best_by_mean"], "best_by_last2": payload["best_by_last2"]}, indent=2))

    test_df = pd.read_parquet(args.test_path)
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    test_feat = prepare(test_df.copy(), origin)
    chosen = payload["best_by_last2"]
    make_submission(
        train_feat=train_feat,
        test_feat=test_feat,
        cfg=cfg,
        mode=str(chosen["mode"]),
        route_weight=float(chosen["route_weight"]),
        out_path=Path(args.submission_out),
    )
    print(f"Saved submission to: {args.submission_out}")


if __name__ == "__main__":
    main()
