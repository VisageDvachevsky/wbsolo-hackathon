"""Segmented same-day calibration experiment.

This experiment stays inside its own write scope:
- new code only in this file
- results only in reports/exp_segment_same_day_eval.json

It compares the current `profile_x_scale` baseline and the best-known
same-day route calibration against more stable segment / hierarchical
same-day calibration schemes:

- route-level same-day calibration
- volume-bucket same-day calibration
- CV-bucket same-day calibration
- hierarchical blend of route + segment + global factors

Evaluation is Saturday-first and uses the same-day morning window as the
calibration signal.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
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


WINDOW_START_HOUR = 11
WINDOW_END_HOUR = 14
DEFAULT_OBS_START_HOUR = 8
DEFAULT_OBS_END_HOUR = 10


@dataclass(frozen=True)
class SegmentConfig:
    name: str
    n_segments: int = 4
    shrink_k: float = 8_000_000.0
    clip_min: float = 0.98
    clip_max: float = 1.02


@dataclass(frozen=True)
class HierConfig:
    name: str
    route_blend_k: float = 6_000_000.0
    segment_blend_k: float = 6_000_000.0
    global_base_weight: float = 4.0
    route_base_weight: float = 1.0
    segment_base_weight: float = 1.0
    clip_min: float = 0.98
    clip_max: float = 1.02


def _saturday_midday_dates(train_feat: pd.DataFrame) -> list[pd.Timestamp]:
    mask = (train_feat["dow"] == 5) & train_feat["hour"].between(WINDOW_START_HOUR, WINDOW_END_HOUR)
    return [pd.Timestamp(x) for x in sorted(train_feat.loc[mask, "timestamp"].dt.normalize().unique())]


def _assign_volume_segments(history_df: pd.DataFrame, n_segments: int) -> pd.Series:
    route_mean = history_df.groupby("route_id")["target_1h"].mean()
    ranks = route_mean.rank(method="first")
    segments = pd.qcut(ranks, q=n_segments, labels=False, duplicates="drop")
    return segments.rename("volume_segment")


def _assign_cv_segments(history_df: pd.DataFrame, n_segments: int) -> pd.Series:
    stats = history_df.groupby("route_id")["target_1h"].agg(["mean", "std"])
    cv = stats["std"] / stats["mean"].replace(0, np.nan)
    cv = cv.replace([np.inf, -np.inf], np.nan)
    if cv.notna().any():
        cv = cv.fillna(float(cv.median()))
    else:
        cv = pd.Series(0.0, index=stats.index)
    ranks = cv.rank(method="first")
    segments = pd.qcut(ranks, q=n_segments, labels=False, duplicates="drop")
    return segments.rename("cv_segment")


def _group_support(obs_df: pd.DataFrame, obs_base_pred: np.ndarray, labels: pd.Series, group_col: str) -> pd.DataFrame:
    tmp = obs_df[["route_id"]].copy()
    tmp[group_col] = tmp["route_id"].map(labels)
    tmp["y_obs"] = obs_df["target_1h"].to_numpy(dtype=float)
    tmp["mu_obs"] = np.asarray(obs_base_pred, dtype=float)
    support = tmp.groupby(group_col, as_index=True).agg(y_sum=("y_obs", "sum"), mu_sum=("mu_obs", "sum"))
    support["support"] = support["mu_sum"].astype(float)
    return support


def _compute_group_multipliers(
    obs_df: pd.DataFrame,
    obs_base_pred: np.ndarray,
    labels: pd.Series,
    config: SameDayCalibrationConfig,
    group_col: str,
) -> tuple[pd.Series, float, pd.DataFrame]:
    if len(obs_df) == 0:
        return pd.Series(dtype=float), 1.0, pd.DataFrame(columns=["y_sum", "mu_sum", "support"])

    global_theta = compute_global_multiplier(obs_df, obs_base_pred, config)
    support = _group_support(obs_df, obs_base_pred, labels, group_col)
    k = float(config.shrink_k)
    multipliers = (support["y_sum"] + global_theta * k) / (support["mu_sum"] + k)
    multipliers = multipliers.clip(lower=config.clip_min, upper=config.clip_max)
    return multipliers.astype(float), global_theta, support


def _apply_group_calibration(
    base_pred: np.ndarray,
    target_df: pd.DataFrame,
    labels: pd.Series,
    multipliers: pd.Series,
    default_multiplier: float,
) -> np.ndarray:
    mapped = target_df["route_id"].map(labels).map(multipliers).fillna(default_multiplier).to_numpy(dtype=float)
    pred = np.asarray(base_pred, dtype=float) * mapped
    return np.clip(pred, 0.0, None)


def _apply_hierarchical_calibration(
    base_pred: np.ndarray,
    target_df: pd.DataFrame,
    route_multipliers: pd.Series,
    segment_multipliers: pd.Series,
    global_multiplier: float,
    route_support: pd.DataFrame,
    segment_support: pd.DataFrame,
    segment_labels: pd.Series,
    cfg: HierConfig,
) -> np.ndarray:
    route_route = target_df["route_id"].map(route_multipliers).fillna(global_multiplier).to_numpy(dtype=float)
    segment_route = target_df["route_id"].map(segment_labels).map(segment_multipliers).fillna(global_multiplier).to_numpy(dtype=float)

    route_mu = target_df["route_id"].map(route_support["support"]).fillna(0.0).to_numpy(dtype=float)
    segment_mu = target_df["route_id"].map(segment_labels).map(segment_support["support"]).fillna(0.0).to_numpy(dtype=float)

    route_w = cfg.route_base_weight * (route_mu / (route_mu + cfg.route_blend_k))
    segment_w = cfg.segment_base_weight * (segment_mu / (segment_mu + cfg.segment_blend_k))
    global_w = np.full(len(target_df), cfg.global_base_weight, dtype=float)

    eps = 1e-8
    route_log = np.log(np.clip(route_route, eps, None))
    segment_log = np.log(np.clip(segment_route, eps, None))
    global_log = np.log(np.clip(global_multiplier, eps, None))

    weights = route_w + segment_w + global_w
    blended_log = (route_w * route_log + segment_w * segment_log + global_w * global_log) / np.maximum(weights, eps)
    blended = np.exp(blended_log)
    blended = np.clip(blended, cfg.clip_min, cfg.clip_max)
    pred = np.asarray(base_pred, dtype=float) * blended
    return np.clip(pred, 0.0, None)


def _evaluate_window(
    hist_df: pd.DataFrame,
    val_df: pd.DataFrame,
    route_cfg: SameDayCalibrationConfig,
    segment_cfg: SegmentConfig,
    hier_cfg: HierConfig,
    segment_kind: str,
) -> dict[str, float]:
    base_pred = build_profile_x_scale(hist_df, val_df, days=14)
    obs_df = hist_df.loc[
        mask_same_day_hours(hist_df, date=val_df["timestamp"].dt.normalize().iloc[0], start_hour=route_cfg.obs_start_hour, end_hour=route_cfg.obs_end_hour)
    ].copy()
    obs_base_pred = build_profile_x_scale(hist_df, obs_df, days=14)

    y_true = val_df["target_1h"].to_numpy(dtype=float)

    route_mult, route_default = compute_route_multipliers_gamma_poisson(obs_df, obs_base_pred, route_cfg)
    route_pred = apply_same_day_calibration(base_pred, val_df, route_mult, default_multiplier=route_default)

    global_mult = compute_global_multiplier(obs_df, obs_base_pred, route_cfg)
    global_pred = np.clip(base_pred * global_mult, 0.0, None)

    if segment_kind == "volume":
        segment_labels = _assign_volume_segments(hist_df, segment_cfg.n_segments)
        segment_group_col = "volume_segment"
    elif segment_kind == "cv":
        segment_labels = _assign_cv_segments(hist_df, segment_cfg.n_segments)
        segment_group_col = "cv_segment"
    else:
        raise ValueError(f"unknown segment_kind: {segment_kind}")

    segment_mult, segment_default, segment_support = _compute_group_multipliers(
        obs_df=obs_df,
        obs_base_pred=obs_base_pred,
        labels=segment_labels,
        config=SameDayCalibrationConfig(
            obs_start_hour=route_cfg.obs_start_hour,
            obs_end_hour=route_cfg.obs_end_hour,
            shrink_k=segment_cfg.shrink_k,
            clip_min=segment_cfg.clip_min,
            clip_max=segment_cfg.clip_max,
        ),
        group_col=segment_group_col,
    )
    segment_pred = _apply_group_calibration(base_pred, val_df, segment_labels, segment_mult, segment_default)

    # Hierarchical blend uses the route-specific and segment-specific factors.
    route_support = _group_support(obs_df, obs_base_pred, pd.Series(index=obs_df["route_id"].unique(), data=obs_df["route_id"].unique()), "route_id")
    # the route mapping above is intentionally identity-like; group_support only needs support by route_id
    route_support.index.name = "route_id"

    hier_pred = _apply_hierarchical_calibration(
        base_pred=base_pred,
        target_df=val_df,
        route_multipliers=route_mult,
        segment_multipliers=segment_mult,
        global_multiplier=global_mult,
        route_support=route_support,
        segment_support=segment_support,
        segment_labels=segment_labels,
        cfg=hier_cfg,
    )

    route_parts = metric_frame(y_true, route_pred)
    global_parts = metric_frame(y_true, global_pred)
    segment_parts = metric_frame(y_true, segment_pred)
    hier_parts = metric_frame(y_true, hier_pred)
    base_parts = metric_frame(y_true, base_pred)

    return {
        "profile_total": float(base_parts["total"]),
        "profile_wape": float(base_parts["wape"]),
        "profile_rbias": float(base_parts["rbias"]),
        "profile_rbias_signed": float(base_parts["rbias_signed"]),
        "global_total": float(global_parts["total"]),
        "global_wape": float(global_parts["wape"]),
        "global_rbias": float(global_parts["rbias"]),
        "global_rbias_signed": float(global_parts["rbias_signed"]),
        "route_total": float(route_parts["total"]),
        "route_wape": float(route_parts["wape"]),
        "route_rbias": float(route_parts["rbias"]),
        "route_rbias_signed": float(route_parts["rbias_signed"]),
        f"{segment_kind}_total": float(segment_parts["total"]),
        f"{segment_kind}_wape": float(segment_parts["wape"]),
        f"{segment_kind}_rbias": float(segment_parts["rbias"]),
        f"{segment_kind}_rbias_signed": float(segment_parts["rbias_signed"]),
        f"hier_{segment_kind}_total": float(hier_parts["total"]),
        f"hier_{segment_kind}_wape": float(hier_parts["wape"]),
        f"hier_{segment_kind}_rbias": float(hier_parts["rbias"]),
        f"hier_{segment_kind}_rbias_signed": float(hier_parts["rbias_signed"]),
    }


def evaluate(n_windows: int) -> dict[str, object]:
    train_df = load_train()
    origin = train_df["timestamp"].min()
    train_feat = prepare(train_df.copy(), origin)
    dates = _saturday_midday_dates(train_feat)[-n_windows:]

    route_cfg = SameDayCalibrationConfig(
        obs_start_hour=DEFAULT_OBS_START_HOUR,
        obs_end_hour=DEFAULT_OBS_END_HOUR,
        shrink_k=2_000_000.0,
        clip_min=0.95,
        clip_max=1.05,
    )
    segment_cfg = SegmentConfig(name="segment", n_segments=4, shrink_k=8_000_000.0, clip_min=0.98, clip_max=1.02)
    hier_cfg = HierConfig(name="hier", route_blend_k=6_000_000.0, segment_blend_k=6_000_000.0, global_base_weight=4.0)

    rows: list[dict[str, float | str]] = []
    for date in dates:
        val_df = train_feat.loc[mask_midday_window(train_feat, date)].copy()
        hist_df = train_feat.loc[train_feat["timestamp"] < val_df["timestamp"].min()].copy()

        volume_row = _evaluate_window(hist_df, val_df, route_cfg, segment_cfg, hier_cfg, segment_kind="volume")
        cv_row = _evaluate_window(hist_df, val_df, route_cfg, segment_cfg, hier_cfg, segment_kind="cv")

        row = {
            "date": str(date.date()),
            **{f"volume_{k}": v for k, v in volume_row.items()},
            **{f"cv_{k}": v for k, v in cv_row.items()},
        }
        rows.append(row)

    def mean_of(key: str) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else float("nan")

    summary = {
        "mean_profile_total": mean_of("volume_profile_total"),
        "mean_route_total": mean_of("volume_route_total"),
        "mean_global_total": mean_of("volume_global_total"),
        "mean_volume_segment_total": mean_of("volume_volume_total"),
        "mean_volume_hier_total": mean_of("volume_hier_volume_total"),
        "mean_cv_segment_total": mean_of("cv_cv_total"),
        "mean_cv_hier_total": mean_of("cv_hier_cv_total"),
    }

    best_candidates = {
        "route_total": summary["mean_route_total"],
        "volume_segment_total": summary["mean_volume_segment_total"],
        "volume_hier_total": summary["mean_volume_hier_total"],
        "cv_segment_total": summary["mean_cv_segment_total"],
        "cv_hier_total": summary["mean_cv_hier_total"],
    }
    best_name = min(best_candidates, key=best_candidates.get)

    return {
        "config": {
            "n_windows": n_windows,
            "obs_start_hour": route_cfg.obs_start_hour,
            "obs_end_hour": route_cfg.obs_end_hour,
            "route_shrink_k": route_cfg.shrink_k,
            "route_clip_min": route_cfg.clip_min,
            "route_clip_max": route_cfg.clip_max,
            "segment_cfg": asdict(segment_cfg),
            "hier_cfg": asdict(hier_cfg),
        },
        "summary": summary,
        "best_candidate": {"name": best_name, "mean_total": float(best_candidates[best_name])},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--report-out", default="reports/exp_segment_same_day_eval.json")
    args = parser.parse_args()

    payload = evaluate(n_windows=args.n_windows)
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
