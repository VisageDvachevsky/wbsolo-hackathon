"""Morning aggregate expected_total calibration experiments.

This script evaluates whether a smarter global scale estimated from a same-day
morning window can improve the current Saturday-like baselines.

It is intentionally self-contained and only writes new exp_total_* artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train.evaluate_baselines import TARGET_COL, build_profile_x_scale, load_train, metric_frame, prepare
from src.train.profile_x_scale_same_day import mask_midday_window, saturday_midday_dates
from src.train.same_day_calibration import (
    SameDayCalibrationConfig,
    apply_same_day_calibration,
    compute_route_multipliers_gamma_poisson,
    mask_same_day_hours,
)


MORNING_START_HOUR = 8
MORNING_END_HOUR = 10


def _safe_log_ratio(value: float, eps: float = 1e-9) -> float:
    return float(np.log(max(value, eps)))


def _fit_log_linear(x: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1:
        return float(y[0]), 0.0
    X = np.column_stack([np.ones_like(x), x])
    xtx = X.T @ X
    xtx += ridge * np.eye(2, dtype=float)
    beta = np.linalg.solve(xtx, X.T @ y)
    return float(beta[0]), float(beta[1])


def _fit_log_linear_multi(x: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return np.zeros(x.shape[1] + 1, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] == 1:
        return np.concatenate([[float(y[0])], np.zeros(x.shape[1], dtype=float)])

    X = np.column_stack([np.ones(x.shape[0]), x])
    xtx = X.T @ X
    xtx += ridge * np.eye(X.shape[1], dtype=float)
    beta = np.linalg.solve(xtx, X.T @ y)
    return beta.astype(float)


def _predict_factor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    x_value: float,
    min_factor: float,
    max_factor: float,
    shrink_strength: float = 4.0,
) -> dict[str, float]:
    """Return global, direct, logreg, and shrink-to-global factors."""
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)

    if train_y.size == 0:
        return {
            "global_factor": 1.0,
            "direct_factor": 1.0,
            "logreg_factor": 1.0,
            "shrink_factor": 1.0,
            "n_train": 0.0,
        }

    global_factor = float(np.exp(np.mean(train_y)))
    direct_factor = float(np.clip(np.exp(x_value), min_factor, max_factor))

    if train_x.size < 2:
        logreg_log_factor = np.log(global_factor)
    else:
        intercept, slope = _fit_log_linear(train_x, train_y)
        logreg_log_factor = intercept + slope * x_value
    logreg_factor = float(np.clip(np.exp(logreg_log_factor), min_factor, max_factor))

    weight = float(shrink_strength / (shrink_strength + train_y.size))
    shrink_log_factor = (1.0 - weight) * np.log(logreg_factor) + weight * np.log(global_factor)
    shrink_factor = float(np.clip(np.exp(shrink_log_factor), min_factor, max_factor))

    return {
        "global_factor": global_factor,
        "direct_factor": direct_factor,
        "logreg_factor": logreg_factor,
        "shrink_factor": shrink_factor,
        "n_train": float(train_y.size),
    }


def _predict_totalreg_factor(
    train_features: np.ndarray,
    train_y: np.ndarray,
    x_features: np.ndarray,
    min_factor: float,
    max_factor: float,
    shrink_strength: float = 4.0,
) -> dict[str, float]:
    """Predict factor from morning aggregate features directly."""
    train_features = np.asarray(train_features, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    x_features = np.asarray(x_features, dtype=float)

    if train_y.size == 0:
        return {
            "global_factor": 1.0,
            "direct_factor": 1.0,
            "logreg_factor": 1.0,
            "shrink_factor": 1.0,
            "n_train": 0.0,
        }
    if train_features.ndim == 1:
        train_features = train_features.reshape(-1, 1)
    if x_features.ndim == 1:
        x_features = x_features.reshape(1, -1)

    global_factor = float(np.exp(np.mean(train_y)))

    if train_features.shape[0] < 2:
        logreg_log_factor = np.log(global_factor)
    else:
        beta = _fit_log_linear_multi(train_features, train_y)
        x_vec = np.concatenate([[1.0], x_features.ravel()])
        logreg_log_factor = float(x_vec @ beta)

    logreg_factor = float(np.clip(np.exp(logreg_log_factor), min_factor, max_factor))
    weight = float(shrink_strength / (shrink_strength + train_y.size))
    shrink_log_factor = (1.0 - weight) * np.log(logreg_factor) + weight * np.log(global_factor)
    shrink_factor = float(np.clip(np.exp(shrink_log_factor), min_factor, max_factor))

    return {
        "global_factor": global_factor,
        "direct_factor": logreg_factor,
        "logreg_factor": logreg_factor,
        "shrink_factor": shrink_factor,
        "n_train": float(train_y.size),
    }


def _same_day_window_mask(df: pd.DataFrame, date: pd.Timestamp, start_hour: int, end_hour: int) -> pd.Series:
    return (df["timestamp"].dt.normalize() == date) & df["hour"].between(start_hour, end_hour)


def _window_history(df: pd.DataFrame, start_ts: pd.Timestamp) -> pd.DataFrame:
    return df.loc[df["timestamp"] < start_ts].copy()


def _build_records(train_feat: pd.DataFrame, dates: list[pd.Timestamp], route_cfg: SameDayCalibrationConfig) -> list[dict]:
    records: list[dict] = []
    for date in dates:
        morning_mask = _same_day_window_mask(train_feat, date, MORNING_START_HOUR, MORNING_END_HOUR)
        midday_mask = mask_midday_window(train_feat, date)

        morning_df = train_feat.loc[morning_mask].copy()
        midday_df = train_feat.loc[midday_mask].copy()

        morning_hist = _window_history(train_feat, morning_df["timestamp"].min())
        midday_hist = _window_history(train_feat, midday_df["timestamp"].min())

        morning_profile_pred = build_profile_x_scale(morning_hist, morning_df, days=14)
        morning_total = float(morning_df[TARGET_COL].sum())
        morning_profile_total = float(np.sum(morning_profile_pred))
        morning_ratio = morning_total / morning_profile_total if morning_profile_total > 0 else 1.0

        profile_base_pred = build_profile_x_scale(midday_hist, midday_df, days=14)
        profile_base_total = float(np.sum(profile_base_pred))
        midday_total = float(midday_df[TARGET_COL].sum())
        profile_factor = midday_total / profile_base_total if profile_base_total > 0 else 1.0
        profile_total = float(metric_frame(midday_df[TARGET_COL].to_numpy(dtype=float), profile_base_pred)["total"])

        obs_mask = mask_same_day_hours(
            midday_hist,
            date=date,
            start_hour=route_cfg.obs_start_hour,
            end_hour=route_cfg.obs_end_hour,
        )
        obs_df = midday_hist.loc[obs_mask].copy()
        obs_base_pred = build_profile_x_scale(midday_hist, obs_df, days=14)
        route_mult, global_theta = compute_route_multipliers_gamma_poisson(
            obs_df=obs_df,
            obs_base_pred=obs_base_pred,
            config=route_cfg,
        )
        route_base_pred = apply_same_day_calibration(
            base_pred=profile_base_pred,
            target_df=midday_df,
            route_multipliers=route_mult,
            default_multiplier=float(np.clip(global_theta, route_cfg.clip_min, route_cfg.clip_max)),
        )
        route_base_total = float(np.sum(route_base_pred))
        route_factor = midday_total / route_base_total if route_base_total > 0 else 1.0
        route_total = float(metric_frame(midday_df[TARGET_COL].to_numpy(dtype=float), route_base_pred)["total"])

        records.append(
            {
                "date": str(date.date()),
                "morning_total": morning_total,
                "morning_profile_total": morning_profile_total,
                "morning_ratio": morning_ratio,
                "midday_total": midday_total,
                "profile_base_pred": profile_base_pred,
                "route_base_pred": route_base_pred,
                "profile_base_total": profile_base_total,
                "route_base_total": route_base_total,
                "profile_factor": profile_factor,
                "route_factor": route_factor,
                "profile_total": profile_total,
                "route_total": route_total,
                "midday_true_vec": midday_df[TARGET_COL].to_numpy(dtype=float),
                "global_theta_obs": float(global_theta),
                "n_route_multipliers": int(route_mult.shape[0]),
            }
        )

    return records


def _score_candidate(
    records: list[dict],
    family: str,
    mode: str,
    min_factor: float,
    max_factor: float,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    evaluation: list[dict[str, object]] = []
    totals: list[float] = []
    wapes: list[float] = []
    rbiases: list[float] = []
    rbias_signed: list[float] = []

    for idx, rec in enumerate(records):
        train_records = records[:idx]
        train_x = np.array([_safe_log_ratio(r["morning_ratio"]) for r in train_records], dtype=float)
        if family == "profile":
            train_y = np.array([_safe_log_ratio(r["profile_factor"]) for r in train_records], dtype=float)
            base_vec = rec["profile_base_pred"]
            train_total_features = np.column_stack(
                [
                    [_safe_log_ratio(r["morning_total"]) for r in train_records],
                    [_safe_log_ratio(r["morning_profile_total"]) for r in train_records],
                ]
            )
            x_total_features = np.array(
                [_safe_log_ratio(float(rec["morning_total"])), _safe_log_ratio(float(rec["morning_profile_total"]))],
                dtype=float,
            )
        elif family == "route":
            train_y = np.array([_safe_log_ratio(r["route_factor"]) for r in train_records], dtype=float)
            base_vec = rec["route_base_pred"]
            train_total_features = np.column_stack(
                [
                    [_safe_log_ratio(r["morning_total"]) for r in train_records],
                    [_safe_log_ratio(r["morning_profile_total"]) for r in train_records],
                ]
            )
            x_total_features = np.array(
                [_safe_log_ratio(float(rec["morning_total"])), _safe_log_ratio(float(rec["morning_profile_total"]))],
                dtype=float,
            )
        else:
            raise ValueError(f"unknown family: {family}")

        x_value = _safe_log_ratio(float(rec["morning_ratio"]))
        if mode == "totalreg":
            factors = _predict_totalreg_factor(
                train_features=train_total_features,
                train_y=train_y,
                x_features=x_total_features,
                min_factor=min_factor,
                max_factor=max_factor,
            )
            factor = factors["logreg_factor"] if "logreg_factor" in factors else factors["global_factor"]
        else:
            factors = _predict_factor(
                train_x=train_x,
                train_y=train_y,
                x_value=x_value,
                min_factor=min_factor,
                max_factor=max_factor,
            )
            factor = factors[f"{mode}_factor"]
        y_pred = np.clip(np.asarray(base_vec, dtype=float) * factor, 0.0, None)
        metric_parts = metric_frame(y_true=np.asarray(rec["midday_true_vec"], dtype=float), y_pred=y_pred)

        totals.append(float(metric_parts["total"]))
        wapes.append(float(metric_parts["wape"]))
        rbiases.append(float(metric_parts["rbias"]))
        rbias_signed.append(float(metric_parts["rbias_signed"]))

        evaluation.append(
            {
                "date": rec["date"],
                "morning_total": rec["morning_total"],
                "morning_ratio": rec["morning_ratio"],
                "factor": factor,
                "global_factor": factors["global_factor"],
                "direct_factor": factors["direct_factor"],
                "logreg_factor": factors["logreg_factor"],
                "shrink_factor": factors["shrink_factor"],
                "profile_total": rec["profile_total"],
                "route_total": rec["route_total"],
                "mean_target": rec["midday_total"],
                "total": float(metric_parts["total"]),
                "wape": float(metric_parts["wape"]),
                "rbias": float(metric_parts["rbias"]),
                "rbias_signed": float(metric_parts["rbias_signed"]),
            }
        )

    summary = {
        "mean_total": float(np.mean(totals)) if totals else float("nan"),
        "mean_wape": float(np.mean(wapes)) if wapes else float("nan"),
        "mean_rbias": float(np.mean(rbiases)) if rbiases else float("nan"),
        "mean_rbias_signed": float(np.mean(rbias_signed)) if rbias_signed else float("nan"),
    }
    return evaluation, summary


def evaluate(train_df: pd.DataFrame, n_windows: int, route_cfg: SameDayCalibrationConfig) -> dict[str, object]:
    origin = train_df["timestamp"].min()
    train_feat = prepare(train_df.copy(), origin)
    dates = saturday_midday_dates(train_feat)[-n_windows:]
    records = _build_records(train_feat, dates, route_cfg=route_cfg)

    candidate_specs = {
        "profile_global": ("profile", "global"),
        "profile_direct": ("profile", "direct"),
        "profile_logreg": ("profile", "logreg"),
        "profile_shrink": ("profile", "shrink"),
        "profile_totalreg": ("profile", "totalreg"),
        "route_global": ("route", "global"),
        "route_direct": ("route", "direct"),
        "route_logreg": ("route", "logreg"),
        "route_shrink": ("route", "shrink"),
        "route_totalreg": ("route", "totalreg"),
    }

    candidate_scores: dict[str, dict[str, float]] = {}
    candidate_evaluations: dict[str, list[dict[str, object]]] = {}
    for name, (family, mode) in candidate_specs.items():
        rows, score = _score_candidate(
            records=records,
            family=family,
            mode=mode,
            min_factor=0.85,
            max_factor=1.15,
        )
        candidate_scores[name] = score
        candidate_evaluations[name] = rows

    best_candidate = min(candidate_scores.items(), key=lambda item: item[1]["mean_total"])[0]

    summary = {
        "n_windows": int(n_windows),
        "route_cfg": {
            "obs_start_hour": route_cfg.obs_start_hour,
            "obs_end_hour": route_cfg.obs_end_hour,
            "shrink_k": route_cfg.shrink_k,
            "clip_min": route_cfg.clip_min,
            "clip_max": route_cfg.clip_max,
        },
        "candidate_scores": candidate_scores,
        "best_candidate": best_candidate,
        "best_candidate_score": candidate_scores[best_candidate],
        "baseline_means": {
            "profile_total": float(np.mean([r["profile_total"] for r in records])) if records else float("nan"),
            "route_total": float(np.mean([r["route_total"] for r in records])) if records else float("nan"),
        },
    }

    evaluation = [
        {
            "date": rec["date"],
            "morning_total": rec["morning_total"],
            "morning_ratio": rec["morning_ratio"],
            "profile_total": rec["profile_total"],
            "route_total": rec["route_total"],
            "profile_base_total": rec["profile_base_total"],
            "route_base_total": rec["route_base_total"],
            "midday_total": rec["midday_total"],
        }
        for rec in records
    ]

    return {
        "summary": summary,
        "evaluation": evaluation,
        "candidate_evaluations": candidate_evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="train_solo_track.parquet")
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--report-out", default="reports/exp_total_morning_scale_eval.json")
    parser.add_argument("--route-obs-start-hour", type=int, default=8)
    parser.add_argument("--route-obs-end-hour", type=int, default=10)
    parser.add_argument("--route-shrink-k", type=float, default=2_000_000.0)
    parser.add_argument("--route-clip-min", type=float, default=0.95)
    parser.add_argument("--route-clip-max", type=float, default=1.05)
    args = parser.parse_args()

    train_df = load_train(args.train_path)
    route_cfg = SameDayCalibrationConfig(
        obs_start_hour=args.route_obs_start_hour,
        obs_end_hour=args.route_obs_end_hour,
        shrink_k=args.route_shrink_k,
        clip_min=args.route_clip_min,
        clip_max=args.route_clip_max,
    )

    payload = evaluate(train_df, n_windows=args.n_windows, route_cfg=route_cfg)
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
