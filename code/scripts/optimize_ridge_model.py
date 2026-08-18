#!/usr/bin/env python3
"""Tune the selected raw-target Ridge model without overwriting its baseline.

The optimization protocol keeps the feature set and target scale used by the
unified benchmark.  In standard scikit-learn Ridge, ``alpha`` is the L2
penalty conventionally denoted by lambda, so there is only one identifiable
regularization hyperparameter.  Hyperparameter selection is nested inside
expanding-origin validation and never uses the isolated 2024 pseudo-holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import compare_branch_models as benchmark
import model_jizhou_tourism_ml as ml_model


ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_UNIFIED_DIR: Final = ROOT / "data/unified"
DEFAULT_OUTPUT_DIR: Final = ROOT / "outputs/ridge_optimization"
DEFAULT_REPORT_PATH: Final = ROOT / "docs/ridge_model_optimization_report.md"
BASELINE_LAMBDA: Final = 0.1
POSITIVE_LAMBDA_GRID: Final = tuple(float(value) for value in np.logspace(-4, 3, 29))
BOUNDARY_DIAGNOSTIC_GRID: Final = (
    0.0,
    1.0e-6,
    1.0e-5,
    *POSITIVE_LAMBDA_GRID,
)
MIN_INNER_TRAIN_RECORDS: Final = 4
MIN_INNER_VALIDATIONS: Final = 3
BOOTSTRAP_REPETITIONS: Final = 10_000
RANDOM_SEED: Final = benchmark.RANDOM_SEED
FORECAST_YEARS: Final = benchmark.FORECAST_YEARS
TARGETS: Final = benchmark.TARGETS
OPTIMIZATION_BRANCH: Final = "ridge"
BASELINE_MODEL: Final = "raw_target_ridge_alpha_0.1"
TUNED_MODEL: Final = "raw_target_ridge_tuned_lambda"
NAIVE_MODEL: Final = "naive_last"
FEATURE_LABELS_CN: Final = {
    "year_index": "时间趋势",
    "pandemic_2020_2022": "2020—2022年阶段指示变量",
    "post_2022": "2023年后恢复阶段指示变量",
}
TARGET_LABELS_CN: Final = {
    "tourist_visits": "旅游接待人次",
    "tourism_comprehensive_income": "旅游综合收入",
}
TARGET_UNITS_CN: Final = {
    "tourist_visits": "万人次",
    "tourism_comprehensive_income": "亿元",
}


@dataclass(frozen=True)
class TuningResult:
    """One train-only hyperparameter selection and its complete audit trail."""

    selected_lambda: float
    status: str
    boundary_hit: bool
    boundary_side: str
    inner_validation_count: int
    candidate_summary: pd.DataFrame
    fold_scores: pd.DataFrame


def build_ridge_model(
    ridge_lambda: float,
    feature_indices: Sequence[int] = (0, 1, 2),
) -> Pipeline:
    """Return the exact StandardScaler + raw-target Ridge baseline family."""
    if ridge_lambda < 0.0 or not math.isfinite(ridge_lambda):
        raise ValueError("ridge lambda must be finite and non-negative")
    if not feature_indices:
        raise ValueError("at least one feature is required")
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=float(ridge_lambda))),
        ]
    )


def make_features(
    years: Iterable[int] | np.ndarray,
    feature_indices: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    return ml_model.make_features(np.asarray(list(years), dtype=int))[:, feature_indices]


def smape_percent(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual_values) + np.abs(predicted_values)
    return np.divide(
        200.0 * np.abs(actual_values - predicted_values),
        denominator,
        out=np.zeros_like(actual_values, dtype=float),
        where=denominator > 0.0,
    )


def _fit_predict(
    train: pd.DataFrame,
    test_years: Iterable[int],
    ridge_lambda: float,
    feature_indices: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    ordered = train.sort_values("year")
    model = build_ridge_model(ridge_lambda, feature_indices)
    model.fit(
        make_features(ordered["year"].astype(int), feature_indices),
        ordered["value"].to_numpy(dtype=float),
    )
    return np.maximum(
        0.0,
        model.predict(make_features(test_years, feature_indices)),
    )


def _inner_training_rows(
    physical_train: pd.DataFrame,
    *,
    validation_year: int,
    simulate_missing_years: bool,
    scope: str,
) -> pd.DataFrame:
    if not simulate_missing_years:
        return physical_train.copy()
    augmented, _ = benchmark.augment_training_rows(
        physical_train,
        test_year=validation_year,
        scope=scope,
        fold_id=f"{scope}__validation_{validation_year}",
    )
    return augmented


def tune_raw_ridge_lambda(
    physical_train: pd.DataFrame,
    *,
    candidate_lambdas: Sequence[float] = POSITIVE_LAMBDA_GRID,
    simulate_missing_years: bool = False,
    min_inner_train_records: int = MIN_INNER_TRAIN_RECORDS,
    min_inner_validations: int = MIN_INNER_VALIDATIONS,
    fallback_lambda: float = BASELINE_LAMBDA,
    scope: str = "train_only_tuning",
) -> TuningResult:
    """Select lambda by expanding-origin inner sMAPE without future leakage.

    When simulated annual rows are requested, each inner fold rebuilds them
    solely from the physical rows visible before that validation year.  The
    complete outer augmented frame is never sliced for inner validation.
    """
    train = physical_train.sort_values("year").reset_index(drop=True).copy()
    if train.empty or train["metric"].nunique() != 1:
        raise ValueError("lambda tuning requires one non-empty target series")
    if train["year"].duplicated().any():
        raise ValueError("lambda tuning received duplicate years")
    if not (train["value"].to_numpy(dtype=float) > 0.0).all():
        raise ValueError("lambda tuning requires positive targets")
    candidates = sorted({float(value) for value in candidate_lambdas})
    if not candidates or candidates[0] < 0.0 or not np.isfinite(candidates).all():
        raise ValueError("candidate lambdas must be finite and non-negative")

    validation_indices = list(range(min_inner_train_records, len(train)))
    fold_rows: list[dict[str, object]] = []
    for ridge_lambda in candidates:
        for validation_index in validation_indices:
            inner_physical = train.iloc[:validation_index].copy()
            validation = train.iloc[validation_index]
            validation_year = int(validation["year"])
            inner_train = _inner_training_rows(
                inner_physical,
                validation_year=validation_year,
                simulate_missing_years=simulate_missing_years,
                scope=scope,
            )
            prediction = float(
                _fit_predict(inner_train, [validation_year], ridge_lambda)[0]
            )
            actual = float(validation["value"])
            fold_rows.append(
                {
                    "metric": str(validation["metric"]),
                    "lambda": ridge_lambda,
                    "alpha_code_parameter": ridge_lambda,
                    "validation_year": validation_year,
                    "inner_physical_train_n": len(inner_physical),
                    "inner_effective_train_n": len(inner_train),
                    "inner_train_end_year": int(inner_physical["year"].max()),
                    "actual": actual,
                    "prediction": prediction,
                    "smape_percent": float(
                        smape_percent(np.array([actual]), np.array([prediction]))[0]
                    ),
                    "absolute_log_error": abs(math.log(actual) - math.log(prediction))
                    if prediction > 0.0
                    else math.inf,
                    "simulated_training": simulate_missing_years,
                }
            )
    fold_scores = pd.DataFrame(fold_rows)
    if fold_scores.empty:
        candidate_summary = pd.DataFrame(
            {
                "metric": [str(train.iloc[0]["metric"])] * len(candidates),
                "lambda": candidates,
                "alpha_code_parameter": candidates,
                "inner_validation_count": 0,
                "mean_inner_smape_percent": math.nan,
                "std_inner_smape_percent": math.nan,
                "mean_inner_absolute_log_error": math.nan,
                "simulated_training": simulate_missing_years,
            }
        )
    else:
        candidate_summary = (
            fold_scores.groupby(
                ["metric", "lambda", "alpha_code_parameter", "simulated_training"],
                as_index=False,
            )
            .agg(
                inner_validation_count=("validation_year", "size"),
                mean_inner_smape_percent=("smape_percent", "mean"),
                std_inner_smape_percent=("smape_percent", "std"),
                mean_inner_absolute_log_error=("absolute_log_error", "mean"),
            )
            .sort_values("lambda", ignore_index=True)
        )

    if len(validation_indices) < min_inner_validations:
        selected = float(fallback_lambda)
        status = "fallback_insufficient_inner_validations"
        boundary_hit = False
        boundary_side = "none"
    else:
        ranked = candidate_summary.sort_values(
            ["mean_inner_smape_percent", "lambda"],
            ascending=[True, False],
            ignore_index=True,
        )
        selected = float(ranked.iloc[0]["lambda"])
        status = "selected_by_nested_expanding_origin_smape"
        if math.isclose(selected, min(candidates), rel_tol=0.0, abs_tol=1e-15):
            boundary_side = "lower"
        elif math.isclose(selected, max(candidates), rel_tol=0.0, abs_tol=1e-15):
            boundary_side = "upper"
        else:
            boundary_side = "none"
        boundary_hit = boundary_side != "none"
    candidate_summary["selected_lambda"] = selected
    candidate_summary["selection_status"] = status
    candidate_summary["boundary_hit"] = boundary_hit
    candidate_summary["boundary_side"] = boundary_side
    candidate_summary["min_required_inner_validations"] = min_inner_validations
    return TuningResult(
        selected_lambda=selected,
        status=status,
        boundary_hit=boundary_hit,
        boundary_side=boundary_side,
        inner_validation_count=len(validation_indices),
        candidate_summary=candidate_summary,
        fold_scores=fold_scores,
    )


def _prediction_record(
    *,
    track: str,
    fold_id: str,
    model: str,
    metric: str,
    test_year: int,
    actual: float,
    prediction: float,
    ridge_lambda: float | None,
    tuning: TuningResult | None,
    physical_train: pd.DataFrame,
    effective_train: pd.DataFrame,
) -> dict[str, object]:
    point_smape = float(
        smape_percent(np.array([actual]), np.array([prediction]))[0]
    )
    return {
        "track": track,
        "branch": OPTIMIZATION_BRANCH,
        "fold_id": fold_id,
        "model": model,
        "metric": metric,
        "year": test_year,
        "actual": actual,
        "prediction": prediction,
        "error": prediction - actual,
        "absolute_error": abs(prediction - actual),
        "point_smape_percent": point_smape,
        "lambda": ridge_lambda,
        "alpha_code_parameter": ridge_lambda,
        "tuning_status": "not_applicable" if tuning is None else tuning.status,
        "inner_validation_count": 0 if tuning is None else tuning.inner_validation_count,
        "lambda_grid_boundary_hit": False if tuning is None else tuning.boundary_hit,
        "physical_train_n": len(physical_train),
        "effective_train_n": len(effective_train),
        "physical_train_end_year": int(physical_train["year"].max()),
        "effective_train_end_year": int(effective_train["year"].max()),
        "uses_test_in_training": bool(
            effective_train["year"].ge(test_year).any()
        ),
    }


def _evaluate_models_for_one_test(
    physical_train: pd.DataFrame,
    effective_train: pd.DataFrame,
    test: pd.Series,
    *,
    track: str,
    fold_id: str,
    simulate_inner_tuning: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tuning = tune_raw_ridge_lambda(
        physical_train,
        simulate_missing_years=simulate_inner_tuning,
        scope=f"{track}__{fold_id}__inner",
    )
    metric = str(test["metric"])
    test_year = int(test["year"])
    actual = float(test["value"])
    naive_prediction = float(effective_train.sort_values("year").iloc[-1]["value"])
    predictions = {
        NAIVE_MODEL: (naive_prediction, None),
        BASELINE_MODEL: (
            float(_fit_predict(effective_train, [test_year], BASELINE_LAMBDA)[0]),
            BASELINE_LAMBDA,
        ),
        TUNED_MODEL: (
            float(
                _fit_predict(
                    effective_train, [test_year], tuning.selected_lambda
                )[0]
            ),
            tuning.selected_lambda,
        ),
    }
    records = [
        _prediction_record(
            track=track,
            fold_id=fold_id,
            model=model,
            metric=metric,
            test_year=test_year,
            actual=actual,
            prediction=prediction,
            ridge_lambda=ridge_lambda,
            tuning=tuning if model == TUNED_MODEL else None,
            physical_train=physical_train,
            effective_train=effective_train,
        )
        for model, (prediction, ridge_lambda) in predictions.items()
    ]
    candidate = tuning.candidate_summary.copy()
    candidate.insert(0, "track", track)
    candidate.insert(1, "outer_fold_id", fold_id)
    folds = tuning.fold_scores.copy()
    if not folds.empty:
        folds.insert(0, "track", track)
        folds.insert(1, "outer_fold_id", fold_id)
    return pd.DataFrame(records), candidate, folds


def evaluate_rolling_track(
    rolling_folds: pd.DataFrame,
    *,
    track: str,
    simulate_missing_years: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    inner_fold_frames: list[pd.DataFrame] = []
    for (_, fold_id), group in rolling_folds.groupby(
        ["metric", "fold_id"], sort=True
    ):
        physical_train = group[group["fold_role"].eq("train")].copy()
        test = group[group["fold_role"].eq("test")].iloc[0]
        test_year = int(test["year"])
        if simulate_missing_years:
            effective_train, _ = benchmark.augment_training_rows(
                physical_train,
                test_year=test_year,
                scope=track,
                fold_id=str(fold_id),
            )
        else:
            effective_train = physical_train.copy()
        predictions, candidates, inner_folds = _evaluate_models_for_one_test(
            physical_train,
            effective_train,
            test,
            track=track,
            fold_id=str(fold_id),
            simulate_inner_tuning=simulate_missing_years,
        )
        prediction_frames.append(predictions)
        candidate_frames.append(candidates)
        if not inner_folds.empty:
            inner_fold_frames.append(inner_folds)
    result = pd.concat(prediction_frames, ignore_index=True)
    if result["uses_test_in_training"].any() or result["year"].gt(2023).any():
        raise AssertionError("rolling optimization leaked an outer test")
    return (
        result,
        pd.concat(candidate_frames, ignore_index=True),
        pd.concat(inner_fold_frames, ignore_index=True)
        if inner_fold_frames
        else pd.DataFrame(),
    )


def evaluate_fixed_scope(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    track: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    inner_fold_frames: list[pd.DataFrame] = []
    for metric in TARGETS:
        metric_train = train[train["metric"].eq(metric)].copy()
        metric_tests = test[test["metric"].eq(metric)].sort_values("year")
        for _, test_row in metric_tests.iterrows():
            test_year = int(test_row["year"])
            visible_train = metric_train[metric_train["year"].lt(test_year)].copy()
            fold_id = f"{track}__{metric}__test_{test_year}"
            effective_train, _ = benchmark.augment_training_rows(
                visible_train,
                test_year=test_year,
                scope=track,
                fold_id=fold_id,
            )
            predictions, candidates, inner_folds = _evaluate_models_for_one_test(
                visible_train,
                effective_train,
                test_row,
                track=track,
                fold_id=fold_id,
                simulate_inner_tuning=True,
            )
            prediction_frames.append(predictions)
            candidate_frames.append(candidates)
            if not inner_folds.empty:
                inner_fold_frames.append(inner_folds)
    result = pd.concat(prediction_frames, ignore_index=True)
    if result["uses_test_in_training"].any():
        raise AssertionError(f"{track} optimization leaked a test")
    return (
        result,
        pd.concat(candidate_frames, ignore_index=True),
        pd.concat(inner_fold_frames, ignore_index=True)
        if inner_fold_frames
        else pd.DataFrame(),
    )


def summarize_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_frames: list[pd.DataFrame] = []
    macro_frames: list[pd.DataFrame] = []
    for track, group in predictions.groupby("track", sort=True):
        by_target, macro = benchmark.summarize_predictions(group)
        by_target.insert(0, "track", track)
        macro.insert(0, "track", track)
        baseline_by_target = (
            by_target[by_target["model"].eq(BASELINE_MODEL)]
            .set_index("metric")["smape_percent"]
            .to_dict()
        )
        by_target["baseline_smape_percent"] = by_target["metric"].map(
            baseline_by_target
        )
        by_target["delta_smape_vs_baseline_percent_points"] = (
            by_target["smape_percent"] - by_target["baseline_smape_percent"]
        )
        if not macro.empty:
            baseline_macro = float(
                macro.loc[
                    macro["model"].eq(BASELINE_MODEL), "macro_smape_percent"
                ].iloc[0]
            )
            macro["baseline_macro_smape_percent"] = baseline_macro
            macro["delta_macro_smape_vs_baseline_percent_points"] = (
                macro["macro_smape_percent"] - baseline_macro
            )
        target_frames.append(by_target)
        macro_frames.append(macro)
    return (
        pd.concat(target_frames, ignore_index=True),
        pd.concat(macro_frames, ignore_index=True),
    )


def paired_fold_bootstrap(
    predictions: pd.DataFrame,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for track, group in predictions.groupby("track", sort=True):
        comparison = group[group["model"].isin([BASELINE_MODEL, TUNED_MODEL])]
        wide = comparison.pivot(
            index=["metric", "fold_id", "year", "actual"],
            columns="model",
            values="prediction",
        ).reset_index()
        target_deltas: dict[str, np.ndarray] = {}
        target_point_estimates: dict[str, float] = {}
        for metric in TARGETS:
            metric_rows = wide[wide["metric"].eq(metric)].reset_index(drop=True)
            baseline_terms = smape_percent(
                metric_rows["actual"].to_numpy(dtype=float),
                metric_rows[BASELINE_MODEL].to_numpy(dtype=float),
            )
            tuned_terms = smape_percent(
                metric_rows["actual"].to_numpy(dtype=float),
                metric_rows[TUNED_MODEL].to_numpy(dtype=float),
            )
            point_delta = float(np.mean(tuned_terms - baseline_terms))
            sample_indices = rng.integers(
                0, len(metric_rows), size=(repetitions, len(metric_rows))
            )
            delta_draws = np.mean(
                (tuned_terms - baseline_terms)[sample_indices], axis=1
            )
            target_deltas[metric] = delta_draws
            target_point_estimates[metric] = point_delta
            rows.append(
                {
                    "track": track,
                    "scope": metric,
                    "delta_smape_tuned_minus_baseline_percent_points": point_delta,
                    "bootstrap_ci95_lower": float(np.quantile(delta_draws, 0.025)),
                    "bootstrap_ci95_upper": float(np.quantile(delta_draws, 0.975)),
                    "bootstrap_probability_delta_below_zero": float(
                        np.mean(delta_draws < 0.0)
                    ),
                    "bootstrap_repetitions": repetitions,
                    "random_seed": seed,
                    "warning": "fold-level descriptive bootstrap; folds overlap and are not independent",
                    "targets_resampled_independently": True,
                    "inferential_confidence_interval": False,
                }
            )
        macro_draws = np.mean(
            np.column_stack([target_deltas[metric] for metric in TARGETS]), axis=1
        )
        macro_point = float(
            np.mean([target_point_estimates[metric] for metric in TARGETS])
        )
        rows.append(
            {
                "track": track,
                "scope": "equal_target_macro",
                "delta_smape_tuned_minus_baseline_percent_points": macro_point,
                "bootstrap_ci95_lower": float(np.quantile(macro_draws, 0.025)),
                "bootstrap_ci95_upper": float(np.quantile(macro_draws, 0.975)),
                "bootstrap_probability_delta_below_zero": float(
                    np.mean(macro_draws < 0.0)
                ),
                "bootstrap_repetitions": repetitions,
                "random_seed": seed,
                "warning": "fold-level descriptive bootstrap; folds overlap and are not independent",
                "targets_resampled_independently": True,
                "inferential_confidence_interval": False,
            }
        )
    return pd.DataFrame(rows)


def build_final_outputs(
    train: pd.DataFrame,
    *,
    ridge_lambda: float,
    model_name: str,
    bootstrap_seed: int = RANDOM_SEED,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one frozen-lambda model and build conditional bootstrap outputs."""
    metric = str(train.iloc[0]["metric"])
    years = train["year"].to_numpy(dtype=int)
    values = train["value"].to_numpy(dtype=float)
    x = make_features(years)
    x_future = make_features(FORECAST_YEARS)
    model = build_ridge_model(ridge_lambda)
    model.fit(x, values)
    scaler = model.named_steps["scale"]
    ridge = model.named_steps["model"]
    z = scaler.transform(x)
    z_future = scaler.transform(x_future)
    fitted = np.asarray(model.predict(x), dtype=float)
    residuals = values - fitted
    centered_residuals = residuals - float(np.mean(residuals))
    penalty_inverse = np.linalg.inv(
        z.T @ z + ridge_lambda * np.eye(z.shape[1])
    )

    rng = np.random.default_rng(bootstrap_seed)
    residual_indices = rng.integers(
        0, len(centered_residuals), size=(bootstrap_repetitions, len(values))
    )
    bootstrap_y = fitted[None, :] + centered_residuals[residual_indices]
    bootstrap_intercepts = bootstrap_y.mean(axis=1)
    bootstrap_centered = bootstrap_y - bootstrap_intercepts[:, None]
    bootstrap_coefficients = (bootstrap_centered @ z) @ penalty_inverse
    bootstrap_mean_forecasts = (
        bootstrap_intercepts[:, None] + bootstrap_coefficients @ z_future.T
    )
    future_residual_indices = rng.integers(
        0,
        len(centered_residuals),
        size=(bootstrap_repetitions, len(FORECAST_YEARS)),
    )
    bootstrap_prediction_draws = (
        bootstrap_mean_forecasts + centered_residuals[future_residual_indices]
    )
    mean_draws_clipped = int((bootstrap_mean_forecasts < 0.0).sum())
    prediction_draws_clipped = int((bootstrap_prediction_draws < 0.0).sum())
    bootstrap_mean_forecasts = np.maximum(0.0, bootstrap_mean_forecasts)
    bootstrap_prediction_draws = np.maximum(0.0, bootstrap_prediction_draws)
    point_forecasts = np.maximum(0.0, model.predict(x_future))

    forecast_rows: list[dict[str, object]] = []
    for index, year in enumerate(FORECAST_YEARS):
        forecast_rows.append(
            {
                "metric": metric,
                "model": model_name,
                "lambda": ridge_lambda,
                "alpha_code_parameter": ridge_lambda,
                "year": year,
                "forecast": float(point_forecasts[index]),
                "mean_ci95_lower": float(
                    np.quantile(bootstrap_mean_forecasts[:, index], 0.025)
                ),
                "mean_ci95_upper": float(
                    np.quantile(bootstrap_mean_forecasts[:, index], 0.975)
                ),
                "prediction_interval95_lower": float(
                    np.quantile(bootstrap_prediction_draws[:, index], 0.025)
                ),
                "prediction_interval95_upper": float(
                    np.quantile(bootstrap_prediction_draws[:, index], 0.975)
                ),
                "interval_method": (
                    f"heuristic fixed-design iid residual bootstrap ({bootstrap_repetitions} repetitions), "
                    f"frozen lambda={ridge_lambda:g}"
                ),
                "interval_warning": (
                    "assumes exchangeable residuals and no serial dependence; coverage is unvalidated; "
                    "excludes tuning, structural, missingness, and scenario uncertainty"
                ),
                "point_semantics": "raw-scale conditional mean under the frozen Ridge specification",
                "bootstrap_repetitions": bootstrap_repetitions,
                "random_seed": bootstrap_seed,
                "training_end_year": int(years.max()),
                "training_n": len(train),
                "simulated_training_n": int(
                    benchmark._coerce_boolean(train["is_simulated"]).sum()
                ),
                "bootstrap_mean_draws_clipped_at_zero": mean_draws_clipped,
                "bootstrap_prediction_draws_clipped_at_zero": prediction_draws_clipped,
                "uses_2025_as_training": bool(np.any(years >= 2025)),
            }
        )

    parameter_names = ["intercept", *ml_model.FEATURE_NAMES]
    estimates = np.concatenate(
        [[float(ridge.intercept_)], np.asarray(ridge.coef_, dtype=float)]
    )
    bootstrap_parameters = np.column_stack(
        [bootstrap_intercepts, bootstrap_coefficients]
    )
    parameter_rows: list[dict[str, object]] = []
    for index, (name, estimate) in enumerate(
        zip(parameter_names, estimates, strict=True)
    ):
        is_intercept = name == "intercept"
        parameter_rows.append(
            {
                "metric": metric,
                "model": model_name,
                "lambda": ridge_lambda,
                "alpha_code_parameter": ridge_lambda,
                "parameter": name,
                "estimate": float(estimate),
                "bootstrap_mean": float(np.mean(bootstrap_parameters[:, index])),
                "bootstrap_standard_deviation": float(
                    np.std(bootstrap_parameters[:, index], ddof=1)
                ),
                "bootstrap_ci95_lower": float(
                    np.quantile(bootstrap_parameters[:, index], 0.025)
                ),
                "bootstrap_ci95_upper": float(
                    np.quantile(bootstrap_parameters[:, index], 0.975)
                ),
                "parameter_scale": (
                    "raw-target intercept"
                    if is_intercept
                    else "coefficient per one training-sample SD of predictor"
                ),
                "feature_training_mean": math.nan
                if is_intercept
                else float(scaler.mean_[index - 1]),
                "feature_training_scale": math.nan
                if is_intercept
                else float(scaler.scale_[index - 1]),
                "bootstrap_repetitions": bootstrap_repetitions,
                "random_seed": bootstrap_seed,
                "training_n": len(train),
                "training_end_year": int(years.max()),
                "uses_2025_as_training": bool(np.any(years >= 2025)),
            }
        )

    n = len(values)
    rss = float(residuals @ residuals)
    tss = float(np.square(values - values.mean()).sum())
    r_squared = 1.0 - rss / tss if tss else math.nan
    effective_parameters = 1.0 + float(np.trace(z @ penalty_inverse @ z.T))
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / max(
        n - effective_parameters, 1.0e-12
    )
    aic = n * math.log(max(rss / n, 1.0e-300)) + 2.0 * effective_parameters
    aicc_denominator = n - effective_parameters - 1.0
    aicc = (
        aic
        + 2.0
        * effective_parameters
        * (effective_parameters + 1.0)
        / aicc_denominator
        if aicc_denominator > 0.0
        else math.nan
    )
    dw = (
        float(np.diff(residuals) @ np.diff(residuals)) / rss
        if rss > 0.0
        else math.nan
    )
    jb, jb_p = benchmark._jarque_bera_diagnostics(residuals)
    diagnostic = pd.DataFrame(
        [
            {
                "metric": metric,
                "model": model_name,
                "lambda": ridge_lambda,
                "alpha_code_parameter": ridge_lambda,
                "target_scale": "raw",
                "training_n": n,
                "physical_canonical_training_n": int(
                    (~benchmark._coerce_boolean(train["is_simulated"])).sum()
                ),
                "simulated_training_n": int(
                    benchmark._coerce_boolean(train["is_simulated"]).sum()
                ),
                "training_start_year": int(years.min()),
                "training_end_year": int(years.max()),
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted_r_squared,
                "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
                "mae": float(np.mean(np.abs(residuals))),
                "mape_percent": float(
                    np.mean(np.abs(residuals) / np.abs(values)) * 100.0
                ),
                "smape_percent": float(np.mean(smape_percent(values, fitted))),
                "aicc": aicc,
                "effective_parameters": effective_parameters,
                "durbin_watson": dw,
                "jarque_bera": jb,
                "jarque_bera_p": jb_p,
                "bootstrap_repetitions": bootstrap_repetitions,
                "random_seed": bootstrap_seed,
                "bootstrap_mean_draws_clipped_at_zero": mean_draws_clipped,
                "bootstrap_prediction_draws_clipped_at_zero": prediction_draws_clipped,
                "diagnostic_note": (
                    "descriptive in-sample diagnostics on augmented targets; "
                    "AICc uses Ridge effective degrees of freedom; time-ordered outer evaluation governs comparison"
                ),
                "uses_2025_as_training": bool(np.any(years >= 2025)),
            }
        ]
    )
    return pd.DataFrame(forecast_rows), diagnostic, pd.DataFrame(parameter_rows)


def build_design_support_diagnostics(
    rolling_folds: pd.DataFrame,
    final_training: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (metric, fold_id), group in rolling_folds.groupby(
        ["metric", "fold_id"], sort=True
    ):
        train = group[group["fold_role"].eq("train")].sort_values("year")
        x = make_features(train["year"].astype(int))
        centered_rank = int(np.linalg.matrix_rank(x - x.mean(axis=0)))
        rows.append(
            {
                "scope": "official_outer_training",
                "metric": metric,
                "fold_id": fold_id,
                "test_year": int(
                    group[group["fold_role"].eq("test")].iloc[0]["year"]
                ),
                "training_n": len(train),
                "centered_feature_rank": centered_rank,
                "year_index_unique": int(np.unique(x[:, 0]).size),
                "pandemic_positive_count": int(x[:, 1].sum()),
                "post_2022_positive_count": int(x[:, 2].sum()),
                "warning": "rank is descriptive; outer folds cannot validate an unseen recovery coefficient",
            }
        )
    for metric in TARGETS:
        train = final_training[final_training["metric"].eq(metric)].sort_values("year")
        x = make_features(train["year"].astype(int))
        rows.append(
            {
                "scope": "final_augmented_training",
                "metric": metric,
                "fold_id": "final_2010_2024",
                "test_year": math.nan,
                "training_n": len(train),
                "centered_feature_rank": int(
                    np.linalg.matrix_rank(x - x.mean(axis=0))
                ),
                "year_index_unique": int(np.unique(x[:, 0]).size),
                "pandemic_positive_count": int(x[:, 1].sum()),
                "post_2022_positive_count": int(x[:, 2].sum()),
                "warning": "full rank does not remove weak support: only 2023-2024 carry the recovery indicator",
            }
        )
    return pd.DataFrame(rows)


def build_lambda_forecast_sensitivity(
    final_training: pd.DataFrame,
) -> pd.DataFrame:
    values = (0.0, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0)
    rows: list[dict[str, object]] = []
    for metric in TARGETS:
        train = final_training[final_training["metric"].eq(metric)]
        for ridge_lambda in values:
            predictions = _fit_predict(
                train, [2026, 2030], ridge_lambda
            )
            for year, prediction in zip((2026, 2030), predictions, strict=True):
                rows.append(
                    {
                        "metric": metric,
                        "lambda": ridge_lambda,
                        "alpha_code_parameter": ridge_lambda,
                        "year": year,
                        "forecast": float(prediction),
                        "formal_ridge_candidate": ridge_lambda
                        in POSITIVE_LAMBDA_GRID,
                        "boundary_diagnostic": ridge_lambda < min(POSITIVE_LAMBDA_GRID),
                    }
                )
    return pd.DataFrame(rows)


def build_feature_ablation(
    rolling_folds: pd.DataFrame,
    final_training: pd.DataFrame,
    selected_lambdas: dict[str, float],
) -> pd.DataFrame:
    feature_sets: dict[str, tuple[int, ...]] = {
        "time_only": (0,),
        "time_plus_pandemic": (0, 1),
        "time_plus_recovery": (0, 2),
        "full_regime_features": (0, 1, 2),
    }
    rows: list[dict[str, object]] = []
    for (metric, fold_id), group in rolling_folds.groupby(
        ["metric", "fold_id"], sort=True
    ):
        train = group[group["fold_role"].eq("train")].sort_values("year")
        test = group[group["fold_role"].eq("test")].iloc[0]
        for name, indices in feature_sets.items():
            prediction = float(
                _fit_predict(
                    train,
                    [int(test["year"])],
                    selected_lambdas[str(metric)],
                    indices,
                )[0]
            )
            rows.append(
                {
                    "scope": "official_outer_fixed_final_lambda_exploratory",
                    "metric": metric,
                    "feature_set": name,
                    "fold_id": fold_id,
                    "year": int(test["year"]),
                    "actual": float(test["value"]),
                    "prediction": prediction,
                    "smape_percent": float(
                        smape_percent(
                            np.array([float(test["value"])]),
                            np.array([prediction]),
                        )[0]
                    ),
                    "lambda": selected_lambdas[str(metric)],
                    "selection_use": "exploratory only; final lambda is retrospectively fixed",
                }
            )
    result = pd.DataFrame(rows)
    summary = (
        result.groupby(["scope", "metric", "feature_set", "lambda"], as_index=False)
        .agg(
            n_test=("year", "size"),
            mean_smape_percent=("smape_percent", "mean"),
            worst_smape_percent=("smape_percent", "max"),
        )
    )
    final_rows: list[dict[str, object]] = []
    for metric in TARGETS:
        train = final_training[final_training["metric"].eq(metric)]
        for name, indices in feature_sets.items():
            prediction_2030 = float(
                _fit_predict(
                    train,
                    [2030],
                    selected_lambdas[metric],
                    indices,
                )[0]
            )
            final_rows.append(
                {
                    "scope": "final_augmented_2030",
                    "metric": metric,
                    "feature_set": name,
                    "lambda": selected_lambdas[metric],
                    "n_test": math.nan,
                    "mean_smape_percent": math.nan,
                    "worst_smape_percent": math.nan,
                    "forecast_2030": prediction_2030,
                    "selection_use": "structural sensitivity only; not a causal estimate",
                }
            )
    summary["forecast_2030"] = math.nan
    summary["selection_use"] = "exploratory only; final lambda is retrospectively fixed"
    return pd.concat([summary, pd.DataFrame(final_rows)], ignore_index=True)


def _format_value(value: object, decimals: int = 3) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    labels: Sequence[str] | None = None,
    decimals: int = 3,
) -> str:
    headers = list(columns) if labels is None else list(labels)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, record in frame[list(columns)].iterrows():
        rows.append(
            "| "
            + " | ".join(
                _format_value(record[column], decimals).replace("|", "\\|")
                for column in columns
            )
            + " |"
        )
    return "\n".join(rows)


def render_figures(
    output_dir: Path,
    *,
    official_predictions: pd.DataFrame,
    final_search: pd.DataFrame,
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    problem1_forecasts: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> list[str]:
    plt.rcParams.update(
        {
            "font.family": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4b5563",
            "text.color": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )
    colors = {BASELINE_MODEL: "#6b7280", TUNED_MODEL: "#2563eb"}
    figures: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, metric in zip(axes, TARGETS, strict=True):
        metric_search = final_search[
            final_search["metric"].eq(metric)
            & final_search["selection_track"].eq(
                "fold_local_augmented_sensitivity"
            )
            & final_search["lambda"].gt(0.0)
        ]
        official_search = final_search[
            final_search["metric"].eq(metric)
            & final_search["selection_track"].eq("official_only")
            & final_search["lambda"].gt(0.0)
        ]
        axis.plot(
            metric_search["lambda"],
            metric_search["mean_inner_smape_percent"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color="#2563eb",
            label="折内增强（部署选择）",
        )
        axis.plot(
            official_search["lambda"],
            official_search["mean_inner_smape_percent"],
            linewidth=1.1,
            linestyle="--",
            color="#9ca3af",
            label="仅物理证据复核",
        )
        axis.axvline(BASELINE_LAMBDA, color="#6b7280", linestyle="-.", label="优化前 λ=0.1")
        selected = float(metric_search["selected_lambda"].iloc[0])
        axis.axvline(selected, color="#dc2626", linestyle=":", label=f"正式选择 λ={selected:g}")
        axis.set_xscale("log")
        axis.set_title(TARGET_LABELS_CN[metric])
        axis.set_xlabel("λ（sklearn 代码参数 alpha）")
        axis.set_ylabel("内层扩展窗口平均 sMAPE（%）")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    search_name = "ridge_lambda_search.png"
    fig.savefig(output_dir / search_name, dpi=180)
    plt.close(fig)
    figures.append(search_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, metric in zip(axes, TARGETS, strict=True):
        metric_rows = official_predictions[official_predictions["metric"].eq(metric)]
        actual = (
            metric_rows[["year", "actual"]]
            .drop_duplicates()
            .sort_values("year")
        )
        axis.plot(actual["year"], actual["actual"], "o-", color="#111827", label="实际值")
        for model in (BASELINE_MODEL, TUNED_MODEL):
            model_rows = metric_rows[metric_rows["model"].eq(model)].sort_values("year")
            axis.plot(
                model_rows["year"],
                model_rows["prediction"],
                marker="o",
                linestyle="--",
                color=colors[model],
                label="优化前 Ridge" if model == BASELINE_MODEL else "严格嵌套调参 Ridge",
            )
        axis.set_title(TARGET_LABELS_CN[metric])
        axis.set_xlabel("外层测试年份")
        axis.set_ylabel(TARGET_UNITS_CN[metric])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    backtest_name = "ridge_optimization_backtest.png"
    fig.savefig(output_dir / backtest_name, dpi=180)
    plt.close(fig)
    figures.append(backtest_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), constrained_layout=True)
    for axis, metric in zip(axes, TARGETS, strict=True):
        history = observations[
            observations["metric"].eq(metric) & observations["value"].notna()
        ].sort_values("year")
        axis.scatter(history["year"], history["value"], color="#111827", s=24, label="官方优选值")
        for model in (BASELINE_MODEL, TUNED_MODEL):
            model_rows = forecasts[
                forecasts["metric"].eq(metric) & forecasts["model"].eq(model)
            ].sort_values("year")
            axis.plot(
                model_rows["year"],
                model_rows["forecast"],
                color=colors[model],
                marker="o",
                label="优化前 Ridge" if model == BASELINE_MODEL else "优化后 Ridge",
            )
            if model == TUNED_MODEL:
                axis.fill_between(
                    model_rows["year"].to_numpy(dtype=float),
                    model_rows["prediction_interval95_lower"].to_numpy(dtype=float),
                    model_rows["prediction_interval95_upper"].to_numpy(dtype=float),
                    color=colors[model],
                    alpha=0.12,
                    label="优化后条件95%预测区间",
                )
        simple = problem1_forecasts[problem1_forecasts["metric"].eq(metric)].sort_values("year")
        axis.plot(
            simple["year"],
            simple["forecast"],
            color="#9ca3af",
            linestyle=":",
            label="题目1疫情前简单增长外推",
        )
        scenario_metric = scenarios[scenarios["metric"].eq(metric)]
        envelope = scenario_metric.groupby("year")["value"].agg(["min", "max"]).reset_index()
        axis.fill_between(
            envelope["year"].to_numpy(dtype=float),
            envelope["min"].to_numpy(dtype=float),
            envelope["max"].to_numpy(dtype=float),
            color="#f59e0b",
            alpha=0.13,
            label="题目3情景包络",
        )
        axis.set_title(TARGET_LABELS_CN[metric])
        axis.set_xlabel("年份")
        axis.set_ylabel(TARGET_UNITS_CN[metric])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    forecast_name = "ridge_optimization_forecast.png"
    fig.savefig(output_dir / forecast_name, dpi=180)
    plt.close(fig)
    figures.append(forecast_name)
    return figures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _markdown_relative_path(target: Path, report_path: Path) -> str:
    return Path(
        os.path.relpath(target.resolve(), start=report_path.resolve().parent)
    ).as_posix()


def _write_output_readme(output_dir: Path) -> None:
    content = """# Ridge optimization artifacts

This directory is generated by `code/scripts/optimize_ridge_model.py`.

The fixed `alpha=0.1` baseline is preserved. The optimized model tunes the
single Ridge penalty `lambda` (the same quantity exposed as `alpha` by
scikit-learn) inside expanding-origin inner validation. The 2024 observations
remain a diagnostic pseudo-holdout and are never used for hyperparameter
selection. Final deployment selection rebuilds augmentation inside every inner
fold so it matches the final 15-row fitting contract; the official-only track
independently selects the same value. Tables named `augmented_outer_*` are the
deployment-consistent backtest and `official_outer_*` preserve strict
continuity with the unified benchmark. Boundary, ablation, and stress tables
are sensitivity analyses.

```bash
MPLCONFIGDIR=/tmp/jizhou-mpl XDG_CACHE_HOME=/tmp/jizhou-xdg \\
  .venv/bin/python code/scripts/optimize_ridge_model.py
```
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def _write_report(
    report_path: Path,
    *,
    output_dir: Path,
    official_by_target: pd.DataFrame,
    official_macro: pd.DataFrame,
    simulated_by_target: pd.DataFrame,
    simulated_macro: pd.DataFrame,
    paired_bootstrap: pd.DataFrame,
    holdout_by_target: pd.DataFrame,
    stress_by_target: pd.DataFrame,
    final_selection: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    parameters: pd.DataFrame,
    final_training: pd.DataFrame,
    implied_spend: pd.DataFrame,
    design_support: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    problem1_summary: pd.DataFrame,
    problem1_diagnostics: pd.DataFrame,
    problem1_forecasts: pd.DataFrame,
    scenarios: pd.DataFrame,
    policy_sensitivity: pd.DataFrame,
) -> None:
    lambda_search_figure = _markdown_relative_path(
        output_dir / "ridge_lambda_search.png", report_path
    )
    backtest_figure = _markdown_relative_path(
        output_dir / "ridge_optimization_backtest.png", report_path
    )
    forecast_figure = _markdown_relative_path(
        output_dir / "ridge_optimization_forecast.png", report_path
    )
    problem1_figure = _markdown_relative_path(
        ROOT / "outputs/unified_model_benchmark/q1_required_indicators.png",
        report_path,
    )
    official_primary = official_by_target[
        official_by_target["model"].isin([BASELINE_MODEL, TUNED_MODEL, NAIVE_MODEL])
    ].copy()
    official_primary["指标"] = official_primary["metric"].map(TARGET_LABELS_CN)
    official_primary["模型"] = official_primary["model"].map(
        {
            BASELINE_MODEL: "优化前 Ridge（λ=0.1）",
            TUNED_MODEL: "优化后严格嵌套 Ridge",
            NAIVE_MODEL: "上一期数值法",
        }
    )
    official_primary["相对上一期法改进率（%）"] = (
        official_primary["smape_skill_vs_naive"] * 100.0
    )
    simulated_primary = simulated_by_target[
        simulated_by_target["model"].isin([BASELINE_MODEL, TUNED_MODEL, NAIVE_MODEL])
    ].copy()
    simulated_primary["指标"] = simulated_primary["metric"].map(TARGET_LABELS_CN)
    simulated_primary["模型"] = simulated_primary["model"].map(
        {
            BASELINE_MODEL: "优化前 Ridge（λ=0.1）",
            TUNED_MODEL: "优化后严格嵌套 Ridge",
            NAIVE_MODEL: "上一期数值法",
        }
    )
    simulated_primary["相对上一期法改进率（%）"] = (
        simulated_primary["smape_skill_vs_naive"] * 100.0
    )
    macro_official_rows = official_macro[
        official_macro["model"].isin([BASELINE_MODEL, TUNED_MODEL, NAIVE_MODEL])
    ].copy()
    macro_official_rows["模型"] = macro_official_rows["model"].map(
        {
            BASELINE_MODEL: "优化前 Ridge（λ=0.1）",
            TUNED_MODEL: "优化后严格嵌套 Ridge",
            NAIVE_MODEL: "上一期数值法",
        }
    )
    macro_official_rows["相对上一期法等权改进（%）"] = (
        macro_official_rows["macro_smape_skill_vs_naive"] * 100.0
    )
    macro_simulated_rows = simulated_macro[
        simulated_macro["model"].isin(
            [BASELINE_MODEL, TUNED_MODEL, NAIVE_MODEL]
        )
    ].copy()
    macro_simulated_rows["模型"] = macro_simulated_rows["model"].map(
        {
            BASELINE_MODEL: "优化前 Ridge（λ=0.1）",
            TUNED_MODEL: "优化后严格嵌套 Ridge",
            NAIVE_MODEL: "上一期数值法",
        }
    )
    macro_simulated_rows["相对上一期法等权改进（%）"] = (
        macro_simulated_rows["macro_smape_skill_vs_naive"] * 100.0
    )
    official_baseline_macro = float(
        macro_official_rows.loc[
            macro_official_rows["model"].eq(BASELINE_MODEL), "macro_smape_percent"
        ].iloc[0]
    )
    official_tuned_macro = float(
        macro_official_rows.loc[
            macro_official_rows["model"].eq(TUNED_MODEL), "macro_smape_percent"
        ].iloc[0]
    )
    delta_macro = official_tuned_macro - official_baseline_macro
    relative_change = delta_macro / official_baseline_macro * 100.0
    macro_boot = paired_bootstrap[
        paired_bootstrap["track"].eq("official_only_nested")
        & paired_bootstrap["scope"].eq("equal_target_macro")
    ].iloc[0]
    deployment_baseline_macro = float(
        macro_simulated_rows.loc[
            macro_simulated_rows["model"].eq(BASELINE_MODEL),
            "macro_smape_percent",
        ].iloc[0]
    )
    deployment_tuned_macro = float(
        macro_simulated_rows.loc[
            macro_simulated_rows["model"].eq(TUNED_MODEL),
            "macro_smape_percent",
        ].iloc[0]
    )
    deployment_delta_macro = (
        deployment_tuned_macro - deployment_baseline_macro
    )
    deployment_relative_change = (
        deployment_delta_macro / deployment_baseline_macro * 100.0
    )
    deployment_macro_boot = paired_bootstrap[
        paired_bootstrap["track"].eq("fold_local_augmented_nested")
        & paired_bootstrap["scope"].eq("equal_target_macro")
    ].iloc[0]

    selection_table = final_selection[
        final_selection["selected_for_forecast"]
    ].copy()
    selection_table["指标"] = selection_table["metric"].map(TARGET_LABELS_CN)
    selection_table["边界命中"] = selection_table["boundary_hit"]
    selection_table["选择轨道"] = selection_table["selection_track"].map(
        {
            "fold_local_augmented_sensitivity": "逐内折重建增强数据（部署一致）",
            "official_only": "仅物理证据",
        }
    )
    selection_table["alpha 与 λ 关系"] = "sklearn alpha = λ"
    selection_table["λ显示"] = selection_table["selected_lambda"].map(
        lambda value: f"{float(value):g}"
    )

    forecast_table = forecasts.copy()
    forecast_table["指标"] = forecast_table["metric"].map(TARGET_LABELS_CN)
    forecast_table["模型"] = forecast_table["model"].map(
        {
            BASELINE_MODEL: "优化前",
            TUNED_MODEL: "优化后",
        }
    )
    forecast_table["λ显示"] = forecast_table["lambda"].map(
        lambda value: f"{float(value):g}"
    )
    diagnostic_table = diagnostics.copy()
    diagnostic_table["指标"] = diagnostic_table["metric"].map(TARGET_LABELS_CN)
    diagnostic_table["模型"] = diagnostic_table["model"].map(
        {BASELINE_MODEL: "优化前", TUNED_MODEL: "优化后"}
    )
    diagnostic_table["λ显示"] = diagnostic_table["lambda"].map(
        lambda value: f"{float(value):g}"
    )
    holdout_table = holdout_by_target[
        holdout_by_target["model"].isin([BASELINE_MODEL, TUNED_MODEL])
    ].copy()
    holdout_table["指标"] = holdout_table["metric"].map(TARGET_LABELS_CN)
    holdout_table["模型"] = holdout_table["model"].map(
        {BASELINE_MODEL: "优化前", TUNED_MODEL: "优化后"}
    )
    stress_table = stress_by_target[
        stress_by_target["model"].isin([BASELINE_MODEL, TUNED_MODEL])
    ].copy()
    stress_table["指标"] = stress_table["metric"].map(TARGET_LABELS_CN)
    stress_table["模型"] = stress_table["model"].map(
        {BASELINE_MODEL: "优化前", TUNED_MODEL: "优化后"}
    )
    stress_table["相对上一期法改进率（%）"] = (
        stress_table["smape_skill_vs_naive"] * 100.0
    )
    spend_table = implied_spend.copy()
    spend_table["模型"] = spend_table["model"].map(
        {BASELINE_MODEL: "优化前", TUNED_MODEL: "优化后"}
    )
    support_summary = (
        design_support[design_support["scope"].eq("official_outer_training")]
        .groupby("metric", as_index=False)
        .agg(
            外层折数=("fold_id", "size"),
            中心化秩最小值=("centered_feature_rank", "min"),
            中心化秩最大值=("centered_feature_rank", "max"),
            恢复期正例最大数=("post_2022_positive_count", "max"),
        )
    )
    support_summary["指标"] = support_summary["metric"].map(TARGET_LABELS_CN)
    ablation_summary = feature_ablation[
        feature_ablation["scope"].eq(
            "official_outer_fixed_final_lambda_exploratory"
        )
    ].copy()
    ablation_summary["指标"] = ablation_summary["metric"].map(TARGET_LABELS_CN)
    ablation_summary["特征规格"] = ablation_summary["feature_set"].map(
        {
            "time_only": "仅时间趋势",
            "time_plus_pandemic": "时间+疫情期",
            "time_plus_recovery": "时间+恢复期",
            "full_regime_features": "完整三特征",
        }
    )
    ablation_final = feature_ablation[
        feature_ablation["scope"].eq("final_augmented_2030")
    ].copy()
    ablation_final["指标"] = ablation_final["metric"].map(TARGET_LABELS_CN)
    ablation_final["特征规格"] = ablation_final["feature_set"].map(
        {
            "time_only": "仅时间趋势",
            "time_plus_pandemic": "时间+疫情期",
            "time_plus_recovery": "时间+恢复期",
            "full_regime_features": "完整三特征",
        }
    )

    parameter_table = parameters[parameters["model"].eq(TUNED_MODEL)].copy()
    parameter_table["指标"] = parameter_table["metric"].map(TARGET_LABELS_CN)
    parameter_table["参数"] = parameter_table["parameter"].map(
        {"intercept": "截距项", **FEATURE_LABELS_CN}
    )

    q1_table = problem1_summary.copy()
    q1_table["单位"] = q1_table["metric"].map(
        {
            "tourist_visits": "万人次",
            "tourism_comprehensive_income": "亿元",
            "jizhou_gdp": "亿元",
            "jizhou_tertiary_value_added": "亿元",
        }
    )
    q1_table["增长口径"] = q1_table.apply(
        lambda row: (
            f"2010—2019 CAGR {float(row['cagr_2010_2019_percent']):.3f}%"
            if pd.notna(row["cagr_2010_2019_percent"])
            else (
                f"2010—2018 {float(row['cagr_2010_2018_percent']):.3f}%；"
                f"2019—2025 {float(row['cagr_2019_2025_percent']):.3f}%"
            )
        ),
        axis=1,
    )
    q1_diagnostics = problem1_diagnostics.copy()
    q1_diagnostics["指标"] = q1_diagnostics["metric"].map(TARGET_LABELS_CN)
    imputation_table = final_training[
        benchmark._coerce_boolean(final_training["is_simulated"])
    ].copy()
    imputation_table["指标"] = imputation_table["metric"].map(TARGET_LABELS_CN)
    imputation_table["单位"] = imputation_table["metric"].map(TARGET_UNITS_CN)
    imputation_table["方法"] = "双侧对数线性插值（仅训练标签）"
    q1_forecast_2030 = problem1_forecasts[
        problem1_forecasts["year"].eq(2030)
    ].set_index("metric")

    scenario_table = scenarios.pivot_table(
        index=["scenario", "scenario_label_cn", "year"],
        columns="metric",
        values="value",
    ).reset_index()
    scenario_table.rename(
        columns={
            "scenario_label_cn": "情景",
            "tourist_visits": "游客量（万人次）",
            "tourism_comprehensive_income": "综合收入（亿元）",
            "nominal_spend_per_visit": "人均次消费（元）",
        },
        inplace=True,
    )
    scenario_table.sort_values(["scenario", "year"], inplace=True)

    policy_table = policy_sensitivity.pivot_table(
        index=["factor", "setting"],
        columns="metric",
        values="delta_percent",
    ).reset_index()
    policy_table["因素"] = policy_table["factor"].map(
        {
            "source_market_growth": "客源市场年增速±2个百分点",
            "new_format_spend_growth": "人均消费年增速2%/4%",
            "policy_coordination_multiplier": "协同水平乘数0.95/1.05",
            "external_shock": "2026外部冲击-15%/0%",
        }
    )
    policy_table["设定"] = policy_table["setting"].map(
        {"low": "低", "high": "高"}
    )
    policy_table.rename(
        columns={
            "tourist_visits": "游客量变化（%）",
            "tourism_comprehensive_income": "收入变化（%）",
        },
        inplace=True,
    )
    policy_table.sort_values(["factor", "setting"], inplace=True)

    recommendation_table = pd.DataFrame(
        [
            {
                "方向": "拓展京津冀客源市场",
                "监测KPI": "客源驱动的游客量年增速相对基准±2个百分点",
                "2030量化效果": "高设定使游客量和收入均约+9.91%；低设定约-9.18%",
            },
            {
                "方向": "发展住宿、文创与夜游等新业态",
                "监测KPI": "名义人均次消费年增速至少3%，争取4%",
                "2030量化效果": "4%设定使收入约+4.95%；2%设定约-4.76%",
            },
            {
                "方向": "强化跨区域政策与产品协同",
                "监测KPI": "协同水平乘数不低于1.00，目标1.05",
                "2030量化效果": "1.05使游客量和收入均+5%；0.95则均-5%",
            },
            {
                "方向": "建立外部冲击预案",
                "监测KPI": "跟踪预订、过夜率和人均消费偏离，触发分级响应",
                "2030量化效果": "持续-15%水平缺口对应游客量3016.5万人次、收入288.50亿元",
            },
        ]
    )

    tourist_2030_before = float(
        forecasts[
            forecasts["metric"].eq("tourist_visits")
            & forecasts["model"].eq(BASELINE_MODEL)
            & forecasts["year"].eq(2030)
        ]["forecast"].iloc[0]
    )
    tourist_2030_after = float(
        forecasts[
            forecasts["metric"].eq("tourist_visits")
            & forecasts["model"].eq(TUNED_MODEL)
            & forecasts["year"].eq(2030)
        ]["forecast"].iloc[0]
    )
    income_2030_before = float(
        forecasts[
            forecasts["metric"].eq("tourism_comprehensive_income")
            & forecasts["model"].eq(BASELINE_MODEL)
            & forecasts["year"].eq(2030)
        ]["forecast"].iloc[0]
    )
    income_2030_after = float(
        forecasts[
            forecasts["metric"].eq("tourism_comprehensive_income")
            & forecasts["model"].eq(TUNED_MODEL)
            & forecasts["year"].eq(2030)
        ]["forecast"].iloc[0]
    )
    spend_after_2030 = float(
        implied_spend[
            implied_spend["model"].eq(TUNED_MODEL)
            & implied_spend["year"].eq(2030)
        ]["implied_spend_cny_per_visit"].iloc[0]
    )
    q1_tourist_2030 = float(q1_forecast_2030.loc["tourist_visits", "forecast"])
    q1_income_2030 = float(
        q1_forecast_2030.loc["tourism_comprehensive_income", "forecast"]
    )
    tourist_ablation_min = float(
        ablation_final.loc[
            ablation_final["metric"].eq("tourist_visits"), "forecast_2030"
        ].min()
    )
    tourist_ablation_max = float(
        ablation_final.loc[
            ablation_final["metric"].eq("tourist_visits"), "forecast_2030"
        ].max()
    )
    income_ablation_min = float(
        ablation_final.loc[
            ablation_final["metric"].eq("tourism_comprehensive_income"),
            "forecast_2030",
        ].min()
    )
    income_ablation_max = float(
        ablation_final.loc[
            ablation_final["metric"].eq("tourism_comprehensive_income"),
            "forecast_2030",
        ].max()
    )

    report = f"""# C题：Ridge 模型正则化调参与优化前后比较报告

## 题目要求覆盖矩阵

| 题面任务 | 本报告对应内容 | 直接产物 | 解释边界 |
| --- | --- | --- | --- |
| 问题1：整理指标并建立简单增长模型 | 独立列出四项指标的数据覆盖、增长口径和疫情前指数增长模型诊断，并作为 Ridge 合理性反例 | 本报告“问题1衔接”表格及原统一报告 `problem1_*` | 疫情前模型只描述2010—2019年，不能机械代表恢复期 |
| 问题2：模型评判、参数估计、2026—2030预测及合理性 | 对选定的原尺度 Ridge 做严格嵌套调参，报告优化前后回测、诊断、系数自助法区间与未来条件区间 | `augmented_outer_*`、`official_outer_*`、`final_*`、三张图 | 2024是伪留出；区间不含选参和结构不确定性 |
| 问题3：三情景、敏感性与对策 | 完整列出三情景五年路径、OAT敏感性、量化建议，并与优化后点预测衔接 | 本报告“问题3衔接”表格、`final_forecasts_2026_2030.csv` | 三情景与OAT是透明会计假设，不是历史因果识别 |
| 资料获取日期 | 沿用统一数据层中实际引用的23个官方优选来源 | 原统一报告 `problem_source_access_dates.csv` | 23项均记录2026-08-17；不推广到来源总表全部条目 |

## 摘要

本次优化保留了统一比较中胜出的模型族：原尺度目标、训练窗内标准化，以及时间趋势、2020—2022年阶段和2023年后恢复阶段三个特征。标准 Ridge 只有一个L2惩罚强度；scikit-learn把它命名为 `alpha`，公式中通常记作 $\\lambda$，所以“调 alpha 和 lambda”在这里是对同一个量调参。若把损失写成 $aL+\\lambda\\|\\beta\\|_2^2$，预测只能识别比值 $\\lambda/a$，不能把二者当成两个独立旋钮。

与最终15行增强拟合机制一致的严格嵌套轨中，训练标签在每个折内重建、外层测试仍只用官方实际值；优化前两个目标等权平均sMAPE为 {deployment_baseline_macro:.3f}%，优化后为 {deployment_tuned_macro:.3f}%，绝对变化 {deployment_delta_macro:.3f} 个百分点、相对变化 {deployment_relative_change:.3f}%。为与统一模型比较的未增强训练口径连续，另报仅物理训练轨：{official_baseline_macro:.3f}%降至{official_tuned_macro:.3f}%，变化 {delta_macro:.3f} 个百分点（{relative_change:.3f}%）。两轨的改善都很小。

部署一致轨按目标分别重采样外层折、再等权合并的描述性2.5%—97.5%分位范围为 [{float(deployment_macro_boot['bootstrap_ci95_lower']):.3f}, {float(deployment_macro_boot['bootstrap_ci95_upper']):.3f}] 个百分点；仅物理训练轨对应范围为 [{float(macro_boot['bootstrap_ci95_lower']):.3f}, {float(macro_boot['bootstrap_ci95_upper']):.3f}]。两目标不是按共同年份联合抽样，滚动折也相互重叠，因此这些都不是一般意义的95%推断区间；两个范围均跨过0，不能据此宣称稳定泛化提升。最终两个目标都选择正式网格下界 $\\lambda=10^{{-4}}$，扩展到 $10^{{-6}}$ 和0后分数仍只向无惩罚边界缓慢改善，说明数据支持的是“惩罚趋近于零”，而不是找到了更好的正则强度。

因此，本报告把 $\\lambda=10^{{-4}}$ 作为可复现的优化后候选，用于展示2026—2030路径，但不建议仅凭这次小样本调参替换原 $\\lambda=0.1$ 的审慎主结论。优化后2030年预测为游客量 {tourist_2030_after:.1f} 万人次、综合收入 {income_2030_after:.2f} 亿元，优化前分别为 {tourist_2030_before:.1f} 万人次和 {income_2030_before:.2f} 亿元；变化幅度远小于模型结构、疫情缺失和情景假设带来的不确定性。

![正则化搜索曲线]({lambda_search_figure})

## 数据与防泄漏协议

两条检验共用 `data/unified/rolling_origin_folds.csv` 的官方优选记录。游客量和综合收入各有6个外层实际测试点，最晚到2023年；每个外层测试年之前的记录才可进入训练。优化前模型始终使用 $\\lambda=0.1$。优化后流程在每个外层训练窗内部再次执行扩展窗口验证，内层至少4条物理记录起步，以原尺度sMAPE选择正的 $\\lambda$；内层验证点少于3个时预先规定回退到0.1，避免用一两个点制造“最优值”。

部署一致的增强选参轨在每个内层折都从当时可见的物理记录重新生成插值或尾部训练值，内层验证年及其后的边界不会进入生成过程；未增强的仅物理训练轨用于严格复核和衔接原统一比较。两条轨的外层评分都只使用官方实际测试值。2024年两项官方值在选参和伪留出评价之前保持隔离；研究者在研究设计前已经见过2024资料，因此该检查不是真正前瞻的未知样本。最终预测的 $\\lambda$ 只根据截至2023年的物理证据及其折内增强训练选择，随后才在冻结超参数的前提下把2024纳入2010—2024最终训练契约重拟合；2025政府目标代理值没有进入训练、调参或区间估计。

## 问题1衔接：指标整理与简单增长模型

四项题面指标的数据覆盖如下。缺失年份在事实层保持缺失；只有题目2模型的训练层在两个已知边界之间做双侧对数线性插值，每个目标生成3个模拟训练标签。地区生产总值和第三产业增加值在2019年存在统计范围断点，因此不跨断点报告2010—2019 CAGR，而分段报告2010—2018与2019—2025增长。

{markdown_table(q1_table, ['indicator_label_cn', '单位', 'nonmissing_count', 'missing_years', 'first_nonmissing_year', 'first_nonmissing_value', 'last_nonmissing_year', 'last_nonmissing_value', '增长口径', 'growth_2023_2024_percent'], ['指标', '单位', '非缺失年数', '缺失年份', '首年', '首值', '末年', '末值', '长期增长口径', '2023—2024增长（%）'])}

用于最终 Ridge 训练的6个补值如下；它们均标记为 `simulated_training_only`，不作为官方事实、外层测试真值或政策成效证据。

{markdown_table(imputation_table, ['指标', 'year', 'value', '单位', 'source_years', 'boundary_left_value', 'boundary_right_value', '方法'], ['指标', '年份', '训练补值', '单位', '边界年份', '左边界值', '右边界值', '方法'])}

![题目1四项指标]({problem1_figure})

题目1的简单模型固定为疫情前指数增长：$\\log(y_t)=\\beta_0+\\beta_1(t-2010)$，年增长率为 $\\exp(\\beta_1)-1$。它只使用2010—2019可用记录，适合描述疫情前趋势，不适合作为疫情后唯一预测。

{markdown_table(q1_diagnostics, ['指标', 'n', 'annual_growth_rate_percent', 'r_squared_log', 'rmse_original_units', 'mape_percent', 'durbin_watson', 'jarque_bera_p'], ['指标', '样本数', '年增长率（%）', '对数尺度R²', '原尺度RMSE', 'MAPE（%）', 'DW', 'JB p值'])}

异常与恢复也必须分开解释：收入从2019年的165亿元降到2021年的110亿元，2023年恢复至191.5亿元、2024年增至221亿元；游客量2023年的2363万人次仍比2019年的2800万人次低15.6%，2024年回升11.85%至2643万人次。缺失的疫情年份不能靠插值反推真实冲击幅度。同一未模拟滚动协议下，简单模型的游客量/综合收入sMAPE为15.600%/26.819%；机械外推到2030年分别达到 {q1_tourist_2030:.1f} 万人次/{q1_income_2030:.1f} 亿元。后文将其作为合理性反例，而不是与恢复期 Ridge 混称同一结构。

## 超参数搜索与可识别性

正式候选为 $10^{{-4}}$ 至 $10^3$ 的29点对数网格，包含旧值0.1。0、$10^{{-6}}$ 和$10^{{-5}}$只用于边界诊断。确定性并列规则在数值完全相同时偏向更强惩罚，但本次两个目标均直接命中最小正式候选。最终选择如下。

{markdown_table(selection_table, ['指标', '选择轨道', 'λ显示', 'inner_validation_count', 'mean_inner_smape_percent', '边界命中', 'boundary_side', 'alpha 与 λ 关系'], ['指标', '最终选择轨道', '正式 λ', '内层验证点', '最小内层sMAPE（%）', '边界命中', '边界方向', '代码记号'])}

最终部署按逐内折重建增强数据的轨道选参，使调参训练机制与最终15行增强拟合一致；仅物理证据轨独立复核后对两个目标也都选择 $\\lambda=10^{{-4}}$。两条正式评估轨都只使用外层官方实际测试值计分，增强标签从不充当外层“事实”得分。

这次搜索的主要限制不是网格精度，而是设计支持。下表显示，在截至2023年的外层训练窗里，三个中心化特征的有效秩大多只有1；`post_2022` 在所有外层训练折中都没有正例。换言之，滚动检验主要验证了时间斜率是否需要收缩，无法验证最终预测所依赖的恢复期系数。最终2010—2024增强设计虽达到满秩，但恢复期只有2023和2024两个年份，且每个目标仍有3个插值训练标签。

{markdown_table(support_summary, ['指标', '外层折数', '中心化秩最小值', '中心化秩最大值', '恢复期正例最大数'], ['指标', '外层折数', '秩最小值', '秩最大值', '恢复期正例最大数'])}

## 优化前后滚动检验

先列与原统一模型比较完全同口径的仅物理训练轨。表中“相对上一期法改进率”大于0才表示模型优于直接沿用上一期实际值；外层测试均为官方实际记录。

{markdown_table(official_primary, ['指标', '模型', 'n_test', 'smape_percent', 'naive_smape_percent', '相对上一期法改进率（%）', 'worst_point_smape_percent', 'delta_smape_vs_baseline_percent_points'], ['指标', '模型', '测试点', 'sMAPE（%）', '上一期法sMAPE（%）', '相对上一期法改进率（%）', '最差单点sMAPE（%）', '较优化前变化（百分点）'])}

{markdown_table(macro_official_rows, ['模型', 'macro_smape_percent', '相对上一期法等权改进（%）', 'beats_naive_all_targets', 'worst_point_smape_percent', 'delta_macro_smape_vs_baseline_percent_points'], ['模型', '两目标等权sMAPE（%）', '相对上一期法等权改进（%）', '两个目标都胜上一期法', '总体最差单点sMAPE（%）', '较优化前变化（百分点）'])}

严格嵌套结果仅有轻微平均改善，并且疫情/恢复期最差点没有同步改善。早期外层折因内层验证不足而按规则保留0.1，后期折选择 $10^{{-4}}$。若事后把最终小 $\\lambda$ 回放到全部外层年份，未模拟轨可得到更低的描述性sMAPE，但该做法利用了后期信息选择早期超参数，只能列作乐观敏感性，不能替代上表。

![优化前后外层预测]({backtest_figure})

下表是与最终拟合机制一致的折内增强训练轨，也是部署流程的直接回测。模拟标签只进入每折训练，外层得分仍由官方实际值计算；插值路径会平滑不可观测的疫情冲击，因此该轨不能把模拟标签升级为官方事实。

{markdown_table(simulated_primary, ['指标', '模型', 'n_test', 'smape_percent', '相对上一期法改进率（%）', 'worst_point_smape_percent', 'delta_smape_vs_baseline_percent_points'], ['指标', '模型', '测试点', 'sMAPE（%）', '相对上一期法改进率（%）', '最差单点sMAPE（%）', '较优化前变化（百分点）'])}

{markdown_table(macro_simulated_rows, ['模型', 'macro_smape_percent', '相对上一期法等权改进（%）', 'beats_naive_all_targets', 'worst_point_smape_percent', 'delta_macro_smape_vs_baseline_percent_points'], ['模型', '两目标等权sMAPE（%）', '相对上一期法等权改进（%）', '两个目标都胜上一期法', '总体最差单点sMAPE（%）', '较优化前变化（百分点）'])}

## 有价值但未采纳为主结论的尝试

边界扩展显示，$\\lambda=10^{{-4}}$ 到0之间的内层误差曲线近乎平坦，最小值仍在0方向；这意味着继续细化正数网格不会得到内部最优的 Ridge 惩罚。把损失权重和惩罚权重分别称作 alpha、lambda 也不能增加自由度，因为只会得到相同的有效比值。采用 Elastic Net 的混合参数会引入L1惩罚并改变模型族，不符合小组已经选定 Ridge 的决定；Bayesian Ridge 的噪声精度和权重精度又属于另一概率模型，均未冒充本次第二个超参数。

一标准误规则也未作为主选择规则。极少数内层折跨越疫情异常，误差标准误很大，会允许明显更强的惩罚而产生任意过度收缩。特征消融结果列在下表；这些数值使用最终小 $\\lambda$ 事后固定到全部外层折，只用于揭示结构敏感性，不能作为重新选模证据。

{markdown_table(ablation_summary, ['指标', '特征规格', 'n_test', 'mean_smape_percent', 'worst_smape_percent'], ['指标', '特征规格', '测试点', '平均sMAPE（%）', '最差sMAPE（%）'])}

同样的特征消融用于最终增强训练时，2030点预测范围为游客量 {tourist_ablation_min:.1f}—{tourist_ablation_max:.1f} 万人次、综合收入 {income_ablation_min:.2f}—{income_ablation_max:.2f} 亿元，明显大于单纯把 $\\lambda$ 从0.1降到0.0001造成的差异。

{markdown_table(ablation_final, ['指标', '特征规格', 'forecast_2030'], ['指标', '特征规格', '2030点预测'])}

已有统一比较还显示，对数目标 `ridge_regime` 与当前原尺度目标产生的差异大于常规 $\\lambda$ 微调。由于小组本轮明确优化已选定的原尺度 Ridge，本报告把目标变换保留为既有敏感性证据，没有悄然换模型来制造更好的分数。

## 2024伪留出与跨疫情压力测试

2024伪留出每个目标只有1个点，只能检查方向，不能用于选择或显著性判断。

{markdown_table(holdout_table, ['指标', '模型', 'n_test', 'mae', 'mape_percent', 'smape_percent'], ['指标', '模型', '测试点', 'MAE', 'MAPE（%）', 'sMAPE（%）'])}

2019截断压力测试只用疫情前记录，逐个预测之后可观察年份。它检验结构突变下会错到什么程度，不是疫情的因果效应估计。

{markdown_table(stress_table, ['指标', '模型', 'n_test', 'mape_percent', 'smape_percent', '相对上一期法改进率（%）'], ['指标', '模型', '测试点', 'MAPE（%）', 'sMAPE（%）', '相对上一期法改进率（%）'])}

## 最终拟合、参数与区间

最终训练契约对每个目标包含2010—2024共15个年度位置，其中12个物理证据值和3个训练期内插值值。对标准化设计矩阵 $Z$ 和原尺度目标 $y$，拟合目标为 $\\min_{{b_0,\\beta}}\\sum_i(y_i-b_0-z_i^T\\beta)^2+\\lambda\\lVert\\beta\\rVert_2^2$，截距不惩罚。下表的样本内指标只能比较同一目标、同一尺度下的优化前后规格，不能跨游客量与收入单位合并。

{markdown_table(diagnostic_table, ['指标', '模型', 'λ显示', 'r_squared', 'adjusted_r_squared', 'rmse', 'mape_percent', 'aicc', 'durbin_watson', 'jarque_bera_p'], ['指标', '模型', 'λ', 'R²', '调整后R²', 'RMSE', 'MAPE（%）', 'AICc', 'DW', 'JB p值'])}

Ridge系数不套用普通最小二乘t检验或p值。优化后标准化系数如下；截距在原目标尺度上，其余系数对应预测变量增加一个训练样本标准差。区间来自固定随机种子 {RANDOM_SEED} 的{BOOTSTRAP_REPETITIONS:,}次固定设计残差自助法。

{markdown_table(parameter_table, ['指标', '参数', 'estimate', 'bootstrap_ci95_lower', 'bootstrap_ci95_upper', 'feature_training_mean', 'feature_training_scale'], ['指标', '参数', '估计值', '条件95%下限', '条件95%上限', '训练均值', '训练标准差'])}

这些系数是预测参数，不是疫情或政策因果效应；尤其恢复期系数只由两个恢复年份和插值路径弱支撑。普通LOOCV在预先构建的增强表上既不遵守时间顺序，留出物理年份的信息也可能经插值标签残留，因此本优化产物不再报告该指标，模型比较以逐折重建训练数据的时间有序外层检验为准。残差自助法还假设固定设计下残差可交换且忽略序列相关，因此以上系数范围及后文预测区间都只是启发式条件区间。

## 2026—2030优化前后预测

{markdown_table(forecast_table, ['指标', '模型', 'year', 'λ显示', 'forecast', 'mean_ci95_lower', 'mean_ci95_upper', 'prediction_interval95_lower', 'prediction_interval95_upper'], ['指标', '模型', '年份', 'λ', '点预测', '平均响应95%下限', '平均响应95%上限', '单次预测95%下限', '单次预测95%上限'])}

所有区间都来自固定设计、残差独立可交换的启发式自助法，是给定目标尺度、特征、插值训练值和冻结 $\\lambda$ 后的模型条件区间；它们不是五年同时置信带，也不覆盖超参数选择、序列相关、结构变化、缺失机制、统计口径或政策变化的不确定性。优化后曲线相对优化前略高，原因是更小的惩罚减弱了对时间斜率的收缩；这不是发现了新的增长机制。

![未来预测、题目1外推与题目3情景包络]({forecast_figure})

## 题目2预测合理性与题目3衔接

题目1疫情前指数增长模型的年增长率为游客量14.389%、综合收入18.487%，未模拟滚动sMAPE分别为15.600%和26.819%。若机械外推，它在2030年给出游客量 {q1_tourist_2030:.1f} 万人次和收入 {q1_income_2030:.1f} 亿元，远高于疫情后恢复期数据与政策情景尺度。优化前后 Ridge 都保留阶段特征和加性原尺度趋势，因此2026—2030路径显著更平缓；这支持继续使用 Ridge 作为题目2主预测族，而不是恢复疫情前指数外推。

题目3以2025政府目标代理为锚，按收入增长、人均次消费增长和冲击假设构造三条透明会计情景；代理值不是2025实际观测。完整路径如下。

{markdown_table(scenario_table, ['情景', 'year', '游客量（万人次）', '综合收入（亿元）', '人均次消费（元）'], ['情景', '年份', '游客量（万人次）', '综合收入（亿元）', '人均次消费（元）'])}

三情景2030年的边际包络为游客量2620.2—4055.8万人次、综合收入238.66—407.10亿元，基准情景为3548.9万人次和339.41亿元。优化后2030年游客量 {tourist_2030_after:.1f} 万人次与收入 {income_2030_after:.2f} 亿元分别位于各自边际范围内；但二者相除得到的人均次消费 {spend_after_2030:.1f} 元，低于悲观情景的910.9元。因此优化后两个独立拟合目标的组合并不对应三条联合会计情景中的任何一条，不能写成“整体落在情景包络内”。它传达的是客流路径偏高、收入转化偏弱的结构风险。

{markdown_table(spend_table, ['模型', 'year', 'implied_spend_cny_per_visit'], ['模型', '年份', '隐含人均次消费（元/人次）'])}

题目3的单因素逐次（OAT）敏感性以基准情景为参照；每次只改一个假设，百分比是2030年相对基准的变化，不是历史因果效应。四类扰动幅度并不统一（增长率百分点、水平乘数和一次性冲击混合），所以绝对变化不能解释为可比的“杠杆效率”排名。

{markdown_table(policy_table, ['因素', '设定', '游客量变化（%）', '收入变化（%）'], ['因素', '设定', '游客量变化（%）', '收入变化（%）'])}

在本表预设幅度下，外部冲击给出最大的下行情景（-15%），但它主要是外生风险；可控因素中，客源市场增速同时影响客流和收入，五年累计变化约-9.18%至+9.91%，是总量端的核心因素；人均消费增速只改变收入，约-4.76%至+4.95%，是把客流转成收入的核心因素；协同乘数对两项目标一比一传导±5%。这个结论是机制和预设情景量级的识别，不是标准化弹性或历史因果排序。由此得到的可监测建议如下。优化只改变题目2的统计基准路径，没有把情景假设变成历史因果系数。

{markdown_table(recommendation_table, ['方向', '监测KPI', '2030量化效果'], ['方向', '监测KPI', '2030量化效果'])}

实际应用应同时监测游客量、综合收入、隐含人均次消费、过夜率与相对预测的恢复偏差；若游客量达到模型路径而人均消费持续走弱，应优先调整产品结构，而不是只追求客流总量。

## 结论与使用建议

这次调参回答了“能否仅靠 alpha/lambda 进一步提升模型”的问题：部署一致轨和仅物理训练轨都只得到很小且不稳定的平均改善，最优点落在无惩罚边界附近，无法证明正则化调参带来可靠能力提升。$\\lambda=10^{{-4}}$ 是本次可复现的优化后部署候选，也可在后续新数据到来时重新检验；当前报告同时保留 $\\lambda=0.1$ 基线。若论文必须给出单一主结论，建议沿用原 Ridge 模型族并把两条参数路径并列，强调结构不确定性高于超参数差异。

后续最有价值的更新不是继续把网格切得更细，而是获得2025及以后真实游客量、综合收入、过夜率和消费结构数据。新增至少数个恢复期年份后，应重新执行同一嵌套协议；届时 `post_2022` 系数和正则强度才可能获得真正的样本外验证。

## 复现

```bash
MPLCONFIGDIR=/tmp/jizhou-mpl XDG_CACHE_HOME=/tmp/jizhou-xdg \\
  .venv/bin/python code/scripts/optimize_ridge_model.py
.venv/bin/python -m unittest discover -s tests -v
```

脚本、网格、随机种子、逐折选择、边界诊断、预测、参数和图均保存在 `code/scripts/optimize_ridge_model.py` 与 `outputs/ridge_optimization/`。本报告在 `ridge` 分支生成；旧的 `raw_target_ridge_alpha_0.1` 及统一比较产物没有被覆盖。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def run_optimization(
    *,
    unified_dir: Path = DEFAULT_UNIFIED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = benchmark.load_benchmark_inputs(unified_dir)

    official_predictions, official_candidates, official_inner_folds = (
        evaluate_rolling_track(
            inputs.rolling_folds,
            track="official_only_nested",
            simulate_missing_years=False,
        )
    )
    simulated_predictions, simulated_candidates, simulated_inner_folds = (
        evaluate_rolling_track(
            inputs.rolling_folds,
            track="fold_local_augmented_nested",
            simulate_missing_years=True,
        )
    )
    all_rolling_predictions = pd.concat(
        [official_predictions, simulated_predictions], ignore_index=True
    )
    rolling_by_target, rolling_macro = summarize_predictions(
        all_rolling_predictions
    )
    official_by_target = rolling_by_target[
        rolling_by_target["track"].eq("official_only_nested")
    ].copy()
    official_macro = rolling_macro[
        rolling_macro["track"].eq("official_only_nested")
    ].copy()
    simulated_by_target = rolling_by_target[
        rolling_by_target["track"].eq("fold_local_augmented_nested")
    ].copy()
    simulated_macro = rolling_macro[
        rolling_macro["track"].eq("fold_local_augmented_nested")
    ].copy()
    fold_bootstrap = paired_fold_bootstrap(all_rolling_predictions)

    holdout_predictions, holdout_candidates, holdout_inner_folds = evaluate_fixed_scope(
        inputs.primary_train,
        inputs.primary_test,
        track="pseudo_holdout_2024",
    )
    holdout_by_target, holdout_macro = summarize_predictions(holdout_predictions)
    stress_predictions, stress_candidates, stress_inner_folds = evaluate_fixed_scope(
        inputs.stress_train,
        inputs.stress_test,
        track="cross_regime_stress_2019_cutoff",
    )
    stress_by_target, stress_macro = summarize_predictions(stress_predictions)

    final_selection_rows: list[dict[str, object]] = []
    final_search_frames: list[pd.DataFrame] = []
    selected_lambdas: dict[str, float] = {}
    for metric in TARGETS:
        physical = inputs.primary_train[
            inputs.primary_train["metric"].eq(metric)
        ].sort_values("year")
        official_selection = tune_raw_ridge_lambda(
            physical,
            simulate_missing_years=False,
            scope=f"final_selection_official__{metric}",
        )
        augmented_selection = tune_raw_ridge_lambda(
            physical,
            simulate_missing_years=True,
            scope=f"final_selection_augmented__{metric}",
        )
        boundary_selection = tune_raw_ridge_lambda(
            physical,
            candidate_lambdas=BOUNDARY_DIAGNOSTIC_GRID,
            simulate_missing_years=False,
            min_inner_validations=MIN_INNER_VALIDATIONS,
            scope=f"final_boundary_diagnostic__{metric}",
        )
        selected_lambdas[metric] = augmented_selection.selected_lambda
        for selection_track, result, selected_for_forecast in (
            ("official_only", official_selection, False),
            ("fold_local_augmented_sensitivity", augmented_selection, True),
            ("zero_boundary_diagnostic", boundary_selection, False),
        ):
            surface = result.candidate_summary.copy()
            surface["selection_track"] = selection_track
            surface["selected_for_forecast"] = selected_for_forecast
            final_search_frames.append(surface)
            best_score = float(
                surface.loc[
                    surface["lambda"].eq(result.selected_lambda),
                    "mean_inner_smape_percent",
                ].iloc[0]
            )
            final_selection_rows.append(
                {
                    "metric": metric,
                    "selection_track": selection_track,
                    "selected_lambda": result.selected_lambda,
                    "alpha_code_parameter": result.selected_lambda,
                    "selection_status": result.status,
                    "boundary_hit": result.boundary_hit,
                    "boundary_side": result.boundary_side,
                    "inner_validation_count": result.inner_validation_count,
                    "mean_inner_smape_percent": best_score,
                    "selected_for_forecast": selected_for_forecast,
                    "training_max_year": int(physical["year"].max()),
                    "uses_2024_for_selection": bool(physical["year"].ge(2024).any()),
                }
            )
    final_selection = pd.DataFrame(final_selection_rows)
    final_search = pd.concat(final_search_frames, ignore_index=True)
    if final_selection["uses_2024_for_selection"].any():
        raise AssertionError("2024 leaked into final lambda selection")

    final_training = benchmark.build_problem2_final_training(inputs.observations)
    forecast_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    for metric in TARGETS:
        metric_train = final_training[final_training["metric"].eq(metric)].copy()
        for model_name, ridge_lambda in (
            (BASELINE_MODEL, BASELINE_LAMBDA),
            (TUNED_MODEL, selected_lambdas[metric]),
        ):
            forecast, diagnostic, parameters = build_final_outputs(
                metric_train,
                ridge_lambda=ridge_lambda,
                model_name=model_name,
            )
            forecast_frames.append(forecast)
            diagnostic_frames.append(diagnostic)
            parameter_frames.append(parameters)
    forecasts = pd.concat(forecast_frames, ignore_index=True).sort_values(
        ["metric", "model", "year"], ignore_index=True
    )
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    if forecasts["uses_2025_as_training"].any() or diagnostics[
        "uses_2025_as_training"
    ].any():
        raise AssertionError("2025 entered final Ridge training")

    baseline_reference_path = (
        ROOT / "outputs/unified_model_benchmark/problem2_forecasts_2026_2030.csv"
    )
    baseline_reference = pd.read_csv(baseline_reference_path)
    baseline_reference = baseline_reference[
        baseline_reference["model"].eq(BASELINE_MODEL)
    ]
    reproduced = forecasts[forecasts["model"].eq(BASELINE_MODEL)]
    check = reproduced.merge(
        baseline_reference,
        on=["metric", "model", "year"],
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )
    validation_columns = [
        "forecast",
        "mean_ci95_lower",
        "mean_ci95_upper",
        "prediction_interval95_lower",
        "prediction_interval95_upper",
    ]
    max_baseline_difference = max(
        float(
            np.max(
                np.abs(
                    check[f"{column}_new"].to_numpy(dtype=float)
                    - check[f"{column}_reference"].to_numpy(dtype=float)
                )
            )
        )
        for column in validation_columns
    )
    if max_baseline_difference > 1.0e-8:
        raise AssertionError(
            f"new pipeline failed to reproduce baseline: {max_baseline_difference}"
        )

    pivot = forecasts.pivot(
        index=["model", "year"], columns="metric", values="forecast"
    ).reset_index()
    pivot["implied_spend_cny_per_visit"] = (
        pivot["tourism_comprehensive_income"]
        / pivot["tourist_visits"]
        * 10_000.0
    )
    implied_spend = pivot[
        [
            "model",
            "year",
            "tourist_visits",
            "tourism_comprehensive_income",
            "implied_spend_cny_per_visit",
        ]
    ].copy()

    design_support = build_design_support_diagnostics(
        inputs.rolling_folds, final_training
    )
    lambda_sensitivity = build_lambda_forecast_sensitivity(final_training)
    feature_ablation = build_feature_ablation(
        inputs.rolling_folds, final_training, selected_lambdas
    )

    fixed_lambda_sensitivity_frames: list[pd.DataFrame] = []
    for simulate, track in (
        (False, "official_fixed_final_lambda_retrospective"),
        (True, "augmented_fixed_final_lambda_retrospective"),
    ):
        records: list[dict[str, object]] = []
        for (_, fold_id), group in inputs.rolling_folds.groupby(
            ["metric", "fold_id"], sort=True
        ):
            metric = str(group.iloc[0]["metric"])
            physical_train = group[group["fold_role"].eq("train")]
            test = group[group["fold_role"].eq("test")].iloc[0]
            if simulate:
                effective_train, _ = benchmark.augment_training_rows(
                    physical_train,
                    test_year=int(test["year"]),
                    scope=track,
                    fold_id=str(fold_id),
                )
            else:
                effective_train = physical_train
            for model, ridge_lambda in (
                (BASELINE_MODEL, BASELINE_LAMBDA),
                (TUNED_MODEL, selected_lambdas[metric]),
            ):
                prediction = float(
                    _fit_predict(
                        effective_train, [int(test["year"])], ridge_lambda
                    )[0]
                )
                records.append(
                    _prediction_record(
                        track=track,
                        fold_id=str(fold_id),
                        model=model,
                        metric=metric,
                        test_year=int(test["year"]),
                        actual=float(test["value"]),
                        prediction=prediction,
                        ridge_lambda=ridge_lambda,
                        tuning=None,
                        physical_train=physical_train,
                        effective_train=effective_train,
                    )
                )
        fixed_lambda_sensitivity_frames.append(pd.DataFrame(records))
    fixed_lambda_predictions = pd.concat(
        fixed_lambda_sensitivity_frames, ignore_index=True
    )
    fixed_lambda_by_target, fixed_lambda_macro = summarize_predictions(
        fixed_lambda_predictions
    )

    problem1_summary_path = (
        ROOT / "outputs/unified_model_benchmark/problem1_indicator_summary.csv"
    )
    problem1_diagnostics_path = (
        ROOT
        / "outputs/unified_model_benchmark/problem1_simple_growth_diagnostics.csv"
    )
    problem1_forecasts_path = (
        ROOT
        / "outputs/unified_model_benchmark/problem1_simple_growth_forecasts_2026_2030.csv"
    )
    scenarios_path = (
        ROOT
        / "outputs/unified_model_benchmark/problem3_scenario_forecasts_2026_2030.csv"
    )
    policy_sensitivity_path = (
        ROOT
        / "outputs/unified_model_benchmark/problem3_policy_sensitivity.csv"
    )
    problem1_summary = pd.read_csv(problem1_summary_path)
    problem1_diagnostics = pd.read_csv(problem1_diagnostics_path)
    problem1_forecasts = pd.read_csv(problem1_forecasts_path)
    scenarios = pd.read_csv(scenarios_path)
    policy_sensitivity = pd.read_csv(policy_sensitivity_path)
    figures = render_figures(
        output_dir,
        official_predictions=official_predictions,
        final_search=final_search,
        forecasts=forecasts,
        observations=inputs.observations,
        problem1_forecasts=problem1_forecasts,
        scenarios=scenarios,
    )

    csv_outputs: dict[str, pd.DataFrame] = {
        "official_outer_predictions.csv": official_predictions,
        "official_outer_metrics_by_target.csv": official_by_target,
        "official_outer_macro_metrics.csv": official_macro,
        "augmented_outer_predictions.csv": simulated_predictions,
        "augmented_outer_metrics_by_target.csv": simulated_by_target,
        "augmented_outer_macro_metrics.csv": simulated_macro,
        "outer_tuning_candidate_scores.csv": pd.concat(
            [official_candidates, simulated_candidates], ignore_index=True
        ),
        "outer_tuning_fold_scores.csv": pd.concat(
            [official_inner_folds, simulated_inner_folds], ignore_index=True
        ),
        "paired_fold_bootstrap.csv": fold_bootstrap,
        "pseudo_holdout_2024_predictions.csv": holdout_predictions,
        "pseudo_holdout_2024_metrics_by_target.csv": holdout_by_target,
        "pseudo_holdout_2024_macro_metrics.csv": holdout_macro,
        "cross_regime_stress_predictions.csv": stress_predictions,
        "cross_regime_stress_metrics_by_target.csv": stress_by_target,
        "cross_regime_stress_macro_metrics.csv": stress_macro,
        "fixed_scope_tuning_candidate_scores.csv": pd.concat(
            [holdout_candidates, stress_candidates], ignore_index=True
        ),
        "fixed_scope_tuning_fold_scores.csv": pd.concat(
            [holdout_inner_folds, stress_inner_folds], ignore_index=True
        ),
        "final_lambda_selection.csv": final_selection,
        "final_lambda_search_surface.csv": final_search,
        "final_training_2010_2024.csv": final_training,
        "final_forecasts_2026_2030.csv": forecasts,
        "final_model_diagnostics.csv": diagnostics,
        "final_standardized_parameters.csv": parameters,
        "implied_spend_forecast.csv": implied_spend,
        "design_support_diagnostics.csv": design_support,
        "lambda_forecast_sensitivity.csv": lambda_sensitivity,
        "feature_ablation.csv": feature_ablation,
        "fixed_final_lambda_retrospective_predictions.csv": fixed_lambda_predictions,
        "fixed_final_lambda_retrospective_metrics_by_target.csv": fixed_lambda_by_target,
        "fixed_final_lambda_retrospective_macro_metrics.csv": fixed_lambda_macro,
    }
    for filename, frame in csv_outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.10f")

    _write_output_readme(output_dir)
    _write_report(
        report_path,
        output_dir=output_dir,
        official_by_target=official_by_target,
        official_macro=official_macro,
        simulated_by_target=simulated_by_target,
        simulated_macro=simulated_macro,
        paired_bootstrap=fold_bootstrap,
        holdout_by_target=holdout_by_target,
        stress_by_target=stress_by_target,
        final_selection=final_selection,
        forecasts=forecasts,
        diagnostics=diagnostics,
        parameters=parameters,
        final_training=final_training,
        implied_spend=implied_spend,
        design_support=design_support,
        feature_ablation=feature_ablation,
        problem1_summary=problem1_summary,
        problem1_diagnostics=problem1_diagnostics,
        problem1_forecasts=problem1_forecasts,
        scenarios=scenarios,
        policy_sensitivity=policy_sensitivity,
    )

    source_files = [
        unified_dir / "benchmark_observations.csv",
        unified_dir / "primary_train.csv",
        unified_dir / "primary_test.csv",
        unified_dir / "rolling_origin_folds.csv",
        unified_dir / "stress_train.csv",
        unified_dir / "stress_test.csv",
        Path(__file__).resolve(),
        ROOT / "code/scripts/compare_branch_models.py",
        ROOT / "code/scripts/model_jizhou_tourism_ml.py",
        baseline_reference_path,
        problem1_summary_path,
        problem1_diagnostics_path,
        problem1_forecasts_path,
        scenarios_path,
        policy_sensitivity_path,
    ]
    official_tuned_macro = official_macro[
        official_macro["model"].eq(TUNED_MODEL)
    ].iloc[0]
    official_baseline_macro = official_macro[
        official_macro["model"].eq(BASELINE_MODEL)
    ].iloc[0]
    deployment_tuned_macro = simulated_macro[
        simulated_macro["model"].eq(TUNED_MODEL)
    ].iloc[0]
    deployment_baseline_macro = simulated_macro[
        simulated_macro["model"].eq(BASELINE_MODEL)
    ].iloc[0]
    summary: dict[str, object] = {
        "branch": OPTIMIZATION_BRANCH,
        "random_seed": RANDOM_SEED,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "parameter_semantics": {
            "formula_symbol": "lambda",
            "scikit_learn_parameter": "alpha",
            "independent_hyperparameters": 1,
            "explanation": "sklearn Ridge alpha is the L2 penalty conventionally written as lambda",
        },
        "protocol": {
            "positive_lambda_grid": list(POSITIVE_LAMBDA_GRID),
            "boundary_diagnostic_grid": list(BOUNDARY_DIAGNOSTIC_GRID),
            "min_inner_train_records": MIN_INNER_TRAIN_RECORDS,
            "min_inner_validations": MIN_INNER_VALIDATIONS,
            "fallback_lambda": BASELINE_LAMBDA,
            "primary_metric": "raw-scale sMAPE",
            "outer_last_test_year": 2023,
            "pseudo_holdout_year": 2024,
            "final_tuning_max_year": 2023,
            "final_refit_max_year": 2024,
            "final_selection_track": "fold_local_augmented_training_with_official_validation",
            "uses_2025_target": False,
            "fold_bootstrap_inferential_confidence_interval": False,
            "forecast_interval_assumptions": (
                "heuristic fixed-design residual bootstrap; iid exchangeable residuals, "
                "no serial-correlation or model-selection uncertainty"
            ),
        },
        "selected_lambdas": selected_lambdas,
        "headline": {
            "deployment_consistent_baseline_macro_smape_percent": float(
                deployment_baseline_macro["macro_smape_percent"]
            ),
            "deployment_consistent_nested_tuned_macro_smape_percent": float(
                deployment_tuned_macro["macro_smape_percent"]
            ),
            "deployment_consistent_delta_percent_points": float(
                deployment_tuned_macro["macro_smape_percent"]
                - deployment_baseline_macro["macro_smape_percent"]
            ),
            "official_baseline_macro_smape_percent": float(
                official_baseline_macro["macro_smape_percent"]
            ),
            "official_nested_tuned_macro_smape_percent": float(
                official_tuned_macro["macro_smape_percent"]
            ),
            "official_delta_percent_points": float(
                official_tuned_macro["macro_smape_percent"]
                - official_baseline_macro["macro_smape_percent"]
            ),
            "conclusion": (
                "marginal and unstable change; selected lambda hits the positive grid boundary, "
                "so no reliable Ridge regularization gain is established"
            ),
        },
        "baseline_reproduction_max_absolute_difference": max_baseline_difference,
        "input_sha256": {
            _summary_path(path): _sha256(path)
            for path in source_files
        },
        "generated_files": sorted(
            [*csv_outputs.keys(), *figures, "README.md", "run_summary.json"]
        ),
        "report": _summary_path(report_path),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-dir", type=Path, default=DEFAULT_UNIFIED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    summary = run_optimization(
        unified_dir=arguments.unified_dir.resolve(),
        output_dir=arguments.output_dir.resolve(),
        report_path=arguments.report.resolve(),
    )
    headline = summary["headline"]
    print(
        "completed Ridge optimization; "
        "deployment-consistent baseline macro sMAPE="
        f"{headline['deployment_consistent_baseline_macro_smape_percent']:.6f}; "
        "nested tuned="
        f"{headline['deployment_consistent_nested_tuned_macro_smape_percent']:.6f}; "
        "delta="
        f"{headline['deployment_consistent_delta_percent_points']:.6f} percentage points"
    )


if __name__ == "__main__":
    main()
