#!/usr/bin/env python3
"""Machine-learning experiments for TJMML problem C.

The annual tourism series are extremely small and have a non-random missing
block during 2020--2022.  This script therefore treats machine learning as a
controlled model comparison rather than as a promise of superior accuracy:

* all validation folds respect time order;
* target gaps are never filled before validation;
* preprocessing and hyperparameter selection are fitted inside each fold;
* naive persistence is reported beside the ML regressors;
* future intervals use the worst observed rolling-origin log error and are
  explicitly labelled empirical rather than guaranteed coverage intervals.

Outputs are deterministic for the pinned environment in requirements-ml.txt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_CACHE_ROOT = Path(tempfile.gettempdir()) / "jizhou-tourism-ml-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/jizhou_tourism_economy/official_annual_summary_2010_2025.csv"
DEFAULT_OUTPUT = ROOT / "outputs/jizhou_tourism_ml"
RANDOM_STATE = 20260817
FEATURE_NAMES = ["year_index", "pandemic_2020_2022", "post_2022"]
ENSEMBLE_MEMBERS = ["ridge_regime", "bayesian_ridge", "huber_regime"]


@dataclass(frozen=True)
class SeriesDefinition:
    metric: str
    value_column: str
    status_column: str
    unit: str
    strict_excluded_statuses: frozenset[str]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[dict[str, Any]], RegressorMixin]
    candidates: tuple[dict[str, Any], ...]
    extrapolation_note: str
    eligible_for_selection: bool


SERIES = {
    "tourist_visits": SeriesDefinition(
        metric="tourist_visits",
        value_column="preferred_visitor_10k_persons",
        status_column="visitor_status",
        unit="10k_persons",
        strict_excluded_statuses=frozenset({"observed_cached"}),
    ),
    "tourism_comprehensive_income": SeriesDefinition(
        metric="tourism_comprehensive_income",
        value_column="preferred_comprehensive_income_100m_cny",
        status_column="comprehensive_status",
        unit="100m_cny",
        strict_excluded_statuses=frozenset(
            {"inferred_from_yoy", "observed_cached", "observed_supporting_attachment"}
        ),
    ),
}


def make_features(years: np.ndarray | list[int]) -> np.ndarray:
    years_array = np.asarray(years, dtype=float)
    return np.column_stack(
        [
            (years_array - 2010.0) / 10.0,
            ((years_array >= 2020.0) & (years_array <= 2022.0)).astype(float),
            (years_array >= 2023.0).astype(float),
        ]
    )


def _ridge_builder(parameters: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=float(parameters["alpha"]))),
        ]
    )


def _bayesian_builder(parameters: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                BayesianRidge(
                    alpha_1=float(parameters["prior"]),
                    alpha_2=float(parameters["prior"]),
                    lambda_1=float(parameters["prior"]),
                    lambda_2=float(parameters["prior"]),
                ),
            ),
        ]
    )


def _huber_builder(parameters: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                HuberRegressor(
                    epsilon=1.35,
                    alpha=float(parameters["alpha"]),
                    max_iter=2000,
                ),
            ),
        ]
    )


def _spline_builder(parameters: dict[str, Any]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "time_spline",
                SplineTransformer(
                    n_knots=int(parameters["n_knots"]),
                    degree=2,
                    include_bias=False,
                    extrapolation="linear",
                ),
                [0],
            ),
            ("regime", StandardScaler(), [1, 2]),
        ]
    )
    return Pipeline(
        [
            ("features", transformer),
            ("model", Ridge(alpha=float(parameters["alpha"]))),
        ]
    )


def _svr_builder(parameters: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                SVR(
                    kernel="rbf",
                    C=float(parameters["C"]),
                    gamma=parameters["gamma"],
                    epsilon=float(parameters["epsilon"]),
                ),
            ),
        ]
    )


def _forest_builder(parameters: dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=parameters["max_depth"],
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _gp_builder(parameters: dict[str, Any]) -> Pipeline:
    if parameters["family"] == "matern":
        smooth_component = Matern(length_scale=np.ones(3), nu=1.5)
    else:
        smooth_component = RBF(length_scale=np.ones(3))
    kernel = (
        ConstantKernel(1.0, (1.0e-2, 1.0e2))
        * (DotProduct(sigma_0=1.0) + smooth_component)
        + WhiteKernel(noise_level=0.02, noise_level_bounds=(1.0e-6, 1.0))
    )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=float(parameters["alpha"]),
                    normalize_y=True,
                    n_restarts_optimizer=0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


MODEL_SPECS = {
    "ridge_regime": ModelSpec(
        name="ridge_regime",
        builder=_ridge_builder,
        candidates=tuple({"alpha": value} for value in (0.01, 0.1, 1.0, 10.0, 100.0)),
        extrapolation_note="linear extrapolation with regularized time, pandemic, and post-2022 features",
        eligible_for_selection=True,
    ),
    "bayesian_ridge": ModelSpec(
        name="bayesian_ridge",
        builder=_bayesian_builder,
        candidates=tuple({"prior": value} for value in (1.0e-6, 1.0e-4, 1.0e-2)),
        extrapolation_note="linear extrapolation with Bayesian shrinkage and conditional model variance",
        eligible_for_selection=True,
    ),
    "huber_regime": ModelSpec(
        name="huber_regime",
        builder=_huber_builder,
        candidates=tuple({"alpha": value} for value in (1.0e-4, 1.0e-2, 1.0)),
        extrapolation_note="robust linear extrapolation that downweights large residuals",
        eligible_for_selection=True,
    ),
    "spline_ridge": ModelSpec(
        name="spline_ridge",
        builder=_spline_builder,
        candidates=tuple(
            {"alpha": alpha, "n_knots": knots}
            for alpha in (0.1, 1.0, 10.0)
            for knots in (3, 4)
        ),
        extrapolation_note="linear spline extrapolation outside the observed year range",
        eligible_for_selection=True,
    ),
    "svr_rbf": ModelSpec(
        name="svr_rbf",
        builder=_svr_builder,
        candidates=tuple(
            {"C": c_value, "gamma": gamma, "epsilon": 0.05}
            for c_value in (1.0, 10.0, 100.0)
            for gamma in ("scale", 0.3)
        ),
        extrapolation_note="RBF predictions tend back toward the learned level outside training support",
        eligible_for_selection=False,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        builder=_forest_builder,
        candidates=tuple(
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth in (2, None)
            for leaf in (1, 2)
        ),
        extrapolation_note="trees cannot extrapolate beyond observed terminal leaves",
        eligible_for_selection=False,
    ),
    "gaussian_process": ModelSpec(
        name="gaussian_process",
        builder=_gp_builder,
        candidates=tuple(
            {"family": family, "alpha": alpha}
            for family in ("rbf", "matern")
            for alpha in (1.0e-6, 1.0e-3)
        ),
        extrapolation_note="linear plus smooth kernel extrapolation; variance is conditional on kernel choice",
        eligible_for_selection=False,
    ),
}


def get_series(data: pd.DataFrame, definition: SeriesDefinition, strict: bool = False) -> pd.DataFrame:
    frame = data[["year", definition.value_column, definition.status_column]].rename(
        columns={definition.value_column: "value", definition.status_column: "status"}
    )
    frame = frame.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
    if strict:
        frame = frame[~frame["status"].isin(definition.strict_excluded_statuses)].reset_index(drop=True)
    return frame


def rolling_origin_splits(n_samples: int, min_train_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.arange(test_index), np.array([test_index]))
        for test_index in range(min_train_size, n_samples)
    ]


def fit_quietly(model: RegressorMixin, x: np.ndarray, y: np.ndarray) -> RegressorMixin:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return model.fit(x, y)


def tune_model(
    spec: ModelSpec,
    years: np.ndarray,
    log_values: np.ndarray,
    min_train_size: int = 4,
) -> tuple[dict[str, Any], float, int]:
    x = make_features(years)
    splits = rolling_origin_splits(len(years), min_train_size)
    if not splits:
        return dict(spec.candidates[0]), math.nan, 0
    best_parameters: dict[str, Any] | None = None
    best_score = math.inf
    for parameters in spec.candidates:
        errors: list[float] = []
        for train_index, validation_index in splits:
            model = fit_quietly(spec.builder(dict(parameters)), x[train_index], log_values[train_index])
            predicted = float(model.predict(x[validation_index])[0])
            errors.append(abs(float(log_values[validation_index][0]) - predicted))
        score = float(np.mean(errors))
        if score < best_score - 1.0e-12:
            best_score = score
            best_parameters = dict(parameters)
    assert best_parameters is not None
    return best_parameters, best_score, len(splits)


def rolling_backtest(
    frame: pd.DataFrame,
    definition: SeriesDefinition,
    min_train_size: int = 5,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    years = frame["year"].to_numpy(dtype=int)
    values = frame["value"].to_numpy(dtype=float)
    log_values = np.log(values)
    records: list[dict[str, Any]] = []
    chosen_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in MODEL_SPECS}

    for test_index in range(min_train_size, len(frame)):
        train_years = years[:test_index]
        train_logs = log_values[:test_index]
        test_year = int(years[test_index])
        actual = float(values[test_index])
        test_regime = (
            "pre_covid" if test_year <= 2019 else "pandemic" if test_year <= 2022 else "recovery"
        )
        x_train = make_features(train_years)
        x_test = make_features([test_year])

        last_prediction = float(np.exp(train_logs[-1]))
        records.append(
            {
                "metric": definition.metric,
                "model": "naive_last",
                "train_end_year": int(train_years[-1]),
                "test_year": test_year,
                "test_regime": test_regime,
                "actual": actual,
                "prediction": last_prediction,
                "log_error": math.log(actual) - math.log(last_prediction),
                "selected_parameters": "{}",
            }
        )

        for name, spec in MODEL_SPECS.items():
            parameters, inner_score, inner_folds = tune_model(spec, train_years, train_logs)
            model = fit_quietly(spec.builder(parameters), x_train, train_logs)
            prediction = float(np.exp(model.predict(x_test)[0]))
            chosen_by_model[name].append(parameters)
            records.append(
                {
                    "metric": definition.metric,
                    "model": name,
                    "train_end_year": int(train_years[-1]),
                    "test_year": test_year,
                    "test_regime": test_regime,
                    "actual": actual,
                    "prediction": prediction,
                    "log_error": math.log(actual) - math.log(prediction),
                    "selected_parameters": json.dumps(parameters, sort_keys=True),
                    "inner_log_mae": inner_score,
                    "inner_folds": inner_folds,
                }
            )

    backtest = pd.DataFrame(records)
    ensemble_source = backtest[backtest["model"].isin(ENSEMBLE_MEMBERS)]
    ensemble = (
        ensemble_source.groupby(
            ["metric", "train_end_year", "test_year", "test_regime", "actual"], as_index=False
        )["prediction"]
        .median()
        .assign(model="robust_ml_ensemble", selected_parameters=json.dumps(ENSEMBLE_MEMBERS))
    )
    ensemble["log_error"] = np.log(ensemble["actual"]) - np.log(ensemble["prediction"])
    backtest = pd.concat([backtest, ensemble[backtest.columns.intersection(ensemble.columns)]], ignore_index=True)
    return backtest, chosen_by_model


def metric_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (metric, model), group in backtest.groupby(["metric", "model"]):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["prediction"].to_numpy(dtype=float)
        absolute_percentage = np.abs(actual - predicted) / actual
        smape = 2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
        rows.append(
            {
                "metric": metric,
                "model": model,
                "n_folds": len(group),
                "first_test_year": int(group["test_year"].min()),
                "last_test_year": int(group["test_year"].max()),
                "mae": mean_absolute_error(actual, predicted),
                "rmse": math.sqrt(mean_squared_error(actual, predicted)),
                "mape_percent": float(np.mean(absolute_percentage) * 100.0),
                "smape_percent": float(np.mean(smape) * 100.0),
                "median_absolute_percentage_error": float(np.median(absolute_percentage) * 100.0),
                "max_absolute_log_error": float(np.max(np.abs(group["log_error"]))),
            }
        )
    result = pd.DataFrame(rows)
    naive_smape = (
        result[result["model"] == "naive_last"]
        .set_index("metric")["smape_percent"]
        .to_dict()
    )
    result["smape_skill_vs_naive"] = result.apply(
        lambda row: 1.0 - row["smape_percent"] / naive_smape[row["metric"]], axis=1
    )
    eligible_names = {
        name for name, spec in MODEL_SPECS.items() if spec.eligible_for_selection
    } | {"robust_ml_ensemble"}
    result["eligible_for_primary_ml"] = result["model"].isin(eligible_names)
    result["beats_naive_smape"] = result["smape_skill_vs_naive"] > 0.0
    return result.sort_values(["metric", "smape_percent"]).reset_index(drop=True)


def metric_summary_by_regime(backtest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (metric, model, regime), group in backtest.groupby(["metric", "model", "test_regime"]):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["prediction"].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "model": model,
                "test_regime": regime,
                "n_folds": len(group),
                "test_years": ";".join(map(str, group["test_year"].astype(int))),
                "mae": mean_absolute_error(actual, predicted),
                "mape_percent": float(np.mean(np.abs(actual - predicted) / actual) * 100.0),
                "smape_percent": float(
                    np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))
                    * 100.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "test_regime", "smape_percent"])


def choose_ml_model(summary: pd.DataFrame, metric: str) -> str:
    eligible_names = {
        name for name, spec in MODEL_SPECS.items() if spec.eligible_for_selection
    } | {"robust_ml_ensemble"}
    candidates = summary[(summary["metric"] == metric) & summary["model"].isin(eligible_names)]
    return str(candidates.sort_values(["smape_percent", "rmse"]).iloc[0]["model"])


def fit_final_models(
    frame: pd.DataFrame,
) -> tuple[dict[str, RegressorMixin], dict[str, dict[str, Any]], pd.DataFrame]:
    years = frame["year"].to_numpy(dtype=int)
    log_values = np.log(frame["value"].to_numpy(dtype=float))
    x = make_features(years)
    models: dict[str, RegressorMixin] = {}
    parameters_by_model: dict[str, dict[str, Any]] = {}
    tuning_rows: list[dict[str, Any]] = []
    for name, spec in MODEL_SPECS.items():
        parameters, inner_score, inner_folds = tune_model(spec, years, log_values)
        models[name] = fit_quietly(spec.builder(parameters), x, log_values)
        parameters_by_model[name] = parameters
        tuning_rows.append(
            {
                "model": name,
                "parameters": json.dumps(parameters, sort_keys=True),
                "rolling_inner_log_mae": inner_score,
                "inner_folds": inner_folds,
            }
        )
    return models, parameters_by_model, pd.DataFrame(tuning_rows)


def predict_model(
    model_name: str,
    models: dict[str, RegressorMixin],
    x: np.ndarray,
    last_observed: float,
) -> np.ndarray:
    if model_name == "naive_last":
        return np.full(len(x), last_observed, dtype=float)
    if model_name == "robust_ml_ensemble":
        member_predictions = np.column_stack(
            [np.exp(models[name].predict(x)) for name in ENSEMBLE_MEMBERS]
        )
        return np.median(member_predictions, axis=1)
    return np.exp(models[model_name].predict(x))


def native_pipeline_interval(
    model: Pipeline,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = model.named_steps["scale"].transform(x)
    mean_log, standard_deviation = model.named_steps["model"].predict(scaled, return_std=True)
    return (
        np.exp(mean_log),
        np.exp(mean_log - 1.96 * standard_deviation),
        np.exp(mean_log + 1.96 * standard_deviation),
    )


def forecast_all_models(
    definition: SeriesDefinition,
    frame: pd.DataFrame,
    models: dict[str, RegressorMixin],
    backtest: pd.DataFrame,
    selected_model: str,
    years: np.ndarray,
) -> pd.DataFrame:
    x = make_features(years)
    last_observed = float(frame.iloc[-1]["value"])
    records: list[dict[str, Any]] = []
    model_names = ["naive_last", *MODEL_SPECS.keys(), "robust_ml_ensemble"]
    for model_name in model_names:
        predictions = predict_model(model_name, models, x, last_observed)
        errors = backtest.loc[backtest["model"] == model_name, "log_error"].abs()
        calibration = float(errors.max()) if len(errors) else math.nan
        for year, prediction in zip(years, predictions, strict=True):
            record: dict[str, Any] = {
                "metric": definition.metric,
                "unit": definition.unit,
                "model": model_name,
                "selected_ml_model": model_name == selected_model,
                "year": int(year),
                "forecast": float(prediction),
                "backtest_stress_lower": float(prediction * math.exp(-calibration)),
                "backtest_stress_upper": float(prediction * math.exp(calibration)),
                "calibration_max_absolute_log_error": calibration,
                "interval_note": "max rolling-origin absolute log error envelope; no coverage guarantee",
            }
            records.append(record)
    forecast = pd.DataFrame(records)

    gp_point, gp_lower, gp_upper = native_pipeline_interval(models["gaussian_process"], x)
    gp_mask = forecast["model"] == "gaussian_process"
    forecast.loc[gp_mask, "native_gp_forecast"] = gp_point
    forecast.loc[gp_mask, "native_gp_interval95_lower"] = gp_lower
    forecast.loc[gp_mask, "native_gp_interval95_upper"] = gp_upper

    bayes_point, bayes_lower, bayes_upper = native_pipeline_interval(models["bayesian_ridge"], x)
    bayes_mask = forecast["model"] == "bayesian_ridge"
    forecast.loc[bayes_mask, "conditional_bayesian_forecast"] = bayes_point
    forecast.loc[bayes_mask, "conditional_bayesian_interval95_lower"] = bayes_lower
    forecast.loc[bayes_mask, "conditional_bayesian_interval95_upper"] = bayes_upper
    return forecast


def impute_missing_years(
    data: pd.DataFrame,
    definition: SeriesDefinition,
    models: dict[str, RegressorMixin],
    backtest: pd.DataFrame,
    selected_model: str,
) -> pd.DataFrame:
    missing_years = data.loc[data[definition.value_column].isna(), "year"].to_numpy(dtype=int)
    x = make_features(missing_years)
    observed = get_series(data, definition)
    predictions = predict_model(selected_model, models, x, float(observed.iloc[-1]["value"]))
    errors = backtest.loc[backtest["model"] == selected_model, "log_error"].abs()
    calibration = float(errors.max())
    return pd.DataFrame(
        {
            "metric": definition.metric,
            "unit": definition.unit,
            "year": missing_years,
            "selected_ml_model": selected_model,
            "imputed_value": predictions,
            "backtest_stress_lower": predictions * math.exp(-calibration),
            "backtest_stress_upper": predictions * math.exp(calibration),
            "use_restriction": "diagnostic only; do not overwrite official missing values",
        }
    )


def strict_evidence_sensitivity(
    data: pd.DataFrame,
    definition: SeriesDefinition,
    selected_model: str,
    full_models: dict[str, RegressorMixin],
) -> pd.DataFrame:
    strict_frame = get_series(data, definition, strict=True)
    strict_models, _, _ = fit_final_models(strict_frame)
    x_2030 = make_features([2030])
    full_prediction = float(
        predict_model(
            selected_model,
            full_models,
            x_2030,
            float(get_series(data, definition).iloc[-1]["value"]),
        )[0]
    )
    strict_prediction = float(
        predict_model(
            selected_model,
            strict_models,
            x_2030,
            float(strict_frame.iloc[-1]["value"]),
        )[0]
    )
    return pd.DataFrame(
        [
            {
                "metric": definition.metric,
                "selected_ml_model": selected_model,
                "full_sample_n": len(get_series(data, definition)),
                "strict_sample_n": len(strict_frame),
                "full_sample_2030": full_prediction,
                "strict_sample_2030": strict_prediction,
                "strict_relative_difference_percent": 100.0
                * (strict_prediction / full_prediction - 1.0),
            }
        ]
    )


def permutation_rows(
    definition: SeriesDefinition,
    frame: pd.DataFrame,
    models: dict[str, RegressorMixin],
) -> pd.DataFrame:
    years = frame["year"].to_numpy(dtype=int)
    x = make_features(years)
    y = np.log(frame["value"].to_numpy(dtype=float))
    rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        result = permutation_importance(
            model,
            x,
            y,
            scoring="neg_mean_absolute_error",
            n_repeats=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        for feature_name, importance, standard_deviation in zip(
            FEATURE_NAMES, result.importances_mean, result.importances_std, strict=True
        ):
            rows.append(
                {
                    "metric": definition.metric,
                    "model": model_name,
                    "feature": feature_name,
                    "permutation_importance_log_mae": float(importance),
                    "importance_std": float(standard_deviation),
                    "warning": "in-sample exploratory importance; correlated time features",
                }
            )
    return pd.DataFrame(rows)


def long_horizon_pressure_test(
    data: pd.DataFrame,
    definition: SeriesDefinition,
) -> pd.DataFrame:
    """One pre-COVID fixed-origin test that exposes sparse long horizons."""
    frame = get_series(data, definition)
    pre_covid = frame[frame["year"] <= 2019].reset_index(drop=True)
    train = pre_covid.iloc[:5]
    test = pre_covid.iloc[5:]
    train_years = train["year"].to_numpy(dtype=int)
    train_logs = np.log(train["value"].to_numpy(dtype=float))
    x_train = make_features(train_years)
    x_test = make_features(test["year"].to_numpy(dtype=int))
    predictions_by_model: dict[str, np.ndarray] = {
        "naive_last": np.full(len(test), float(train.iloc[-1]["value"]), dtype=float)
    }
    for name, spec in MODEL_SPECS.items():
        parameters, _, _ = tune_model(spec, train_years, train_logs)
        model = fit_quietly(spec.builder(parameters), x_train, train_logs)
        predictions_by_model[name] = np.exp(model.predict(x_test))
    predictions_by_model["robust_ml_ensemble"] = np.median(
        np.column_stack([predictions_by_model[name] for name in ENSEMBLE_MEMBERS]), axis=1
    )
    rows: list[dict[str, Any]] = []
    for model_name, predictions in predictions_by_model.items():
        for (_, test_row), prediction in zip(test.iterrows(), predictions, strict=True):
            actual = float(test_row["value"])
            rows.append(
                {
                    "metric": definition.metric,
                    "model": model_name,
                    "train_start_year": int(train.iloc[0]["year"]),
                    "train_end_year": int(train.iloc[-1]["year"]),
                    "test_year": int(test_row["year"]),
                    "calendar_horizon_years": int(test_row["year"] - train.iloc[-1]["year"]),
                    "actual": actual,
                    "prediction": float(prediction),
                    "absolute_percentage_error": abs(actual - float(prediction)) / actual * 100.0,
                }
            )
    return pd.DataFrame(rows)


def raw_target_ridge_sensitivity(
    frame: pd.DataFrame,
    definition: SeriesDefinition,
    forecast_years: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare additive raw-target Ridge with the primary log-target specification."""
    years = frame["year"].to_numpy(dtype=int)
    values = frame["value"].to_numpy(dtype=float)
    records: list[dict[str, Any]] = []
    for test_index in range(5, len(frame)):
        model = _ridge_builder({"alpha": 0.1})
        model.fit(make_features(years[:test_index]), values[:test_index])
        prediction = max(0.0, float(model.predict(make_features([years[test_index]]))[0]))
        actual = float(values[test_index])
        records.append(
            {
                "metric": definition.metric,
                "model": "raw_target_ridge_alpha_0.1",
                "train_end_year": int(years[test_index - 1]),
                "test_year": int(years[test_index]),
                "actual": actual,
                "prediction": prediction,
                "absolute_percentage_error": abs(actual - prediction) / actual * 100.0,
            }
        )
    final_model = _ridge_builder({"alpha": 0.1})
    final_model.fit(make_features(years), values)
    predictions = np.maximum(0.0, final_model.predict(make_features(forecast_years)))
    forecast = pd.DataFrame(
        {
            "metric": definition.metric,
            "unit": definition.unit,
            "target_transform": "raw_additive",
            "model": "raw_target_ridge_alpha_0.1",
            "year": forecast_years,
            "forecast": predictions,
            "interpretation": "recommended additive sensitivity path; same time/regime features",
        }
    )
    return pd.DataFrame(records), forecast


def data_availability_audit(data: pd.DataFrame) -> pd.DataFrame:
    related = pd.read_csv(ROOT / "data/jizhou_tourism_economy/official_related_observations_2014_2025.csv")
    rows: list[dict[str, Any]] = []
    for definition in SERIES.values():
        frame = get_series(data, definition)
        rows.append(
            {
                "feature_group": definition.metric,
                "observations": len(frame),
                "years": ";".join(map(str, frame["year"].astype(int))),
                "future_known_2026_2030": False,
                "decision": "target series only; retain missing years",
            }
        )
    for metric, group in related.groupby("metric"):
        rows.append(
            {
                "feature_group": metric,
                "observations": len(group),
                "years": ";".join(map(str, sorted(group["year"].astype(int).unique()))),
                "future_known_2026_2030": False,
                "decision": "excluded from final forecast; sparse and future path unavailable",
            }
        )
    rows.extend(
        [
            {
                "feature_group": "gdp",
                "observations": int(data["preferred_gdp_100m_cny"].notna().sum()),
                "years": ";".join(
                    map(str, data.loc[data["preferred_gdp_100m_cny"].notna(), "year"].astype(int))
                ),
                "future_known_2026_2030": False,
                "decision": "excluded from final forecast; 2019 level break and future values unknown",
            },
            {
                "feature_group": "tertiary_value_added",
                "observations": int(data["preferred_tertiary_100m_cny"].notna().sum()),
                "years": ";".join(
                    map(str, data.loc[data["preferred_tertiary_100m_cny"].notna(), "year"].astype(int))
                ),
                "future_known_2026_2030": False,
                "decision": "excluded from final forecast; five historical gaps and future values unknown",
            },
        ]
    )
    return pd.DataFrame(rows)


def create_forecast_figure(
    data: pd.DataFrame,
    selected_log_forecasts: pd.DataFrame,
    recommended_forecasts: pd.DataFrame,
    all_forecasts: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    titles = {
        "tourist_visits": ("Tourist visits", "10,000 person-visits"),
        "tourism_comprehensive_income": ("Tourism comprehensive income", "100 million CNY"),
    }
    colors = {
        "ridge_regime": "#4c78a8",
        "bayesian_ridge": "#72b7b2",
        "huber_regime": "#9c755f",
        "spline_ridge": "#f58518",
        "svr_rbf": "#54a24b",
        "random_forest": "#b279a2",
        "gaussian_process": "#e45756",
        "robust_ml_ensemble": "#222222",
    }
    for axis, (metric, (title, ylabel)) in zip(axes, titles.items(), strict=True):
        definition = SERIES[metric]
        observed = get_series(data, definition)
        axis.plot(observed["year"], observed["value"], "o-", color="#1f4e79", label="official preferred")
        metric_forecasts = all_forecasts[(all_forecasts["metric"] == metric) & (all_forecasts["year"] >= 2025)]
        for model_name in [*MODEL_SPECS.keys(), "robust_ml_ensemble"]:
            model_rows = metric_forecasts[metric_forecasts["model"] == model_name]
            axis.plot(
                model_rows["year"],
                model_rows["forecast"],
                "--",
                color=colors[model_name],
                alpha=0.55,
                linewidth=1.2,
                label=model_name,
            )
        selected_log = selected_log_forecasts[selected_log_forecasts["metric"] == metric]
        axis.plot(
            selected_log["year"],
            selected_log["forecast"],
            ":",
            color="black",
            linewidth=1.8,
            label="selected log-target ML",
        )
        recommended = recommended_forecasts[recommended_forecasts["metric"] == metric]
        axis.plot(
            recommended["year"],
            recommended["forecast"],
            "o-",
            color="black",
            linewidth=2.4,
            label="recommended raw-target ML",
        )
        axis.fill_between(
            recommended["year"].to_numpy(dtype=float),
            recommended["backtest_stress_lower"].to_numpy(dtype=float),
            recommended["backtest_stress_upper"].to_numpy(dtype=float),
            color="grey",
            alpha=0.2,
            label="backtest stress envelope",
        )
        axis.axvspan(2020, 2022, color="grey", alpha=0.12, label="pandemic / missing block")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=7)
    axes[1].set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_backtest_figure(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = list(SERIES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for axis, metric in zip(axes, metrics, strict=True):
        frame = summary[summary["metric"] == metric].sort_values("smape_percent")
        axis.barh(frame["model"], frame["smape_percent"], color="#4c78a8")
        axis.invert_yaxis()
        axis.set_title(metric)
        axis.set_xlabel("Rolling-origin sMAPE (%)")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    all_backtests: list[pd.DataFrame] = []
    all_forecasts: list[pd.DataFrame] = []
    all_imputations: list[pd.DataFrame] = []
    all_tuning: list[pd.DataFrame] = []
    all_strict: list[pd.DataFrame] = []
    all_importance: list[pd.DataFrame] = []
    all_raw_backtests: list[pd.DataFrame] = []
    all_raw_forecasts: list[pd.DataFrame] = []
    final_models: dict[str, dict[str, RegressorMixin]] = {}

    for metric, definition in SERIES.items():
        frame = get_series(data, definition)
        backtest, _ = rolling_backtest(frame, definition)
        all_backtests.append(backtest)

    backtests = pd.concat(all_backtests, ignore_index=True)
    comparison = metric_summary(backtests)
    comparison_by_regime = metric_summary_by_regime(backtests)
    selected_models = {metric: choose_ml_model(comparison, metric) for metric in SERIES}
    comparison["selected_ml_model"] = comparison.apply(
        lambda row: row["model"] == selected_models[row["metric"]], axis=1
    )
    pressure_tests = pd.concat(
        [long_horizon_pressure_test(data, definition) for definition in SERIES.values()],
        ignore_index=True,
    )

    for metric, definition in SERIES.items():
        frame = get_series(data, definition)
        metric_backtest = backtests[backtests["metric"] == metric]
        models, parameters, tuning = fit_final_models(frame)
        final_models[metric] = models
        tuning.insert(0, "metric", metric)
        tuning["selected_ml_model"] = tuning["model"] == selected_models[metric]
        all_tuning.append(tuning)
        forecast = forecast_all_models(
            definition,
            frame,
            models,
            metric_backtest,
            selected_models[metric],
            np.arange(2025, 2031),
        )
        all_forecasts.append(forecast)
        all_imputations.append(
            impute_missing_years(data, definition, models, metric_backtest, selected_models[metric])
        )
        all_strict.append(
            strict_evidence_sensitivity(data, definition, selected_models[metric], models)
        )
        all_importance.append(permutation_rows(definition, frame, models))
        raw_backtest, raw_forecast = raw_target_ridge_sensitivity(
            frame, definition, np.arange(2025, 2031)
        )
        all_raw_backtests.append(raw_backtest)
        all_raw_forecasts.append(raw_forecast)

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    selected_forecasts = forecasts[forecasts["selected_ml_model"]].copy()
    imputations = pd.concat(all_imputations, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    strict = pd.concat(all_strict, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True)
    raw_backtests = pd.concat(all_raw_backtests, ignore_index=True)
    raw_forecasts = pd.concat(all_raw_forecasts, ignore_index=True)
    raw_backtests["log_error"] = np.log(raw_backtests["actual"]) - np.log(
        raw_backtests["prediction"].clip(lower=1.0e-12)
    )
    raw_backtests["test_regime"] = raw_backtests["test_year"].map(
        lambda year: "pre_covid" if year <= 2019 else "pandemic" if year <= 2022 else "recovery"
    )
    raw_calibration = raw_backtests.groupby("metric")["log_error"].apply(
        lambda values: float(values.abs().max())
    )
    recommended_forecasts = raw_forecasts.copy()
    recommended_forecasts["calibration_max_absolute_log_error"] = recommended_forecasts[
        "metric"
    ].map(raw_calibration)
    recommended_forecasts["backtest_stress_lower"] = recommended_forecasts["forecast"] * np.exp(
        -recommended_forecasts["calibration_max_absolute_log_error"]
    )
    recommended_forecasts["backtest_stress_upper"] = recommended_forecasts["forecast"] * np.exp(
        recommended_forecasts["calibration_max_absolute_log_error"]
    )
    recommended_forecasts["recommendation_basis"] = (
        "lower rolling-origin MAPE than selected log-target Ridge; additive path remains a sensitivity"
    )
    comparison["recommended_point_model"] = False
    raw_summary_rows: list[dict[str, Any]] = []
    for metric, group in raw_backtests.groupby("metric"):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["prediction"].to_numpy(dtype=float)
        smape = float(
            np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))
            * 100.0
        )
        naive_smape = float(
            comparison.loc[
                (comparison["metric"] == metric) & (comparison["model"] == "naive_last"),
                "smape_percent",
            ].iloc[0]
        )
        raw_summary_rows.append(
            {
                "metric": metric,
                "model": "raw_target_ridge_alpha_0.1",
                "n_folds": len(group),
                "first_test_year": int(group["test_year"].min()),
                "last_test_year": int(group["test_year"].max()),
                "mae": mean_absolute_error(actual, predicted),
                "rmse": math.sqrt(mean_squared_error(actual, predicted)),
                "mape_percent": float(np.mean(np.abs(actual - predicted) / actual) * 100.0),
                "smape_percent": smape,
                "median_absolute_percentage_error": float(
                    np.median(np.abs(actual - predicted) / actual) * 100.0
                ),
                "max_absolute_log_error": float(group["log_error"].abs().max()),
                "smape_skill_vs_naive": 1.0 - smape / naive_smape,
                "eligible_for_primary_ml": True,
                "beats_naive_smape": smape < naive_smape,
                "selected_ml_model": False,
                "recommended_point_model": True,
            }
        )
    comparison = pd.concat([comparison, pd.DataFrame(raw_summary_rows)], ignore_index=True).sort_values(
        ["metric", "smape_percent"]
    )
    raw_regime_rows: list[dict[str, Any]] = []
    for (metric, regime), group in raw_backtests.groupby(["metric", "test_regime"]):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["prediction"].to_numpy(dtype=float)
        raw_regime_rows.append(
            {
                "metric": metric,
                "model": "raw_target_ridge_alpha_0.1",
                "test_regime": regime,
                "n_folds": len(group),
                "test_years": ";".join(map(str, group["test_year"].astype(int))),
                "mae": mean_absolute_error(actual, predicted),
                "mape_percent": float(np.mean(np.abs(actual - predicted) / actual) * 100.0),
                "smape_percent": float(
                    np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))
                    * 100.0
                ),
            }
        )
    comparison_by_regime = pd.concat(
        [comparison_by_regime, pd.DataFrame(raw_regime_rows)], ignore_index=True
    ).sort_values(["metric", "test_regime", "smape_percent"])
    availability = data_availability_audit(data)

    behavior_rows: list[dict[str, Any]] = []
    for (metric, model), group in forecasts.groupby(["metric", "model"]):
        start = float(group.loc[group["year"] == 2025, "forecast"].iloc[0])
        end = float(group.loc[group["year"] == 2030, "forecast"].iloc[0])
        behavior_rows.append(
            {
                "metric": metric,
                "model": model,
                "forecast_2025": start,
                "forecast_2030": end,
                "five_year_change_percent": 100.0 * (end / start - 1.0),
                "extrapolation_note": (
                    "constant persistence baseline"
                    if model == "naive_last"
                    else "median of Ridge, BayesianRidge, and Huber"
                    if model == "robust_ml_ensemble"
                    else MODEL_SPECS[model].extrapolation_note
                ),
            }
        )
    behavior = pd.DataFrame(behavior_rows)

    backtests.to_csv(output_dir / "rolling_backtest_predictions.csv", index=False, float_format="%.8f")
    comparison.to_csv(output_dir / "model_comparison.csv", index=False, float_format="%.8f")
    comparison_by_regime.to_csv(
        output_dir / "model_comparison_by_regime.csv", index=False, float_format="%.8f"
    )
    pressure_tests.to_csv(
        output_dir / "pre_covid_long_horizon_pressure_test.csv", index=False, float_format="%.8f"
    )
    tuning.to_csv(output_dir / "selected_hyperparameters.csv", index=False, float_format="%.8f")
    forecasts.to_csv(output_dir / "ml_forecasts_2025_2030_all_models.csv", index=False, float_format="%.6f")
    selected_forecasts.to_csv(
        output_dir / "selected_ml_forecasts_2025_2030.csv", index=False, float_format="%.6f"
    )
    forecasts[forecasts["model"] == "bayesian_ridge"].to_csv(
        output_dir / "bayesian_conditional_forecasts_2025_2030.csv",
        index=False,
        float_format="%.6f",
    )
    imputations.to_csv(output_dir / "ml_missing_year_imputations.csv", index=False, float_format="%.6f")
    strict.to_csv(output_dir / "strict_evidence_sensitivity.csv", index=False, float_format="%.6f")
    importance.to_csv(output_dir / "permutation_importance_exploratory.csv", index=False, float_format="%.8f")
    raw_backtests.to_csv(
        output_dir / "raw_target_ridge_backtest.csv", index=False, float_format="%.8f"
    )
    raw_forecasts.to_csv(
        output_dir / "target_transform_sensitivity_forecast.csv", index=False, float_format="%.6f"
    )
    recommended_forecasts.to_csv(
        output_dir / "recommended_ml_forecasts_2025_2030.csv",
        index=False,
        float_format="%.6f",
    )
    availability.to_csv(output_dir / "ml_feature_availability_audit.csv", index=False)
    behavior.to_csv(output_dir / "forecast_behavior.csv", index=False, float_format="%.6f")
    create_forecast_figure(
        data,
        selected_forecasts,
        recommended_forecasts,
        forecasts,
        output_dir / "ml_forecast_comparison.png",
    )
    create_backtest_figure(comparison, output_dir / "rolling_backtest_smape.png")

    run_summary = {
        "input": str(input_path.relative_to(ROOT)),
        "python": os.sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "validation": "expanding-window rolling origin; all preprocessing and tuning inside each outer fold",
        "target_gap_policy": "never pre-impute targets; model-based imputations are diagnostic outputs only",
        "features": FEATURE_NAMES,
        "selected_ml_models": selected_models,
        "recommended_point_forecast": "raw_target_ridge_alpha_0.1",
        "ensemble_members": ENSEMBLE_MEMBERS,
        "interval_warning": (
            "BayesianRidge exports a model-conditional Gaussian 95% interval. The selected-model stress "
            "envelope uses the maximum rolling-origin absolute log error and has no coverage guarantee; "
            "there are too few calibration residuals for finite-sample 95% conformal coverage."
        ),
        "target_transform_warning": (
            "Raw-target Ridge is the recommended planning point path because it has lower rolling-origin "
            "MAPE here, but the large additive-versus-log difference remains structural uncertainty."
        ),
        "resource_decision": (
            "CPU sequential fitting (n_jobs=1): only 12 observations per target; available memory was low, "
            "and parallelism would add overhead without analytical benefit."
        ),
        "generated_files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input.resolve(), arguments.output_dir.resolve())
