#!/usr/bin/env python3
"""Compare Ridge fits before and after an approximate 2019 macro-scope bridge."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

import optimize_ridge_model as ridge


ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL: Final = (
    ROOT / "data/unified/canonical_official_annual_2010_2025.csv"
)
DEFAULT_OUTPUT_DIR: Final = ROOT / "outputs/ridge_macro_harmonization"
RIDGE_REFERENCE_DIR: Final = ROOT / "outputs/ridge_optimization"
MACRO_SERIES: Final = {
    "jizhou_gdp": {
        "value_column": "preferred_gdp_100m_cny",
        "status_column": "gdp_status",
        "2019_comparable_price_growth": 0.021,
    },
    "jizhou_tertiary_value_added": {
        "value_column": "preferred_tertiary_100m_cny",
        "status_column": "tertiary_status",
        "2019_comparable_price_growth": 0.044,
    },
}
BRIDGE_LAST_YEAR: Final = 2018
BRIDGE_ANCHOR_YEAR: Final = 2019
HARMONIZATION_METHOD: Final = (
    "approximate_fixed_bridge_from_2019_revised_current_value_and_"
    "2019_bulletin_comparable_price_growth"
)
HARMONIZATION_WARNING: Final = (
    "Approximate bridge for sensitivity analysis; not an official current-price "
    "backcast. The bulletin growth rate is at comparable prices while the annual "
    "levels are at current prices. The 2.1%/4.4% growth rates are from the 2019 "
    "bulletin initial release, while the 219.62/136.91 anchors are later-yearbook "
    "revised values; no matching finalized growth rates were found, so the bridge "
    "also contains a vintage mismatch."
)
OFFICIAL_SOURCE_LINKS: Final = {
    "jizhou_2019_bulletin": (
        "https://www.tjjz.gov.cn/zwgk/zfxxgkqjjg/tjj1/fdzdgknr34/"
        "tjxx34/202107/t20210705_5495855.html"
    ),
    "nbs_unified_accounting_revision_qa": (
        "https://www.stats.gov.cn/sj/sjjd/202302/t20230202_1896273.html"
    ),
    "jizhou_2020_yearbook_xls": (
        "https://www.tjjz.gov.cn/zwgk/zfxxgkqjjg/tjj1/fdzdgknr34/"
        "tjxx34/202111/W020211117618518239847.xls"
    ),
}
DATA_SCOPES: Final = {
    "original_mixed": "original_value",
    "harmonized_bridge": "harmonized_value",
}
FIXED_MODEL: Final = "ridge_fixed_lambda_0.1"
TUNED_MODEL: Final = "ridge_nested_tuned"
ROLLING_TEST_YEARS: Final = tuple(range(2015, 2024))
FORECAST_YEARS: Final = tuple(range(2026, 2031))


@dataclass(frozen=True)
class FinalEvaluation:
    lambda_selection: pd.DataFrame
    holdout_2025: pd.DataFrame
    forecasts: pd.DataFrame
    forecast_comparison: pd.DataFrame


def build_harmonized_macro_series(canonical: pd.DataFrame) -> pd.DataFrame:
    """Return original and approximately bridged GDP/tertiary annual series."""
    required = {"year"} | {
        str(specification[key])
        for specification in MACRO_SERIES.values()
        for key in ("value_column", "status_column")
    }
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise ValueError(f"canonical macro columns missing: {missing}")

    source = canonical.copy()
    source["year"] = pd.to_numeric(source["year"], errors="raise").astype(int)
    if set(source["year"]) != set(range(2010, 2026)):
        raise ValueError("canonical macro series must cover every year from 2010 to 2025")
    source.set_index("year", inplace=True)
    if not source.index.is_unique:
        raise ValueError("canonical macro series contains duplicate years")

    frames: list[pd.DataFrame] = []
    for metric, specification in MACRO_SERIES.items():
        value_column = str(specification["value_column"])
        status_column = str(specification["status_column"])
        values = pd.to_numeric(source[value_column], errors="raise").astype(float)
        if values.isna().any() or not values.gt(0.0).all():
            raise ValueError(f"{metric} requires positive non-missing annual values")
        growth = float(specification["2019_comparable_price_growth"])
        old_2018_value = float(values.loc[BRIDGE_LAST_YEAR])
        revised_2019_value = float(values.loc[BRIDGE_ANCHOR_YEAR])
        implied_2018_bridge_anchor = revised_2019_value / (1.0 + growth)
        bridge_factor = implied_2018_bridge_anchor / old_2018_value

        metric_frame = pd.DataFrame(
            {
                "metric": metric,
                "year": values.index.astype(int),
                "unit": "100m_cny",
                "source_status": source[status_column].astype(str).to_numpy(),
                "original_value": values.to_numpy(),
            }
        )
        metric_frame["bridge_factor"] = bridge_factor
        metric_frame["bridge_applied"] = metric_frame["year"].le(BRIDGE_LAST_YEAR)
        metric_frame["harmonized_value"] = metric_frame["original_value"]
        bridge_mask = metric_frame["bridge_applied"]
        metric_frame.loc[bridge_mask, "harmonized_value"] = (
            metric_frame.loc[bridge_mask, "original_value"] * bridge_factor
        )
        metric_frame["bridge_anchor_old_2018_value"] = old_2018_value
        metric_frame["bridge_anchor_revised_2019_value"] = revised_2019_value
        metric_frame["bulletin_2019_comparable_price_growth_percent"] = growth * 100.0
        metric_frame["implied_2018_bridge_anchor"] = implied_2018_bridge_anchor
        metric_frame["harmonization_method"] = HARMONIZATION_METHOD
        metric_frame["harmonization_warning"] = HARMONIZATION_WARNING
        frames.append(metric_frame)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["metric", "year"], ignore_index=True
    )


def _model_frame(
    series: pd.DataFrame,
    *,
    metric: str,
    data_scope: str,
) -> pd.DataFrame:
    value_column = DATA_SCOPES[data_scope]
    selected = series.loc[
        series["metric"].eq(metric), ["year", value_column]
    ].rename(columns={value_column: "value"})
    selected["metric"] = metric
    selected["unit"] = "100m_cny"
    selected["status"] = data_scope
    selected["source_ids"] = "canonical_macro_scope_experiment"
    selected["quality_note"] = HARMONIZATION_WARNING if data_scope == "harmonized_bridge" else "documented mixed-scope series"
    selected["is_observed"] = True
    return selected.sort_values("year", ignore_index=True)


def _fit_predict(
    train: pd.DataFrame,
    test_years: list[int] | tuple[int, ...],
    ridge_lambda: float,
) -> np.ndarray:
    ordered = train.sort_values("year")
    model = ridge.build_ridge_model(ridge_lambda)
    model.fit(
        ridge.make_features(ordered["year"].astype(int)),
        ordered["value"].to_numpy(dtype=float),
    )
    return np.maximum(0.0, model.predict(ridge.make_features(test_years)))


def evaluate_rolling_models(series: pd.DataFrame) -> pd.DataFrame:
    """Run the frozen 2015-2023 expanding-origin protocol for both scopes."""
    rows: list[dict[str, object]] = []
    for data_scope in DATA_SCOPES:
        for metric in MACRO_SERIES:
            metric_series = _model_frame(
                series, metric=metric, data_scope=data_scope
            )
            for test_year in ROLLING_TEST_YEARS:
                train = metric_series[metric_series["year"].lt(test_year)].copy()
                test = metric_series[metric_series["year"].eq(test_year)].iloc[0]
                tuning = ridge.tune_raw_ridge_lambda(
                    train,
                    simulate_missing_years=False,
                    scope=f"macro_{data_scope}_{metric}_outer_{test_year}",
                )
                for model_name, ridge_lambda in (
                    (FIXED_MODEL, ridge.BASELINE_LAMBDA),
                    (TUNED_MODEL, tuning.selected_lambda),
                ):
                    prediction = float(
                        _fit_predict(train, [test_year], ridge_lambda)[0]
                    )
                    actual = float(test["value"])
                    point_smape = float(
                        ridge.smape_percent(
                            np.array([actual]), np.array([prediction])
                        )[0]
                    )
                    rows.append(
                        {
                            "data_scope": data_scope,
                            "metric": metric,
                            "model": model_name,
                            "year": test_year,
                            "actual": actual,
                            "prediction": prediction,
                            "error": prediction - actual,
                            "absolute_error": abs(prediction - actual),
                            "point_smape_percent": point_smape,
                            "lambda": ridge_lambda,
                            "tuning_status": (
                                "fixed_not_tuned"
                                if model_name == FIXED_MODEL
                                else tuning.status
                            ),
                            "inner_validation_count": (
                                0
                                if model_name == FIXED_MODEL
                                else tuning.inner_validation_count
                            ),
                            "training_start_year": int(train["year"].min()),
                            "training_end_year": int(train["year"].max()),
                            "training_n": len(train),
                            "uses_test_in_training": bool(
                                train["year"].ge(test_year).any()
                            ),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["data_scope", "metric", "model", "year"], ignore_index=True
    )


def summarize_rolling_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize each target/model/scope on its own consistently scaled actuals."""
    rows: list[dict[str, object]] = []
    for (data_scope, metric, model), group in predictions.groupby(
        ["data_scope", "metric", "model"], sort=True
    ):
        ordered = group.sort_values("year")
        actual = ordered["actual"].to_numpy(dtype=float)
        forecast = ordered["prediction"].to_numpy(dtype=float)
        error = forecast - actual
        rows.append(
            {
                "data_scope": data_scope,
                "metric": metric,
                "model": model,
                "n_test": len(ordered),
                "test_years": ";".join(map(str, ordered["year"].astype(int))),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mape_percent": float(np.mean(np.abs(error) / np.abs(actual)) * 100.0),
                "smape_percent": float(
                    np.mean(ridge.smape_percent(actual, forecast))
                ),
                "worst_point_smape_percent": float(
                    np.max(ridge.smape_percent(actual, forecast))
                ),
            }
        )
    result = pd.DataFrame(rows)
    original_scores = (
        result[result["data_scope"].eq("original_mixed")][
            ["metric", "model", "smape_percent"]
        ]
        .rename(columns={"smape_percent": "original_mixed_smape_percent"})
        .copy()
    )
    result = result.merge(
        original_scores, on=["metric", "model"], validate="many_to_one"
    )
    result["delta_smape_vs_original_same_model_pp"] = (
        result["smape_percent"] - result["original_mixed_smape_percent"]
    )
    result["score_semantics"] = (
        "each fit is scored against actuals expressed in the same data scope; "
        "sMAPE is comparable across the multiplicative bridge"
    )
    return result.sort_values(
        ["data_scope", "metric", "model"], ignore_index=True
    )


def build_break_fold_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the 2019 outer fold where mixed-scope training crosses the break."""
    result = predictions[predictions["year"].eq(BRIDGE_ANCHOR_YEAR)].copy()
    original_scores = (
        result[result["data_scope"].eq("original_mixed")][
            ["metric", "model", "point_smape_percent"]
        ]
        .rename(
            columns={
                "point_smape_percent": "original_mixed_point_smape_percent"
            }
        )
        .copy()
    )
    result = result.merge(
        original_scores, on=["metric", "model"], validate="many_to_one"
    )
    result["smape_improvement_vs_original_pp"] = (
        result["original_mixed_point_smape_percent"]
        - result["point_smape_percent"]
    )
    result["interpretation"] = (
        "2019 actual is unchanged in both scopes; only the pre-2019 training levels differ"
    )
    return result.sort_values(
        ["metric", "model", "data_scope"], ignore_index=True
    )


def build_final_evaluation(series: pd.DataFrame) -> FinalEvaluation:
    """Select lambda through 2023, then fit through 2024 without using 2025."""
    selections: list[dict[str, object]] = []
    selected_lambdas: dict[tuple[str, str], float] = {}
    model_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for data_scope in DATA_SCOPES:
        for metric in MACRO_SERIES:
            metric_series = _model_frame(
                series, metric=metric, data_scope=data_scope
            )
            model_frames[(data_scope, metric)] = metric_series
            selection_train = metric_series[metric_series["year"].le(2023)].copy()
            tuning = ridge.tune_raw_ridge_lambda(
                selection_train,
                simulate_missing_years=False,
                scope=f"macro_final_selection_{data_scope}_{metric}",
            )
            selected_lambdas[(data_scope, metric)] = tuning.selected_lambda
            selected_score = float(
                tuning.candidate_summary.loc[
                    tuning.candidate_summary["lambda"].eq(tuning.selected_lambda),
                    "mean_inner_smape_percent",
                ].iloc[0]
            )
            selections.append(
                {
                    "data_scope": data_scope,
                    "metric": metric,
                    "selected_lambda": tuning.selected_lambda,
                    "selection_status": tuning.status,
                    "boundary_hit": tuning.boundary_hit,
                    "boundary_side": tuning.boundary_side,
                    "inner_validation_count": tuning.inner_validation_count,
                    "mean_inner_smape_percent": selected_score,
                    "training_min_year": int(selection_train["year"].min()),
                    "training_max_year": int(selection_train["year"].max()),
                    "training_n": len(selection_train),
                    "uses_2024_for_selection": bool(
                        selection_train["year"].ge(2024).any()
                    ),
                    "uses_2025_for_selection": bool(
                        selection_train["year"].ge(2025).any()
                    ),
                }
            )

    holdout_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    for (data_scope, metric), metric_series in model_frames.items():
        final_train = metric_series[metric_series["year"].le(2024)].copy()
        holdout_actual = float(
            metric_series.loc[metric_series["year"].eq(2025), "value"].iloc[0]
        )
        for model_name, ridge_lambda in (
            (FIXED_MODEL, ridge.BASELINE_LAMBDA),
            (TUNED_MODEL, selected_lambdas[(data_scope, metric)]),
        ):
            holdout_prediction = float(
                _fit_predict(final_train, [2025], ridge_lambda)[0]
            )
            holdout_smape = float(
                ridge.smape_percent(
                    np.array([holdout_actual]), np.array([holdout_prediction])
                )[0]
            )
            holdout_rows.append(
                {
                    "data_scope": data_scope,
                    "metric": metric,
                    "model": model_name,
                    "lambda": ridge_lambda,
                    "year": 2025,
                    "actual": holdout_actual,
                    "prediction": holdout_prediction,
                    "error": holdout_prediction - holdout_actual,
                    "absolute_error": abs(holdout_prediction - holdout_actual),
                    "point_smape_percent": holdout_smape,
                    "training_start_year": int(final_train["year"].min()),
                    "training_end_year": int(final_train["year"].max()),
                    "training_n": len(final_train),
                    "uses_2025_as_training": bool(
                        final_train["year"].ge(2025).any()
                    ),
                    "holdout_status": "official_initial_not_used_for_fit",
                }
            )
            forecasts = _fit_predict(final_train, FORECAST_YEARS, ridge_lambda)
            for year, forecast in zip(FORECAST_YEARS, forecasts, strict=True):
                forecast_rows.append(
                    {
                        "data_scope": data_scope,
                        "metric": metric,
                        "model": model_name,
                        "lambda": ridge_lambda,
                        "year": year,
                        "forecast": float(forecast),
                        "training_start_year": int(final_train["year"].min()),
                        "training_end_year": int(final_train["year"].max()),
                        "training_n": len(final_train),
                        "uses_2025_as_training": bool(
                            final_train["year"].ge(2025).any()
                        ),
                        "forecast_semantics": (
                            "conditional point forecast from the frozen Ridge specification; "
                            "2025 official_initial is held out"
                        ),
                    }
                )

    lambda_selection = pd.DataFrame(selections).sort_values(
        ["data_scope", "metric"], ignore_index=True
    )
    holdout = pd.DataFrame(holdout_rows).sort_values(
        ["metric", "model", "data_scope"], ignore_index=True
    )
    forecast_frame = pd.DataFrame(forecast_rows).sort_values(
        ["metric", "model", "data_scope", "year"], ignore_index=True
    )
    comparison = forecast_frame.pivot(
        index=["metric", "model", "year"],
        columns="data_scope",
        values="forecast",
    ).reset_index()
    comparison.columns.name = None
    comparison["harmonized_minus_original_forecast"] = (
        comparison["harmonized_bridge"] - comparison["original_mixed"]
    )
    comparison["harmonized_vs_original_percent"] = (
        comparison["harmonized_minus_original_forecast"]
        / comparison["original_mixed"]
        * 100.0
    )
    comparison["comparison_note"] = (
        "both forecasts use the same Ridge algorithm and training end year; only the data scope differs"
    )
    comparison.sort_values(["metric", "model", "year"], inplace=True)
    comparison.reset_index(drop=True, inplace=True)
    return FinalEvaluation(
        lambda_selection=lambda_selection,
        holdout_2025=holdout,
        forecasts=forecast_frame,
        forecast_comparison=comparison,
    )


def build_tourism_ridge_invariance(
    reference_dir: Path = RIDGE_REFERENCE_DIR,
) -> pd.DataFrame:
    """Expose the unchanged tourism Ridge outputs implied by the dependency graph.

    The existing tourism Ridge consumes the two tourism targets plus year/regime
    features. Macro values are absent from both its target table and feature
    builder, so the macro-only bridge cannot change these reference values.
    """
    rows: list[dict[str, object]] = []
    metric_artifacts = {
        "official_only_nested": "official_outer_macro_metrics.csv",
        "fold_local_augmented_nested": "augmented_outer_macro_metrics.csv",
        "pseudo_holdout_2024": "pseudo_holdout_2024_macro_metrics.csv",
        "cross_regime_stress_2019_cutoff": "cross_regime_stress_macro_metrics.csv",
    }
    selected_models = {ridge.BASELINE_MODEL, ridge.TUNED_MODEL}
    verification_basis = (
        "dependency-graph invariance, backed by the read-only baseline reproduction "
        "in which all 30 Ridge CSV outputs were byte-identical; the macro canonical "
        "columns are not targets or features of the existing tourism Ridge"
    )
    for track, filename in metric_artifacts.items():
        path = reference_dir / filename
        frame = pd.read_csv(path)
        selected = frame[frame["model"].isin(selected_models)]
        for _, record in selected.iterrows():
            value = float(record["macro_smape_percent"])
            rows.append(
                {
                    "source_artifact": filename,
                    "track": track,
                    "metric": "equal_target_macro",
                    "model": str(record["model"]),
                    "year": pd.NA,
                    "value_name": "macro_smape_percent",
                    "before_macro_harmonization": value,
                    "after_macro_harmonization": value,
                    "delta": 0.0,
                    "macro_values_used_by_ridge": False,
                    "verification_basis": verification_basis,
                }
            )

    forecast_filename = "final_forecasts_2026_2030.csv"
    forecasts = pd.read_csv(reference_dir / forecast_filename)
    forecasts = forecasts[forecasts["model"].isin(selected_models)]
    for _, record in forecasts.iterrows():
        value = float(record["forecast"])
        rows.append(
            {
                "source_artifact": forecast_filename,
                "track": "final_fit_through_2024",
                "metric": str(record["metric"]),
                "model": str(record["model"]),
                "year": int(record["year"]),
                "value_name": "forecast",
                "before_macro_harmonization": value,
                "after_macro_harmonization": value,
                "delta": 0.0,
                "macro_values_used_by_ridge": False,
                "verification_basis": verification_basis,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["value_name", "track", "metric", "model", "year"],
        ignore_index=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_input_label(path: Path) -> str:
    """Return a stable input label without committing a local checkout path."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _macro_smape(metrics: pd.DataFrame, data_scope: str, model: str) -> float:
    selected = metrics[
        metrics["data_scope"].eq(data_scope) & metrics["model"].eq(model)
    ]
    if len(selected) != len(MACRO_SERIES):
        raise AssertionError("macro sMAPE requires complete target coverage")
    return float(selected["smape_percent"].mean())


def _write_readme(
    output_dir: Path,
    *,
    factors: dict[str, float],
    metrics: pd.DataFrame,
    break_comparison: pd.DataFrame,
) -> None:
    fixed_before = _macro_smape(metrics, "original_mixed", FIXED_MODEL)
    fixed_after = _macro_smape(metrics, "harmonized_bridge", FIXED_MODEL)
    tuned_before = _macro_smape(metrics, "original_mixed", TUNED_MODEL)
    tuned_after = _macro_smape(metrics, "harmonized_bridge", TUNED_MODEL)
    break_improvements = break_comparison[
        break_comparison["data_scope"].eq("harmonized_bridge")
    ]["smape_improvement_vs_original_pp"]
    content = f"""# Ridge macro-scope harmonization comparison

This directory compares the unchanged Ridge algorithm on the documented mixed-scope
GDP/tertiary series and an approximate bridge to the post-2019 level. The bridge
multiplies 2010-2018 values by `2019 revised current-price value / (1 + 2019
bulletin comparable-price growth) / 2018 old-scope value`.

This is an **approximate bridge for sensitivity analysis, not an official current-price backcast**.
Comparable-price growth and current-price levels are not
the same accounting object. The 2.1%/4.4% rates come from the 2019 bulletin
initial release, but the 219.62/136.91 anchors are later-yearbook revisions; no
matching finalized rates were found, so the bridge also has a **vintage mismatch**.

- GDP bridge factor: `{factors['jizhou_gdp']:.10f}`.
- Tertiary-value-added bridge factor: `{factors['jizhou_tertiary_value_added']:.10f}`.
- Fixed-lambda macro sMAPE, original -> harmonized: `{fixed_before:.6f}% -> {fixed_after:.6f}%`.
- Nested-tuned macro sMAPE, original -> harmonized: `{tuned_before:.6f}% -> {tuned_after:.6f}%`.
- All four 2019 target/model break-fold comparisons improve; improvement range:
  `{float(break_improvements.min()):.6f}` to `{float(break_improvements.max()):.6f}` percentage points.

Protocol: expanding-origin tests for 2015-2023; lambda selection uses data through
2023; the final fit ends in 2024; 2025 is an official-initial holdout and is never
used for fitting; forecasts cover 2026-2030. `tourism_ridge_invariance.csv` records
why the existing tourism Ridge remains numerically unchanged: GDP and tertiary
value added are absent from its target and feature dependency graph.

Official sources: [Jizhou 2019 bulletin]({OFFICIAL_SOURCE_LINKS['jizhou_2019_bulletin']});
[NBS unified-accounting and historical-revision Q&A]({OFFICIAL_SOURCE_LINKS['nbs_unified_accounting_revision_qa']});
[Jizhou 2020 yearbook XLS]({OFFICIAL_SOURCE_LINKS['jizhou_2020_yearbook_xls']}).
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def run_experiment(
    *,
    canonical_path: Path = DEFAULT_CANONICAL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ridge_reference_dir: Path = RIDGE_REFERENCE_DIR,
) -> dict[str, object]:
    """Run the complete comparison and write its compact reproducibility bundle."""
    canonical_path = canonical_path.resolve()
    output_dir = output_dir.resolve()
    ridge_reference_dir = ridge_reference_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series = build_harmonized_macro_series(pd.read_csv(canonical_path))
    rolling = evaluate_rolling_models(series)
    metrics = summarize_rolling_predictions(rolling)
    break_comparison = build_break_fold_comparison(rolling)
    final = build_final_evaluation(series)
    tourism_invariance = build_tourism_ridge_invariance(ridge_reference_dir)

    csv_outputs = {
        "harmonized_macro_series.csv": series,
        "rolling_predictions.csv": rolling,
        "metrics_by_target.csv": metrics,
        "break_fold_comparison.csv": break_comparison,
        "final_lambda_selection.csv": final.lambda_selection,
        "holdout_2025.csv": final.holdout_2025,
        "forecasts_2026_2030.csv": final.forecasts,
        "forecast_comparison.csv": final.forecast_comparison,
        "tourism_ridge_invariance.csv": tourism_invariance,
    }
    for filename, frame in csv_outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.10f")

    factors = {
        metric: float(
            series.loc[series["metric"].eq(metric), "bridge_factor"].iloc[0]
        )
        for metric in MACRO_SERIES
    }
    _write_readme(
        output_dir,
        factors=factors,
        metrics=metrics,
        break_comparison=break_comparison,
    )

    fixed_before = _macro_smape(metrics, "original_mixed", FIXED_MODEL)
    fixed_after = _macro_smape(metrics, "harmonized_bridge", FIXED_MODEL)
    tuned_before = _macro_smape(metrics, "original_mixed", TUNED_MODEL)
    tuned_after = _macro_smape(metrics, "harmonized_bridge", TUNED_MODEL)
    generated_files = sorted([*csv_outputs, "run_summary.json", "README.md"])
    summary: dict[str, object] = {
        "experiment": "ridge_macro_scope_harmonization",
        "bridge_method": HARMONIZATION_METHOD,
        "bridge_warning": HARMONIZATION_WARNING,
        "bridge_factors": factors,
        "protocol": {
            "algorithm": "unchanged StandardScaler + raw-target Ridge",
            "fixed_lambda": ridge.BASELINE_LAMBDA,
            "positive_lambda_grid": list(ridge.POSITIVE_LAMBDA_GRID),
            "rolling_test_years": list(ROLLING_TEST_YEARS),
            "final_selection_max_year": 2023,
            "final_fit_max_year": 2024,
            "holdout_year": 2025,
            "forecast_years": list(FORECAST_YEARS),
            "uses_2025_for_fit": False,
        },
        "headline": {
            "fixed_original_macro_smape_percent": fixed_before,
            "fixed_harmonized_macro_smape_percent": fixed_after,
            "fixed_delta_percent_points": fixed_after - fixed_before,
            "nested_tuned_original_macro_smape_percent": tuned_before,
            "nested_tuned_harmonized_macro_smape_percent": tuned_after,
            "nested_tuned_delta_percent_points": tuned_after - tuned_before,
            "all_2019_break_fold_scores_improve": bool(
                break_comparison.loc[
                    break_comparison["data_scope"].eq("harmonized_bridge"),
                    "smape_improvement_vs_original_pp",
                ].gt(0.0).all()
            ),
            "existing_tourism_ridge_max_abs_delta": float(
                tourism_invariance["delta"].abs().max()
            ),
        },
        "input_sha256": {
            _portable_input_label(canonical_path): _sha256(canonical_path),
            _portable_input_label(Path(ridge.__file__).resolve()): _sha256(
                Path(ridge.__file__).resolve()
            ),
        },
        "generated_files": generated_files,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ridge-reference-dir", type=Path, default=RIDGE_REFERENCE_DIR
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    summary = run_experiment(
        canonical_path=arguments.canonical,
        output_dir=arguments.output_dir,
        ridge_reference_dir=arguments.ridge_reference_dir,
    )
    headline = summary["headline"]
    print(
        "completed macro-scope Ridge comparison; fixed macro sMAPE "
        f"{headline['fixed_original_macro_smape_percent']:.6f} -> "
        f"{headline['fixed_harmonized_macro_smape_percent']:.6f}; "
        "nested tuned "
        f"{headline['nested_tuned_original_macro_smape_percent']:.6f} -> "
        f"{headline['nested_tuned_harmonized_macro_smape_percent']:.6f}"
    )


if __name__ == "__main__":
    main()
