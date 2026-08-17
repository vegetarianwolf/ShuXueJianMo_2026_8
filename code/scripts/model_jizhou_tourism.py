#!/usr/bin/env python3
"""Exploratory, reproducible models for TJMML problem C.

The script intentionally keeps the 2020--2022 tourism gaps as missing.  It
fits transparent log-linear regressions to non-pandemic annual observations,
adds a post-2022 level break for the primary model, and exports diagnostics,
forecasts, scenario calculations, and a missingness audit.

Only NumPy, pandas, and matplotlib are required; these are available in the
project's standard Python environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Keep plotting caches out of the repository and support restricted runners.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "jizhou-tourism-model-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/jizhou_tourism_economy/official_annual_summary_2010_2025.csv"
DEFAULT_OUTPUT = ROOT / "outputs/jizhou_tourism_model"


@dataclass
class OLSFit:
    model_name: str
    metric: str
    columns: list[str]
    beta: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    fitted_log: np.ndarray
    observed_log: np.ndarray
    years: np.ndarray
    xtx_inv: np.ndarray
    sigma: float
    df_resid: int
    r_squared: float
    adjusted_r_squared: float
    aicc: float
    loocv_log_rmse: float
    durbin_watson: float
    jarque_bera: float
    jarque_bera_p: float


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    max_iter = 300
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(value: float, df: int) -> float:
    if df <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if value == 0.0:
        return 0.5
    x = df / (df + value * value)
    tail_twice = regularized_beta(x, df / 2.0, 0.5)
    return 1.0 - 0.5 * tail_twice if value > 0 else 0.5 * tail_twice


def student_t_p_two_sided(value: float, df: int) -> float:
    return min(1.0, 2.0 * (1.0 - student_t_cdf(abs(value), df)))


def student_t_critical_975(df: int) -> float:
    low, high = 0.0, 20.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if student_t_cdf(middle, df) < 0.975:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def design_matrix(years: np.ndarray, interrupted: bool) -> tuple[np.ndarray, list[str]]:
    t = years.astype(float) - 2010.0
    columns = ["intercept", "year_index"]
    pieces = [np.ones_like(t), t]
    if interrupted:
        pieces.append((years >= 2023).astype(float))
        columns.append("post_2022_level")
    return np.column_stack(pieces), columns


def _loocv_log_rmse(x: np.ndarray, y: np.ndarray) -> float:
    errors: list[float] = []
    for held_out in range(len(y)):
        keep = np.arange(len(y)) != held_out
        x_train = x[keep]
        y_train = y[keep]
        if np.linalg.matrix_rank(x_train) < x_train.shape[1]:
            continue
        beta = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        errors.append(float(y[held_out] - x[held_out] @ beta))
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else math.nan


def fit_log_ols(
    years: np.ndarray,
    values: np.ndarray,
    *,
    metric: str,
    model_name: str,
    interrupted: bool,
) -> OLSFit:
    if np.any(values <= 0):
        raise ValueError(f"{metric} contains non-positive values and cannot be log transformed")
    y = np.log(values.astype(float))
    x, columns = design_matrix(years, interrupted)
    n, k = x.shape
    if n <= k + 1:
        raise ValueError(f"{model_name} has too few observations: n={n}, k={k}")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    residuals = y - fitted
    rss = float(residuals @ residuals)
    df_resid = n - k
    sigma2 = rss / df_resid
    covariance = sigma2 * xtx_inv
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - rss / tss
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / df_resid
    aic = n * math.log(max(rss / n, 1.0e-300)) + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    dw_denom = float(residuals @ residuals)
    durbin_watson = (
        float(np.diff(residuals) @ np.diff(residuals)) / dw_denom if dw_denom else math.nan
    )
    centered = residuals - residuals.mean()
    variance = float(np.mean(centered**2))
    if variance > 0:
        skewness = float(np.mean(centered**3) / variance**1.5)
        excess_kurtosis = float(np.mean(centered**4) / variance**2 - 3.0)
        jarque_bera = n / 6.0 * (skewness**2 + 0.25 * excess_kurtosis**2)
        # A chi-square variable with 2 df has survival function exp(-x/2).
        jarque_bera_p = math.exp(-jarque_bera / 2.0)
    else:
        jarque_bera = 0.0
        jarque_bera_p = 1.0
    return OLSFit(
        model_name=model_name,
        metric=metric,
        columns=columns,
        beta=beta,
        covariance=covariance,
        residuals=residuals,
        fitted_log=fitted,
        observed_log=y,
        years=years,
        xtx_inv=xtx_inv,
        sigma=math.sqrt(sigma2),
        df_resid=df_resid,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        aicc=aicc,
        loocv_log_rmse=_loocv_log_rmse(x, y),
        durbin_watson=durbin_watson,
        jarque_bera=jarque_bera,
        jarque_bera_p=jarque_bera_p,
    )


def parameter_rows(fit: OLSFit) -> list[dict[str, float | int | str]]:
    critical = student_t_critical_975(fit.df_resid)
    standard_errors = np.sqrt(np.diag(fit.covariance))
    rows: list[dict[str, float | int | str]] = []
    for name, estimate, se in zip(fit.columns, fit.beta, standard_errors, strict=True):
        t_value = float(estimate / se)
        rows.append(
            {
                "metric": fit.metric,
                "model": fit.model_name,
                "parameter": name,
                "estimate": float(estimate),
                "standard_error": float(se),
                "t_value": t_value,
                "p_value": student_t_p_two_sided(t_value, fit.df_resid),
                "ci95_lower": float(estimate - critical * se),
                "ci95_upper": float(estimate + critical * se),
                "df_resid": fit.df_resid,
            }
        )
    return rows


def diagnostic_row(fit: OLSFit) -> dict[str, float | int | str]:
    values = np.exp(fit.observed_log)
    fitted = np.exp(fit.fitted_log)
    return {
        "metric": fit.metric,
        "model": fit.model_name,
        "n": len(fit.years),
        "first_year": int(fit.years.min()),
        "last_year": int(fit.years.max()),
        "r_squared_log": fit.r_squared,
        "adjusted_r_squared_log": fit.adjusted_r_squared,
        "rmse_original_units": float(np.sqrt(np.mean((values - fitted) ** 2))),
        "mape_percent": float(np.mean(np.abs((values - fitted) / values)) * 100.0),
        "sigma_log": fit.sigma,
        "aicc_log": fit.aicc,
        "loocv_log_rmse": fit.loocv_log_rmse,
        "durbin_watson": fit.durbin_watson,
        "jarque_bera": fit.jarque_bera,
        "jarque_bera_p": fit.jarque_bera_p,
    }


def forecast_rows(fit: OLSFit, years: np.ndarray) -> list[dict[str, float | int | str]]:
    interrupted = "post_2022_level" in fit.columns
    x, _ = design_matrix(years, interrupted)
    predicted_log = x @ fit.beta
    critical = student_t_critical_975(fit.df_resid)
    rows: list[dict[str, float | int | str]] = []
    for year, row, mean_log in zip(years, x, predicted_log, strict=True):
        leverage = float(row @ fit.xtx_inv @ row)
        mean_se = fit.sigma * math.sqrt(leverage)
        prediction_se = fit.sigma * math.sqrt(1.0 + leverage)
        rows.append(
            {
                "metric": fit.metric,
                "model": fit.model_name,
                "year": int(year),
                "forecast": math.exp(float(mean_log)),
                "mean_ci95_lower": math.exp(float(mean_log) - critical * mean_se),
                "mean_ci95_upper": math.exp(float(mean_log) + critical * mean_se),
                "prediction_interval95_lower": math.exp(float(mean_log) - critical * prediction_se),
                "prediction_interval95_upper": math.exp(float(mean_log) + critical * prediction_se),
            }
        )
    return rows


def observed_series(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "tourist_visits":
        value_column = "preferred_visitor_10k_persons"
        status_column = "visitor_status"
    elif metric == "tourism_comprehensive_income":
        value_column = "preferred_comprehensive_income_100m_cny"
        status_column = "comprehensive_status"
    else:
        raise KeyError(metric)
    return data[["year", value_column, status_column]].rename(
        columns={value_column: "value", status_column: "status"}
    )


def fit_metric_models(data: pd.DataFrame, metric: str) -> list[OLSFit]:
    series = observed_series(data, metric).dropna(subset=["value"]).copy()

    # Normal-trend models exclude the pandemic years.  The 2021 revenue value
    # remains in the audit and shock calculations, but is not used to estimate
    # a normal-growth slope.
    structural = series[~series["year"].between(2020, 2022)]
    pre_covid = series[series["year"] <= 2019]
    if metric == "tourist_visits":
        excluded_statuses = {"observed_cached"}
    else:
        excluded_statuses = {
            "inferred_from_yoy",
            "observed_cached",
            "observed_supporting_attachment",
        }
    strict = structural[~structural["status"].isin(excluded_statuses)]

    return [
        fit_log_ols(
            pre_covid["year"].to_numpy(),
            pre_covid["value"].to_numpy(),
            metric=metric,
            model_name="pre_covid_exponential",
            interrupted=False,
        ),
        fit_log_ols(
            structural["year"].to_numpy(),
            structural["value"].to_numpy(),
            metric=metric,
            model_name="no_break_log_linear",
            interrupted=False,
        ),
        fit_log_ols(
            structural["year"].to_numpy(),
            structural["value"].to_numpy(),
            metric=metric,
            model_name="post_2022_level_break",
            interrupted=True,
        ),
        fit_log_ols(
            strict["year"].to_numpy(),
            strict["value"].to_numpy(),
            metric=metric,
            model_name="strict_evidence_level_break",
            interrupted=True,
        ),
    ]


def scenario_rows(years: np.ndarray) -> pd.DataFrame:
    """Create policy-anchored planning scenarios.

    These are transparent accounting assumptions rather than causal estimates.
    The 2025 visitor/revenue targets are used only as provisional anchors because
    annual 2025 actuals are unavailable.  The baseline revenue growth rate is the
    8% target stated in the 2026 district government work report.  Visitor counts
    follow from revenue divided by nominal spending per visit.
    """
    anchor_revenue = 231.0  # 100m CNY, 2025 target (not actual)
    anchor_visits = 2800.0  # 10k person-visits, 2025 target (not actual)
    anchor_spend = anchor_revenue / anchor_visits * 10000.0  # CNY per visit
    rows: list[dict[str, float | int | str]] = []
    scenario_parameters = {
        "baseline_policy_anchor": {
            "revenue_growth": 0.08,
            "spend_growth": 0.03,
            "initial_shock": 0.0,
            "assumption": "2025 targets as proxies; revenue +8%/yr; spend per visit +3%/yr",
        },
        "optimistic_assumption": {
            "revenue_growth": 0.12,
            "spend_growth": 0.04,
            "initial_shock": 0.0,
            "assumption": "2025 targets as proxies; revenue +12%/yr; spend per visit +4%/yr",
        },
        "pessimistic_assumption": {
            "revenue_growth": 0.05,
            "spend_growth": 0.02,
            "initial_shock": 0.15,
            "assumption": "2025 targets as proxies; 2026 -15% shock, then revenue +5%/yr; spend +2%/yr",
        },
    }
    for scenario, parameters in scenario_parameters.items():
        for year in years:
            horizon = int(year) - 2025
            shock_multiplier = 1.0 - parameters["initial_shock"]
            if parameters["initial_shock"] > 0:
                revenue = (
                    anchor_revenue
                    * shock_multiplier
                    * (1.0 + parameters["revenue_growth"]) ** (horizon - 1)
                )
            else:
                revenue = anchor_revenue * (1.0 + parameters["revenue_growth"]) ** horizon
            spend = anchor_spend * (1.0 + parameters["spend_growth"]) ** horizon
            visits = revenue / spend * 10000.0
            common = {
                "year": int(year),
                "scenario": scenario,
                "assumption": parameters["assumption"],
            }
            rows.extend(
                [
                    {
                        **common,
                        "metric": "tourist_visits",
                        "value": visits,
                        "unit": "10k_persons",
                    },
                    {
                        **common,
                        "metric": "tourism_comprehensive_income",
                        "value": revenue,
                        "unit": "100m_cny",
                    },
                    {
                        **common,
                        "metric": "nominal_spend_per_visit",
                        "value": spend,
                        "unit": "cny_per_visit",
                    },
                ]
            )
    return pd.DataFrame(rows)


def missingness_audit(data: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("tourist_visits", "preferred_visitor_10k_persons", "visitor_status", "10k_persons"),
        (
            "tourism_comprehensive_income",
            "preferred_comprehensive_income_100m_cny",
            "comprehensive_status",
            "100m_cny",
        ),
        ("gdp", "preferred_gdp_100m_cny", "gdp_status", "100m_cny"),
        ("tertiary_value_added", "preferred_tertiary_100m_cny", "tertiary_status", "100m_cny"),
    ]
    rows: list[dict[str, str | int | float]] = []
    for metric, value_col, status_col, unit in definitions:
        missing_years = data.loc[data[value_col].isna(), "year"].astype(int).tolist()
        observed = int(data[value_col].notna().sum())
        questionable = int(
            data[status_col]
            .fillna("")
            .str.contains("inferred|provisional|cached|supporting", regex=True)
            .sum()
        )
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "years_total": len(data),
                "years_with_value": observed,
                "coverage_percent": 100.0 * observed / len(data),
                "missing_years": ";".join(map(str, missing_years)),
                "questionable_or_provisional_years": questionable,
            }
        )
    return pd.DataFrame(rows)


def aggregate_constraint_rows(data: pd.DataFrame, primary_2025: pd.DataFrame) -> pd.DataFrame:
    visits_2023_2024 = float(
        data.loc[data["year"].isin([2023, 2024]), "preferred_visitor_10k_persons"].sum()
    )
    revenue_known = float(
        data.loc[data["year"].isin([2021, 2023, 2024]), "preferred_comprehensive_income_100m_cny"].sum()
    )
    visit_2025 = float(
        primary_2025.loc[primary_2025["metric"] == "tourist_visits", "forecast"].iloc[0]
    )
    revenue_2025 = float(
        primary_2025.loc[
            primary_2025["metric"] == "tourism_comprehensive_income", "forecast"
        ].iloc[0]
    )
    return pd.DataFrame(
        [
            {
                "metric": "tourist_visits",
                "five_year_total_2021_2025": 10500.0,
                "known_annual_values_sum": visits_2023_2024,
                "known_years": "2023;2024",
                "model_2025": visit_2025,
                "residual_for_other_missing_years": 10500.0 - visits_2023_2024 - visit_2025,
                "other_missing_years": "2021;2022",
            },
            {
                "metric": "tourism_comprehensive_income",
                "five_year_total_2021_2025": 792.5,
                "known_annual_values_sum": revenue_known,
                "known_years": "2021;2023;2024",
                "model_2025": revenue_2025,
                "residual_for_other_missing_years": 792.5 - revenue_known - revenue_2025,
                "other_missing_years": "2022",
            },
        ]
    )


def create_figure(data: pd.DataFrame, forecasts: pd.DataFrame, output_path: Path) -> None:
    labels = {
        "tourist_visits": ("Tourist visits", "10,000 person-visits"),
        "tourism_comprehensive_income": ("Tourism comprehensive income", "100 million CNY"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for axis, (metric, (title, ylabel)) in zip(axes, labels.items(), strict=True):
        series = observed_series(data, metric)
        axis.plot(series["year"], series["value"], "o-", color="#1f4e79", label="official preferred")
        primary = forecasts[
            (forecasts["metric"] == metric) & (forecasts["model"] == "post_2022_level_break")
        ]
        axis.plot(primary["year"], primary["forecast"], "o--", color="#c55a11", label="baseline forecast")
        axis.fill_between(
            primary["year"].to_numpy(dtype=float),
            primary["prediction_interval95_lower"].to_numpy(dtype=float),
            primary["prediction_interval95_upper"].to_numpy(dtype=float),
            color="#f4b183",
            alpha=0.3,
            label="95% prediction interval",
        )
        axis.axvspan(2020, 2022, color="grey", alpha=0.15, label="pandemic / missing block")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    metrics = ["tourist_visits", "tourism_comprehensive_income"]
    all_fits: list[OLSFit] = []
    for metric in metrics:
        all_fits.extend(fit_metric_models(data, metric))

    diagnostics = pd.DataFrame([diagnostic_row(fit) for fit in all_fits])
    parameters = pd.DataFrame([row for fit in all_fits for row in parameter_rows(fit)])

    forecast_years = np.arange(2025, 2031)
    forecasts = pd.DataFrame(
        [row for fit in all_fits for row in forecast_rows(fit, forecast_years)]
    )
    primary = forecasts[forecasts["model"] == "post_2022_level_break"].copy()
    scenarios = scenario_rows(np.arange(2026, 2031))
    gaps = missingness_audit(data)
    aggregate_constraints = aggregate_constraint_rows(data, primary[primary["year"] == 2025])

    diagnostics.to_csv(output_dir / "model_diagnostics.csv", index=False, float_format="%.8f")
    parameters.to_csv(output_dir / "parameter_estimates.csv", index=False, float_format="%.8f")
    forecasts.to_csv(output_dir / "forecasts_2025_2030_all_models.csv", index=False, float_format="%.6f")
    primary[primary["year"] >= 2026].to_csv(
        output_dir / "baseline_forecast_2026_2030.csv", index=False, float_format="%.6f"
    )
    scenarios.to_csv(output_dir / "scenario_forecasts_2026_2030.csv", index=False, float_format="%.6f")
    gaps.to_csv(output_dir / "missingness_audit.csv", index=False, float_format="%.2f")
    aggregate_constraints.to_csv(
        output_dir / "five_year_aggregate_constraints.csv", index=False, float_format="%.6f"
    )
    create_figure(data, forecasts, output_dir / "trend_and_baseline_forecast.png")

    summary = {
        "input": str(input_path.relative_to(ROOT)),
        "primary_model": "post_2022_level_break",
        "excluded_from_normal_trend_fit": "2020-2022",
        "forecast_origin_note": "2025 annual actual is unavailable; 2025 is predicted, forecasts requested start in 2026",
        "scenario_note": (
            "planning scenarios use 2025 targets as provisional anchors; optimistic and pessimistic "
            "cases are transparent assumptions, not causal estimates"
        ),
        "generated_files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input.resolve(), arguments.output_dir.resolve())
