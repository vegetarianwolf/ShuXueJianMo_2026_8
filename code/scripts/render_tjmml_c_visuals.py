#!/usr/bin/env python3
"""Render deterministic, problem-aligned figures for the C-problem report."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_NAMES: Final = (
    "q1_required_indicators.png",
    "q2_model_judgement.png",
    "q2_forecast_2026_2030.png",
    "q3_scenarios_sensitivity.png",
)

COLORS: Final = {
    "ridge": "#0072B2",
    "ols": "#D55E00",
    "naive": "#6B7280",
    "baseline": "#0072B2",
    "optimistic": "#009E73",
    "pessimistic": "#D55E00",
    "actual": "#222222",
    "simulated": "#CC79A7",
    "q1_simple": "#7A5195",
    "pandemic": "#E5E7EB",
    "grid": "#D1D5DB",
}

MODEL_LABELS: Final = {
    "raw_target_ridge_alpha_0.1": "Raw-target Ridge",
    "no_break_log_linear_common_rows": "无断点 Log-linear OLS",
    "naive_last": "Naive-last",
    "pre_covid_exponential": "Q1 疫情前指数模型",
}

MODEL_COLORS: Final = {
    "raw_target_ridge_alpha_0.1": COLORS["ridge"],
    "no_break_log_linear_common_rows": COLORS["ols"],
    "naive_last": COLORS["naive"],
    "pre_covid_exponential": COLORS["q1_simple"],
}

METRIC_LABELS: Final = {
    "tourist_visits": "旅游接待人次",
    "tourism_comprehensive_income": "旅游综合收入",
}

METRIC_UNITS: Final = {
    "tourist_visits": "万人次",
    "tourism_comprehensive_income": "亿元",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "axes.titleweight": "normal",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
        }
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.14,
        metadata={"Software": "TJMML-C unified benchmark"},
    )
    plt.close(fig)


def _format_axes(ax: plt.Axes, *, x_label: str = "年份", y_label: str = "") -> None:
    ax.grid(axis="y")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines[["top", "right"]].set_visible(False)


def _annotate_values(ax: plt.Axes, bars, *, decimals: int = 1) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            f"{value:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )


def _render_q1(canonical: pd.DataFrame, augmented: pd.DataFrame, path: Path) -> None:
    definitions = (
        (
            "preferred_visitor_10k_persons",
            "tourist_visits",
            "旅游接待人次",
            "万人次",
        ),
        (
            "preferred_comprehensive_income_100m_cny",
            "tourism_comprehensive_income",
            "旅游综合收入",
            "亿元",
        ),
        ("preferred_gdp_100m_cny", None, "地区生产总值（GDP）", "亿元"),
        ("preferred_tertiary_100m_cny", None, "第三产业增加值", "亿元"),
    )
    _require_columns(
        canonical,
        {"year", *(item[0] for item in definitions)},
        "canonical annual indicators",
    )
    _require_columns(
        augmented,
        {"metric", "year", "value", "is_simulated"},
        "augmented training contract",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8), sharex=True)
    for ax, (column, metric, title, unit) in zip(axes.flat, definitions, strict=True):
        years = canonical["year"].to_numpy(dtype=int)
        values = pd.to_numeric(canonical[column], errors="coerce").to_numpy(dtype=float)
        coverage = int(np.isfinite(values).sum())
        ax.axvspan(2019.5, 2022.5, color=COLORS["pandemic"], alpha=0.75, zorder=0)
        ax.plot(
            years,
            values,
            color=COLORS["actual"],
            linewidth=1.6,
            marker="o",
            markersize=4.2,
            label="canonical 值",
            zorder=3,
        )
        if metric is not None:
            metric_augmented = augmented[augmented["metric"].eq(metric)].copy()
            metric_augmented["is_simulated"] = metric_augmented["is_simulated"].astype(bool)
            ax.plot(
                metric_augmented["year"],
                metric_augmented["value"],
                color=COLORS["simulated"],
                linewidth=1.1,
                linestyle="--",
                alpha=0.8,
                label="增强训练路径",
                zorder=2,
            )
            simulated = metric_augmented[metric_augmented["is_simulated"]]
            ax.scatter(
                simulated["year"],
                simulated["value"],
                marker="X",
                s=48,
                color=COLORS["simulated"],
                edgecolor="white",
                linewidth=0.6,
                label="模拟训练点",
                zorder=4,
            )
        if column in {"preferred_gdp_100m_cny", "preferred_tertiary_100m_cny"}:
            ax.axvline(2019, color=COLORS["naive"], linewidth=1.0, linestyle=":")
            ax.text(
                2019.15,
                0.96,
                "宏观口径变化",
                transform=ax.get_xaxis_transform(),
                fontsize=7.5,
                va="top",
                color=COLORS["naive"],
            )
        ax.set_title(f"{title}｜覆盖 {coverage}/16")
        _format_axes(ax, y_label=unit)
        ax.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])

    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0, 1].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle(
        "问题1：题目要求的四项核心指标、数据覆盖与疫情缺口",
        fontsize=14,
        fontweight="normal",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "灰色区间为 2020—2022 年；紫色叉号是训练期内生成的模拟值，不是官方观测。",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    _save(fig, path)


def _metric_smape(frame: pd.DataFrame, metric: str, model: str) -> float:
    row = frame[frame["metric"].eq(metric) & frame["model"].eq(model)]
    if len(row) != 1:
        raise ValueError(f"expected one sMAPE row for {metric}/{model}, got {len(row)}")
    return float(row.iloc[0]["smape_percent"])


def _macro_smape(frame: pd.DataFrame, model: str) -> float:
    rows = frame[frame["model"].eq(model)]
    if set(rows["metric"]) != set(METRIC_LABELS):
        raise ValueError(f"incomplete target coverage for {model}")
    return float(rows["smape_percent"].mean())


def _render_q2_judgement(output_dir: Path, path: Path) -> None:
    simulated = pd.read_csv(output_dir / "stability_metrics_by_target.csv")
    official = pd.read_csv(
        output_dir / "stability_official_only_common_row_metrics_by_target.csv"
    )
    holdout = pd.read_csv(output_dir / "final_holdout_2024_metrics_by_target.csv")
    required = {"metric", "model", "smape_percent"}
    for frame, label in ((simulated, "simulated metrics"), (official, "canonical metrics"), (holdout, "holdout metrics")):
        _require_columns(frame, required, label)

    models = [
        "raw_target_ridge_alpha_0.1",
        "no_break_log_linear_common_rows",
        "naive_last",
    ]
    tracks = [("模拟增强", simulated), ("未模拟 canonical", official)]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8))

    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.32
    for offset, model in zip((-width / 2, width / 2), models[:2], strict=True):
        values = [_macro_smape(frame, model) for _, frame in tracks]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        _annotate_values(ax, bars)
    ax.set_xticks(x, [item[0] for item in tracks])
    ax.set_title("两目标等权 macro-sMAPE")
    _format_axes(ax, x_label="训练轨", y_label="sMAPE（%，越低越好）")

    for ax, metric in zip((axes[0, 1], axes[1, 0]), METRIC_LABELS, strict=True):
        x = np.arange(2)
        width = 0.24
        for index, model in enumerate(models):
            values = [_metric_smape(frame, metric, model) for _, frame in tracks]
            bars = ax.bar(
                x + (index - 1) * width,
                values,
                width,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
            _annotate_values(ax, bars)
        ax.set_xticks(x, [item[0] for item in tracks])
        ax.set_title(f"{METRIC_LABELS[metric]}滚动回测")
        _format_axes(ax, x_label="训练轨", y_label="sMAPE（%）")

    ax = axes[1, 1]
    x = np.arange(2)
    width = 0.24
    for index, model in enumerate(models):
        values = [_metric_smape(holdout, metric, model) for metric in METRIC_LABELS]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        _annotate_values(ax, bars)
    ax.set_xticks(x, [METRIC_LABELS[item] for item in METRIC_LABELS])
    ax.set_title("2024 pseudo-holdout（每目标 n=1）")
    _format_axes(ax, x_label="目标", y_label="sMAPE（%）")

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("问题2：用题目两个预测目标评判模型", fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.012,
        "sMAPE、naive 基线与 pseudo-holdout 是为题目的“适用性、合理性和模型优劣”设置的评价工具，并非题面指定指标。",
        ha="center",
        fontsize=8.2,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    _save(fig, path)


def _render_q2_forecast(
    canonical: pd.DataFrame,
    forecasts: pd.DataFrame,
    q1_simple_forecasts: pd.DataFrame,
    path: Path,
) -> None:
    required = {
        "metric",
        "model",
        "year",
        "forecast",
        "mean_ci95_lower",
        "mean_ci95_upper",
        "prediction_interval95_lower",
        "prediction_interval95_upper",
    }
    _require_columns(forecasts, required, "problem2 forecasts")
    _require_columns(
        q1_simple_forecasts,
        {"metric", "model", "year", "forecast"},
        "problem1 simple-growth forecasts",
    )
    definitions = {
        "tourist_visits": "preferred_visitor_10k_persons",
        "tourism_comprehensive_income": "preferred_comprehensive_income_100m_cny",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for ax, metric in zip(axes, METRIC_LABELS, strict=True):
        history = canonical[["year", definitions[metric]]].rename(
            columns={definitions[metric]: "value"}
        )
        history = history[history["year"].between(2018, 2024) & history["value"].notna()]
        ax.plot(
            history["year"],
            history["value"],
            color=COLORS["actual"],
            marker="o",
            linewidth=1.5,
            label="canonical 实际",
            zorder=5,
        )
        for model in (
            "raw_target_ridge_alpha_0.1",
            "no_break_log_linear_common_rows",
        ):
            model_frame = forecasts[
                forecasts["metric"].eq(metric) & forecasts["model"].eq(model)
            ].sort_values("year")
            if model_frame.empty:
                raise ValueError(f"missing future forecasts for {metric}/{model}")
            color = MODEL_COLORS[model]
            years = model_frame["year"].to_numpy(dtype=float)
            ax.fill_between(
                years,
                model_frame["prediction_interval95_lower"].to_numpy(dtype=float),
                model_frame["prediction_interval95_upper"].to_numpy(dtype=float),
                color=color,
                alpha=0.08,
                linewidth=0,
            )
            ax.fill_between(
                years,
                model_frame["mean_ci95_lower"].to_numpy(dtype=float),
                model_frame["mean_ci95_upper"].to_numpy(dtype=float),
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                years,
                model_frame["forecast"].to_numpy(dtype=float),
                color=color,
                marker="o" if model.startswith("raw") else "s",
                linewidth=1.8,
                label=MODEL_LABELS[model],
                zorder=4,
            )
        q1_line = q1_simple_forecasts[
            q1_simple_forecasts["metric"].eq(metric)
            & q1_simple_forecasts["model"].eq("pre_covid_exponential")
        ].sort_values("year")
        if q1_line.empty:
            raise ValueError(f"missing Q1 simple-growth forecasts for {metric}")
        ax.plot(
            q1_line["year"].to_numpy(dtype=float),
            q1_line["forecast"].to_numpy(dtype=float),
            color=COLORS["q1_simple"],
            marker="^",
            markersize=4.5,
            linestyle="--",
            linewidth=1.3,
            label=MODEL_LABELS["pre_covid_exponential"],
            zorder=3,
        )
        ax.axvspan(2024.5, 2025.5, color=COLORS["pandemic"], alpha=0.55)
        ax.text(
            2025,
            0.97,
            "2025实际缺失",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            color="#4B5563",
        )
        ax.set_title(METRIC_LABELS[metric])
        _format_axes(ax, y_label=METRIC_UNITS[metric])
        ax.set_xticks([2018, 2020, 2022, 2024, 2026, 2028, 2030])
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "问题2：2026—2030 预测区间及与问题1初步模型的对比",
        fontsize=14,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "深色带：OLS为 log 均值指数化（原尺度条件中位数）区间，Ridge为条件均值区间；浅色带为单次预测区间。Q1虚线仅作机械外推对照。",
        ha="center",
        fontsize=8.2,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    _save(fig, path)


def _render_q3(scenarios: pd.DataFrame, sensitivity: pd.DataFrame, path: Path) -> None:
    _require_columns(
        scenarios,
        {"year", "scenario", "metric", "value"},
        "problem3 scenario forecasts",
    )
    _require_columns(
        sensitivity,
        {
            "factor",
            "setting",
            "metric",
            "baseline_2030",
            "scenario_2030",
            "delta_percent",
        },
        "problem3 sensitivity",
    )
    scenario_labels = {
        "baseline_policy_anchor": "基准",
        "optimistic_assumption": "乐观",
        "pessimistic_assumption": "悲观",
    }
    scenario_colors = {
        "baseline_policy_anchor": COLORS["baseline"],
        "optimistic_assumption": COLORS["optimistic"],
        "pessimistic_assumption": COLORS["pessimistic"],
    }
    factor_labels = {
        "source_market_growth": "客源市场年增速 ±2pp",
        "new_format_spend_growth": "新业态/人均消费增速 ±1pp",
        "policy_coordination_multiplier": "政策与协同水平 ±5%",
        "external_shock": "2026突发冲击 0%~-15%",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0))
    for ax, metric in zip(axes[0], METRIC_LABELS, strict=True):
        metric_frame = scenarios[scenarios["metric"].eq(metric)]
        for scenario in scenario_labels:
            line = metric_frame[metric_frame["scenario"].eq(scenario)].sort_values("year")
            if line.empty:
                raise ValueError(f"missing scenario {scenario}/{metric}")
            ax.plot(
                line["year"],
                line["value"],
                marker="o",
                linewidth=2.0,
                color=scenario_colors[scenario],
                label=scenario_labels[scenario],
            )
        ax.set_title(f"{METRIC_LABELS[metric]}三情景路径")
        _format_axes(ax, y_label=METRIC_UNITS[metric])
        ax.set_xticks([2026, 2027, 2028, 2029, 2030])
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")

    for ax, metric in zip(axes[1], METRIC_LABELS, strict=True):
        metric_frame = sensitivity[sensitivity["metric"].eq(metric)].copy()
        factors = list(dict.fromkeys(metric_frame["factor"].astype(str)))
        y = np.arange(len(factors))
        for index, factor in enumerate(factors):
            rows = metric_frame[metric_frame["factor"].eq(factor)]
            low = rows[rows["setting"].eq("low")]
            high = rows[rows["setting"].eq("high")]
            if len(low) != 1 or len(high) != 1:
                raise ValueError(f"sensitivity factor {factor}/{metric} needs low and high")
            low_value = float(low.iloc[0]["delta_percent"])
            high_value = float(high.iloc[0]["delta_percent"])
            low_value = 0.0 if abs(low_value) < 0.05 else low_value
            high_value = 0.0 if abs(high_value) < 0.05 else high_value
            left, right = sorted((low_value, high_value))
            if np.isclose(left, right, atol=1.0e-12):
                ax.scatter(
                    [left],
                    [index],
                    color=COLORS["naive"],
                    marker="D",
                    s=30,
                    zorder=3,
                )
                ax.text(
                    left,
                    index - 0.18,
                    f"{left:+.1f}%" if left else "0.0%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
                continue
            ax.plot([left, right], [index, index], color=COLORS["naive"], linewidth=6, solid_capstyle="butt")
            ax.scatter([low_value], [index], color=COLORS["pessimistic"], marker="<", s=42, zorder=3)
            ax.scatter([high_value], [index], color=COLORS["optimistic"], marker=">", s=42, zorder=3)
            ax.text(left, index - 0.18, f"{left:+.1f}%", ha="right", va="center", fontsize=7.5)
            ax.text(right, index - 0.18, f"{right:+.1f}%", ha="left", va="center", fontsize=7.5)
        ax.axvline(0.0, color=COLORS["actual"], linewidth=1.0)
        ax.set_yticks(y, [factor_labels.get(item, item) for item in factors])
        ax.set_title(f"{METRIC_LABELS[metric]}：2030单因素敏感性")
        _format_axes(ax, x_label="相对基准变化（%）", y_label="")
        ax.invert_yaxis()

    fig.suptitle("问题3：三情景预测与关键情景杠杆", fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.01,
        "敏感性结果来自透明的情景假设与收入恒等式，不是从历史样本识别出的因果政策效应。",
        ha="center",
        fontsize=8.2,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    _save(fig, path)


def render_problem_visuals(output_dir: Path, canonical_path: Path) -> list[str]:
    """Render the four deterministic figures embedded by the Markdown report."""
    _configure_style()
    output_dir = Path(output_dir)
    canonical = pd.read_csv(canonical_path)
    augmented = pd.read_csv(output_dir / "primary_train_augmented.csv")
    forecasts = pd.read_csv(output_dir / "problem2_forecasts_2026_2030.csv")
    q1_simple_forecasts = pd.read_csv(
        output_dir / "problem1_simple_growth_forecasts_2026_2030.csv"
    )
    scenarios = pd.read_csv(output_dir / "problem3_scenario_forecasts_2026_2030.csv")
    sensitivity = pd.read_csv(output_dir / "problem3_policy_sensitivity.csv")

    _render_q1(canonical, augmented, output_dir / FIGURE_NAMES[0])
    _render_q2_judgement(output_dir, output_dir / FIGURE_NAMES[1])
    _render_q2_forecast(
        canonical,
        forecasts,
        q1_simple_forecasts,
        output_dir / FIGURE_NAMES[2],
    )
    _render_q3(scenarios, sensitivity, output_dir / FIGURE_NAMES[3])
    return list(FIGURE_NAMES)
