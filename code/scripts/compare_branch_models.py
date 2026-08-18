#!/usr/bin/env python3
"""Run a leakage-controlled benchmark of executable models merged from branches.

The main ranking uses common expanding-origin outer folds that end in 2023.
The 2024 observations are isolated as a one-year pseudo-holdout for this rerun,
while a fixed 2019 cutoff is reported separately as a cross-regime stress test.
The module also exposes its split builders so the data contract can be tested.
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
from typing import Any
from typing import Final

import numpy as np
import pandas as pd

import model_jizhou_tourism as traditional_model
import model_jizhou_tourism_ml as ml_model


ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_UNIFIED_DIR: Final = ROOT / "data/unified"
DEFAULT_CANONICAL: Final = (
    ROOT / "data/jizhou_tourism_economy/official_annual_summary_2010_2025.csv"
)
DEFAULT_OUTPUT: Final = ROOT / "outputs/unified_model_benchmark"
DEFAULT_REPORT: Final = ROOT / "docs/unified_branch_model_comparison.md"
DEFAULT_PROBLEM_CANONICAL: Final = (
    DEFAULT_UNIFIED_DIR / "canonical_official_annual_2010_2025.csv"
)
RANDOM_SEED: Final = 20260817
RIDGE_BOOTSTRAP_REPETITIONS: Final = 10_000
FORECAST_YEARS: Final = tuple(range(2026, 2031))
PRIMARY_SPLIT_ID: Final = "primary_holdout_2024"
TARGETS: Final = ("tourist_visits", "tourism_comprehensive_income")
ROLLING_SPLIT_ID: Final = "rolling_origin_min5_max2023"
STRESS_SPLIT_ID: Final = "stress_cutoff_2019_through_2023"
TRADITIONAL_BRANCH: Final = "codex/jizhou-tourism-modeling"
ML_BRANCH: Final = "codex/jizhou-tourism-ml"
MODEL_SOURCE_SPECS: Final = (
    {
        "branch": TRADITIONAL_BRANCH,
        "path": "code/scripts/model_jizhou_tourism.py",
        "pinned_commit": "66e27eb5a29bbf3abd51dc2dc1af4b8e41fc349c",
        "git_blob_oid": "86ed05d5ccb2086532293d81ba02640aa35dd3de",
        "sha256": "8622be9dee45afdb10458eaa38501053fd3035913e77ab78bb67a2d51fe2cdc2",
    },
    {
        "branch": ML_BRANCH,
        "path": "code/scripts/model_jizhou_tourism_ml.py",
        "pinned_commit": "3709084fc84614223ee00979494aa82b458296fe",
        "git_blob_oid": "81ba267b066c0d60a446e862ae100be0767ce6e9",
        "sha256": "688195945f5e35ec585962484fb30d43884f2a32d008cf4c08b6e354e98c7355",
    },
)
EXPECTED_ROLLING_TEST_YEARS: Final = {
    "tourist_visits": (2015, 2016, 2017, 2018, 2019, 2023),
    "tourism_comprehensive_income": (2015, 2017, 2018, 2019, 2021, 2023),
}
NON_EVIDENCE_STATUS_PATTERN: Final = (
    r"imput|scenario|diagnostic|forecast|proxy|target|secondary_reported|aggregate_constraint"
)
PROBLEM1_INDICATORS: Final = {
    "tourist_visits": {
        "label_cn": "旅游接待人次",
        "value_column": "preferred_visitor_10k_persons",
        "status_column": "visitor_status",
        "unit": "10k_persons",
    },
    "tourism_comprehensive_income": {
        "label_cn": "旅游综合收入",
        "value_column": "preferred_comprehensive_income_100m_cny",
        "status_column": "comprehensive_status",
        "unit": "100m_cny",
    },
    "jizhou_gdp": {
        "label_cn": "地区生产总值",
        "value_column": "preferred_gdp_100m_cny",
        "status_column": "gdp_status",
        "unit": "100m_cny",
    },
    "jizhou_tertiary_value_added": {
        "label_cn": "第三产业增加值",
        "value_column": "preferred_tertiary_100m_cny",
        "status_column": "tertiary_status",
        "unit": "100m_cny",
    },
}
SCENARIO_ANCHOR_YEAR: Final = 2025
SCENARIO_VISITOR_PROXY_10K: Final = 2800.0
SCENARIO_INCOME_PROXY_100M: Final = 231.0


@dataclass(frozen=True)
class BenchmarkInputs:
    observations: pd.DataFrame
    primary_train: pd.DataFrame
    primary_test: pd.DataFrame
    rolling_folds: pd.DataFrame
    stress_train: pd.DataFrame
    stress_test: pd.DataFrame
    source_mode: str
    source_files: tuple[Path, ...]


def _coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    result = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
            "nan": False,
            "none": False,
            "": False,
        }
    )
    if result.isna().any():
        unexpected = sorted(normalized[result.isna()].unique())
        raise ValueError(f"unrecognized boolean values: {unexpected}")
    return result.astype(bool)


def _training_evidence_mask(frame: pd.DataFrame) -> pd.Series:
    status = frame.get("status", pd.Series("", index=frame.index)).fillna("").astype(str)
    return frame["value"].notna() & ~status.str.contains(
        NON_EVIDENCE_STATUS_PATTERN, case=False, regex=True
    )


def build_primary_split(
    observations: pd.DataFrame, cutoff_year: int = 2023
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return physical training and execution-isolated 2024 pseudo-holdout frames."""
    required = {"metric", "year", "value", "is_observed"}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"benchmark observations are missing columns: {missing}")

    frame = observations.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["is_observed"] = _coerce_boolean(frame["is_observed"])

    train = frame[
        (frame["year"] <= cutoff_year)
        & _training_evidence_mask(frame)
    ].copy()
    test = frame[
        frame["year"].eq(cutoff_year + 1)
        & frame["value"].notna()
        & frame["is_observed"].eq(True)
    ].copy()
    for split, name in ((train, "train"), (test, "test")):
        split["split_id"] = PRIMARY_SPLIT_ID
        split["split"] = name
        split["cutoff_year"] = cutoff_year
        split.sort_values(["metric", "year"], inplace=True)
        split.reset_index(drop=True, inplace=True)
    return train, test


def build_rolling_origin_folds(
    observations: pd.DataFrame, min_train_size: int = 5, last_test_year: int = 2023
) -> pd.DataFrame:
    """Materialize common expanding-origin outer folds for both targets."""
    required = {"metric", "year", "value", "is_observed"}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"benchmark observations are missing columns: {missing}")
    records: list[dict[str, object]] = []
    for metric in TARGETS:
        series = observations[
            observations["metric"].eq(metric)
            & _training_evidence_mask(observations)
            & observations["year"].le(last_test_year)
        ].copy()
        series["year"] = pd.to_numeric(series["year"], errors="raise").astype(int)
        series["is_observed"] = _coerce_boolean(series["is_observed"])
        series.sort_values("year", inplace=True)
        series.drop_duplicates(["metric", "year"], inplace=True)
        series.reset_index(drop=True, inplace=True)
        if len(series) <= min_train_size:
            raise ValueError(
                f"{metric} needs more than {min_train_size} observations for rolling validation"
            )
        for test_index in range(min_train_size, len(series)):
            if not bool(series.iloc[test_index]["is_observed"]):
                continue
            test_year = int(series.iloc[test_index]["year"])
            fold_id = f"{metric}_test_{test_year}"
            for row_index in range(test_index + 1):
                source = series.iloc[row_index].to_dict()
                source.update(
                    {
                        "split_id": ROLLING_SPLIT_ID,
                        "fold_id": fold_id,
                        "fold_role": "test" if row_index == test_index else "train",
                        "outer_test_year": test_year,
                        "train_end_year": int(series.iloc[test_index - 1]["year"]),
                        "min_train_records": min_train_size,
                        "outer_test_max_year": last_test_year,
                    }
                )
                records.append(source)
    return pd.DataFrame(records).sort_values(
        ["metric", "outer_test_year", "fold_role", "year"], ignore_index=True
    )


def build_stress_split(
    observations: pd.DataFrame, cutoff_year: int = 2019, last_test_year: int = 2023
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive the fixed-origin cross-regime stress rows without touching 2024."""
    frame = _normalize_long_frame(observations)
    train = frame[
        frame["year"].le(cutoff_year) & _training_evidence_mask(frame)
    ].copy()
    test = frame[
        frame["year"].between(cutoff_year + 1, last_test_year)
        & frame["value"].notna()
        & frame["is_observed"]
    ].copy()
    for result, role in ((train, "train"), (test, "test")):
        result["split_id"] = STRESS_SPLIT_ID
        result["split"] = role
        result["cutoff_year"] = cutoff_year
        result.sort_values(["metric", "year"], inplace=True)
        result.reset_index(drop=True, inplace=True)
    return train, test


def augment_training_rows(
    physical_train: pd.DataFrame,
    *,
    test_year: int,
    scope: str,
    fold_id: str,
    recent_growth_intervals: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill annual training gaps using only official rows visible before one outer test."""
    train = _normalize_long_frame(physical_train).sort_values("year").reset_index(drop=True)
    if train.empty or train["metric"].nunique() != 1:
        raise ValueError("augmentation requires one non-empty target series")
    if train.duplicated("year").any():
        raise ValueError("augmentation received duplicate training years")
    if not train["year"].lt(test_year).all():
        raise ValueError("augmentation cannot see the outer test year or future rows")
    if not (train["value"] > 0.0).all():
        raise ValueError("log-scale augmentation requires strictly positive target values")

    metric = str(train.iloc[0]["metric"])
    known_through_year = int(train["year"].max())
    first_year = int(train["year"].min())
    official_by_year = {int(row["year"]): row for _, row in train.iterrows()}
    official_years = np.array(sorted(official_by_year), dtype=int)
    official_values = np.array(
        [float(official_by_year[year]["value"]) for year in official_years], dtype=float
    )
    interval_growths = np.diff(np.log(official_values)) / np.diff(official_years)
    interval_pairs = list(zip(official_years[:-1], official_years[1:], strict=True))
    recent_count = min(recent_growth_intervals, len(interval_growths))
    if recent_count == 0 and known_through_year < test_year - 1:
        raise ValueError("tail augmentation requires at least two official training rows")
    recent_growths = interval_growths[-recent_count:] if recent_count else np.array([])
    tail_growth = float(np.median(recent_growths)) if recent_count else math.nan
    recent_pairs = interval_pairs[-recent_count:] if recent_count else []
    growth_source_intervals = ";".join(f"{left}-{right}" for left, right in recent_pairs)

    rows: list[dict[str, object]] = []
    for year in range(first_year, test_year):
        if year in official_by_year:
            record = official_by_year[year].to_dict()
            record.update(
                {
                    "is_simulated": False,
                    "method": "physical_canonical_training_value",
                    "source_years": str(year),
                    "boundary_left_year": year,
                    "boundary_left_value": float(record["value"]),
                    "boundary_right_year": year,
                    "boundary_right_value": float(record["value"]),
                    "annualized_log_growth": math.nan,
                    "growth_source_intervals": "",
                }
            )
        elif year < known_through_year:
            left_year = int(official_years[official_years < year].max())
            right_year = int(official_years[official_years > year].min())
            left_value = float(official_by_year[left_year]["value"])
            right_value = float(official_by_year[right_year]["value"])
            fraction = (year - left_year) / (right_year - left_year)
            annualized_growth = (math.log(right_value) - math.log(left_value)) / (
                right_year - left_year
            )
            value = math.exp(math.log(left_value) + fraction * (math.log(right_value) - math.log(left_value)))
            record = {
                "metric": metric,
                "year": year,
                "value": value,
                "unit": str(train.iloc[0]["unit"]),
                "status": "simulated_training_only",
                "source_ids": f"training_interpolation:{left_year};{right_year}",
                "quality_note": "log-linear interpolation using two official outer-training boundaries",
                "is_observed": False,
                "is_simulated": True,
                "method": "log_linear_interpolation",
                "source_years": f"{left_year};{right_year}",
                "boundary_left_year": left_year,
                "boundary_left_value": left_value,
                "boundary_right_year": right_year,
                "boundary_right_value": right_value,
                "annualized_log_growth": annualized_growth,
                "growth_source_intervals": "",
            }
        else:
            last_year = known_through_year
            last_value = float(official_by_year[last_year]["value"])
            value = math.exp(math.log(last_value) + tail_growth * (year - last_year))
            record = {
                "metric": metric,
                "year": year,
                "value": value,
                "unit": str(train.iloc[0]["unit"]),
                "status": "simulated_training_only",
                "source_ids": f"training_tail_growth:{growth_source_intervals}",
                "quality_note": (
                    "tail extrapolation using median of recent official-training annualized log growth"
                ),
                "is_observed": False,
                "is_simulated": True,
                "method": "tail_median_annualized_log_growth",
                "source_years": ";".join(
                    map(str, sorted({item for pair in recent_pairs for item in pair}))
                ),
                "boundary_left_year": last_year,
                "boundary_left_value": last_value,
                "boundary_right_year": math.nan,
                "boundary_right_value": math.nan,
                "annualized_log_growth": tail_growth,
                "growth_source_intervals": growth_source_intervals,
            }
        record.update(
            {
                "evaluation_scope": scope,
                "fold_id": fold_id,
                "test_year": test_year,
                "known_through_year": known_through_year,
            }
        )
        rows.append(record)
    augmented = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    augmented["physical_train_n"] = len(train)
    augmented["augmented_train_n"] = len(augmented)
    audit_columns = [
        "evaluation_scope",
        "fold_id",
        "metric",
        "year",
        "value",
        "unit",
        "status",
        "source_ids",
        "quality_note",
        "is_observed",
        "is_simulated",
        "method",
        "known_through_year",
        "test_year",
        "source_years",
        "boundary_left_year",
        "boundary_left_value",
        "boundary_right_year",
        "boundary_right_value",
        "annualized_log_growth",
        "growth_source_intervals",
        "physical_train_n",
        "augmented_train_n",
    ]
    return augmented, augmented[audit_columns].copy()


def _assert_exact_derivation(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
    identity_columns: list[str],
) -> None:
    evidence_columns = [
        "metric",
        "year",
        "value",
        "unit",
        "status",
        "source_ids",
        "quality_note",
        "is_observed",
    ]
    columns = list(dict.fromkeys([*identity_columns, *evidence_columns]))
    missing_actual = sorted(set(columns) - set(actual.columns))
    missing_expected = sorted(set(columns) - set(expected.columns))
    if missing_actual or missing_expected:
        raise ValueError(
            f"{label} derivation columns missing: actual={missing_actual}, expected={missing_expected}"
        )

    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[columns].copy()
        result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
        result["value"] = pd.to_numeric(result["value"], errors="raise").astype(float)
        result["is_observed"] = _coerce_boolean(result["is_observed"])
        for column in set(columns) - {"year", "value", "is_observed", "cutoff_year"}:
            result[column] = result[column].fillna("").astype(str)
        if "cutoff_year" in result:
            result["cutoff_year"] = pd.to_numeric(
                result["cutoff_year"], errors="raise"
            ).astype(int)
        return result.sort_values(identity_columns).reset_index(drop=True)

    actual_normalized = normalized(actual)
    expected_normalized = normalized(expected)
    try:
        pd.testing.assert_frame_equal(
            actual_normalized,
            expected_normalized,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            f"{label} is not an exact derivation of benchmark_observations.csv: {error}"
        ) from error


def canonical_to_long(canonical: pd.DataFrame) -> pd.DataFrame:
    """Convert the merged official annual wide table to the benchmark contract."""
    definitions = {
        "tourist_visits": (
            "preferred_visitor_10k_persons",
            "visitor_status",
            "10k_persons",
        ),
        "tourism_comprehensive_income": (
            "preferred_comprehensive_income_100m_cny",
            "comprehensive_status",
            "100m_cny",
        ),
    }
    required = {"year", "source_ids", "quality_note"} | {
        column for value_status_unit in definitions.values() for column in value_status_unit[:2]
    }
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise ValueError(f"canonical annual CSV is missing columns: {missing}")
    rows: list[pd.DataFrame] = []
    for metric, (value_column, status_column, unit) in definitions.items():
        frame = canonical[
            ["year", value_column, status_column, "source_ids", "quality_note"]
        ].rename(columns={value_column: "value", status_column: "status"})
        frame.insert(0, "metric", metric)
        frame["unit"] = unit
        status = frame["status"].fillna("").astype(str)
        frame["is_observed"] = frame["value"].notna() & status.str.startswith("observed")
        rows.append(frame)
    return _normalize_long_frame(pd.concat(rows, ignore_index=True))


def _normalize_long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"metric", "year", "value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"long benchmark frame is missing columns: {missing}")
    result = frame.copy()
    result["metric"] = result["metric"].astype(str)
    unexpected_metrics = sorted(set(result["metric"]) - set(TARGETS))
    if unexpected_metrics:
        raise ValueError(f"unexpected benchmark metrics: {unexpected_metrics}")
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    defaults: dict[str, object] = {
        "unit": "",
        "status": "",
        "source_ids": "",
        "quality_note": "",
        "is_observed": False,
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
    result["is_observed"] = _coerce_boolean(result["is_observed"])
    for column in ("unit", "status", "source_ids", "quality_note"):
        result[column] = result[column].fillna("").astype(str)
    return result


def _validate_primary_files(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if train.empty or test.empty:
        raise ValueError("primary train and 2024 holdout must both be non-empty")
    if not train["year"].le(2023).all():
        raise ValueError("primary_train.csv contains a year after 2023")
    if not _training_evidence_mask(train).all():
        raise ValueError("primary_train.csv contains a non-evidence/imputed target value")
    if not test["year"].eq(2024).all():
        raise ValueError("primary_test.csv must contain only 2024")
    if not (test["value"].notna() & test["is_observed"]).all():
        raise ValueError("primary_test.csv must contain observed, non-missing actuals only")
    if set(test["metric"]) != set(TARGETS):
        raise ValueError("primary_test.csv must contain one 2024 actual for each target")
    for name, frame in (("train", train), ("test", test)):
        duplicated = frame.duplicated(["metric", "year"], keep=False)
        if duplicated.any():
            raise ValueError(f"primary {name} has duplicate metric/year rows")


def _normalize_rolling_folds(frame: pd.DataFrame) -> pd.DataFrame:
    result = _normalize_long_frame(frame)
    if "fold_role" not in result and "role" in result:
        result = result.rename(columns={"role": "fold_role"})
    required = {"fold_id", "fold_role"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"rolling_origin_folds.csv is missing columns: {missing}")
    result["fold_id"] = result["fold_id"].astype(str)
    result["fold_role"] = result["fold_role"].astype(str).str.lower()
    if not set(result["fold_role"]).issubset({"train", "test"}):
        raise ValueError("rolling fold role must be train or test")

    result["outer_test_year"] = result.groupby(["metric", "fold_id"])["year"].transform(
        lambda years: int(
            result.loc[years.index]
            .loc[result.loc[years.index, "fold_role"].eq("test"), "year"]
            .iloc[0]
        )
    )
    result["train_end_year"] = result.groupby(["metric", "fold_id"])["year"].transform(
        lambda years: int(
            result.loc[years.index]
            .loc[result.loc[years.index, "fold_role"].eq("train"), "year"]
            .max()
        )
    )
    for (metric, fold_id), group in result.groupby(["metric", "fold_id"], sort=False):
        train = group[group["fold_role"].eq("train")]
        test = group[group["fold_role"].eq("test")]
        if len(test) != 1:
            raise ValueError(f"{fold_id} must have exactly one outer test row")
        if len(train) < 5:
            raise ValueError(f"{fold_id} has fewer than five training records")
        test_year = int(test.iloc[0]["year"])
        if test_year > 2023:
            raise ValueError(f"{fold_id} leaks the 2024 final holdout")
        if not train["year"].lt(test_year).all():
            raise ValueError(f"{fold_id} has non-past rows in outer training")
        if not bool(test.iloc[0]["is_observed"]):
            raise ValueError(f"{fold_id} outer test is not an observed actual")
        if not _training_evidence_mask(train).all():
            raise ValueError(f"{fold_id} outer training contains imputed/scenario values")
        if str(metric) != str(test.iloc[0]["metric"]):
            raise ValueError(f"{fold_id} mixes target metrics")

    actual_years = {
        metric: tuple(
            sorted(
                result[
                    result["metric"].eq(metric) & result["fold_role"].eq("test")
                ]["year"].astype(int)
            )
        )
        for metric in TARGETS
    }
    if actual_years != EXPECTED_ROLLING_TEST_YEARS:
        raise ValueError(
            f"rolling outer-test years do not match the frozen contract: {actual_years}"
        )
    return result.sort_values(
        ["metric", "outer_test_year", "fold_role", "year"], ignore_index=True
    )


def load_benchmark_inputs(
    unified_dir: Path = DEFAULT_UNIFIED_DIR,
    canonical_path: Path = DEFAULT_CANONICAL,
) -> BenchmarkInputs:
    """Load physical splits/folds when present, otherwise derive them from canonical CSV."""
    observations_path = unified_dir / "benchmark_observations.csv"
    train_path = unified_dir / "primary_train.csv"
    test_path = unified_dir / "primary_test.csv"
    rolling_path = unified_dir / "rolling_origin_folds.csv"
    stress_train_path = unified_dir / "stress_train.csv"
    stress_test_path = unified_dir / "stress_test.csv"

    source_files: list[Path] = []
    if observations_path.exists():
        observations = _normalize_long_frame(pd.read_csv(observations_path))
        source_files.append(observations_path)
        observations_source = "data/unified/benchmark_observations.csv"
    else:
        observations = canonical_to_long(pd.read_csv(canonical_path))
        source_files.append(canonical_path)
        observations_source = "canonical_fallback"
    if observations.duplicated(["metric", "year"], keep=False).any():
        raise ValueError("benchmark_observations.csv has duplicate metric/year rows")
    if not _training_evidence_mask(observations).all():
        raise ValueError(
            "benchmark_observations.csv must contain canonical non-missing target evidence only"
        )

    expected_primary_train, expected_primary_test = build_primary_split(observations)

    if train_path.exists() and test_path.exists():
        primary_train = _normalize_long_frame(pd.read_csv(train_path))
        primary_test = _normalize_long_frame(pd.read_csv(test_path))
        source_files.extend([train_path, test_path])
        split_source = "physical_unified_files"
    elif train_path.exists() != test_path.exists():
        raise ValueError("primary_train.csv and primary_test.csv must exist together")
    else:
        primary_train, primary_test = expected_primary_train, expected_primary_test
        split_source = "derived_from_observations"
    _validate_primary_files(primary_train, primary_test)
    _assert_exact_derivation(
        primary_train,
        expected_primary_train,
        label="primary_train.csv",
        identity_columns=["split_id", "split", "cutoff_year", "metric", "year"],
    )
    _assert_exact_derivation(
        primary_test,
        expected_primary_test,
        label="primary_test.csv",
        identity_columns=["split_id", "split", "cutoff_year", "metric", "year"],
    )

    expected_rolling = _normalize_rolling_folds(
        build_rolling_origin_folds(observations, min_train_size=5, last_test_year=2023)
    )

    if rolling_path.exists():
        rolling_folds = _normalize_rolling_folds(pd.read_csv(rolling_path))
        source_files.append(rolling_path)
        rolling_source = "physical_unified_file"
    else:
        rolling_folds = expected_rolling
        rolling_source = "derived_from_observations"
    _assert_exact_derivation(
        rolling_folds,
        expected_rolling,
        label="rolling_origin_folds.csv",
        identity_columns=["split_id", "fold_id", "fold_role", "metric", "year"],
    )

    expected_stress_train, expected_stress_test = build_stress_split(observations)

    if stress_train_path.exists() and stress_test_path.exists():
        stress_train = _normalize_long_frame(pd.read_csv(stress_train_path))
        stress_test = _normalize_long_frame(pd.read_csv(stress_test_path))
        source_files.extend([stress_train_path, stress_test_path])
        stress_source = "physical_unified_files"
    elif stress_train_path.exists() != stress_test_path.exists():
        raise ValueError("stress_train.csv and stress_test.csv must exist together")
    else:
        stress_train, stress_test = expected_stress_train, expected_stress_test
        stress_source = "derived_from_observations"
    if not stress_train["year"].le(2019).all():
        raise ValueError("2019 stress training includes later data")
    if not stress_test["year"].between(2020, 2023).all():
        raise ValueError("2019 stress test must be 2020-2023 and exclude 2024")
    if not (stress_test["value"].notna() & stress_test["is_observed"]).all():
        raise ValueError("2019 stress test must contain observed actuals only")
    _assert_exact_derivation(
        stress_train,
        expected_stress_train,
        label="stress_train.csv",
        identity_columns=["split_id", "split", "cutoff_year", "metric", "year"],
    )
    _assert_exact_derivation(
        stress_test,
        expected_stress_test,
        label="stress_test.csv",
        identity_columns=["split_id", "split", "cutoff_year", "metric", "year"],
    )

    return BenchmarkInputs(
        observations=observations,
        primary_train=primary_train.sort_values(["metric", "year"], ignore_index=True),
        primary_test=primary_test.sort_values(["metric", "year"], ignore_index=True),
        rolling_folds=rolling_folds,
        stress_train=stress_train.sort_values(["metric", "year"], ignore_index=True),
        stress_test=stress_test.sort_values(["metric", "year"], ignore_index=True),
        source_mode=(
            f"observations={observations_source};primary={split_source};rolling={rolling_source};"
            f"stress={stress_source}"
        ),
        source_files=tuple(dict.fromkeys(path.resolve() for path in source_files)),
    )


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute target metrics and an equal-target-weighted sMAPE ranking."""
    required = {"branch", "model", "metric", "year", "actual", "prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction table is missing columns: {missing}")
    if predictions.empty:
        columns = [
            "branch",
            "model",
            "metric",
            "n_test",
            "test_years",
            "mae",
            "rmse",
            "mape_percent",
            "smape_percent",
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (branch, model, metric), group in predictions.groupby(
        ["branch", "model", "metric"], sort=True
    ):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["prediction"].to_numpy(dtype=float)
        error = forecast - actual
        mape_terms = np.divide(
            np.abs(error),
            np.abs(actual),
            out=np.full_like(actual, np.nan),
            where=np.abs(actual) > 0.0,
        )
        denominator = np.abs(actual) + np.abs(forecast)
        smape_terms = np.divide(
            2.0 * np.abs(error),
            denominator,
            out=np.zeros_like(actual),
            where=denominator > 0.0,
        )
        positive = (actual > 0.0) & (forecast > 0.0)
        absolute_log_errors = np.full_like(actual, np.nan)
        absolute_log_errors[positive] = np.abs(
            np.log(actual[positive]) - np.log(forecast[positive])
        )
        rows.append(
            {
                "branch": branch,
                "model": model,
                "metric": metric,
                "n_test": len(group),
                "test_years": ";".join(map(str, sorted(group["year"].astype(int).unique()))),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(math.sqrt(float(np.mean(np.square(error))))),
                "mape_percent": float(np.nanmean(mape_terms) * 100.0),
                "smape_percent": float(np.mean(smape_terms) * 100.0),
                "point_smape_std_percent": float(np.std(smape_terms * 100.0)),
                "worst_absolute_percentage_error_percent": float(
                    np.nanmax(mape_terms) * 100.0
                ),
                "worst_point_smape_percent": float(np.max(smape_terms) * 100.0),
                "max_absolute_log_error": float(np.nanmax(absolute_log_errors)),
            }
        )
    by_target = pd.DataFrame(rows).sort_values(
        ["metric", "smape_percent", "branch", "model"], ignore_index=True
    )
    expected_counts = by_target.groupby("metric")["n_test"].max().to_dict()
    by_target["expected_test_points"] = by_target["metric"].map(expected_counts).astype(int)
    by_target["coverage_status"] = np.where(
        by_target["n_test"].eq(by_target["expected_test_points"]), "complete", "partial"
    )
    naive_by_target = (
        by_target[by_target["model"].eq("naive_last")]
        .groupby("metric")["smape_percent"]
        .mean()
        .to_dict()
    )
    by_target["naive_smape_percent"] = by_target["metric"].map(naive_by_target)
    by_target["smape_skill_vs_naive"] = 1.0 - (
        by_target["smape_percent"] / by_target["naive_smape_percent"]
    )

    macro_rows: list[dict[str, object]] = []
    for (branch, model), group in by_target.groupby(["branch", "model"], sort=True):
        complete = set(group["metric"]) == set(TARGETS) and group["coverage_status"].eq(
            "complete"
        ).all()
        if not complete:
            continue
        worst_target_row = group.loc[group["smape_percent"].idxmax()]
        model_points = predictions[
            predictions["branch"].eq(branch) & predictions["model"].eq(model)
        ]
        point_denominator = model_points["actual"].abs() + model_points["prediction"].abs()
        point_smape = np.divide(
            2.0 * (model_points["actual"] - model_points["prediction"]).abs(),
            point_denominator,
            out=np.zeros(len(model_points), dtype=float),
            where=point_denominator.to_numpy(dtype=float) > 0.0,
        )
        macro_smape = float(group["smape_percent"].mean())
        macro_naive = (
            float(np.mean([naive_by_target[target] for target in TARGETS]))
            if all(target in naive_by_target for target in TARGETS)
            else math.nan
        )
        macro_rows.append(
            {
                "branch": branch,
                "model": model,
                "macro_smape_percent": macro_smape,
                "macro_naive_smape_percent": macro_naive,
                "macro_smape_skill_vs_naive": 1.0 - macro_smape / macro_naive
                if math.isfinite(macro_naive) and macro_naive != 0.0
                else math.nan,
                "worst_target": worst_target_row["metric"],
                "worst_target_smape_percent": float(worst_target_row["smape_percent"]),
                "worst_point_smape_percent": float(np.max(point_smape) * 100.0),
                "worst_absolute_percentage_error_percent": float(
                    group["worst_absolute_percentage_error_percent"].max()
                ),
                "mean_target_smape_std_percent": float(
                    group["point_smape_std_percent"].mean()
                ),
                "beats_naive_all_targets": bool(
                    group["smape_skill_vs_naive"].notna().all()
                    and group["smape_skill_vs_naive"].gt(0.0).all()
                ),
                "target_count": len(group),
                "total_test_points": int(group["n_test"].sum()),
                "complete_target_coverage": True,
            }
        )
    macro = pd.DataFrame(macro_rows)
    if macro.empty:
        return by_target, macro
    macro.sort_values(
        ["macro_smape_percent", "branch", "model"], inplace=True, ignore_index=True
    )
    rounded_score = macro["macro_smape_percent"].round(10)
    macro["rank"] = rounded_score.rank(method="min").astype(int)
    macro["tie_group_size"] = rounded_score.groupby(rounded_score).transform("size").astype(int)
    macro["stability_flag"] = np.where(macro["tie_group_size"].gt(1), "tied", "unique")
    return by_target, macro


def assess_level_break(years: np.ndarray | list[int]) -> dict[str, object]:
    """Report whether an intercept/trend/post-2022 OLS design is identifiable."""
    year_array = np.asarray(years, dtype=int)
    if not np.any(year_array >= 2023):
        return {
            "status": "not_applicable",
            "reason": "no post-2022 training observation; the level-break coefficient is not identifiable",
        }
    time = year_array.astype(float) - 2010.0
    design = np.column_stack(
        [np.ones(len(year_array)), time, (year_array >= 2023).astype(float)]
    )
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        return {
            "status": "not_applicable",
            "reason": f"rank-deficient level-break design (rank={rank}, columns={design.shape[1]})",
        }
    return {"status": "applicable", "reason": "full-rank level-break design"}


def _regime(year: int) -> str:
    if year <= 2019:
        return "pre_covid"
    if year <= 2022:
        return "pandemic"
    return "recovery"


def _traditional_effective_train(train: pd.DataFrame, model: str) -> pd.DataFrame:
    if model == "no_break_log_linear_common_rows":
        return train.copy()
    if model == "pre_covid_exponential":
        return train[train["year"].le(2019)].copy()
    if model in {
        "no_break_log_linear",
        "post_2022_level_break",
        "strict_evidence_level_break",
    }:
        return train[~train["year"].between(2020, 2022)].copy()
    raise KeyError(model)


def _point_prediction_rows(
    *,
    scope: str,
    split_id: str,
    fold_id: str,
    branch: str,
    model: str,
    model_group: str,
    role: str,
    physical_train: pd.DataFrame,
    effective_train: pd.DataFrame,
    training_input: pd.DataFrame | None = None,
    test: pd.DataFrame,
    predictions: np.ndarray,
    parameters: dict[str, Any],
    data_policy: str,
    feature_support: str = "not_applicable",
) -> list[dict[str, object]]:
    if len(test) != len(predictions):
        raise ValueError(f"prediction length mismatch for {model}")
    if not np.isfinite(predictions).all():
        raise ValueError(f"{model} produced a non-finite prediction")
    rows: list[dict[str, object]] = []
    input_frame = physical_train if training_input is None else training_input
    simulated_mask = (
        _coerce_boolean(input_frame["is_simulated"])
        if "is_simulated" in input_frame
        else pd.Series(False, index=input_frame.index)
    )
    physical_observed_mask = (
        _coerce_boolean(physical_train["is_observed"])
        if "is_observed" in physical_train
        else pd.Series(False, index=physical_train.index)
    )
    effective_years = set(effective_train["year"].astype(int))
    input_years = set(input_frame["year"].astype(int))
    effective_end = int(effective_train["year"].max())
    for (_, actual_row), prediction in zip(test.iterrows(), predictions, strict=True):
        actual = float(actual_row["value"])
        prediction_value = float(prediction)
        denominator = abs(actual) + abs(prediction_value)
        rows.append(
            {
                "evaluation_scope": scope,
                "split_id": split_id,
                "fold_id": fold_id,
                "branch": branch,
                "model": model,
                "model_group": model_group,
                "comparison_role": role,
                "metric": str(actual_row["metric"]),
                "unit": str(actual_row.get("unit", "")),
                "physical_train_n": len(physical_train),
                "observed_train_n": int(physical_observed_mask.sum()),
                "physical_canonical_train_n": int((~simulated_mask).sum()),
                "simulated_train_n": int(simulated_mask.sum()),
                "augmented_train_n": len(input_frame),
                "effective_train_n": len(effective_train),
                "uses_all_augmented_training_rows": effective_years == input_years
                and len(effective_train) == len(input_frame),
                "train_start_year": int(effective_train["year"].min()),
                "train_end_year": effective_end,
                "test_year": int(actual_row["year"]),
                "calendar_horizon_years": int(actual_row["year"]) - effective_end,
                "test_regime": _regime(int(actual_row["year"])),
                "actual": actual,
                "prediction": prediction_value,
                "error": prediction_value - actual,
                "absolute_error": abs(prediction_value - actual),
                "absolute_percentage_error_percent": abs(prediction_value - actual)
                / abs(actual)
                * 100.0,
                "point_smape_percent": 200.0
                * abs(prediction_value - actual)
                / denominator
                if denominator
                else 0.0,
                "absolute_log_error": abs(math.log(actual) - math.log(prediction_value))
                if actual > 0.0 and prediction_value > 0.0
                else math.nan,
                "selected_parameters": json.dumps(parameters, sort_keys=True),
                "data_policy": data_policy,
                "feature_support": feature_support,
                "actual_status": str(actual_row.get("status", "")),
                "actual_source_ids": str(actual_row.get("source_ids", "")),
            }
        )
    return rows


def _applicability_row(
    *,
    scope: str,
    split_id: str,
    fold_id: str,
    branch: str,
    model: str,
    metric: str,
    status: str,
    reason: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, object]:
    return {
        "evaluation_scope": scope,
        "split_id": split_id,
        "fold_id": fold_id,
        "branch": branch,
        "model": model,
        "metric": metric,
        "status": status,
        "reason": reason,
        "physical_train_n": len(train),
        "test_n": len(test),
        "physical_train_end_year": int(train["year"].max()),
        "test_years": ";".join(map(str, sorted(test["year"].astype(int).unique()))),
    }


def _strict_evidence_train(train: pd.DataFrame, metric: str) -> pd.DataFrame:
    excluded = (
        {"observed_cached"}
        if metric == "tourist_visits"
        else {"inferred_from_yoy", "observed_cached", "observed_supporting_attachment"}
    )
    return train[~train["status"].isin(excluded)].copy()


def _evaluate_traditional_models(
    train_input: pd.DataFrame,
    test: pd.DataFrame,
    *,
    physical_train: pd.DataFrame,
    scope: str,
    split_id: str,
    fold_id: str,
    model_names: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric = str(test.iloc[0]["metric"])
    if (
        set(train_input["metric"]) != {metric}
        or set(physical_train["metric"]) != {metric}
        or set(test["metric"]) != {metric}
    ):
        raise ValueError("traditional evaluation frames must contain one matching metric")
    predictions: list[dict[str, object]] = []
    applicability: list[dict[str, object]] = []
    roles = {
        "pre_covid_exponential": "native_policy_sensitivity",
        "no_break_log_linear": "native_policy_sensitivity",
        "no_break_log_linear_common_rows": "user_protocol_common_row_adapter",
    }
    for model_name in model_names:
        source_train = (
            train_input
            if model_name == "no_break_log_linear_common_rows"
            else physical_train
        )
        effective = _traditional_effective_train(source_train, model_name)
        years = effective["year"].to_numpy(dtype=int)
        values = effective["value"].to_numpy(dtype=float)
        fit = traditional_model.fit_log_ols(
            years,
            values,
            metric=metric,
            model_name=model_name,
            interrupted=False,
        )
        design, _ = traditional_model.design_matrix(
            test["year"].to_numpy(dtype=int), interrupted=False
        )
        if model_name == "pre_covid_exponential":
            data_policy = "native sensitivity: official pre-COVID observations only"
        elif model_name == "no_break_log_linear":
            data_policy = "native sensitivity: official structural rows; excludes 2020-2022"
        elif "is_simulated" in train_input and _coerce_boolean(
            train_input["is_simulated"]
        ).any():
            data_policy = "user protocol: all common official+simulated annual training rows; no breakpoint"
        else:
            data_policy = "no-simulation sensitivity: all common physical canonical rows; no breakpoint"
        predictions.extend(
            _point_prediction_rows(
                scope=scope,
                split_id=split_id,
                fold_id=fold_id,
                branch=TRADITIONAL_BRANCH,
                model=model_name,
                model_group="traditional_log_ols",
                role=roles[model_name],
                physical_train=physical_train,
                effective_train=effective,
                training_input=train_input,
                test=test,
                predictions=np.exp(design @ fit.beta),
                parameters={
                    "target_transform": "log",
                    "columns": fit.columns,
                    "interrupted": False,
                },
                data_policy=data_policy,
            )
        )
        applicability.append(
            _applicability_row(
                scope=scope,
                split_id=split_id,
                fold_id=fold_id,
                branch=TRADITIONAL_BRANCH,
                model=model_name,
                metric=metric,
                status="applicable",
                reason=data_policy,
                train=physical_train,
                test=test,
            )
        )

    structural = _traditional_effective_train(physical_train, "post_2022_level_break")
    break_inputs = {
        "post_2022_level_break": structural,
        "strict_evidence_level_break": _strict_evidence_train(structural, metric),
    }
    for break_model, break_train in break_inputs.items():
        assessment = assess_level_break(break_train["year"].to_numpy(dtype=int))
        support_note = str(assessment["reason"])
        if assessment["status"] == "applicable":
            post_count = int(break_train["year"].ge(2023).sum())
            support_note += f"; post-2022 training points={post_count} (weak support)"
        applicability.append(
            _applicability_row(
                scope=scope,
                split_id=split_id,
                fold_id=fold_id,
                branch=TRADITIONAL_BRANCH,
                model=break_model,
                metric=metric,
                status="not_executed_user_protocol",
                reason=(
                    "user-directed unified protocol forbids all breakpoint models; "
                    f"mathematical diagnostic only: {support_note}; no prediction or error generated"
                ),
                train=physical_train,
                test=test,
            )
        )
    return predictions, applicability


def _ml_feature_support(train_years: np.ndarray, test_years: np.ndarray) -> str:
    train_features = ml_model.make_features(train_years)
    test_features = ml_model.make_features(test_years)
    unseen: list[str] = []
    for index, name in enumerate(ml_model.FEATURE_NAMES[1:], start=1):
        if np.ptp(train_features[:, index]) == 0.0 and np.any(
            test_features[:, index] != train_features[0, index]
        ):
            unseen.append(name)
    return "within_training_feature_support" if not unseen else "unseen:" + ";".join(unseen)


def _evaluate_ml_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    physical_train: pd.DataFrame,
    scope: str,
    split_id: str,
    fold_id: str,
    roster: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    metric = str(test.iloc[0]["metric"])
    years = train["year"].to_numpy(dtype=int)
    values = train["value"].to_numpy(dtype=float)
    test_years = test["year"].to_numpy(dtype=int)
    logs = np.log(values)
    x_train = ml_model.make_features(years)
    x_test = ml_model.make_features(test_years)
    feature_support = _ml_feature_support(years, test_years)
    predictions_by_model: dict[str, np.ndarray] = {
        "naive_last": np.full(len(test), values[-1], dtype=float)
    }
    parameters_by_model: dict[str, dict[str, Any]] = {"naive_last": {}}
    inner_scores: dict[str, float] = {}
    tuning_rows: list[dict[str, object]] = []
    if roster not in {"full", "fixed_representatives"}:
        raise ValueError(f"unknown ML roster: {roster}")
    simulated_count = (
        int(_coerce_boolean(train["is_simulated"]).sum())
        if "is_simulated" in train
        else 0
    )

    if roster == "full":
        for name, spec in ml_model.MODEL_SPECS.items():
            parameters, inner_score, inner_folds = ml_model.tune_model(
                spec, years, logs, min_train_size=4
            )
            fitted = ml_model.fit_quietly(spec.builder(parameters), x_train, logs)
            predictions_by_model[name] = np.exp(fitted.predict(x_test))
            parameters_by_model[name] = parameters
            inner_scores[name] = float(inner_score)
            tuning_rows.append(
                {
                    "evaluation_scope": scope,
                    "split_id": split_id,
                    "fold_id": fold_id,
                    "metric": metric,
                    "outer_test_years": ";".join(map(str, test_years)),
                    "outer_train_end_year": int(years.max()),
                    "physical_train_n": len(physical_train),
                    "augmented_train_n": len(train),
                    "simulated_train_n": simulated_count,
                    "model": name,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "selection_method": (
                        "rolling-origin inner log-MAE inside outer-training rows; exploratory only"
                    ),
                    "inner_log_mae": float(inner_score),
                    "inner_folds": int(inner_folds),
                    "max_inner_validation_year": int(years[-1]),
                    "random_seed": RANDOM_SEED,
                }
            )

        ensemble_members = list(ml_model.ENSEMBLE_MEMBERS)
        predictions_by_model["robust_ml_ensemble"] = np.median(
            np.column_stack([predictions_by_model[name] for name in ensemble_members]), axis=1
        )
        parameters_by_model["robust_ml_ensemble"] = {"members": ensemble_members}

    raw_alpha = 0.1
    raw_model = ml_model._ridge_builder({"alpha": raw_alpha})
    raw_model = ml_model.fit_quietly(raw_model, x_train, values)
    predictions_by_model["raw_target_ridge_alpha_0.1"] = np.maximum(
        0.0, raw_model.predict(x_test)
    )
    parameters_by_model["raw_target_ridge_alpha_0.1"] = {
        "alpha": raw_alpha,
        "target_transform": "raw_additive",
    }

    if roster == "full":
        selector_candidates = sorted(
            name
            for name, spec in ml_model.MODEL_SPECS.items()
            if spec.eligible_for_selection and math.isfinite(inner_scores[name])
        )
        if not selector_candidates:
            raise RuntimeError("ML inner selector has no finite train-only candidate score")
        selected_name = min(selector_candidates, key=lambda name: (inner_scores[name], name))
        predictions_by_model["ml_inner_selector"] = predictions_by_model[selected_name]
        parameters_by_model["ml_inner_selector"] = {
            "selected_model": selected_name,
            "selected_parameters": parameters_by_model[selected_name],
            "selection_score_inner_log_mae": inner_scores[selected_name],
            "eligible_models": selector_candidates,
        }
        tuning_rows.append(
            {
                "evaluation_scope": scope,
                "split_id": split_id,
                "fold_id": fold_id,
                "metric": metric,
                "outer_test_years": ";".join(map(str, test_years)),
                "outer_train_end_year": int(years.max()),
                "physical_train_n": len(physical_train),
                "augmented_train_n": len(train),
                "simulated_train_n": simulated_count,
                "model": "ml_inner_selector",
                "parameters": json.dumps(
                    parameters_by_model["ml_inner_selector"], sort_keys=True
                ),
                "selection_method": (
                    "benchmark-created inner selector; exploratory adapter only"
                ),
                "inner_log_mae": inner_scores[selected_name],
                "inner_folds": next(
                    int(row["inner_folds"])
                    for row in tuning_rows
                    if row["model"] == selected_name
                ),
                "max_inner_validation_year": int(years[-1]),
                "random_seed": RANDOM_SEED,
            }
        )

    roles = {
        "naive_last": "baseline",
        "raw_target_ridge_alpha_0.1": "user_protocol_fixed_ml_adapter",
        "ml_inner_selector": "benchmark_created_exploratory_adapter",
        "robust_ml_ensemble": "exploratory_candidate",
    }
    prediction_rows: list[dict[str, object]] = []
    applicability_rows: list[dict[str, object]] = []
    model_order = (
        [
            "naive_last",
            *ml_model.MODEL_SPECS.keys(),
            "robust_ml_ensemble",
            "raw_target_ridge_alpha_0.1",
            "ml_inner_selector",
        ]
        if roster == "full"
        else ["naive_last", "raw_target_ridge_alpha_0.1"]
    )
    for name in model_order:
        role = roles.get(name, "exploratory_candidate")
        model_support = (
            "not_feature_based" if name == "naive_last" else feature_support
        )
        prediction_rows.extend(
            _point_prediction_rows(
                scope=scope,
                split_id=split_id,
                fold_id=fold_id,
                branch=ML_BRANCH,
                model=name,
                model_group="ml_branch",
                role=role,
                physical_train=physical_train,
                effective_train=train,
                training_input=train,
                test=test,
                predictions=predictions_by_model[name],
                parameters=parameters_by_model[name],
                data_policy=(
                    "user protocol: common official+simulated annual training rows"
                    if simulated_count
                    else "no-simulation common physical canonical training rows"
                ),
                feature_support=model_support,
            )
        )
        applicability_rows.append(
            _applicability_row(
                scope=scope,
                split_id=split_id,
                fold_id=fold_id,
                branch=ML_BRANCH,
                model=name,
                metric=metric,
                status="applicable"
                if not model_support.startswith("unseen:")
                else "partial_out_of_support",
                reason=model_support,
                train=physical_train,
                test=test,
            )
        )
    return prediction_rows, applicability_rows, tuning_rows


def evaluate_one_split(
    train_input: pd.DataFrame,
    test: pd.DataFrame,
    *,
    physical_train: pd.DataFrame | None = None,
    scope: str,
    split_id: str,
    fold_id: str,
    traditional_model_names: tuple[str, ...] = ("no_break_log_linear_common_rows",),
    ml_roster: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete executable roster for one target/split without test leakage."""
    if set(train_input["metric"]) != set(test["metric"]) or len(set(test["metric"])) != 1:
        raise ValueError("one-split evaluation requires one matching target")
    train_input = train_input.sort_values("year").reset_index(drop=True)
    physical_train = (
        train_input.copy()
        if physical_train is None
        else physical_train.sort_values("year").reset_index(drop=True)
    )
    test = test.sort_values("year").reset_index(drop=True)
    if not train_input["year"].lt(test["year"].min()).all() or not physical_train[
        "year"
    ].lt(test["year"].min()).all():
        raise ValueError("outer training must be strictly earlier than outer test")
    traditional_rows, traditional_applicability = _evaluate_traditional_models(
        train_input,
        test,
        physical_train=physical_train,
        scope=scope,
        split_id=split_id,
        fold_id=fold_id,
        model_names=traditional_model_names,
    )
    ml_rows, ml_applicability, tuning = _evaluate_ml_models(
        train_input,
        test,
        physical_train=physical_train,
        scope=scope,
        split_id=split_id,
        fold_id=fold_id,
        roster=ml_roster,
    )
    return (
        pd.DataFrame([*traditional_rows, *ml_rows]),
        pd.DataFrame([*traditional_applicability, *ml_applicability]),
        pd.DataFrame(tuning),
    )


def _summarize_scope(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_input = predictions.rename(columns={"test_year": "year"})
    by_target, macro = summarize_predictions(summary_input)
    roles = predictions[["branch", "model", "comparison_role"]].drop_duplicates()
    by_target = by_target.merge(roles, on=["branch", "model"], how="left")
    if not macro.empty:
        macro = macro.merge(roles, on=["branch", "model"], how="left")
    return by_target, macro


def _evaluate_rolling_track(
    folds: pd.DataFrame,
    *,
    scope: str,
    simulate_missing_years: bool,
    include_native_sensitivities: bool,
    ml_roster: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one rolling track while keeping the physical outer folds frozen."""
    predictions: list[pd.DataFrame] = []
    applicability: list[pd.DataFrame] = []
    tuning: list[pd.DataFrame] = []
    simulated_points: list[pd.DataFrame] = []
    traditional_models = (
        (
            "pre_covid_exponential",
            "no_break_log_linear",
            "no_break_log_linear_common_rows",
        )
        if include_native_sensitivities
        else ("no_break_log_linear_common_rows",)
    )
    for (_, fold_id), group in folds.groupby(["metric", "fold_id"], sort=True):
        physical_train = group[group["fold_role"].eq("train")].copy()
        test = group[group["fold_role"].eq("test")].copy()
        test_year = int(test.iloc[0]["year"])
        if simulate_missing_years:
            train_input, audit = augment_training_rows(
                physical_train,
                test_year=test_year,
                scope=scope,
                fold_id=str(fold_id),
            )
            simulated_points.append(audit)
        else:
            train_input = physical_train.copy()
        result = evaluate_one_split(
            train_input,
            test,
            physical_train=physical_train,
            scope=scope,
            split_id=(
                str(group["split_id"].iloc[0])
                if "split_id" in group
                else ROLLING_SPLIT_ID
            ),
            fold_id=str(fold_id),
            traditional_model_names=traditional_models,
            ml_roster=ml_roster,
        )
        predictions.append(result[0])
        applicability.append(result[1])
        if not result[2].empty:
            tuning.append(result[2])
    result_predictions = pd.concat(predictions, ignore_index=True)
    if result_predictions["test_year"].gt(2023).any():
        raise AssertionError("2024 leaked into rolling stability predictions")
    return (
        result_predictions,
        pd.concat(applicability, ignore_index=True),
        pd.concat(tuning, ignore_index=True) if tuning else pd.DataFrame(),
        pd.concat(simulated_points, ignore_index=True)
        if simulated_points
        else pd.DataFrame(),
    )


def _evaluate_augmented_fixed_scope(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    scope: str,
    split_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate each fixed-scope actual independently after train-only augmentation."""
    predictions: list[pd.DataFrame] = []
    applicability: list[pd.DataFrame] = []
    tuning: list[pd.DataFrame] = []
    simulated_points: list[pd.DataFrame] = []
    for metric in TARGETS:
        metric_train = train[train["metric"].eq(metric)].copy()
        metric_tests = test[test["metric"].eq(metric)].sort_values("year")
        for _, test_row in metric_tests.iterrows():
            one_test = test_row.to_frame().T
            one_test["year"] = pd.to_numeric(one_test["year"]).astype(int)
            one_test["value"] = pd.to_numeric(one_test["value"]).astype(float)
            test_year = int(test_row["year"])
            visible_train = metric_train[metric_train["year"].lt(test_year)].copy()
            fold_id = f"{scope}__{metric}__test_{test_year}"
            train_input, audit = augment_training_rows(
                visible_train,
                test_year=test_year,
                scope=scope,
                fold_id=fold_id,
            )
            result = evaluate_one_split(
                train_input,
                one_test,
                physical_train=visible_train,
                scope=scope,
                split_id=split_id,
                fold_id=fold_id,
                traditional_model_names=("no_break_log_linear_common_rows",),
                ml_roster="full",
            )
            predictions.append(result[0])
            applicability.append(result[1])
            if not result[2].empty:
                tuning.append(result[2])
            simulated_points.append(audit)
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(applicability, ignore_index=True),
        pd.concat(tuning, ignore_index=True) if tuning else pd.DataFrame(),
        pd.concat(simulated_points, ignore_index=True),
    )


USER_PROTOCOL_ADAPTERS: Final = {
    (TRADITIONAL_BRANCH, "no_break_log_linear_common_rows"),
    (ML_BRANCH, "raw_target_ridge_alpha_0.1"),
}


def _exploratory_ranking(macro: pd.DataFrame, ranking_scope: str) -> pd.DataFrame:
    exploratory = macro.copy().rename(columns={"rank": "exploratory_rank"})
    exploratory.insert(0, "ranking_scope", ranking_scope)
    return exploratory


def _adapter_comparison_table(
    macro: pd.DataFrame,
    *,
    ranking_scope: str,
    training_track: str,
) -> pd.DataFrame:
    """Re-rank only the two fixed adapters without inheriting exploratory metadata."""
    mask = pd.Series(
        [
            (str(branch), str(model)) in USER_PROTOCOL_ADAPTERS
            for branch, model in zip(macro["branch"], macro["model"], strict=True)
        ],
        index=macro.index,
    )
    comparison = macro.loc[mask].drop(
        columns=["rank", "tie_group_size", "stability_flag"], errors="ignore"
    ).copy()
    if set(zip(comparison["branch"], comparison["model"], strict=True)) != USER_PROTOCOL_ADAPTERS:
        raise AssertionError("both fixed common-row adapters must be present")
    comparison.sort_values(
        ["macro_smape_percent", "branch", "model"], inplace=True, ignore_index=True
    )
    rounded = comparison["macro_smape_percent"].round(10)
    comparison.insert(0, "ranking_scope", ranking_scope)
    comparison["descriptive_rank"] = rounded.rank(method="min").astype(int)
    comparison["tie_group_size"] = rounded.groupby(rounded).transform("size").astype(int)
    comparison["tie_status"] = np.where(
        comparison["tie_group_size"].gt(1), "tied", "unique"
    )
    comparison["training_track"] = training_track
    comparison["same_effective_training_rows_required"] = True
    comparison["original_branch_declared_process"] = False
    comparison["joint_original_branch_rank"] = False
    comparison["adapter_track_first"] = comparison["descriptive_rank"].eq(1)
    comparison["robust_winner"] = False
    comparison["robustness_status"] = (
        "descriptive adapter ordering only; original branch winner unavailable"
    )
    return comparison


def _validate_common_row_adapter_inputs(
    predictions: pd.DataFrame, *, scope: str
) -> None:
    pair = predictions[
        [
            (str(branch), str(model)) in USER_PROTOCOL_ADAPTERS
            for branch, model in zip(
                predictions["branch"], predictions["model"], strict=True
            )
        ]
    ].copy()
    keys = ["fold_id", "metric", "test_year"]
    counts = pair.groupby(keys).size()
    if not counts.eq(2).all():
        raise AssertionError(f"{scope}: each test point must have both fixed adapters")
    invariant_columns = [
        "physical_train_n",
        "observed_train_n",
        "physical_canonical_train_n",
        "simulated_train_n",
        "augmented_train_n",
        "effective_train_n",
    ]
    for column in invariant_columns:
        if pair.groupby(keys)[column].nunique().gt(1).any():
            raise AssertionError(f"{scope}: fixed adapters disagree on {column}")
    if not pair["uses_all_augmented_training_rows"].all():
        raise AssertionError(f"{scope}: fixed adapters did not use every common input row")
    if not pair["effective_train_n"].eq(pair["augmented_train_n"]).all():
        raise AssertionError(f"{scope}: effective and supplied training counts differ")


def _validate_original_declarations(root: Path = ROOT) -> dict[str, object]:
    traditional_path = root / "outputs/jizhou_tourism_model/run_summary.json"
    ml_path = root / "outputs/jizhou_tourism_ml/run_summary.json"
    traditional = json.loads(traditional_path.read_text(encoding="utf-8"))
    machine_learning = json.loads(ml_path.read_text(encoding="utf-8"))
    if traditional.get("primary_model") != "post_2022_level_break":
        raise RuntimeError("traditional branch declaration drifted")
    selected = machine_learning.get("selected_ml_models", {})
    if set(selected.values()) != {"ridge_regime"}:
        raise RuntimeError("ML branch selected-model declaration drifted")
    if machine_learning.get("recommended_point_forecast") != "raw_target_ridge_alpha_0.1":
        raise RuntimeError("ML branch recommended-point declaration drifted")
    return {
        "traditional_summary": _relative(traditional_path),
        "traditional_primary_model": traditional["primary_model"],
        "ml_summary": _relative(ml_path),
        "ml_selected_models": selected,
        "ml_recommended_point_forecast": machine_learning["recommended_point_forecast"],
    }


def _declared_representative_table(
    simulated_macro: pd.DataFrame,
    official_macro: pd.DataFrame,
) -> pd.DataFrame:
    def score(frame: pd.DataFrame, branch: str, model: str) -> float:
        row = frame[frame["branch"].eq(branch) & frame["model"].eq(model)]
        return float(row.iloc[0]["macro_smape_percent"]) if len(row) == 1 else math.nan

    rows = [
        {
            "branch": TRADITIONAL_BRANCH,
            "model": "post_2022_level_break",
            "original_declaration": "primary_model",
            "declaration_source": "outputs/jizhou_tourism_model/run_summary.json",
            "rolling_execution_status": "not_executed_user_protocol",
            "simulated_track_macro_smape_percent": math.nan,
            "official_track_macro_smape_percent": math.nan,
            "jointly_rankable_original_declared_representatives": False,
            "reason": "user protocol forbids all breakpoint models; no rolling prediction/error",
        },
        {
            "branch": ML_BRANCH,
            "model": "raw_target_ridge_alpha_0.1",
            "original_declaration": "recommended_point_forecast",
            "declaration_source": "outputs/jizhou_tourism_ml/run_summary.json",
            "rolling_execution_status": "fixed_adapter_executed_but_historical_selection_saw_2024",
            "simulated_track_macro_smape_percent": score(
                simulated_macro, ML_BRANCH, "raw_target_ridge_alpha_0.1"
            ),
            "official_track_macro_smape_percent": score(
                official_macro, ML_BRANCH, "raw_target_ridge_alpha_0.1"
            ),
            "jointly_rankable_original_declared_representatives": False,
            "reason": "traditional declared counterpart was not executed; historical recommendation post-dates 2024 data",
        },
        {
            "branch": ML_BRANCH,
            "model": "ridge_regime",
            "original_declaration": "selected_ml_models for both targets",
            "declaration_source": "outputs/jizhou_tourism_ml/run_summary.json",
            "rolling_execution_status": "exploratory_candidate_retrained_with_inner_tuning",
            "simulated_track_macro_smape_percent": score(
                simulated_macro, ML_BRANCH, "ridge_regime"
            ),
            "official_track_macro_smape_percent": math.nan,
            "jointly_rankable_original_declared_representatives": False,
            "reason": "reported for declaration audit only; not the fixed user-protocol ML adapter",
        },
    ]
    return pd.DataFrame(rows)


def _ranking_status(
    predictions: pd.DataFrame,
    applicability: pd.DataFrame,
    exploratory: pd.DataFrame,
    adapter_comparison: pd.DataFrame,
) -> pd.DataFrame:
    expected_points = sum(len(years) for years in EXPECTED_ROLLING_TEST_YEARS.values())
    count_map = predictions.groupby(["branch", "model"]).size().to_dict()
    exploratory_rank = exploratory.set_index(["branch", "model"])[
        "exploratory_rank"
    ].to_dict()
    adapter_rank = adapter_comparison.set_index(["branch", "model"])[
        "descriptive_rank"
    ].to_dict()
    role_map = (
        predictions[["branch", "model", "comparison_role"]]
        .drop_duplicates()
        .set_index(["branch", "model"])["comparison_role"]
        .to_dict()
    )
    keys = sorted(
        set(zip(applicability["branch"], applicability["model"], strict=True))
        | set(count_map)
    )
    rows: list[dict[str, object]] = []
    for branch, model in keys:
        model_app = applicability[
            applicability["branch"].eq(branch) & applicability["model"].eq(model)
        ]
        statuses = sorted(model_app["status"].unique())
        count = int(count_map.get((branch, model), 0))
        if count == expected_points:
            coverage = (
                "complete_predictions_with_partial_feature_support"
                if "partial_out_of_support" in statuses
                else "complete"
            )
        elif count == 0 and statuses == ["not_executed_user_protocol"]:
            coverage = "not_executed_user_protocol"
        else:
            coverage = "partial"
        rows.append(
            {
                "branch": branch,
                "model": model,
                "comparison_role": role_map.get(
                    (branch, model), "non_executed_breakpoint_diagnostic"
                ),
                "rolling_prediction_points": count,
                "expected_prediction_points": expected_points,
                "coverage_status": coverage,
                "applicability_statuses": ";".join(statuses),
                "exploratory_rank": exploratory_rank.get((branch, model), math.nan),
                "adapter_descriptive_rank": adapter_rank.get((branch, model), math.nan),
                "ranking_status": "not_executed_user_protocol"
                if coverage == "not_executed_user_protocol"
                else "user_protocol_fixed_adapter"
                if (branch, model) in adapter_rank
                else "baseline_only"
                if model == "naive_last"
                else "exploratory_only",
                "reason": " | ".join(sorted(model_app["reason"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["ranking_status", "branch", "model"], ignore_index=True
    )


def _branch_inventory(source_provenance: pd.DataFrame) -> pd.DataFrame:
    pinned = source_provenance.set_index("branch").to_dict(orient="index")

    def source_fields(branch: str) -> dict[str, object]:
        record = pinned.get(branch, {})
        return {
            "model_source_path": record.get("path", ""),
            "model_source_pinned_commit": record.get("pinned_commit", ""),
            "model_source_git_blob_oid": record.get("git_blob_oid", ""),
            "model_source_sha256": record.get("sha256", ""),
            "model_source_validated": record.get("validated", False),
        }

    return pd.DataFrame(
        [
            {
                "ref": "main",
                "role": "canonical target truth and merged reporting destination",
                "executable_model_implementation": False,
                "original_declared_model": "",
                "benchmark_decision": "data source; no standalone branch model",
                **source_fields("main"),
            },
            {
                "ref": TRADITIONAL_BRANCH,
                "role": "traditional log-linear model source",
                "executable_model_implementation": True,
                "original_declared_model": "post_2022_level_break",
                "benchmark_decision": (
                    "user protocol executes no-break common-row OLS; native no-break/pre-COVID are "
                    "no-simulation canonical sensitivities; all breakpoint models are not executed"
                ),
                **source_fields(TRADITIONAL_BRANCH),
            },
            {
                "ref": ML_BRANCH,
                "role": "machine-learning candidate source",
                "executable_model_implementation": True,
                "original_declared_model": (
                    "ridge_regime (selected family); raw_target_ridge_alpha_0.1 "
                    "(recommended point path)"
                ),
                "benchmark_decision": (
                    "fixed raw-target Ridge is the user-protocol adapter; all candidates and "
                    "inner selector are exploratory"
                ),
                **source_fields(ML_BRANCH),
            },
            {
                "ref": "origin/111",
                "role": "example paper/workbooks",
                "executable_model_implementation": False,
                "original_declared_model": "",
                "benchmark_decision": "not benchmarked: no executable model source code",
                **source_fields("origin/111"),
            },
            {
                "ref": "origin/邱志烨-数据搜索",
                "role": "data preprocessing and model recommendations",
                "executable_model_implementation": False,
                "original_declared_model": "",
                "benchmark_decision": "not benchmarked: no executable model implementation",
                **source_fields("origin/邱志烨-数据搜索"),
            },
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_oid(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def validate_model_source_provenance(root: Path = ROOT) -> pd.DataFrame:
    """Refuse to benchmark if either executable branch source drifts from its pinned blob."""
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for spec in MODEL_SOURCE_SPECS:
        path = root / str(spec["path"])
        if not path.is_file():
            failures.append(f"{spec['branch']}: missing {spec['path']}")
            continue
        actual_sha256 = _sha256(path)
        actual_blob = _git_blob_oid(path)
        validated = (
            actual_sha256 == spec["sha256"] and actual_blob == spec["git_blob_oid"]
        )
        rows.append(
            {
                **spec,
                "actual_sha256": actual_sha256,
                "actual_git_blob_oid": actual_blob,
                "validated": validated,
            }
        )
        if not validated:
            failures.append(
                f"{spec['branch']}: expected sha256={spec['sha256']} blob={spec['git_blob_oid']}; "
                f"got sha256={actual_sha256} blob={actual_blob}"
            )
    if failures:
        raise RuntimeError("model source drift detected; " + " | ".join(failures))
    return pd.DataFrame(rows)


def _compressed_status_runs(years: pd.Series, statuses: pd.Series) -> str:
    pairs = [
        (int(year), "missing" if pd.isna(status) or str(status).strip() == "" else str(status))
        for year, status in zip(years, statuses, strict=True)
    ]
    if not pairs:
        return ""
    runs: list[str] = []
    start_year, previous_year = pairs[0][0], pairs[0][0]
    current_status = pairs[0][1]
    for year, status in pairs[1:]:
        if status == current_status and year == previous_year + 1:
            previous_year = year
            continue
        year_label = str(start_year) if start_year == previous_year else f"{start_year}-{previous_year}"
        runs.append(f"{year_label}:{current_status}")
        start_year = previous_year = year
        current_status = status
    year_label = str(start_year) if start_year == previous_year else f"{start_year}-{previous_year}"
    runs.append(f"{year_label}:{current_status}")
    return ";".join(runs)


def build_problem1_indicator_summary(canonical: pd.DataFrame) -> pd.DataFrame:
    """Summarize the four indicators explicitly requested by Problem 1."""
    required = {"year"} | {
        str(definition[key])
        for definition in PROBLEM1_INDICATORS.values()
        for key in ("value_column", "status_column")
    }
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise ValueError(f"Problem 1 canonical columns missing: {missing}")
    frame = canonical.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if set(frame["year"]) != set(range(2010, 2026)):
        raise ValueError("Problem 1 indicator summary requires one row for every year 2010-2025")
    frame.sort_values("year", inplace=True)
    rows: list[dict[str, object]] = []
    for metric, definition in PROBLEM1_INDICATORS.items():
        value_column = str(definition["value_column"])
        status_column = str(definition["status_column"])
        values = pd.to_numeric(frame[value_column], errors="coerce")
        available = frame.loc[values.notna(), ["year", status_column]].copy()
        available["value"] = values[values.notna()].to_numpy()
        if available.empty:
            raise ValueError(f"Problem 1 indicator {metric} has no available values")
        first = available.iloc[0]
        last = available.iloc[-1]
        value_by_year = dict(zip(frame["year"], values, strict=True))
        start_2010 = float(value_by_year[2010])
        end_2019 = float(value_by_year[2019])
        value_2023 = float(value_by_year[2023])
        value_2024 = float(value_by_year[2024])
        cagr = ((end_2019 / start_2010) ** (1.0 / 9.0) - 1.0) * 100.0
        value_2018 = float(value_by_year[2018])
        value_2025 = float(value_by_year[2025])
        cagr_2010_2018 = (
            ((value_2018 / start_2010) ** (1.0 / 8.0) - 1.0) * 100.0
            if np.isfinite(value_2018) and np.isfinite(start_2010)
            else math.nan
        )
        cagr_2019_2025 = (
            ((value_2025 / end_2019) ** (1.0 / 6.0) - 1.0) * 100.0
            if np.isfinite(value_2025) and np.isfinite(end_2019)
            else math.nan
        )
        crosses_documented_macro_break = metric in {
            "jizhou_gdp",
            "jizhou_tertiary_value_added",
        }
        if crosses_documented_macro_break:
            cagr = math.nan
        recovery_growth = (value_2024 / value_2023 - 1.0) * 100.0
        missing_years = frame.loc[values.isna(), "year"].astype(int).tolist()
        status_runs = _compressed_status_runs(frame["year"], frame[status_column])
        boundary_note = (
            "2019 macro series has a documented scope break; 2025 is official_initial"
            if metric in {"jizhou_gdp", "jizhou_tertiary_value_added"}
            else "2010 is inferred_from_yoy; 2016,2020,2022,2025 are missing"
            if metric == "tourism_comprehensive_income"
            else "2020-2022 and 2025 are missing"
        )
        rows.append(
            {
                "metric": metric,
                "indicator_label_cn": definition["label_cn"],
                "unit": definition["unit"],
                "coverage_start_year": 2010,
                "coverage_end_year": 2025,
                "total_calendar_years": 16,
                "nonmissing_count": int(values.notna().sum()),
                "missing_count": int(values.isna().sum()),
                "missing_years": ";".join(map(str, missing_years)),
                "first_nonmissing_year": int(first["year"]),
                "first_nonmissing_value": float(first["value"]),
                "first_status": str(first[status_column]),
                "last_nonmissing_year": int(last["year"]),
                "last_nonmissing_value": float(last["value"]),
                "last_status": str(last[status_column]),
                "cagr_2010_2019_percent": cagr,
                "cagr_2010_2018_percent": cagr_2010_2018,
                "cagr_2019_2025_percent": cagr_2019_2025,
                "cagr_interpretation": (
                    "2010-2019 CAGR suppressed because it crosses the documented macro scope break; use the two segment CAGRs"
                    if crosses_documented_macro_break
                    else "2010-2019 CAGR is within the target-series pre-COVID segment"
                ),
                "growth_2023_2024_percent": recovery_growth,
                "status_runs": status_runs,
                "status_boundary_note": boundary_note,
            }
        )
    return pd.DataFrame(rows)


def build_canonical_source_access_audit(
    canonical: pd.DataFrame, source_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Verify retrieval dates for exactly the sources cited by the canonical table."""
    if "source_ids" not in canonical or "source_id" not in source_metadata:
        raise ValueError("canonical/source metadata is missing source identifier columns")
    cited_ids = sorted(
        {
            token.strip()
            for cell in canonical["source_ids"].dropna().astype(str)
            for token in cell.split(";")
            if token.strip()
        }
    )
    if not cited_ids:
        raise ValueError("canonical table cites no source IDs")
    if source_metadata["source_id"].duplicated().any():
        raise ValueError("source metadata contains duplicate source_id rows")
    selected = source_metadata[source_metadata["source_id"].isin(cited_ids)].copy()
    missing = sorted(set(cited_ids) - set(selected["source_id"]))
    if missing:
        raise ValueError(f"canonical source IDs are absent from sources.csv: {missing}")
    selected["accessed_date"] = selected["notes"].astype(str).str.extract(
        r"accessed=(\d{4}-\d{2}-\d{2})", expand=False
    )
    if selected["accessed_date"].isna().any():
        undated = sorted(selected.loc[selected["accessed_date"].isna(), "source_id"])
        raise ValueError(f"canonical-cited sources lack an accessed date: {undated}")
    selected["used_by_canonical"] = True
    return selected[
        [
            "source_id",
            "topic",
            "source_label",
            "url",
            "status",
            "accessed_date",
            "used_by_canonical",
        ]
    ].sort_values("source_id", ignore_index=True)


def build_problem1_simple_growth_outputs(
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the pre-COVID exponential baseline requested as the Q1 simple model."""
    parameters: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for metric in TARGETS:
        definition = PROBLEM1_INDICATORS[metric]
        frame = canonical.loc[
            canonical["year"].between(2010, 2019),
            ["year", str(definition["value_column"])],
        ].rename(columns={str(definition["value_column"]): "value"})
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame.dropna(subset=["value"], inplace=True)
        fit = traditional_model.fit_log_ols(
            frame["year"].to_numpy(dtype=int),
            frame["value"].to_numpy(dtype=float),
            metric=metric,
            model_name="pre_covid_exponential",
            interrupted=False,
        )
        for row in traditional_model.parameter_rows(fit):
            row.update(
                {
                    "unit": definition["unit"],
                    "fit_scope": "available canonical values, 2010-2019 only",
                    "model_equation": "log(value)=intercept+year_index*(year-2010)",
                    "problem_role": "Q1 simple growth model; not the Q2 selected forecast model",
                }
            )
            parameters.append(row)
        diagnostic = traditional_model.diagnostic_row(fit)
        year_coefficient = float(fit.beta[fit.columns.index("year_index")])
        diagnostic.update(
            {
                "unit": definition["unit"],
                "fit_years": ";".join(map(str, fit.years.astype(int))),
                "annual_growth_rate_percent": (math.exp(year_coefficient) - 1.0) * 100.0,
                "fit_scope": "available canonical values, 2010-2019 only",
                "applicability": (
                    "describes the pre-COVID exponential trend; unsuitable as a unique post-2020 forecast"
                ),
            }
        )
        diagnostics.append(diagnostic)
    return pd.DataFrame(parameters), pd.DataFrame(diagnostics)


def build_problem1_simple_growth_forecasts(canonical: pd.DataFrame) -> pd.DataFrame:
    """Mechanically extrapolate the Q1 pre-COVID simple model for Q2 comparison."""
    rows: list[dict[str, object]] = []
    for metric in TARGETS:
        definition = PROBLEM1_INDICATORS[metric]
        frame = canonical.loc[
            canonical["year"].between(2010, 2019),
            ["year", str(definition["value_column"])],
        ].rename(columns={str(definition["value_column"]): "value"})
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame.dropna(subset=["value"], inplace=True)
        fit = traditional_model.fit_log_ols(
            frame["year"].to_numpy(dtype=int),
            frame["value"].to_numpy(dtype=float),
            metric=metric,
            model_name="pre_covid_exponential",
            interrupted=False,
        )
        for row in traditional_model.forecast_rows(
            fit, np.asarray(FORECAST_YEARS, dtype=int)
        ):
            row.update(
                {
                    "unit": definition["unit"],
                    "training_start_year": int(frame["year"].min()),
                    "training_end_year": int(frame["year"].max()),
                    "training_n": len(frame),
                    "forecast_role": (
                        "Q1 simple-model mechanical extrapolation for Q2 comparison; not selected as the Q2 main forecast"
                    ),
                    "point_semantics": (
                        "exponentiated conditional mean on the log scale; conditional median on the original scale"
                    ),
                    "uses_post_2019_training": False,
                    "uses_2025_as_training": False,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["metric", "year"], ignore_index=True
    )


def build_problem2_final_training(observations: pd.DataFrame) -> pd.DataFrame:
    """Build the final 2010-2024 model matrix without creating a 2025 label."""
    frames: list[pd.DataFrame] = []
    for metric in TARGETS:
        physical = _normalize_long_frame(
            observations[
                observations["metric"].eq(metric)
                & observations["year"].le(2024)
                & _training_evidence_mask(observations)
            ].copy()
        )
        augmented, _ = augment_training_rows(
            physical,
            test_year=2025,
            scope="problem2_final_fit_through_2024",
            fold_id=f"problem2_final_fit__{metric}",
        )
        simulated = augmented[_coerce_boolean(augmented["is_simulated"])]
        if simulated["method"].ne("log_linear_interpolation").any():
            raise AssertionError(
                "Problem 2 final refit may fill internal gaps only; tail extrapolation is forbidden"
            )
        if (
            simulated["boundary_right_year"].isna().any()
            or simulated["boundary_right_year"].gt(2024).any()
        ):
            raise AssertionError("Problem 2 interpolation boundary reaches beyond 2024")
        # ``test_year=2025`` above is only the exclusive range bound needed by the
        # generic fold-local augmenter.  Remove it from the reusable training
        # contract so 2025 cannot be mistaken for a label, boundary or input.
        augmented.drop(columns=["test_year"], inplace=True)
        augmented["split_id"] = "problem2_final_fit_through_2024"
        augmented["split"] = "train_final_augmented"
        augmented["training_end_year"] = 2024
        augmented["uses_2025_as_training"] = False
        frames.append(augmented)
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["metric", "year"], ignore_index=True
    )
    if result["year"].max() != 2024 or result["year"].eq(2025).any():
        raise AssertionError("Problem 2 final training must end at 2024 and exclude 2025")
    expected_years = set(range(2010, 2025))
    expected_simulated_years = {
        "tourist_visits": {2020, 2021, 2022},
        "tourism_comprehensive_income": {2016, 2020, 2022},
    }
    for metric, group in result.groupby("metric"):
        if set(group["year"].astype(int)) != expected_years:
            raise AssertionError(f"Problem 2 final training is not annual for {metric}")
        simulated_years = set(
            group.loc[_coerce_boolean(group["is_simulated"]), "year"].astype(int)
        )
        if simulated_years != expected_simulated_years[str(metric)]:
            raise AssertionError(
                f"Problem 2 final simulated-year contract drifted for {metric}: "
                f"{sorted(simulated_years)}"
            )
    return result


def _jarque_bera_diagnostics(residuals: np.ndarray) -> tuple[float, float]:
    centered = residuals - float(np.mean(residuals))
    variance = float(np.mean(centered**2))
    if variance <= 0.0:
        return 0.0, 1.0
    n = len(centered)
    skewness = float(np.mean(centered**3) / variance**1.5)
    excess_kurtosis = float(np.mean(centered**4) / variance**2 - 3.0)
    statistic = n / 6.0 * (skewness**2 + 0.25 * excess_kurtosis**2)
    return statistic, math.exp(-statistic / 2.0)


def _ridge_loocv_rmse(years: np.ndarray, values: np.ndarray) -> float:
    errors: list[float] = []
    for held_out in range(len(values)):
        keep = np.arange(len(values)) != held_out
        model = ml_model._ridge_builder({"alpha": 0.1})
        model = ml_model.fit_quietly(
            model, ml_model.make_features(years[keep]), values[keep]
        )
        prediction = float(model.predict(ml_model.make_features([years[held_out]]))[0])
        errors.append(float(values[held_out] - prediction))
    return float(np.sqrt(np.mean(np.square(errors))))


def _ridge_final_outputs(
    train: pd.DataFrame,
    *,
    bootstrap_seed: int = RANDOM_SEED,
    bootstrap_repetitions: int = RIDGE_BOOTSTRAP_REPETITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric = str(train.iloc[0]["metric"])
    years = train["year"].to_numpy(dtype=int)
    values = train["value"].to_numpy(dtype=float)
    x = ml_model.make_features(years)
    x_future = ml_model.make_features(np.asarray(FORECAST_YEARS, dtype=int))
    model = ml_model._ridge_builder({"alpha": 0.1})
    model = ml_model.fit_quietly(model, x, values)
    scaler = model.named_steps["scale"]
    ridge = model.named_steps["model"]
    z = scaler.transform(x)
    z_future = scaler.transform(x_future)
    fitted = np.asarray(model.predict(x), dtype=float)
    residuals = values - fitted
    centered_residuals = residuals - float(np.mean(residuals))
    penalty_inverse = np.linalg.inv(z.T @ z + 0.1 * np.eye(z.shape[1]))

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
                "model": "raw_target_ridge_alpha_0.1",
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
                    "fixed-design residual bootstrap (10000 repetitions), fixed alpha=0.1"
                ),
                "interval_warning": (
                    "model-conditional bootstrap interval; repeated-sampling 95% coverage is not guaranteed"
                ),
                "point_semantics": "raw-scale conditional mean under the fixed Ridge specification",
                "bootstrap_repetitions": bootstrap_repetitions,
                "random_seed": bootstrap_seed,
                "training_end_year": 2024,
                "training_n": len(train),
                "simulated_training_n": int(_coerce_boolean(train["is_simulated"]).sum()),
                "bootstrap_mean_draws_clipped_at_zero": mean_draws_clipped,
                "bootstrap_prediction_draws_clipped_at_zero": prediction_draws_clipped,
                "uses_2025_as_training": False,
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
                "model": "raw_target_ridge_alpha_0.1",
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
                "alpha": 0.1,
                "bootstrap_repetitions": bootstrap_repetitions,
                "random_seed": bootstrap_seed,
                "training_n": len(train),
                "training_end_year": 2024,
                "uses_2025_as_training": False,
            }
        )

    n = len(values)
    rss = float(residuals @ residuals)
    tss = float(np.square(values - values.mean()).sum())
    r_squared = 1.0 - rss / tss if tss else math.nan
    effective_parameters = 1.0 + float(
        np.trace(z @ penalty_inverse @ z.T)
    )
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / max(
        n - effective_parameters, 1.0e-12
    )
    aic = n * math.log(max(rss / n, 1.0e-300)) + 2.0 * effective_parameters
    aicc = aic + (
        2.0 * effective_parameters * (effective_parameters + 1.0)
        / (n - effective_parameters - 1.0)
    )
    dw = (
        float(np.diff(residuals) @ np.diff(residuals)) / rss
        if rss > 0.0
        else math.nan
    )
    jb, jb_p = _jarque_bera_diagnostics(residuals)
    diagnostic = pd.DataFrame(
        [
            {
                "metric": metric,
                "model": "raw_target_ridge_alpha_0.1",
                "target_scale": "raw",
                "training_n": n,
                "physical_canonical_training_n": int(
                    (~_coerce_boolean(train["is_simulated"])).sum()
                ),
                "simulated_training_n": int(
                    _coerce_boolean(train["is_simulated"]).sum()
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
                "smape_percent": float(
                    np.mean(
                        200.0
                        * np.abs(residuals)
                        / (np.abs(values) + np.abs(fitted))
                    )
                ),
                "aicc": aicc,
                "effective_parameters": effective_parameters,
                "loocv_rmse": _ridge_loocv_rmse(years, values),
                "loocv_scale": "raw",
                "durbin_watson": dw,
                "jarque_bera": jb,
                "jarque_bera_p": jb_p,
                "bootstrap_repetitions": bootstrap_repetitions,
                "random_seed": bootstrap_seed,
                "bootstrap_mean_draws_clipped_at_zero": mean_draws_clipped,
                "bootstrap_prediction_draws_clipped_at_zero": prediction_draws_clipped,
                "diagnostic_note": (
                    "descriptive in-sample diagnostics on augmented targets; AICc uses ridge effective degrees of freedom"
                ),
                "uses_2025_as_training": False,
            }
        ]
    )
    return pd.DataFrame(forecast_rows), diagnostic, pd.DataFrame(parameter_rows)


def build_problem2_outputs(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_training = build_problem2_final_training(observations)
    forecasts: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    ridge_parameters: list[pd.DataFrame] = []
    for metric in TARGETS:
        train = final_training[final_training["metric"].eq(metric)].copy()
        years = train["year"].to_numpy(dtype=int)
        values = train["value"].to_numpy(dtype=float)
        fit = traditional_model.fit_log_ols(
            years,
            values,
            metric=metric,
            model_name="no_break_log_linear_common_rows",
            interrupted=False,
        )
        ols_forecasts = pd.DataFrame(
            traditional_model.forecast_rows(
                fit, np.asarray(FORECAST_YEARS, dtype=int)
            )
        )
        ols_forecasts["interval_method"] = "log-OLS Student-t model-conditional intervals"
        ols_forecasts["interval_warning"] = (
            "conditional on the no-break log-linear specification and augmented training targets; no lognormal mean correction"
        )
        ols_forecasts["point_semantics"] = (
            "exponentiated conditional mean on the log scale; conditional median on the original scale"
        )
        ols_forecasts["bootstrap_repetitions"] = 0
        ols_forecasts["random_seed"] = RANDOM_SEED
        ols_forecasts["training_end_year"] = 2024
        ols_forecasts["training_n"] = len(train)
        ols_forecasts["simulated_training_n"] = int(
            _coerce_boolean(train["is_simulated"]).sum()
        )
        ols_forecasts["uses_2025_as_training"] = False
        forecasts.append(ols_forecasts)

        ols_diagnostic = traditional_model.diagnostic_row(fit)
        ols_diagnostic_frame = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "model": "no_break_log_linear_common_rows",
                    "target_scale": "log",
                    "training_n": len(train),
                    "physical_canonical_training_n": int(
                        (~_coerce_boolean(train["is_simulated"])).sum()
                    ),
                    "simulated_training_n": int(
                        _coerce_boolean(train["is_simulated"]).sum()
                    ),
                    "training_start_year": int(years.min()),
                    "training_end_year": int(years.max()),
                    "r_squared": ols_diagnostic["r_squared_log"],
                    "adjusted_r_squared": ols_diagnostic[
                        "adjusted_r_squared_log"
                    ],
                    "rmse": ols_diagnostic["rmse_original_units"],
                    "mae": float(
                        np.mean(np.abs(values - np.exp(fit.fitted_log)))
                    ),
                    "mape_percent": ols_diagnostic["mape_percent"],
                    "smape_percent": float(
                        np.mean(
                            200.0
                            * np.abs(values - np.exp(fit.fitted_log))
                            / (np.abs(values) + np.abs(np.exp(fit.fitted_log)))
                        )
                    ),
                    "aicc": ols_diagnostic["aicc_log"],
                    "effective_parameters": len(fit.beta),
                    "loocv_rmse": ols_diagnostic["loocv_log_rmse"],
                    "loocv_scale": "log",
                    "durbin_watson": ols_diagnostic["durbin_watson"],
                    "jarque_bera": ols_diagnostic["jarque_bera"],
                    "jarque_bera_p": ols_diagnostic["jarque_bera_p"],
                    "bootstrap_repetitions": 0,
                    "random_seed": RANDOM_SEED,
                    "bootstrap_mean_draws_clipped_at_zero": 0,
                    "bootstrap_prediction_draws_clipped_at_zero": 0,
                    "diagnostic_note": (
                        "descriptive in-sample diagnostics on augmented targets; inference is model-conditional"
                    ),
                    "uses_2025_as_training": False,
                }
            ]
        )
        diagnostics.append(ols_diagnostic_frame)

        ridge_forecast, ridge_diagnostic, ridge_parameter = _ridge_final_outputs(
            train, bootstrap_seed=RANDOM_SEED
        )
        forecasts.append(ridge_forecast)
        diagnostics.append(ridge_diagnostic)
        ridge_parameters.append(ridge_parameter)

    result_forecasts = pd.concat(forecasts, ignore_index=True).sort_values(
        ["metric", "model", "year"], ignore_index=True
    )
    if result_forecasts["uses_2025_as_training"].any():
        raise AssertionError("Problem 2 forecasts used the 2025 proxy as training")
    return (
        final_training,
        result_forecasts,
        pd.concat(diagnostics, ignore_index=True),
        pd.concat(ridge_parameters, ignore_index=True),
    )


def build_problem3_scenario_forecasts() -> pd.DataFrame:
    """Recompute the legacy policy-anchor paths with explicit non-causal metadata."""
    anchor_spend = (
        SCENARIO_INCOME_PROXY_100M / SCENARIO_VISITOR_PROXY_10K * 10_000.0
    )
    definitions = {
        "baseline_policy_anchor": {
            "label_cn": "基准情景",
            "income_growth": 0.08,
            "spend_growth": 0.03,
            "shock_2026": 0.0,
            "assumption": "2025政府目标作proxy；收入年增8%；人均消费年增3%",
        },
        "optimistic_assumption": {
            "label_cn": "乐观情景",
            "income_growth": 0.12,
            "spend_growth": 0.04,
            "shock_2026": 0.0,
            "assumption": "2025政府目标作proxy；收入年增12%；人均消费年增4%",
        },
        "pessimistic_assumption": {
            "label_cn": "悲观情景",
            "income_growth": 0.05,
            "spend_growth": 0.02,
            "shock_2026": -0.15,
            "assumption": "2025政府目标作proxy；2026收入冲击-15%；其后收入年增5%；人均消费年增2%",
        },
    }
    units = {
        "tourist_visits": "10k_persons",
        "tourism_comprehensive_income": "100m_cny",
        "nominal_spend_per_visit": "cny_per_visit",
    }
    anchor_values = {
        "tourist_visits": SCENARIO_VISITOR_PROXY_10K,
        "tourism_comprehensive_income": SCENARIO_INCOME_PROXY_100M,
        "nominal_spend_per_visit": anchor_spend,
    }
    rows: list[dict[str, object]] = []
    for scenario, definition in definitions.items():
        for year in FORECAST_YEARS:
            horizon = year - SCENARIO_ANCHOR_YEAR
            if scenario == "pessimistic_assumption":
                income = (
                    SCENARIO_INCOME_PROXY_100M
                    * (1.0 + float(definition["shock_2026"]))
                    * (1.0 + float(definition["income_growth"])) ** (horizon - 1)
                )
            else:
                income = SCENARIO_INCOME_PROXY_100M * (
                    1.0 + float(definition["income_growth"])
                ) ** horizon
            spend = anchor_spend * (
                1.0 + float(definition["spend_growth"])
            ) ** horizon
            visitors = income / spend * 10_000.0
            values = {
                "tourist_visits": visitors,
                "tourism_comprehensive_income": income,
                "nominal_spend_per_visit": spend,
            }
            for metric, value in values.items():
                rows.append(
                    {
                        "year": year,
                        "scenario": scenario,
                        "scenario_label_cn": definition["label_cn"],
                        "assumption": definition["assumption"],
                        "metric": metric,
                        "value": value,
                        "unit": units[metric],
                        "anchor_year": SCENARIO_ANCHOR_YEAR,
                        "anchor_value": anchor_values[metric],
                        "anchor_status": "government_target_proxy_not_actual",
                        "anchor_is_observed": False,
                        "income_growth_assumption": definition["income_growth"],
                        "spend_growth_assumption": definition["spend_growth"],
                        "shock_2026_assumption": definition["shock_2026"],
                        "historically_identified_causal_effect": False,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["scenario", "year", "metric"], ignore_index=True
    )


def build_problem3_policy_sensitivity(
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    """One-at-a-time policy sensitivity around the transparent baseline identity."""
    baseline = scenarios[
        scenarios["scenario"].eq("baseline_policy_anchor")
        & scenarios["year"].eq(2030)
    ].set_index("metric")["value"]
    baseline_visitors = float(baseline["tourist_visits"])
    baseline_income = float(baseline["tourism_comprehensive_income"])
    anchor_spend = (
        SCENARIO_INCOME_PROXY_100M / SCENARIO_VISITOR_PROXY_10K * 10_000.0
    )
    baseline_visitor_growth = 1.08 / 1.03 - 1.0
    horizon = 2030 - SCENARIO_ANCHOR_YEAR
    configurations = [
        (
            "source_market_growth",
            "low",
            baseline_visitor_growth - 0.02,
            0.03,
            1.0,
            "visitor annual growth is baseline implied growth minus 2 percentage points; spend growth fixed at 3%",
        ),
        (
            "source_market_growth",
            "high",
            baseline_visitor_growth + 0.02,
            0.03,
            1.0,
            "visitor annual growth is baseline implied growth plus 2 percentage points; spend growth fixed at 3%",
        ),
        (
            "new_format_spend_growth",
            "low",
            baseline_visitor_growth,
            0.02,
            1.0,
            "nominal spend-per-visit growth is 2% (baseline 3% minus 1 percentage point); visitor growth fixed",
        ),
        (
            "new_format_spend_growth",
            "high",
            baseline_visitor_growth,
            0.04,
            1.0,
            "nominal spend-per-visit growth is 4% (baseline 3% plus 1 percentage point); visitor growth fixed",
        ),
        (
            "policy_coordination_multiplier",
            "low",
            baseline_visitor_growth,
            0.03,
            0.95,
            "persistent visitor and income level multiplier from 2026 is 0.95; subsequent growth unchanged",
        ),
        (
            "policy_coordination_multiplier",
            "high",
            baseline_visitor_growth,
            0.03,
            1.05,
            "persistent visitor and income level multiplier from 2026 is 1.05; subsequent growth unchanged",
        ),
        (
            "external_shock",
            "low",
            baseline_visitor_growth,
            0.03,
            0.85,
            "2026 unexpected shock is -15% and leaves a persistent level gap; subsequent growth unchanged",
        ),
        (
            "external_shock",
            "high",
            baseline_visitor_growth,
            0.03,
            1.0,
            "2026 unexpected shock is 0%; baseline path is retained",
        ),
    ]
    rows: list[dict[str, object]] = []
    for factor, setting, visitor_growth, spend_growth, level_multiplier, assumption in configurations:
        visitors = (
            SCENARIO_VISITOR_PROXY_10K
            * (1.0 + visitor_growth) ** horizon
            * level_multiplier
        )
        spend = anchor_spend * (1.0 + spend_growth) ** horizon
        income = visitors * spend / 10_000.0
        for metric, baseline_value, scenario_value, unit in (
            ("tourist_visits", baseline_visitors, visitors, "10k_persons"),
            (
                "tourism_comprehensive_income",
                baseline_income,
                income,
                "100m_cny",
            ),
        ):
            delta = scenario_value - baseline_value
            if abs(delta) < 1.0e-10:
                scenario_value = baseline_value
                delta = 0.0
            rows.append(
                {
                    "factor": factor,
                    "setting": setting,
                    "metric": metric,
                    "baseline_2030": baseline_value,
                    "scenario_2030": scenario_value,
                    "delta_2030": delta,
                    "delta_percent": delta / baseline_value * 100.0,
                    "assumption": assumption,
                    "unit": unit,
                    "historically_identified_causal_effect": False,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["factor", "setting", "metric"], ignore_index=True
    )


def _render_problem_visuals_if_available(
    output_dir: Path, canonical_path: Path
) -> list[str]:
    try:
        from render_tjmml_c_visuals import render_problem_visuals
    except ModuleNotFoundError as error:
        if error.name != "render_tjmml_c_visuals":
            raise
        return []
    rendered = list(render_problem_visuals(output_dir, canonical_path))
    expected = {
        "q1_required_indicators.png",
        "q2_model_judgement.png",
        "q2_forecast_2026_2030.png",
        "q3_scenarios_sensitivity.png",
    }
    unexpected = sorted(set(rendered) - expected)
    if unexpected:
        raise ValueError(f"visual renderer returned unexpected files: {unexpected}")
    missing = [name for name in rendered if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"visual renderer did not create returned files: {missing}")
    return rendered


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


REPORT_COLUMN_LABELS: Final = {
    "indicator_label_cn": "指标",
    "unit": "单位",
    "nonmissing_count": "有数据年份数",
    "missing_years": "缺失年份",
    "first_nonmissing_year": "首个有数据年份",
    "first_nonmissing_value": "首个数值",
    "last_nonmissing_year": "最近有数据年份",
    "last_nonmissing_value": "最近数值",
    "cagr_2010_2019_percent": "2010—2019年均复合增长率（CAGR，%）",
    "cagr_2010_2018_percent": "2010—2018年均复合增长率（CAGR，%）",
    "cagr_2019_2025_percent": "2019—2025年均复合增长率（CAGR，%）",
    "growth_2023_2024_percent": "2023—2024增长率（%）",
    "status_boundary_note": "数据状态与口径说明",
    "metric": "预测指标",
    "parameter": "参数或解释变量",
    "estimate": "估计值",
    "standard_error": "标准误",
    "t_value": "t统计量",
    "p_value": "p值",
    "ci95_lower": "95%置信区间下限",
    "ci95_upper": "95%置信区间上限",
    "df_resid": "残差自由度",
    "n": "样本数",
    "annual_growth_rate_percent": "年增长率（%）",
    "r_squared_log": "对数尺度决定系数（R²）",
    "adjusted_r_squared_log": "对数尺度调整后决定系数（调整后R²）",
    "rmse_original_units": "原尺度均方根误差（RMSE）",
    "mape_percent": "平均绝对百分比误差（MAPE，%）",
    "aicc_log": "对数尺度小样本修正赤池信息准则（AICc）",
    "loocv_log_rmse": "对数尺度留一交叉验证均方根误差（LOOCV RMSE）",
    "durbin_watson": "德宾—沃森统计量（DW）",
    "jarque_bera_p": "雅克—贝拉检验p值（JB p值）",
    "model": "模型",
    "year": "年份",
    "forecast": "点预测",
    "mean_ci95_lower": "平均响应95%置信区间下限",
    "mean_ci95_upper": "平均响应95%置信区间上限",
    "prediction_interval95_lower": "单次预测95%区间下限",
    "prediction_interval95_upper": "单次预测95%区间上限",
    "n_test": "测试点数",
    "smape_percent": "对称平均绝对百分比误差（sMAPE，%）",
    "naive_smape_percent": "上一期数值法sMAPE（%）",
    "smape_skill_vs_naive": "相对上一期数值法的改进率",
    "worst_point_smape_percent": "最差单点sMAPE（%）",
    "training_n": "训练样本数",
    "simulated_training_n": "模拟训练样本数",
    "target_scale": "因变量计算尺度",
    "r_squared": "决定系数（R²）",
    "rmse": "均方根误差（RMSE）",
    "aicc": "小样本修正赤池信息准则（AICc）",
    "loocv_rmse": "留一交叉验证均方根误差（LOOCV RMSE）",
    "loocv_scale": "留一交叉验证计算尺度",
    "bootstrap_ci95_lower": "自助法95%区间下限",
    "bootstrap_ci95_upper": "自助法95%区间上限",
    "feature_training_mean": "解释变量训练均值",
    "feature_training_scale": "解释变量训练标准差",
    "scenario_label_cn": "情景",
    "scenario": "情景代码",
    "tourist_visits": "旅游接待人次（万人次）",
    "tourism_comprehensive_income": "旅游综合收入（亿元）",
    "nominal_spend_per_visit": "名义人均次消费（元/人次）",
    "factor": "影响因素",
    "setting": "扰动方向",
    "baseline_2030": "2030年基准值",
    "scenario_2030": "2030年扰动后数值",
    "delta_2030": "2030年变化量",
    "delta_percent": "相对基准变化率（%）",
    "descriptive_rank": "描述性排序",
    "branch": "来源分支",
    "macro_smape_percent": "两个目标等权平均sMAPE（%）",
    "macro_smape_skill_vs_naive": "相对上一期数值法的等权平均改进率",
    "beats_naive_all_targets": "两个目标是否均优于上一期数值法",
    "worst_target": "误差较大的目标",
    "worst_target_smape_percent": "误差较大目标的sMAPE（%）",
    "tie_status": "并列状态",
    "training_track": "训练数据方案",
    "original_declaration": "原分支声明角色",
    "rolling_execution_status": "滚动检验执行状态",
    "simulated_track_macro_smape_percent": "模拟增强方案等权平均sMAPE（%）",
    "official_track_macro_smape_percent": "未模拟方案等权平均sMAPE（%）",
    "jointly_rankable_original_declared_representatives": "原声明模型能否共同排名",
    "reason": "原因",
    "exploratory_rank": "探索性排序",
    "stability_flag": "排序稳定性标记",
    "mae": "平均绝对误差（MAE）",
    "test_years": "测试年份",
    "path": "文件路径",
    "pinned_commit": "固定提交",
    "git_blob_oid": "Git对象标识",
    "sha256": "SHA-256哈希",
    "validated": "是否校验通过",
}

REPORT_VALUE_LABELS: Final = {
    "10k_persons": "万人次",
    "100m_cny": "亿元",
    "tourist_visits": "旅游接待人次",
    "tourism_comprehensive_income": "旅游综合收入",
    "intercept": "截距项",
    "year_index": "时间趋势项",
    "pandemic_2020_2022": "2020—2022年疫情期指示变量",
    "post_2022": "2023年后恢复期指示变量",
    "raw": "原尺度",
    "log": "对数尺度",
    "raw_target_ridge_alpha_0.1": "原尺度岭回归（Ridge，惩罚参数α=0.1）",
    "no_break_log_linear_common_rows": "相同训练年份的无断点对数线性回归（OLS）",
    "no_break_log_linear": "原生无断点对数线性回归（OLS）",
    "pre_covid_exponential": "疫情前指数增长模型",
    "post_2022_level_break": "2023年后水平断点模型",
    "strict_evidence_level_break": "严格证据口径的水平断点模型",
    "ridge_regime": "分阶段岭回归",
    "gaussian_process": "高斯过程回归",
    "ml_inner_selector": "训练内部模型选择器",
    "svr_rbf": "径向基核支持向量回归（RBF-SVR）",
    "robust_ml_ensemble": "稳健机器学习集成模型",
    "bayesian_ridge": "贝叶斯岭回归",
    "huber_regime": "分阶段Huber稳健回归",
    "naive_last": "上一期数值法（朴素基准）",
    "random_forest": "随机森林回归",
    "spline_ridge": "样条岭回归",
    "official_only_physical_rows": "未模拟的官方来源训练行",
    "user_simulated_augmentation": "模拟增强训练行",
    "baseline_policy_anchor": "基准情景",
    "optimistic_assumption": "乐观情景",
    "pessimistic_assumption": "悲观情景",
    "external_shock": "突发事件冲击",
    "new_format_spend_growth": "新业态与人均消费增速",
    "policy_coordination_multiplier": "政策协同水平",
    "source_market_growth": "客源市场增长",
    "high": "上行情景",
    "low": "下行情景",
    "unique": "无并列",
    "primary_model": "原分支主模型",
    "recommended_point_forecast": "原分支推荐点预测",
    "selected_ml_models for both targets": "原分支为两个目标选定的机器学习模型",
    "not_executed_user_protocol": "按本次约定不执行",
    "fixed_adapter_executed_but_historical_selection_saw_2024": "本次固定规格已执行，但原历史选择曾使用2024年数据",
    "exploratory_candidate_retrained_with_inner_tuning": "仅作探索，并在每折训练内部重新调参",
    "exploratory_candidate": "探索性候选模型",
    "user_protocol_fixed_ml_adapter": "本次约定的固定机器学习规格",
    "user_protocol_common_row_adapter": "本次约定的相同训练年份规格",
    "benchmark_created_exploratory_adapter": "本次比较新增的探索性规格",
    "baseline": "基准模型",
    "physical canonical training rows only; no simulated labels": "仅使用未模拟的官方来源训练行",
    "2020-2022 and 2025 are missing": "2020—2022年及2025年缺失",
    "2010 is inferred_from_yoy; 2016,2020,2022,2025 are missing": "2010年由同比反推；2016、2020、2022、2025年缺失",
    "2019 macro series has a documented scope break; 2025 is official_initial": "2019年宏观序列存在已记录的统计口径断点；2025年为官方初值",
}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()

    def format_cell(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (bool, np.bool_)):
            return "是" if bool(value) else "否"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        text = REPORT_VALUE_LABELS.get(str(value), str(value))
        return text.replace("|", "\\|").replace("\n", " ")

    rows = [[format_cell(value) for value in row] for row in selected.itertuples(index=False)]
    headers = [REPORT_COLUMN_LABELS.get(column, column) for column in columns]
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _write_report(
    report_path: Path,
    *,
    inputs: BenchmarkInputs,
    simulated_comparison: pd.DataFrame,
    official_comparison: pd.DataFrame,
    simulated_by_target: pd.DataFrame,
    official_by_target: pd.DataFrame,
    declared: pd.DataFrame,
    exploratory: pd.DataFrame,
    holdout_by_target: pd.DataFrame,
    stress_by_target: pd.DataFrame,
    source_provenance: pd.DataFrame,
    problem_canonical: pd.DataFrame,
    problem1_indicator_summary: pd.DataFrame,
    problem1_growth_parameters: pd.DataFrame,
    problem1_growth_diagnostics: pd.DataFrame,
    problem1_growth_forecasts: pd.DataFrame,
    problem2_forecasts: pd.DataFrame,
    problem2_diagnostics: pd.DataFrame,
    problem2_ridge_parameters: pd.DataFrame,
    problem3_scenarios: pd.DataFrame,
    problem3_sensitivity: pd.DataFrame,
    canonical_source_access: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    simulated_view = simulated_comparison[
        [
            "descriptive_rank",
            "branch",
            "model",
            "macro_smape_percent",
            "macro_smape_skill_vs_naive",
            "beats_naive_all_targets",
            "worst_target",
            "worst_target_smape_percent",
            "worst_point_smape_percent",
            "tie_status",
        ]
    ]
    official_view = official_comparison[list(simulated_view.columns)]
    declared_view = declared[
        [
            "branch",
            "model",
            "original_declaration",
            "rolling_execution_status",
            "simulated_track_macro_smape_percent",
            "official_track_macro_smape_percent",
            "jointly_rankable_original_declared_representatives",
        ]
    ]
    exploratory_view = exploratory[
        [
            "exploratory_rank",
            "branch",
            "model",
            "macro_smape_percent",
            "macro_smape_skill_vs_naive",
            "worst_point_smape_percent",
            "stability_flag",
        ]
    ]
    report_models = {
        "naive_last",
        "no_break_log_linear_common_rows",
        "raw_target_ridge_alpha_0.1",
        "ml_inner_selector",
    }
    holdout_view = holdout_by_target[holdout_by_target["model"].isin(report_models)][
        ["metric", "branch", "model", "n_test", "mae", "mape_percent", "smape_percent"]
    ].sort_values(["metric", "smape_percent"])
    stress_view = stress_by_target[stress_by_target["model"].isin(report_models)][
        [
            "metric",
            "branch",
            "model",
            "n_test",
            "test_years",
            "mape_percent",
            "smape_percent",
            "smape_skill_vs_naive",
        ]
    ].sort_values(["metric", "smape_percent"])
    simulated_first = simulated_comparison[
        simulated_comparison["descriptive_rank"].eq(1)
    ].iloc[0]
    official_first = official_comparison[
        official_comparison["descriptive_rank"].eq(1)
    ].iloc[0]
    ordering_consistent = bool(
        simulated_comparison["cross_track_order_consistent"].iloc[0]
    )
    ordering_sentence = (
        "两种训练数据方案下，固定模型规格的相对顺序一致，但这仍不能替代原分支声明模型的共同排名。"
        if ordering_consistent
        else "两种训练数据方案下，固定模型规格的相对顺序发生反转，因此不能给出稳健赢家。"
    )
    track_target_view = pd.concat(
        [
            simulated_by_target[
                [
                    "branch",
                    "model",
                    "metric",
                    "n_test",
                    "smape_percent",
                    "naive_smape_percent",
                    "smape_skill_vs_naive",
                ]
            ].assign(training_track="user_simulated_augmentation"),
            official_by_target[
                [
                    "branch",
                    "model",
                    "metric",
                    "n_test",
                    "smape_percent",
                    "naive_smape_percent",
                    "smape_skill_vs_naive",
                ]
            ].assign(training_track="official_only_physical_rows"),
        ],
        ignore_index=True,
    )
    track_target_view = track_target_view[
        [
            (str(branch), str(model)) in USER_PROTOCOL_ADAPTERS
            for branch, model in zip(
                track_target_view["branch"], track_target_view["model"], strict=True
            )
        ]
    ].sort_values(["training_track", "metric", "smape_percent"])
    native_view = official_by_target[
        official_by_target["model"].isin(
            {
                "naive_last",
                "pre_covid_exponential",
                "no_break_log_linear",
                "no_break_log_linear_common_rows",
                "raw_target_ridge_alpha_0.1",
            }
        )
    ][
        [
            "metric",
            "branch",
            "model",
            "n_test",
            "smape_percent",
            "smape_skill_vs_naive",
        ]
    ].sort_values(["metric", "smape_percent"])
    provenance_view = source_provenance[
        [
            "branch",
            "path",
            "pinned_commit",
            "git_blob_oid",
            "sha256",
            "validated",
        ]
    ]
    train_counts = inputs.primary_train.groupby("metric").size().to_dict()
    rolling_years = {
        metric: list(years) for metric, years in EXPECTED_ROLLING_TEST_YEARS.items()
    }
    coverage_matrix = pd.DataFrame(
        [
            {
                "题目": "问题1",
                "题面任务": "整理四项指标并建立简单增长模型",
                "直接产物": (
                    "problem1_indicator_summary.csv；problem1_simple_growth_parameters.csv；"
                    "problem1_simple_growth_diagnostics.csv；"
                    "problem1_simple_growth_forecasts_2026_2030.csv；"
                    "q1_required_indicators.png"
                ),
                "解释边界": "疫情前指数增长模型只描述2010—2019年疫情前可用值",
            },
            {
                "题目": "问题2",
                "题面任务": "比较模型、预测2026—2030并评价合理性",
                "直接产物": (
                    "problem2_forecasts_2026_2030.csv；"
                    "problem2_final_model_diagnostics.csv；"
                    "两张Q2图"
                ),
                "解释边界": "2025年政府目标代理值不进训练；所有区间均为模型条件区间",
            },
            {
                "题目": "问题3",
                "题面任务": "三情景预测、因素敏感性和政策建议",
                "直接产物": (
                    "problem3_scenario_forecasts_2026_2030.csv；"
                    "problem3_policy_sensitivity.csv；q3_scenarios_sensitivity.png"
                ),
                "解释边界": "透明会计情景，不是历史因果识别",
            },
            {
                "题目": "资料说明",
                "题面任务": "论文注明所有数据的实际获取日期",
                "直接产物": (
                    f"problem_source_access_dates.csv 核验官方优选数据引用的 "
                    f"{len(canonical_source_access)} 个唯一来源"
                ),
                "解释边界": (
                    f"这{len(canonical_source_access)}项均记录实际获取日期为2026-08-17；"
                    "不推广到来源总表的全部条目"
                ),
            },
        ]
    )
    problem1_summary_view = problem1_indicator_summary[
        [
            "indicator_label_cn",
            "unit",
            "nonmissing_count",
            "missing_years",
            "first_nonmissing_year",
            "first_nonmissing_value",
            "last_nonmissing_year",
            "last_nonmissing_value",
            "cagr_2010_2019_percent",
            "cagr_2010_2018_percent",
            "cagr_2019_2025_percent",
            "growth_2023_2024_percent",
            "status_boundary_note",
        ]
    ]
    problem1_parameter_view = problem1_growth_parameters[
        [
            "metric",
            "parameter",
            "estimate",
            "standard_error",
            "t_value",
            "p_value",
            "ci95_lower",
            "ci95_upper",
            "df_resid",
        ]
    ]
    problem1_diagnostic_view = problem1_growth_diagnostics[
        [
            "metric",
            "n",
            "annual_growth_rate_percent",
            "r_squared_log",
            "adjusted_r_squared_log",
            "rmse_original_units",
            "mape_percent",
            "aicc_log",
            "loocv_log_rmse",
            "durbin_watson",
            "jarque_bera_p",
        ]
    ]
    problem1_forecast_view = problem1_growth_forecasts[
        [
            "metric",
            "model",
            "year",
            "forecast",
            "mean_ci95_lower",
            "mean_ci95_upper",
            "prediction_interval95_lower",
            "prediction_interval95_upper",
        ]
    ]
    problem2_forecast_view = problem2_forecasts[
        [
            "metric",
            "model",
            "year",
            "forecast",
            "mean_ci95_lower",
            "mean_ci95_upper",
            "prediction_interval95_lower",
            "prediction_interval95_upper",
        ]
    ].sort_values(["metric", "year", "model"])
    problem2_diagnostic_view = problem2_diagnostics[
        [
            "metric",
            "model",
            "training_n",
            "simulated_training_n",
            "target_scale",
            "r_squared",
            "rmse",
            "mape_percent",
            "aicc",
            "loocv_rmse",
            "loocv_scale",
            "durbin_watson",
            "jarque_bera_p",
        ]
    ]
    problem2_parameter_view = problem2_ridge_parameters[
        [
            "metric",
            "parameter",
            "estimate",
            "bootstrap_ci95_lower",
            "bootstrap_ci95_upper",
            "feature_training_mean",
            "feature_training_scale",
        ]
    ]
    problem2_validation_view = official_by_target[
        official_by_target["model"].isin(
            {
                "naive_last",
                "pre_covid_exponential",
                "no_break_log_linear_common_rows",
                "raw_target_ridge_alpha_0.1",
            }
        )
    ][
        [
            "metric",
            "model",
            "n_test",
            "smape_percent",
            "naive_smape_percent",
            "smape_skill_vs_naive",
            "worst_point_smape_percent",
        ]
    ].sort_values(["metric", "smape_percent"])
    scenario_view = (
        problem3_scenarios.pivot_table(
            index=["scenario_label_cn", "scenario", "year"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )[
        [
            "scenario_label_cn",
            "scenario",
            "year",
            "tourist_visits",
            "tourism_comprehensive_income",
            "nominal_spend_per_visit",
        ]
    ]
    sensitivity_view = problem3_sensitivity[
        [
            "factor",
            "setting",
            "metric",
            "baseline_2030",
            "scenario_2030",
            "delta_2030",
            "delta_percent",
        ]
    ]

    def sensitivity_row(factor: str, setting: str, metric: str) -> pd.Series:
        selected = problem3_sensitivity[
            problem3_sensitivity["factor"].eq(factor)
            & problem3_sensitivity["setting"].eq(setting)
            & problem3_sensitivity["metric"].eq(metric)
        ]
        if len(selected) != 1:
            raise AssertionError(f"missing Q3 sensitivity row: {factor}/{setting}/{metric}")
        return selected.iloc[0]

    source_high_visits = sensitivity_row(
        "source_market_growth", "high", "tourist_visits"
    )
    source_high_income = sensitivity_row(
        "source_market_growth", "high", "tourism_comprehensive_income"
    )
    spend_high_income = sensitivity_row(
        "new_format_spend_growth", "high", "tourism_comprehensive_income"
    )
    coordination_high_visits = sensitivity_row(
        "policy_coordination_multiplier", "high", "tourist_visits"
    )
    coordination_high_income = sensitivity_row(
        "policy_coordination_multiplier", "high", "tourism_comprehensive_income"
    )
    shock_low_visits = sensitivity_row("external_shock", "low", "tourist_visits")
    shock_low_income = sensitivity_row(
        "external_shock", "low", "tourism_comprehensive_income"
    )
    baseline_2030 = problem3_scenarios[
        problem3_scenarios["scenario"].eq("baseline_policy_anchor")
        & problem3_scenarios["year"].eq(2030)
    ].set_index("metric")["value"]
    canonical_access_dates = "、".join(
        sorted(canonical_source_access["accessed_date"].astype(str).unique())
    )

    def final_forecast_value(metric: str, model: str, year: int) -> float:
        selected = problem2_forecasts[
            problem2_forecasts["metric"].eq(metric)
            & problem2_forecasts["model"].eq(model)
            & problem2_forecasts["year"].eq(year)
        ]
        if len(selected) != 1:
            raise AssertionError(f"missing Q2 forecast row: {metric}/{model}/{year}")
        return float(selected.iloc[0]["forecast"])

    ridge_2026_visits = final_forecast_value(
        "tourist_visits", "raw_target_ridge_alpha_0.1", 2026
    )
    ridge_2030_visits = final_forecast_value(
        "tourist_visits", "raw_target_ridge_alpha_0.1", 2030
    )
    ridge_2026_income = final_forecast_value(
        "tourism_comprehensive_income", "raw_target_ridge_alpha_0.1", 2026
    )
    ridge_2030_income = final_forecast_value(
        "tourism_comprehensive_income", "raw_target_ridge_alpha_0.1", 2030
    )
    ols_2030_visits = final_forecast_value(
        "tourist_visits", "no_break_log_linear_common_rows", 2030
    )
    ols_2030_income = final_forecast_value(
        "tourism_comprehensive_income", "no_break_log_linear_common_rows", 2030
    )
    q1_growth_by_metric = problem1_growth_diagnostics.set_index("metric")[
        "annual_growth_rate_percent"
    ]
    q1_summary_by_metric = problem1_indicator_summary.set_index("metric")
    q1_forecast_keyed = problem1_growth_forecasts.set_index(["metric", "year"])[
        "forecast"
    ]
    canonical_by_year = problem_canonical.set_index("year")

    def relative_change_percent(column: str, start_year: int, end_year: int) -> float:
        start_value = float(canonical_by_year.loc[start_year, column])
        end_value = float(canonical_by_year.loc[end_year, column])
        return (end_value / start_value - 1.0) * 100.0

    visitor_2023_vs_2019 = relative_change_percent(
        "preferred_visitor_10k_persons", 2019, 2023
    )
    visitor_2024_vs_2019 = relative_change_percent(
        "preferred_visitor_10k_persons", 2019, 2024
    )
    income_2021_vs_2019 = relative_change_percent(
        "preferred_comprehensive_income_100m_cny", 2019, 2021
    )
    income_2023_vs_2019 = relative_change_percent(
        "preferred_comprehensive_income_100m_cny", 2019, 2023
    )
    gdp_2019_vs_2018 = relative_change_percent(
        "preferred_gdp_100m_cny", 2018, 2019
    )
    tertiary_2019_vs_2018 = relative_change_percent(
        "preferred_tertiary_100m_cny", 2018, 2019
    )
    actual_2024_spend = (
        float(
            canonical_by_year.loc[
                2024, "preferred_comprehensive_income_100m_cny"
            ]
        )
        / float(canonical_by_year.loc[2024, "preferred_visitor_10k_persons"])
        * 10_000.0
    )
    ridge_2026_implied_spend = ridge_2026_income / ridge_2026_visits * 10_000.0
    ridge_2030_implied_spend = ridge_2030_income / ridge_2030_visits * 10_000.0
    ridge_spend_2030_vs_2024 = (
        ridge_2030_implied_spend / actual_2024_spend - 1.0
    ) * 100.0
    scenario_2030_pivot = problem3_scenarios[
        problem3_scenarios["year"].eq(2030)
        & problem3_scenarios["metric"].isin(TARGETS)
    ].pivot(index="scenario", columns="metric", values="value")
    scenario_2030_envelope = {
        metric: (
            float(scenario_2030_pivot[metric].min()),
            float(scenario_2030_pivot[metric].max()),
        )
        for metric in TARGETS
    }

    problem_sections = f"""## 题目要求覆盖矩阵

{_markdown_table(coverage_matrix, list(coverage_matrix.columns))}

本报告把题面要求与计算产物逐项对齐。对称平均绝对百分比误差（sMAPE）、相对上一期数值法的改进率、滚动检验和单年留出检验，是为回答“模型是否适用、合理、哪个更好”而选用的审计工具，**不是题面直接指定的评价指标**。

官方优选年度汇总表实际引用的 {len(canonical_source_access)} 个唯一来源编号（代码字段 `source_id`）均能在 `data/metadata/sources.csv` 中定位，且备注字段（`notes`）记录的实际获取日期均为 {canonical_access_dates}；逐项核对见 `problem_source_access_dates.csv`。这个结论只覆盖官方优选数据真正引用的来源，不声称来源总表全部条目都有日期。

## 指标、缩写与技术用语对照

为避免同一缩写在不同学科中含义不同，正文和表格均以中文名称为主；英文缩写只在首次出现或需要复现代码时保留。

| 名称或缩写 | 本报告中的中文含义 | 用来判断什么 |
| --- | --- | --- |
| GDP | 国内生产总值（Gross Domestic Product） | 描述地区总体经济规模；本报告只作背景核对，不与旅游目标混合评分。 |
| OLS | 普通最小二乘法（Ordinary Least Squares） | 估计线性或对数线性趋势；本报告中的传统模型是不设断点的对数线性回归。 |
| Ridge | 岭回归（Ridge Regression） | 在最小二乘损失中加入系数惩罚，降低小样本、多解释变量下系数过度波动。 |
| MAE | 平均绝对误差（Mean Absolute Error） | 平均相差多少原单位；游客量与收入单位不同，不能直接合并。 |
| RMSE | 均方根误差（Root Mean Squared Error） | 对大误差惩罚更重；同样只能在同一指标、同一单位内比较。 |
| MAPE | 平均绝对百分比误差（Mean Absolute Percentage Error） | 平均相差实际值的百分之多少；实际值接近零时会不稳定。 |
| sMAPE | 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error） | 用实际值与预测值绝对值之和作分母，便于对游客量和收入进行无量纲比较；越低越好。 |
| 两个目标等权平均 sMAPE | 游客量 sMAPE 与综合收入 sMAPE 的算术平均 | 本报告选主预测模型的核心汇总指标；两个目标各占 50%，不是另一个国际通用缩写。 |
| 相对上一期数值法的改进率 | `1－模型sMAPE÷上一期数值法sMAPE` | 大于0表示优于“下一年等于上一期实际值”的简单基准，小于0表示还不如该基准。 |
| R² / 调整后R² | 决定系数 / 调整后决定系数 | 描述样本内拟合程度；调整后R²对解释变量数量作惩罚，不能替代样本外预测误差。 |
| AICc | 小样本修正赤池信息准则（Corrected Akaike Information Criterion） | 在同一因变量、同一尺度的候选模型间权衡拟合与复杂度；越低越好，不能跨尺度比较。 |
| LOOCV | 留一交叉验证（Leave-One-Out Cross-Validation） | 每次留出一个样本，观察模型对未参与拟合样本的误差；本报告只作同尺度诊断。 |
| DW | 德宾—沃森统计量（Durbin–Watson Statistic） | 检查残差是否存在一阶自相关；明显低于2通常提示正自相关。 |
| JB | 雅克—贝拉正态性检验（Jarque–Bera Test） | 检查残差的偏度和峰度是否偏离正态；本报告样本很小，只作提示，不作“已证明正态”的结论。 |
| 95% CI | 95%置信区间（Confidence Interval） | 表示模型平均响应或参数估计的不确定范围。 |
| 95% PI | 95%预测区间（Prediction Interval） | 表示单个未来观测可能落入的范围，通常比平均响应置信区间宽。 |
| CAGR | 年均复合增长率（Compound Annual Growth Rate） | 把一段时期的首尾变化换算为等效年增长率；跨统计口径断点时不计算。 |
| OAT | 单因素逐次敏感性分析（One-at-a-Time） | 每次只改变一个情景假设，观察结果变化；不同因素扰动幅度不同，不能当作标准化因果贡献。 |
| KPI | 关键绩效指标（Key Performance Indicator） | 把情景结果转成可监测目标；本报告中的KPI是管理建议，不是模型估计出的因果系数。 |

| 报告用语 | 中文解释 |
| --- | --- |
| 官方优选数据（代码中称 `canonical`） | 从多来源中按证据规则选定的年度值；保留来源、状态和修订信息。 |
| 原始证据训练行（代码中称 `physical`） | 来自官方优选数据、没有由本次模型插补生成的训练行。 |
| 模拟训练值（`simulated`） | 仅为补齐训练年份而按训练期信息生成的数值，不是官方事实。 |
| 代理锚值（`proxy`） | 用政府目标或替代资料作情景起点，不是实际观测。 |
| 扩展窗口滚动检验（`expanding-origin`） | 按年份向前推进，每次只用测试年以前的数据训练，再预测下一可观测年份，模拟真实预测顺序。 |
| 上一期数值法（代码中称 `naive_last`） | 直接把最近一期实际值作为下一期预测，是判断复杂模型是否真正有增益的最低基准。 |
| 相同训练年份比较（代码中称 `common-row`） | 两个模型在每一折使用完全相同的有效训练年份，避免由删样规则不同造成不公平。 |
| 单年留出检验（代码中称 `pseudo-holdout`） | 把2024年从本次调参与排序流程中隔离，仅作最后一年检查；由于研究者此前已见过2024年数据，它不是真正前瞻检验。 |
| 跨疫情阶段压力测试 | 只用截至2019年的数据，直接预测疫情及恢复阶段的可观测年份，用来检查模型遇到结构突变时会错到什么程度；它不是疫情效应的因果识别。 |
| 最终修订版回溯数据（代码中称 `final-vintage`） | 使用当前能获得的历史修订值重建过去；不等同于当年预测时实际可获得的数据版本。 |
| 自助法（`bootstrap`） | 对模型残差反复重抽样以近似参数或预测的不确定范围；本报告岭回归固定随机种子并重复10,000次。 |

## 问题1：指标整理与简单增长模型

四项题目指标统一到 2010—2025 年历；“覆盖数”只统计官方优选数据的非空值，不把模型比较中生成的模拟点冒充事实。游客量在 2020—2022 和 2025 缺失，综合收入在 2016、2020、2022、2025 缺失；宏观指标虽覆盖 16/16，但 2019 存在资料口径边界，不能把整段机械解释为同口径因果趋势。国内生产总值（GDP）和第三产业增加值只用于历史背景与预测合理性核对，不参与跨单位模型评分；旅游综合收入是收入总量口径，不是增加值，不能把“综合收入/GDP”解释为旅游增加值贡献率。

清洗规则是：每个指标每年只保留官方优选值，强制转为数值并保留数据状态和来源编号（代码字段 `status/source_ids`）；游客量统一为万人次（代码单位 `10k_persons`），收入、GDP、第三产业增加值统一为亿元（代码单位 `100m_cny`）。缺失保持缺失，不用 0 代替。双侧对数插值只在问题2的模型训练契约中生成，均标记为模拟值（`is_simulated=true`），不是问题1的指标事实或官方观测。

{_markdown_table(problem1_summary_view, list(problem1_summary_view.columns))}

2010—2019年均复合增长率（CAGR）的代码字段 `cagr_2010_2019_percent` 因跨越口径断点而置空：GDP 的 2010—2018 / 2019—2025 分段 CAGR 为 {float(q1_summary_by_metric.loc['jizhou_gdp', 'cagr_2010_2018_percent']):.3f}% / {float(q1_summary_by_metric.loc['jizhou_gdp', 'cagr_2019_2025_percent']):.3f}%，第三产业为 {float(q1_summary_by_metric.loc['jizhou_tertiary_value_added', 'cagr_2010_2018_percent']):.3f}% / {float(q1_summary_by_metric.loc['jizhou_tertiary_value_added', 'cagr_2019_2025_percent']):.3f}%。2018→2019 的 GDP 与第三产业表观变化分别为 {gdp_2019_vs_2018:.3f}% 和 {tertiary_2019_vs_2018:.3f}%，应读作资料口径断裂，不是经济活动骤降。

旅游目标的恢复路径也不是简单回到疫情前趋势：游客量 2023 较 2019 仍为 {visitor_2023_vs_2019:.3f}%，2024 较 2019 为 {visitor_2024_vs_2019:.3f}%，但 2023→2024 同比增长 {float(q1_summary_by_metric.loc['tourist_visits', 'growth_2023_2024_percent']):.3f}%；综合收入 2021 较 2019 为 {income_2021_vs_2019:.3f}%，2023 已较 2019 高 {income_2023_vs_2019:.3f}%，2023→2024 再增长 {float(q1_summary_by_metric.loc['tourism_comprehensive_income', 'growth_2023_2024_percent']):.3f}%。这些断点是后续模型不能只延长疫情前指数曲线的直接证据。

题目1的简单增长模型固定为疫情前指数增长模型（代码标识 `pre_covid_exponential`）：对 2010—2019 年可用官方优选值拟合 `ln(指标值)=β0+β1×(年份−2010)`。参数表给出模型条件标准误、t检验和95%置信区间：

{_markdown_table(problem1_parameter_view, list(problem1_parameter_view.columns))}

{_markdown_table(problem1_diagnostic_view, list(problem1_diagnostic_view.columns))}

诊断量只描述疫情前小样本拟合；决定系数（R²）按对数尺度计算，均方根误差（RMSE）和平均绝对百分比误差（MAPE）按原尺度计算，小样本修正赤池信息准则（AICc）与留一交叉验证（LOOCV）仅供同一目标规格核对，不能证明疫情后的结构稳定性。

若把问题1简单模型不加结构修正地机械外推到 2026—2030，结果如下；点值是对数条件均值指数化后在原尺度上的条件中位数，不是经过对数正态偏差修正的算术均值，也不作为问题2主预测：

{_markdown_table(problem1_forecast_view, list(problem1_forecast_view.columns))}

![问题1四项指标](../outputs/unified_model_benchmark/q1_required_indicators.png)

## 问题2：模型评判与 2026—2030 预测

模型形式先由截至2023年的统一滚动检验冻结，再把2024年官方优选目标值加入最终系数重拟合。最终训练契约为每个目标2010—2024共15个年度位置：12个原始证据值和3个训练期内双侧对数插值；**不生成、不读取、不使用2025年目标值**。其中2010年综合收入虽是官方来源的非模拟行，但状态为“由同比反推”（代码状态 `inferred_from_yoy`），不是严格意义上的直接观测。

模型评判首先采用两个题面目标的未模拟官方数据扩展窗口滚动检验（`expanding-origin`，每个目标6个外层实际测试点）：

{_markdown_table(problem2_validation_view, list(problem2_validation_view.columns))}

问题1的疫情前指数增长模型在游客量/综合收入上的未模拟滚动对称平均绝对百分比误差（sMAPE）分别为 {float(official_by_target.loc[(official_by_target['model'].eq('pre_covid_exponential')) & (official_by_target['metric'].eq('tourist_visits')), 'smape_percent'].iloc[0]):.3f}% / {float(official_by_target.loc[(official_by_target['model'].eq('pre_covid_exponential')) & (official_by_target['metric'].eq('tourism_comprehensive_income')), 'smape_percent'].iloc[0]):.3f}%，固定的原尺度岭回归（Ridge）则为 {float(official_by_target.loc[(official_by_target['model'].eq('raw_target_ridge_alpha_0.1')) & (official_by_target['metric'].eq('tourist_visits')), 'smape_percent'].iloc[0]):.3f}% / {float(official_by_target.loc[(official_by_target['model'].eq('raw_target_ridge_alpha_0.1')) & (official_by_target['metric'].eq('tourism_comprehensive_income')), 'smape_percent'].iloc[0]):.3f}%；这直接说明问题1疫情前简单模型在疫情后不再适合作为主预测。它机械外推到2026年已达游客量 {float(q1_forecast_keyed.loc[('tourist_visits', 2026)]):.1f} 万人次、收入 {float(q1_forecast_keyed.loc[('tourism_comprehensive_income', 2026)]):.1f} 亿元，到2030年达 {float(q1_forecast_keyed.loc[('tourist_visits', 2030)]):.1f} 万人次和 {float(q1_forecast_keyed.loc[('tourism_comprehensive_income', 2030)]):.1f} 亿元，远离恢复期实际与政策情景尺度。

在未模拟方案上，固定原尺度岭回归的两个目标等权平均sMAPE为 {float(official_comparison.loc[official_comparison['model'].eq('raw_target_ridge_alpha_0.1'), 'macro_smape_percent'].iloc[0]):.3f}%，使用相同训练年份的无断点对数线性普通最小二乘模型（OLS）为 {float(official_comparison.loc[official_comparison['model'].eq('no_break_log_linear_common_rows'), 'macro_smape_percent'].iloc[0]):.3f}%；在模拟增强方案上二者分别为 {float(simulated_comparison.loc[simulated_comparison['model'].eq('raw_target_ridge_alpha_0.1'), 'macro_smape_percent'].iloc[0]):.3f}% 和 {float(simulated_comparison.loc[simulated_comparison['model'].eq('no_break_log_linear_common_rows'), 'macro_smape_percent'].iloc[0]):.3f}%。但岭回归在游客量上的未模拟sMAPE {float(official_by_target.loc[(official_by_target['model'].eq('raw_target_ridge_alpha_0.1')) & (official_by_target['metric'].eq('tourist_visits')), 'smape_percent'].iloc[0]):.3f}% 仍略高于上一期数值法的 {float(official_by_target.loc[(official_by_target['model'].eq('naive_last')) & (official_by_target['metric'].eq('tourist_visits')), 'smape_percent'].iloc[0]):.3f}%，因此不能称为全目标稳健赢家。这里并列给出两个冻结模型的未来路径，供趋势型与恢复特征型假设交叉核对。所有选型只比较同一目标原尺度上的外层滚动sMAPE；下方在不同因变量尺度和留一交叉验证尺度上计算的拟合诊断，不能横向替代该选型依据。

两种固定规格分别是使用相同训练年份的无断点对数线性普通最小二乘模型（OLS）和原尺度岭回归（代码标识 `raw_target_ridge_alpha_0.1`）。OLS区间直接采用分支源码的学生t分布公式，是平均对数响应区间指数化后的结果；岭回归使用固定种子 `{RANDOM_SEED}` 的 {RIDGE_BOOTSTRAP_REPETITIONS:,} 次固定设计残差自助法（bootstrap）。两者都把插值当作给定训练值，都是模型条件区间，不保证在重复抽样意义下达到95%覆盖率，也不是五年同时置信带。

{_markdown_table(problem2_forecast_view, list(problem2_forecast_view.columns))}

{_markdown_table(problem2_diagnostic_view, list(problem2_diagnostic_view.columns))}

表中的“因变量计算尺度”和“留一交叉验证计算尺度”（代码字段 `target_scale/loocv_scale`）说明：普通最小二乘模型的决定系数（R²）、小样本修正赤池信息准则（AICc）与留一交叉验证（LOOCV）在对数尺度计算，岭回归在原尺度计算，二者及两个不同单位的预测目标之间不可直接比较。OLS点预测是对数条件均值指数化后的原尺度条件中位数；没有做对数正态均值修正。

岭回归的标准化参数及自助法（bootstrap）百分位区间如下；没有为岭回归系数伪造t检验或p值：

{_markdown_table(problem2_parameter_view, list(problem2_parameter_view.columns))}

岭回归的解释变量固定为时间趋势项（`year_index=(year-2010)/10`）、2020—2022年疫情期指示变量（`pandemic_2020_2022`）和2023年后恢复期指示变量（`post_2022`），再按最终训练样本标准化。系数只用于预测，不是因果效应；最终2020—2022年目标值是训练期双侧对数插值，因此疫情期指示变量尤其受模拟路径驱动，不能解释为已识别的疫情冲击。

![问题2模型评判](../outputs/unified_model_benchmark/q2_model_judgement.png)

![问题2预测](../outputs/unified_model_benchmark/q2_forecast_2026_2030.png)

**题目导向的选模结论：** 两种统一滚动检验方案中，固定原尺度岭回归的两个目标等权平均sMAPE都低于无断点相同训练年份OLS，因此把原尺度岭回归（代码标识 `raw_target_ridge_alpha_0.1`）作为2026—2030年的主点预测：游客量从 {ridge_2026_visits:.1f} 万人次增至 {ridge_2030_visits:.1f} 万人次，综合收入从 {ridge_2026_income:.2f} 亿元增至 {ridge_2030_income:.2f} 亿元；普通最小二乘模型保留为问题1简单增长模型和问题2的解释性趋势对照。到2030年，OLS给出 {ols_2030_visits:.1f} 万人次和 {ols_2030_income:.2f} 亿元，明显高于岭回归，原因是OLS把全期平均对数增长持续外推，而岭回归在原尺度上同时估计时间趋势、疫情期和2023年后恢复水平，外推更平缓。两条路径都为正且随时间增长，但OLS区间更宽、对长期指数趋势更敏感；再考虑岭回归的游客量未模拟回测没有胜过上一期数值法、样本很小，主预测只是相对更审慎的固定规格，不能称为稳健胜者。

合理性衔接上，岭回归2030年游客量 {ridge_2030_visits:.1f} 万人次和收入 {ridge_2030_income:.2f} 亿元均落在问题3三情景包络（游客量 {scenario_2030_envelope['tourist_visits'][0]:.1f}—{scenario_2030_envelope['tourist_visits'][1]:.1f} 万人次；收入 {scenario_2030_envelope['tourism_comprehensive_income'][0]:.2f}—{scenario_2030_envelope['tourism_comprehensive_income'][1]:.2f} 亿元）内，但呈现“客流高于政策基准、收入低于政策基准”的组合。其隐含名义人均次消费由2026年 {ridge_2026_implied_spend:.1f} 元降至2030年 {ridge_2030_implied_spend:.1f} 元，相比2024年官方优选值的 {actual_2024_spend:.1f} 元下降 {abs(ridge_spend_2030_vs_2024):.2f}%；因此把岭回归作为偏保守风险主线时，应同步监测客单价，而不能只看客流。

## 问题3：政策锚定三情景与敏感性

三情景沿用已合并分支的透明政策锚定口径：2025年游客量2800万人次、综合收入231亿元均是政府目标代理锚值（proxy），**不是实际观测**；对应名义人均次消费为825元/人次。基准情景取收入年增8%、人均消费年增3%；乐观情景取12%/4%；悲观情景先在2026年施加收入水平冲击−15%，随后收入年增5%、人均消费年增2%。所有游客量均由“收入=游客量×人均次消费÷10000”的恒等式反推。

{_markdown_table(scenario_view, list(scenario_view.columns))}

单因素逐次敏感性分析（OAT）只围绕基准情景，每次只改变一项假设：客源年增速±2个百分点、人均消费年增速±1个百分点、2026年政策协同水平乘数±5%，以及突发冲击从0到−15%。水平乘数和冲击在该恒等式中都表现为持续水平位移，不能解释成两个独立可加机制。

各因素的扰动宽度和单位不同（±2 个百分点、±1 个百分点、±5%、0 至 −15%），所以图中的影响幅度不是标准化弹性，不能据此作严格“杠杆效率”排序；这里只展示在指定假设幅度下的非因果压力结果。

{_markdown_table(sensitivity_view, list(sensitivity_view.columns))}

量化建议（均为情景计算，不是历史因果效应）：

- **客源拓展关键绩效指标（KPI）**：把隐含客源年增速从4.854%提高到6.854%；情景计算的2030年游客量为 {source_high_visits['scenario_2030']:.1f} 万人次，较基准增加 {source_high_visits['delta_2030']:.1f} 万人次，收入增加 {source_high_income['delta_2030']:.2f} 亿元。该增量是单因素逐次敏感性分析假设结果，不证明投放会因果地产生同等增量。
- **新业态消费关键绩效指标（KPI）**：把名义人均次消费年增速从3%提高到4%；在客源路径不变时，2030年收入为 {spend_high_income['scenario_2030']:.2f} 亿元，较基准增加 {spend_high_income['delta_2030']:.2f} 亿元。该计算未识别业态投资的历史弹性或成本。
- **政策协同关键绩效指标（KPI）**：以2026年路径水平乘数1.05作压力目标；2030年游客量和收入分别较基准增加 {coordination_high_visits['delta_2030']:.1f} 万人次、{coordination_high_income['delta_2030']:.2f} 亿元。乘数是外生假设，不能当作政策因果系数。
- **风险预案关键绩效指标（KPI）**：监测预订量、客单价和交通可达性，使2026年冲击幅度尽量高于−15%；若形成持续−15%水平缺口，2030年游客量和收入将比基准少 {abs(shock_low_visits['delta_2030']):.1f} 万人次、{abs(shock_low_income['delta_2030']):.2f} 亿元。损失仅为压力测试，不是风险概率预测。
- **年度校准关键绩效指标（KPI）**：以基准2030年的 {float(baseline_2030['tourist_visits']):.1f} 万人次和 {float(baseline_2030['tourism_comprehensive_income']):.2f} 亿元作为可滚动修订的假设锚，而非硬承诺；每年用新实际更新偏差。锚值来自2025年政府目标代理值和设定增速，不具有因果或概率保证。

![问题3情景与敏感性](../outputs/unified_model_benchmark/q3_scenarios_sensitivity.png)
"""
    content = f"""# C题：蓟州区旅游经济趋势预测与对策分析——统一分支模型比较报告

{problem_sections}

## 附录：统一分支模型回测与审计

### 结论先行

**不能判定原分支赢家。** 原传统分支声明的主模型是2023年后水平断点模型（`post_2022_level_break`），而用户本轮统一协议明确不运行任何断点模型；原机器学习分支声明的是分阶段岭回归（`ridge_regime`）模型族和原尺度岭回归（`raw_target_ridge_alpha_0.1`）点预测路径。因此原声明代表没有一组可共同排名的滚动预测。

在用户指定的模拟增强训练方案上，描述性第一是{REPORT_VALUE_LABELS.get(str(simulated_first['model']), str(simulated_first['model']))}（两个目标等权平均sMAPE为 {simulated_first['macro_smape_percent']:.3f}%）；在更适合作为稳健性依据的未模拟官方数据相同训练年份方案上，描述性第一是{REPORT_VALUE_LABELS.get(str(official_first['model']), str(official_first['model']))}（{official_first['macro_smape_percent']:.3f}%）。{ordering_sentence} 这些表只比较“相同训练年份上的固定模型规格”，不是原分支胜负。

模拟增强方案是假设性分析，不是真实观测证据：对数插值和对数增长率尾部外推在结构上更贴近对数线性普通最小二乘模型，而且会把游客量2020—2022年的未知疫情路径平滑成趋势点。报告因此优先用未模拟、相同训练年份的方案判断稳健性，并把两种方案并列展示。

滚动稳定性框架采用截至2023年的扩展窗口滚动外层检验（`expanding-origin`）；2024年仅作最终单年留出检验（代码中称 `pseudo-holdout`，每个目标测试点数为1），不用于本次执行的排序。模型族代码是在2024年数据已经存在后形成，统一数据又是最终修订版回溯数据（`final-vintage`），因此2024年不是研究设计层真正“未见”的前瞻测试。2019年截断结果仅作跨疫情及恢复阶段压力测试，不与单年留出检验混称。

### 分支数据合并结论

- 共审计并合入5个唯一分支版本。`main`、传统模型分支和机器学习分支的核心数据目录完全相同，因此没有重复拼接同一批年度观测。
- `origin/111` 的独有天津市 GDP（2010—2025）与天津市旅游基准已规范化为独立辅助表；天津市口径不会覆盖蓟州区 GDP，也不直接充当蓟州目标标签。
- `origin/邱志烨-数据搜索` 的8条补全值保存在 `sensitivity_imputations.csv`，全部标记为非观测并排除在统一训练、测试和排名之外；其全样本标准化、异常检测等派生列也没有直接进入回测。
- 两个分支中的旧版工作簿、预测和情景交付均保留在Git历史和原路径，但不会覆盖集成前 `main` 固定提交的官方优选真值。逐文件来源、Git对象标识、SHA-256哈希和纳入决策见 `data/unified/branch_data_inventory.csv`。

### 可比协议

- 随机种子固定为 `{RANDOM_SEED}`；所有模型读取同一统一数据层。
- 滚动外层最小训练记录数为 5，游客量测试年为 `{rolling_years['tourist_visits']}`，综合收入测试年为 `{rolling_years['tourism_comprehensive_income']}`；外层测试最晚到 2023 年。
- 每折模拟只读取该折测试年前已存在的原始证据训练行：内部缺口用两侧训练边界在对数尺度插值；训练尾部到测试年前一年，用最近至多3个训练区间的年化对数增长率中位数外推。逐点方法、源年、边界见 `simulated_training_points.csv`；绝不读取外层测试、未来官方值或邱分支的辅助数据表。
- 固定主比较规格是传统分支的相同训练年份无断点对数线性回归（`no_break_log_linear_common_rows`）与机器学习分支的原尺度岭回归（`raw_target_ridge_alpha_0.1`）。两者逐折使用完全相同的原始证据行、模拟行和最终有效行；脚本运行时强制验证这一不变量。
- 机器学习全部候选模型及新增的训练内部模型选择器（`ml_inner_selector`）只作探索；其预处理、调参和选择均限定在每个外层折训练内部，但不能据其事后名次宣布分支赢家。
- 2024年单年留出检验的训练集每个目标记录数为 `{train_counts}`，测试集严格为游客量和综合收入各一条2024年实际值；隔离只成立于本次重跑的执行流程。
- `data/unified/primary_train.csv` 是2024年前的原始证据层；`outputs/unified_model_benchmark/primary_train_augmented.csv` 是用户指定的2010—2023年建模训练层，包含原始证据行与模拟行，并保留是否模拟、生成方法和已知数据截止年等代码字段，可直接与 `data/unified/primary_test.csv` 配对。
- 主排序指标是先按目标计算对称平均绝对百分比误差（sMAPE），再对两个目标各赋50%权重；不跨单位汇总均方根误差（RMSE）。表中同时给出相对上一期数值法（`naive_last`）的改进率与最差单点误差。
- 传统分支的2023年后水平断点模型（`post_2022_level_break`）和严格证据口径水平断点模型（`strict_evidence_level_break`）在所有评价范围内都按本次约定不执行，不生成预测或误差。原生疫情前指数增长模型和会排除2020—2022年训练行的无断点对数线性模型，仅保留在未模拟官方数据敏感性表。

### 用户指定的模拟增强固定模型排序（假设性）

{_markdown_table(simulated_view, list(simulated_view.columns))}

该表的“描述性排序”只描述用户指定规格在模拟伪标签上的相对误差；“稳健胜者”标志固定为否。模拟点会改变目标路径，不能当成新增事实或原分支的历史声明流程。

### 未模拟官方数据的相同训练年份排序（稳健性优先）

{_markdown_table(official_view, list(official_view.columns))}

这里不生成任何模型比较伪标签。传统分支的相同训练年份普通最小二乘模型与原尺度岭回归都使用每一折全部未模拟官方来源训练行，测试仍只用官方实际值；官方优选数据可包含官方来源的同比反推、回列或辅助值，并用数据状态和是否直接观测字段（`status/is_observed`）明示，例如2010年综合收入是“由同比反推”（`inferred_from_yoy`），不是严格意义上的直接观测。该方案优先于模拟增强方案，用于判断结论是否由伪标签驱动；即使两种方案顺序一致，也只能称固定规格的相对排序。

### 两轨逐目标核对

{_markdown_table(track_target_view, list(track_target_view.columns))}

### 原分支声明方法核对

{_markdown_table(declared_view, list(declared_view.columns))}

传统原声明模型没有执行值，因此原声明代表不能共同排名（代码字段 `jointly_rankable_original_declared_representatives=false`）。原尺度岭回归本次每折固定惩罚参数α=0.1且不读取外层测试，但原机器学习分支推荐它时已检查包含2024年的回测；分阶段岭回归（`ridge_regime`）仅作模型族声明审计。因此没有合法的原分支级胜者。

### 全候选探索性排名

{_markdown_table(exploratory_view, list(exploratory_view.columns))}

全候选表用于诊断，不可从中事后挑一个最优候选再宣称无偏胜者。并列通过1e-10精度判定；最差单点sMAPE用于暴露平均值可能掩盖的失稳。两种断点模型无论代数上是否可识别，都因用户协议禁止而不执行、不展示误差。

### 未模拟官方数据的分支原生策略敏感性

{_markdown_table(native_view, list(native_view.columns))}

疫情前指数增长模型（`pre_covid_exponential`）与原生无断点对数线性回归（`no_break_log_linear`）会删除部分统一训练行，故只能用来解释“分支原生过滤策略会怎样”，不能与相同训练年份的固定规格混作纯算法比较。

### 2024年最终单年留出检验（非真正前瞻）

{_markdown_table(holdout_view, list(holdout_view.columns))}

2024年单年留出检验每个目标只有一个测试点，此时平均绝对误差（MAE）就等于该点的绝对误差，平均绝对百分比误差（MAPE）和对称平均绝对百分比误差（sMAPE）都没有稳定性含义；它只检查本次重跑中冻结后的方法能否跨到下一年。两种断点模型仍不执行。任何2024年误差都没有回流到本次重跑的超参数、候选选择或主排序；但由于模型代码和方法讨论形成时2024年数据已存在，它不能支持真正的前瞻泛化声明。

### 2019 截断跨阶段压力测试（不含 2024）

{_markdown_table(stress_view, list(stress_view.columns))}

压力测试的原始证据训练数据只到2019年，测试使用2020—2023年间现有的官方实际值：综合收入为2021年和2023年，游客量为2023年；每个测试点单独生成截至测试年前一年的模拟尾部，2021年实际值不会流入2023年压力测试训练。机器学习模型的疫情期和2023年后恢复期指示变量在原始训练期中没有出现过变化，输出属于跨阶段外推，不等于识别疫情或恢复效应。

### 模型源码固定

{_markdown_table(provenance_view, list(provenance_view.columns))}

比较程序在建模前校验两个可执行模型文件的Git对象标识与SHA-256哈希；任一字节漂移都会拒绝运行。原声明摘要也做字段校验。

### 分支覆盖与未纳入原因

- `codex/jizhou-tourism-modeling`：存在可执行Python实现；本次统一方案只运行相同训练年份的无断点普通最小二乘模型，原生疫情前/无断点规格仅作敏感性分析，所有断点模型均不执行。
- `codex/jizhou-tourism-ml`：存在可执行Python实现，实际运行全部7个候选模型族、稳健集成模型、原尺度岭回归、上一期数值法和训练内部模型选择器。
- `origin/111`：只有示例论文和工作簿，**缺少可执行模型源代码**；其中数据只能作统一数据层的旁证，不能把论文数值冒充为同一训练—测试划分下的重跑结果。
- `origin/邱志烨-数据搜索`：有数据预处理、补值和模型建议，**没有模型实现**；补值辅助表明确排除在目标标签、外层训练和测试之外。
- `main`：提供官方优选目标真值并接收最终报告；没有另一套可独立执行的分支模型。

### 复现与限制

输入模式代码为 `{inputs.source_mode}`。数据读取程序会逐行验证原始证据训练—测试划分及滚动折，与 `benchmark_observations.csv` 中的目标值、状态、来源、观测标志、年份和训练/测试角色完全一致。`train_only_hyperparameters.csv` 可核对只在训练期内确定的超参数；`model_applicability.csv` 保留禁用断点模型与机器学习模型支持状态；逐点预测可复算所有指标。

获取日期核对严格限定在官方优选年度汇总表实际引用的 {len(canonical_source_access)} 个唯一来源：它们在 `data/metadata/sources.csv` 的备注字段中都记录实际获取日期为 {canonical_access_dates}，详见 `problem_source_access_dates.csv`。未被官方优选数据使用的来源元数据不在这项完整性声明内。

```bash
.venv/bin/python scripts/build_unified_branch_data.py
MPLCONFIGDIR=/tmp/jizhou-mpl XDG_CACHE_HOME=/tmp/jizhou-xdg .venv/bin/python code/scripts/compare_branch_models.py
.venv/bin/python -m unittest discover -s tests -v
```

样本总量很小、测试年份不规则，2020—2022年又存在目标缺口；官方优选数据还是包含同比反推和回列修订的最终修订版回溯数据。模拟伪标签会低估结构冲击，不增加信息量；对称平均绝对百分比误差（sMAPE）和时间外推也无法替代结构解释。应联合查看未模拟官方数据相同训练年份方案、模拟增强方案、逐目标相对上一期数值法改进率、最差误差、2019年跨疫情阶段压力测试和2024年单年留出检验，不能引用“稳健冠军”或“原分支赢家”结论。

## 题目导向综合结论

- **问题1：增长背景。** 疫情前指数增长模型估计游客量年增长 {float(q1_growth_by_metric['tourist_visits']):.3f}%、综合收入年增长 {float(q1_growth_by_metric['tourism_comprehensive_income']):.3f}%；2023—2024年官方优选实际值的增长率分别为 {float(q1_summary_by_metric.loc['tourist_visits', 'growth_2023_2024_percent']):.3f}% 和 {float(q1_summary_by_metric.loc['tourism_comprehensive_income', 'growth_2023_2024_percent']):.3f}%。恢复仍为正但结构已变，问题1曲线机械外推到2030年会达到 {float(q1_forecast_keyed.loc[('tourist_visits', 2030)]):.1f} 万人次和 {float(q1_forecast_keyed.loc[('tourism_comprehensive_income', 2030)]):.1f} 亿元，故只能作反例对照；国内生产总值（GDP）和第三产业增加值按2019年口径断点分段核对，综合收入不是增加值。
- **问题2：主预测与不确定性。** 两种统一滚动检验方案均由固定原尺度岭回归取得较低的两个目标等权平均sMAPE，故主点预测采用该规格：游客量 {ridge_2026_visits:.1f}→{ridge_2030_visits:.1f} 万人次，综合收入 {ridge_2026_income:.2f}→{ridge_2030_income:.2f} 亿元；无断点普通最小二乘模型作为更陡的趋势对照。岭回归的游客量未模拟回测仍略逊上一期数值法，且其2030年隐含人均次消费较2024年低 {abs(ridge_spend_2030_vs_2024):.2f}%，所以结论是“带客单价下行风险的相对审慎主线”，不是稳健冠军。
- **问题3：情景与行动。** 政策基准到2030年为 {float(baseline_2030['tourist_visits']):.1f} 万人次和 {float(baseline_2030['tourism_comprehensive_income']):.2f} 亿元；在指定扰动下，客源年增速提高2个百分点对应游客量增加 {source_high_visits['delta_2030']:.1f} 万人次、收入增加 {source_high_income['delta_2030']:.2f} 亿元，同时应为−15%持续冲击下的 {abs(shock_low_income['delta_2030']):.2f} 亿元收入缺口准备预案。各单因素逐次敏感性分析（OAT）的扰动宽度不同，不能当作标准化杠杆排名；全部增量来自2025年政府目标代理值与透明假设，不代表政策因果效应或实现概率。
"""
    report_path.write_text(content, encoding="utf-8")


def _write_output_readme(output_dir: Path) -> None:
    content = """# Unified model benchmark artifacts

This directory is generated by `code/scripts/compare_branch_models.py`.

- `problem1_indicator_summary.csv`, `problem1_simple_growth_parameters.csv`, `problem1_simple_growth_diagnostics.csv`, and `problem1_simple_growth_forecasts_2026_2030.csv` answer Q1 with four-indicator coverage and the 2010-2019 `pre_covid_exponential` simple model; the forecast is a mechanical comparison path, not the selected Q2 forecast.
- `problem_source_access_dates.csv` lists the exact canonical-cited sources whose retrieval dates are verified from `data/metadata/sources.csv`.
- `problem2_final_training_2010_2024.csv` is the Q2 final-refit contract. It contains only 2010-2024 canonical/internal-interpolation rows and never uses the 2025 target proxy.
- `problem2_forecasts_2026_2030.csv`, `problem2_final_model_diagnostics.csv`, and `problem2_ridge_standardized_parameters.csv` contain the two frozen Q2 models, model-conditional intervals, and Ridge bootstrap inference.
- `problem3_scenario_forecasts_2026_2030.csv` and `problem3_policy_sensitivity.csv` are transparent, non-causal policy-anchor calculations; the 2025 anchor is a government-target proxy, not an actual observation.
- `q1_required_indicators.png`, `q2_model_judgement.png`, `q2_forecast_2026_2030.png`, and `q3_scenarios_sensitivity.png` are the fixed report figures.
- `stability_rolling_predictions.csv` and `stability_simulated_augmentation_*` are the user-directed, hypothetical simulated-augmentation rolling track through 2023.
- `stability_official_only_common_row_*` is the no-simulation common-effective-row robustness track.
- `stability_official_only_native_sensitivity_*` retains branch-native pre-COVID / 2020-2022 filtering only as sensitivity analysis.
- `stability_predeclared_representatives.csv` audits original branch declarations; those representatives are not jointly rankable.
- `simulated_training_points.csv` records every physical and simulated training point, method, source years, boundaries, and outer-test cutoff.
- `primary_train_augmented.csv` is the directly reusable 2010-2023 modeling contract paired with `data/unified/primary_test.csv`; `data/unified/primary_train.csv` remains the evidence-layer contract.
- `final_holdout_2024_*` is an execution-isolated one-point-per-target pseudo-holdout, not a genuinely prospective holdout.
- `cross_regime_stress_*` is a 2019-cutoff diagnostic and excludes 2024.
- `model_source_provenance.csv` pins and validates executable model source blobs.
- `model_applicability.csv`, `train_only_hyperparameters.csv`, and point predictions provide the audit trail.

Breakpoint models are never executed under the user protocol and have no prediction/error rows.

```bash
.venv/bin/python scripts/build_unified_branch_data.py
MPLCONFIGDIR=/tmp/jizhou-mpl XDG_CACHE_HOME=/tmp/jizhou-xdg .venv/bin/python code/scripts/compare_branch_models.py
.venv/bin/python -m unittest discover -s tests -v
```
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def run_benchmark(
    *,
    unified_dir: Path = DEFAULT_UNIFIED_DIR,
    canonical_path: Path = DEFAULT_CANONICAL,
    output_dir: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, object]:
    """Execute all frozen evaluations and write deterministic audit artifacts."""
    np.random.seed(RANDOM_SEED)
    source_provenance = validate_model_source_provenance(ROOT)
    original_declarations = _validate_original_declarations(ROOT)
    inputs = load_benchmark_inputs(unified_dir, canonical_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_canonical_path = unified_dir / "canonical_official_annual_2010_2025.csv"
    if not problem_canonical_path.is_file():
        problem_canonical_path = canonical_path
    problem_canonical = pd.read_csv(problem_canonical_path)
    source_metadata_path = ROOT / "data/metadata/sources.csv"
    canonical_source_access = build_canonical_source_access_audit(
        problem_canonical, pd.read_csv(source_metadata_path)
    )
    problem1_indicator_summary = build_problem1_indicator_summary(problem_canonical)
    (
        problem1_growth_parameters,
        problem1_growth_diagnostics,
    ) = build_problem1_simple_growth_outputs(problem_canonical)
    problem1_growth_forecasts = build_problem1_simple_growth_forecasts(
        problem_canonical
    )

    (
        rolling_predictions,
        rolling_applicability,
        rolling_tuning,
        rolling_simulated_points,
    ) = _evaluate_rolling_track(
        inputs.rolling_folds,
        scope="user_simulated_augmentation_rolling",
        simulate_missing_years=True,
        include_native_sensitivities=False,
        ml_roster="full",
    )
    rolling_by_target, rolling_macro = _summarize_scope(rolling_predictions)
    exploratory = _exploratory_ranking(
        rolling_macro, "user_simulated_augmentation_exploratory_all_models"
    )
    simulated_comparison = _adapter_comparison_table(
        rolling_macro,
        ranking_scope="user_simulated_augmentation_fixed_adapters",
        training_track="common physical rows plus fold-local simulated annual rows",
    )
    _validate_common_row_adapter_inputs(
        rolling_predictions, scope="user_simulated_augmentation_rolling"
    )

    (
        official_predictions,
        official_applicability,
        official_tuning,
        _,
    ) = _evaluate_rolling_track(
        inputs.rolling_folds,
        scope="official_only_common_row_rolling",
        simulate_missing_years=False,
        include_native_sensitivities=True,
        ml_roster="fixed_representatives",
    )
    official_by_target, official_macro = _summarize_scope(official_predictions)
    official_comparison = _adapter_comparison_table(
        official_macro,
        ranking_scope="official_only_common_effective_rows_fixed_adapters",
        training_track="physical canonical training rows only; no simulated labels",
    )
    _validate_common_row_adapter_inputs(
        official_predictions, scope="official_only_common_row_rolling"
    )

    simulated_first = set(
        zip(
            simulated_comparison.loc[
                simulated_comparison["descriptive_rank"].eq(1), "branch"
            ],
            simulated_comparison.loc[
                simulated_comparison["descriptive_rank"].eq(1), "model"
            ],
            strict=True,
        )
    )
    official_first = set(
        zip(
            official_comparison.loc[
                official_comparison["descriptive_rank"].eq(1), "branch"
            ],
            official_comparison.loc[
                official_comparison["descriptive_rank"].eq(1), "model"
            ],
            strict=True,
        )
    )
    cross_track_consistent = simulated_first == official_first
    winner_reason = (
        "original declared representatives are not jointly executable under the user protocol; "
        "adapter ordering is consistent but remains descriptive"
        if cross_track_consistent
        else "original declared representatives are not jointly executable and adapter ordering reverses across tracks"
    )
    for comparison in (simulated_comparison, official_comparison):
        comparison["cross_track_order_consistent"] = cross_track_consistent
        comparison["winner_determination"] = "unavailable"
        comparison["winner_determination_reason"] = winner_reason

    declared = _declared_representative_table(rolling_macro, official_macro)
    ranking_status = _ranking_status(
        rolling_predictions,
        rolling_applicability,
        exploratory,
        simulated_comparison,
    )

    (
        holdout_predictions,
        holdout_applicability,
        holdout_tuning,
        holdout_simulated_points,
    ) = _evaluate_augmented_fixed_scope(
        inputs.primary_train,
        inputs.primary_test,
        scope="user_simulated_augmentation_holdout_2024",
        split_id=PRIMARY_SPLIT_ID,
    )
    if not holdout_predictions["test_year"].eq(2024).all():
        raise AssertionError("final holdout contains a non-2024 point")
    holdout_by_target, holdout_macro = _summarize_scope(holdout_predictions)
    holdout_macro = holdout_macro.drop(
        columns=["rank", "tie_group_size", "stability_flag"], errors="ignore"
    )
    holdout_macro["selection_use"] = "diagnostic only; n=1 per target; never tune/select"

    (
        stress_predictions,
        stress_applicability,
        stress_tuning,
        stress_simulated_points,
    ) = _evaluate_augmented_fixed_scope(
        inputs.stress_train,
        inputs.stress_test,
        scope="user_simulated_augmentation_stress_cutoff_2019",
        split_id=STRESS_SPLIT_ID,
    )
    if stress_predictions["test_year"].ge(2024).any():
        raise AssertionError("2019 stress evaluation includes the 2024 holdout")
    stress_by_target, stress_macro = _summarize_scope(stress_predictions)
    stress_macro = stress_macro.drop(
        columns=["rank", "tie_group_size", "stability_flag"], errors="ignore"
    )
    stress_macro["selection_use"] = "cross-regime diagnostic only; not primary ranking"

    applicability = pd.concat(
        [
            rolling_applicability,
            official_applicability,
            holdout_applicability,
            stress_applicability,
        ],
        ignore_index=True,
    )
    tuning_frames = [
        frame
        for frame in (rolling_tuning, official_tuning, holdout_tuning, stress_tuning)
        if not frame.empty
    ]
    tuning = pd.concat(tuning_frames, ignore_index=True)
    if (
        tuning["evaluation_scope"].eq("user_simulated_augmentation_holdout_2024")
        & tuning["outer_train_end_year"].gt(2023)
    ).any():
        raise AssertionError("2024 leaked into holdout tuning")
    simulated_points = pd.concat(
        [rolling_simulated_points, holdout_simulated_points, stress_simulated_points],
        ignore_index=True,
    )
    if not simulated_points["year"].lt(simulated_points["test_year"]).all():
        raise AssertionError("simulated training audit reaches the outer test or future")
    if not simulated_points["known_through_year"].lt(
        simulated_points["test_year"]
    ).all():
        raise AssertionError("augmentation boundary sees the outer test or future")
    inventory = _branch_inventory(source_provenance)
    primary_train_augmented = holdout_simulated_points.copy()
    primary_train_augmented.insert(0, "split_id", PRIMARY_SPLIT_ID)
    primary_train_augmented.insert(1, "split", "train_augmented")
    primary_train_augmented.insert(2, "cutoff_year", 2023)
    primary_train_augmented.sort_values(
        ["metric", "year"], inplace=True, ignore_index=True
    )
    official_native_macro = official_macro.copy()
    official_native_macro["selection_use"] = (
        "no-simulation canonical sensitivity; native filtering is not a common-row algorithm comparison"
    )

    (
        problem2_final_training,
        problem2_forecasts,
        problem2_diagnostics,
        problem2_ridge_parameters,
    ) = build_problem2_outputs(inputs.observations)
    problem3_scenarios = build_problem3_scenario_forecasts()
    problem3_sensitivity = build_problem3_policy_sensitivity(problem3_scenarios)

    csv_outputs: dict[str, pd.DataFrame] = {
        "stability_rolling_predictions.csv": rolling_predictions,
        "stability_simulated_augmentation_predictions.csv": rolling_predictions,
        "stability_metrics_by_target.csv": rolling_by_target,
        "stability_exploratory_model_ranking.csv": exploratory,
        "stability_clean_branch_ranking.csv": simulated_comparison,
        "stability_simulated_augmentation_adapter_comparison.csv": simulated_comparison,
        "stability_official_only_common_row_predictions.csv": official_predictions,
        "stability_official_only_common_row_metrics_by_target.csv": official_by_target,
        "stability_official_only_common_row_comparison.csv": official_comparison,
        "stability_official_only_native_sensitivity_macro.csv": official_native_macro,
        "stability_predeclared_representatives.csv": declared,
        "stability_ranking_status.csv": ranking_status,
        "final_holdout_2024_predictions.csv": holdout_predictions,
        "final_holdout_2024_metrics_by_target.csv": holdout_by_target,
        "final_holdout_2024_macro_diagnostic.csv": holdout_macro,
        "cross_regime_stress_predictions.csv": stress_predictions,
        "cross_regime_stress_metrics_by_target.csv": stress_by_target,
        "cross_regime_stress_macro_diagnostic.csv": stress_macro,
        "model_applicability.csv": applicability,
        "train_only_hyperparameters.csv": tuning,
        "simulated_training_points.csv": simulated_points,
        "primary_train_augmented.csv": primary_train_augmented,
        "model_source_provenance.csv": source_provenance,
        "branch_implementation_inventory.csv": inventory,
        "problem1_indicator_summary.csv": problem1_indicator_summary,
        "problem_source_access_dates.csv": canonical_source_access,
        "problem1_simple_growth_parameters.csv": problem1_growth_parameters,
        "problem1_simple_growth_diagnostics.csv": problem1_growth_diagnostics,
        "problem1_simple_growth_forecasts_2026_2030.csv": problem1_growth_forecasts,
        "problem2_final_training_2010_2024.csv": problem2_final_training,
        "problem2_forecasts_2026_2030.csv": problem2_forecasts,
        "problem2_final_model_diagnostics.csv": problem2_diagnostics,
        "problem2_ridge_standardized_parameters.csv": problem2_ridge_parameters,
        "problem3_scenario_forecasts_2026_2030.csv": problem3_scenarios,
        "problem3_policy_sensitivity.csv": problem3_sensitivity,
    }
    for filename, frame in csv_outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.10f")

    rendered_files = _render_problem_visuals_if_available(
        output_dir, problem_canonical_path
    )

    _write_report(
        report_path,
        inputs=inputs,
        simulated_comparison=simulated_comparison,
        official_comparison=official_comparison,
        simulated_by_target=rolling_by_target,
        official_by_target=official_by_target,
        declared=declared,
        exploratory=exploratory,
        holdout_by_target=holdout_by_target,
        stress_by_target=stress_by_target,
        source_provenance=source_provenance,
        problem_canonical=problem_canonical,
        problem1_indicator_summary=problem1_indicator_summary,
        problem1_growth_parameters=problem1_growth_parameters,
        problem1_growth_diagnostics=problem1_growth_diagnostics,
        problem1_growth_forecasts=problem1_growth_forecasts,
        problem2_forecasts=problem2_forecasts,
        problem2_diagnostics=problem2_diagnostics,
        problem2_ridge_parameters=problem2_ridge_parameters,
        problem3_scenarios=problem3_scenarios,
        problem3_sensitivity=problem3_sensitivity,
        canonical_source_access=canonical_source_access,
    )
    _write_output_readme(output_dir)

    source_files = tuple(
        dict.fromkeys(
            [*inputs.source_files, problem_canonical_path, source_metadata_path]
        )
    )
    source_hashes = {
        _relative(path): _sha256(path) for path in source_files if path.exists()
    }
    simulated_first_row = simulated_comparison.sort_values("descriptive_rank").iloc[0]
    official_first_row = official_comparison.sort_values("descriptive_rank").iloc[0]
    break_holdout = applicability[
        applicability["evaluation_scope"].eq(
            "user_simulated_augmentation_holdout_2024"
        )
        & applicability["model"].isin(
            ["post_2022_level_break", "strict_evidence_level_break"]
        )
    ]
    summary: dict[str, object] = {
        "random_seed": RANDOM_SEED,
        "input_mode": inputs.source_mode,
        "input_sha256": source_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": ml_model.sklearn.__version__,
        },
        "model_source_provenance": source_provenance.to_dict(orient="records"),
        "original_branch_declarations": original_declarations,
        "headline": {
            "original_branch_winner_determined": False,
            "reason": winner_reason,
            "cross_track_order_consistent": cross_track_consistent,
            "simulated_augmentation_descriptive_first": {
                "branch": str(simulated_first_row["branch"]),
                "model": str(simulated_first_row["model"]),
                "macro_smape_percent": float(
                    simulated_first_row["macro_smape_percent"]
                ),
            },
            "official_only_descriptive_first": {
                "branch": str(official_first_row["branch"]),
                "model": str(official_first_row["model"]),
                "macro_smape_percent": float(
                    official_first_row["macro_smape_percent"]
                ),
            },
        },
        "primary_stability_protocol": {
            "design": "expanding-origin outer validation; tests end at 2023",
            "min_outer_train_records": 5,
            "outer_test_years": {
                metric: list(years) for metric, years in EXPECTED_ROLLING_TEST_YEARS.items()
            },
            "tuning": "nested rolling-origin inner CV inside each augmented outer-training frame; exploratory models only",
            "ranking_metric": "equal-target-weighted macro sMAPE",
            "naive_skill": "1 - model sMAPE / naive_last sMAPE",
            "all_candidate_ranking_status": "exploratory only; cannot declare a branch winner",
            "user_protocol_fixed_adapters": {
                TRADITIONAL_BRANCH: "no_break_log_linear_common_rows",
                ML_BRANCH: "raw_target_ridge_alpha_0.1",
            },
            "simulated_augmentation": {
                "status": "user-directed hypothetical track; not observational evidence",
                "internal_gap_method": "log-linear interpolation from visible outer-training boundaries",
                "tail_method": "median of up to three recent visible annualized log-growth intervals",
                "bias_warning": "structurally favors log-linear paths and smooths unobserved pandemic shocks",
            },
            "official_only_common_row": {
                "status": "priority robustness track; no simulated labels",
                "same_effective_rows_verified": True,
            },
        },
        "final_holdout_2024": {
            "train_max_year": 2023,
            "test_year": 2024,
            "test_n_by_target": inputs.primary_test.groupby("metric").size().to_dict(),
            "selection_use": (
                "simulated-augmentation pseudo-holdout for this rerun only; no 2024 row enters tuning, selection, or augmentation"
            ),
            "pseudo_holdout": True,
            "prospectively_unseen_at_research_design_level": False,
            "level_break_status": sorted(break_holdout["status"].unique()),
            "n_per_target": 1,
        },
        "cross_regime_stress": {
            "train_max_year": 2019,
            "test_years": sorted(inputs.stress_test["year"].astype(int).unique()),
            "contains_2024": False,
            "selection_use": "diagnostic only; not primary ranking",
        },
        "level_break_policy": (
            "post_2022_level_break and strict_evidence_level_break are not executed in any scope "
            "under the user-directed no-break protocol; no predictions or errors are generated"
        ),
        "branch_coverage": inventory.to_dict(orient="records"),
        "known_absences": {
            "origin/111": "no executable model source code",
            "origin/邱志烨-数据搜索": "no executable model implementation",
        },
        "historical_selection_warning": (
            "raw_target_ridge_alpha_0.1 is a fixed user-protocol adapter here, but the original ML "
            "branch recommendation was made after a backtest that included 2024"
        ),
        "research_design_warning": (
            "The model-family code and method discussion were created after 2024 data existed, and the "
            "canonical table is final-vintage retrospective data with backfilled/revised values. Therefore "
            "2024 is isolated only in this rerun's execution graph, not a genuinely prospective unseen holdout."
        ),
        "problem_aligned_outputs": {
            "problem1": {
                "simple_growth_model": "pre_covid_exponential",
                "fit_year_range": "2010-2019 available canonical values",
                "mechanical_forecast_years": list(FORECAST_YEARS),
                "mechanical_forecast_selected_for_q2": False,
                "indicator_count": len(problem1_indicator_summary),
                "canonical_cited_source_count": len(canonical_source_access),
                "canonical_source_access_dates": sorted(
                    canonical_source_access["accessed_date"].unique()
                ),
            },
            "problem2": {
                "final_training_years": "2010-2024",
                "excluded_training_year": 2025,
                "forecast_years": list(FORECAST_YEARS),
                "models": [
                    "no_break_log_linear_common_rows",
                    "raw_target_ridge_alpha_0.1",
                ],
                "ridge_bootstrap_repetitions": RIDGE_BOOTSTRAP_REPETITIONS,
                "ridge_bootstrap_seed": RANDOM_SEED,
                "interval_scope": "model-conditional; repeated-sampling 95% coverage is not guaranteed",
            },
            "problem3": {
                "anchor_year": SCENARIO_ANCHOR_YEAR,
                "anchor_status": "government_target_proxy_not_actual",
                "scenario_count": int(problem3_scenarios["scenario"].nunique()),
                "sensitivity_is_causal": False,
            },
        },
        "generated_files": sorted(
            [*csv_outputs, *rendered_files, "README.md", "run_summary.json"]
        ),
        "report_artifact_name": report_path.name,
        "default_report_location": "docs/unified_branch_model_comparison.md",
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-dir", type=Path, default=DEFAULT_UNIFIED_DIR)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    summary = run_benchmark(
        unified_dir=arguments.unified_dir.resolve(),
        canonical_path=arguments.canonical.resolve(),
        output_dir=arguments.output_dir.resolve(),
        report_path=arguments.report.resolve(),
    )
    headline = summary["headline"]  # type: ignore[index]
    simulated_first = headline["simulated_augmentation_descriptive_first"]  # type: ignore[index]
    official_first = headline["official_only_descriptive_first"]  # type: ignore[index]
    print(
        "completed unified benchmark; original branch winner unavailable; "
        f"simulated adapter first={simulated_first['model']} "  # type: ignore[index]
        f"({simulated_first['macro_smape_percent']:.3f}%), "  # type: ignore[index]
        f"no-simulation adapter first={official_first['model']} "  # type: ignore[index]
        f"({official_first['macro_smape_percent']:.3f}%)"  # type: ignore[index]
    )


if __name__ == "__main__":
    main()
