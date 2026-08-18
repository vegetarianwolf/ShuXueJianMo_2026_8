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
RANDOM_SEED: Final = 20260817
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


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()

    def format_cell(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [[format_cell(value) for value in row] for row in selected.itertuples(index=False)]
    header = "| " + " | ".join(columns) + " |"
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
        "两轨的适配器相对顺序一致，但这仍不能替代原分支声明模型的共同排名。"
        if ordering_consistent
        else "两轨的适配器相对顺序发生反转，因此连统一适配器层面也不能给出稳健赢家。"
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
    content = f"""# 统一分支模型比较报告

## 结论先行

**不能判定原分支赢家。** 原传统分支声明的主模型是 `post_2022_level_break`，而用户本轮统一协议明确不运行任何断点模型；原 ML 分支声明的是 `ridge_regime` 模型族和 `raw_target_ridge_alpha_0.1` 点路径。因此原声明代表没有一组可共同排名的滚动预测。

在用户指定的模拟增强适配器轨上，描述性第一是 `{simulated_first['model']}`（macro-sMAPE {simulated_first['macro_smape_percent']:.3f}%）；在更适合作为稳健性依据的未模拟 canonical 共同行轨（official-source）上，描述性第一是 `{official_first['model']}`（{official_first['macro_smape_percent']:.3f}%）。{ordering_sentence} 这些表只比较“共同训练行上的固定规格适配器”，不是原分支胜负。

模拟增强轨是假设性分析，不是真实观测证据：log 插值和 log 增长尾推在结构上更贴近 log-linear OLS，而且会把游客量 2020—2022 的未知疫情路径平滑成趋势点。报告因此优先用未模拟共同行轨判断稳健性，并把两轨并列展示。

滚动稳定性框架采用截至 2023 年的 expanding-origin 外层验证；2024 年仅作最终单年 pseudo-holdout（每个目标 n=1），不用于本次执行的排序。模型族代码是在 2024 数据已经存在后形成，统一数据又是 final-vintage 回溯版，因此 2024 不是研究设计层真正“未见”的前瞻测试。2019 截断结果仅作跨疫情/恢复阶段压力测试，不与 pseudo-holdout 混称。

## 分支数据合并结论

- 共审计并合入 5 个唯一业务 tip。`main`、传统模型分支和 ML 分支的核心 `data/` 树完全相同，因此没有重复拼接同一批年度观测。
- `origin/111` 的独有天津市 GDP（2010—2025）与天津市旅游基准已规范化为独立辅助表；天津市口径不会覆盖蓟州区 GDP，也不直接充当蓟州目标标签。
- `origin/邱志烨-数据搜索` 的 8 条补全值保存在 `sensitivity_imputations.csv`，全部标记为非观测并排除在统一训练、测试和排名之外；其全样本标准化、异常检测等派生列也没有直接进入回测。
- 两个分支中的旧版工作簿、预测和情景交付均保留在 Git 历史/原路径，但不会覆盖集成前 `main` 固定提交的 canonical 真值。逐文件来源、blob、SHA-256 和纳入决策见 `data/unified/branch_data_inventory.csv`。

## 可比协议

- 随机种子固定为 `{RANDOM_SEED}`；所有模型读取同一统一数据层。
- 滚动外层最小训练记录数为 5，游客量测试年为 `{rolling_years['tourist_visits']}`，综合收入测试年为 `{rolling_years['tourism_comprehensive_income']}`；外层测试最晚到 2023 年。
- 每折模拟只读取该折测试年前已存在的 physical training rows：内部缺口用两侧训练边界在 log 尺度插值；训练尾部到 `test_year-1` 用最近至多 3 个训练区间的 annualized log-growth 中位数外推。逐点方法、源年、边界见 `simulated_training_points.csv`；绝不读取外层测试、未来官方值或邱分支 sidecar。
- 固定主适配器是传统 `no_break_log_linear_common_rows` 与 ML `raw_target_ridge_alpha_0.1`。两者逐折使用完全相同的 physical/simulated/effective 行；脚本运行时强制验证这一不变量。
- ML 全候选及新建 `ml_inner_selector` 只作探索；其预处理、调参和选择均限定在每个外层折训练内部，但不能据其事后名次宣布分支赢家。
- 2024 pseudo-holdout 训练集每目标记录数为 `{train_counts}`，测试集严格为游客量和综合收入各一条 2024 实际值；隔离只成立于本次重跑的执行流程。
- `data/unified/primary_train.csv` 是 2024 前的 physical 证据层；`outputs/unified_model_benchmark/primary_train_augmented.csv` 是用户指定的 2010—2023 建模训练层，包含 physical 与 simulated 行并保留 `is_simulated/method/known_through_year`，可直接与 `data/unified/primary_test.csv` 配对。
- 主排序指标是先按目标计算 sMAPE，再对两个目标 50/50 等权；不跨单位汇总 RMSE。表中同时给出相对 `naive_last` 的 skill 与最坏点误差。
- 传统断点模型 `post_2022_level_break`、`strict_evidence_level_break` 在所有 scope 均为 `not_executed_user_protocol`，不生成预测或误差。原生 `pre_covid_exponential` 和会排除 2020—2022 的 `no_break_log_linear` 仅保留在未模拟 canonical 敏感性表。

## 用户指定的模拟增强固定适配器排序（假设性）

{_markdown_table(simulated_view, list(simulated_view.columns))}

该表的 `descriptive_rank` 只描述用户指定规格在模拟伪标签上的相对误差；`robust_winner` 固定为 false。模拟点会改变目标路径，不能当成新增事实或原分支的历史声明流程。

## 未模拟 canonical 共同行排序（official-source，稳健性优先）

{_markdown_table(official_view, list(official_view.columns))}

这里不生成任何 benchmark 伪标签。传统 common-row OLS 与 raw Ridge 都吃每折全部 physical canonical training rows，测试仍只用官方实际；canonical 可包含官方来源的同比反推、回列或 supporting 值，并以 `status/is_observed` 明示，例如 2010 年综合收入是 `inferred_from_yoy`，不是 strict observed。该轨优先于模拟轨用于判断结论是否由伪标签驱动；即使两轨顺序一致，也只能称适配器相对排序。

## 两轨逐目标核对

{_markdown_table(track_target_view, list(track_target_view.columns))}

## 原分支声明方法核对

{_markdown_table(declared_view, list(declared_view.columns))}

传统原声明模型没有执行值，`jointly_rankable_original_declared_representatives=false`。`raw_target_ridge_alpha_0.1` 本次每折固定 alpha 且不读取外层测试，但原 ML 分支推荐它时已检查含 2024 的回测；`ridge_regime` 仅作模型族声明审计。因此没有合法的原分支级胜者。

## 全候选探索性排名

{_markdown_table(exploratory_view, list(exploratory_view.columns))}

全候选表用于诊断，不可从中事后挑一个最优候选再宣称无偏胜者。并列通过 1e-10 精度判定；`worst_point_smape_percent` 暴露平均值掩盖的失稳。两种断点模型无论代数上是否可识别，都因用户协议禁止而不执行、不展示误差。

## 未模拟 canonical 分支原生策略敏感性

{_markdown_table(native_view, list(native_view.columns))}

`pre_covid_exponential` 与原生 `no_break_log_linear` 会删除部分统一训练行，故只能用来解释“分支原生过滤策略会怎样”，不能与共同行适配器混作纯算法比较。

## 2024 最终单年 pseudo-holdout

{_markdown_table(holdout_view, list(holdout_view.columns))}

2024 pseudo-holdout 每个目标只有一个点，`MAE = absolute error`，MAPE/sMAPE 都没有稳定性含义；它只检查本次重跑中冻结后的方法能否跨到下一年。两种断点模型仍不执行。任何 2024 误差都没有回流到本次重跑的超参数、候选选择或主排序；但由于模型代码和方法讨论形成时 2024 已存在，它不能支持真正的前瞻泛化声明。

## 2019 截断跨阶段压力测试（不含 2024）

{_markdown_table(stress_view, list(stress_view.columns))}

压力测试的 physical training 只到 2019，测试为 2020—2023 中现有官方实际值：综合收入 2021/2023、游客量 2023；每个测试点单独生成截至 `test_year-1` 的模拟尾部，2021 实际不会流入 2023 压力测试训练。ML 的 pandemic/post-2022 特征在 physical training 中未见，输出属于跨阶段外推，不等于识别疫情或恢复效应。

## 模型源码固定

{_markdown_table(provenance_view, list(provenance_view.columns))}

runner 在建模前校验两个可执行模型文件的 Git blob 与 SHA-256；任一字节漂移都会拒绝运行。原声明摘要也做字段校验。

## 分支覆盖与未纳入原因

- `codex/jizhou-tourism-modeling`：存在可执行 Python 实现；用户轨只跑无断点共同行 OLS，原生 pre-COVID/no-break 仅作敏感性，所有断点均不执行。
- `codex/jizhou-tourism-ml`：存在可执行 Python 实现，实跑全部 7 个候选族、稳健集成、raw-target Ridge、naive 和训练内选择器。
- `origin/111`：只有示例论文和工作簿，**缺少可执行模型源代码**；其中数据只能作统一数据层的旁证，不能把论文数值冒充为同一 split 下的重跑结果。
- `origin/邱志烨-数据搜索`：有数据预处理、补值和模型建议，**没有模型实现**；补值 sidecar 明确排除在标签、外层训练和测试之外。
- `main`：提供 canonical 目标真值并接收最终报告；没有另一套可独立执行的分支模型。

## 复现与限制

输入模式：`{inputs.source_mode}`。loader 会逐行验证 physical split/fold 与 `benchmark_observations.csv` 的目标值、状态、来源、观测标志、年份和 role 完全一致。`train_only_hyperparameters.csv` 可核对调参上限；`model_applicability.csv` 保留禁用断点与 ML 支持状态；逐点预测可复算所有指标。

```bash
.venv/bin/python scripts/build_unified_branch_data.py
MPLCONFIGDIR=/tmp/jizhou-mpl XDG_CACHE_HOME=/tmp/jizhou-xdg .venv/bin/python code/scripts/compare_branch_models.py
.venv/bin/python -m unittest discover -s tests -v
```

样本总量很小、测试年份不规则，2020—2022 又存在目标缺口；canonical 还是包含同比反推、回列修订的 final-vintage 数据。模拟伪标签会低估结构冲击，不增加信息量；sMAPE 和时间外推也无法替代结构解释。应联合查看未模拟 canonical 共同行轨、模拟轨、逐目标 naive skill、最坏误差、2019 stress 和 2024 单点，不能引用“稳健冠军”或“原分支赢家”结论。
"""
    report_path.write_text(content, encoding="utf-8")


def _write_output_readme(output_dir: Path) -> None:
    content = """# Unified model benchmark artifacts

This directory is generated by `code/scripts/compare_branch_models.py`.

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
    }
    for filename, frame in csv_outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.10f")

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
    )
    _write_output_readme(output_dir)

    source_hashes = {
        _relative(path): _sha256(path) for path in inputs.source_files if path.exists()
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
        "generated_files": sorted([*csv_outputs, "README.md", "run_summary.json"]),
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
