from __future__ import annotations

import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))

import compare_branch_models as benchmark  # noqa: E402


class UnifiedSplitContractTest(unittest.TestCase):
    def test_primary_split_holds_out_only_observed_2024_actuals(self) -> None:
        observations = pd.DataFrame(
            [
                {
                    "metric": "tourist_visits",
                    "year": 2019,
                    "value": 100.0,
                    "unit": "10k_persons",
                    "status": "observed",
                    "source_ids": "a",
                    "quality_note": "",
                    "is_observed": True,
                },
                {
                    "metric": "tourist_visits",
                    "year": 2020,
                    "value": 80.0,
                    "unit": "10k_persons",
                    "status": "shock_proxy_scenario",
                    "source_ids": "b",
                    "quality_note": "diagnostic only",
                    "is_observed": False,
                },
                {
                    "metric": "tourist_visits",
                    "year": 2023,
                    "value": 120.0,
                    "unit": "10k_persons",
                    "status": "observed",
                    "source_ids": "c",
                    "quality_note": "",
                    "is_observed": True,
                },
                {
                    "metric": "tourist_visits",
                    "year": 2024,
                    "value": 130.0,
                    "unit": "10k_persons",
                    "status": "observed",
                    "source_ids": "d",
                    "quality_note": "",
                    "is_observed": True,
                },
            ]
        )

        train, test = benchmark.build_primary_split(observations, cutoff_year=2023)

        self.assertEqual(train["year"].tolist(), [2019, 2023])
        self.assertEqual(test["year"].tolist(), [2024])
        self.assertTrue(test["is_observed"].all())
        self.assertEqual(set(train["split"]), {"train"})
        self.assertEqual(set(test["split"]), {"test"})

    def test_expanding_outer_folds_stop_before_2024_and_start_with_five_rows(self) -> None:
        observations = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "year": year,
                    "value": float(year - 2000),
                    "unit": "u",
                    "status": "observed",
                    "source_ids": "s",
                    "quality_note": "",
                    "is_observed": True,
                }
                for metric in benchmark.TARGETS
                for year in range(2010, 2017)
            ]
        )

        folds = benchmark.build_rolling_origin_folds(observations, min_train_size=5)

        tests = folds[folds["fold_role"] == "test"]
        self.assertEqual(set(tests["year"]), {2015, 2016})
        self.assertLessEqual(int(tests["year"].max()), 2023)
        train_sizes = (
            folds[folds["fold_role"] == "train"]
            .groupby(["metric", "fold_id"])
            .size()
            .sort_values()
            .unique()
            .tolist()
        )
        self.assertEqual(train_sizes, [5, 6])

    def test_loader_rejects_shape_valid_physical_split_content_tampering(self) -> None:
        filenames = (
            "benchmark_observations.csv",
            "primary_train.csv",
            "primary_test.csv",
            "rolling_origin_folds.csv",
            "stress_train.csv",
            "stress_test.csv",
        )
        with tempfile.TemporaryDirectory() as directory:
            unified = Path(directory)
            for filename in filenames:
                shutil.copy2(ROOT / "data/unified" / filename, unified / filename)
            primary = pd.read_csv(unified / "primary_train.csv")
            primary.loc[0, "source_ids"] = "shape-valid-but-wrong-source"
            primary.to_csv(unified / "primary_train.csv", index=False)

            with self.assertRaisesRegex(ValueError, "primary_train"):
                benchmark.load_benchmark_inputs(unified_dir=unified)

    def test_loader_rejects_shape_valid_rolling_fold_content_tampering(self) -> None:
        filenames = (
            "benchmark_observations.csv",
            "primary_train.csv",
            "primary_test.csv",
            "rolling_origin_folds.csv",
            "stress_train.csv",
            "stress_test.csv",
        )
        with tempfile.TemporaryDirectory() as directory:
            unified = Path(directory)
            for filename in filenames:
                shutil.copy2(ROOT / "data/unified" / filename, unified / filename)
            rolling = pd.read_csv(unified / "rolling_origin_folds.csv")
            rolling.loc[0, "status"] = "shape-valid-but-wrong-status"
            rolling.to_csv(unified / "rolling_origin_folds.csv", index=False)

            with self.assertRaisesRegex(ValueError, "rolling_origin_folds"):
                benchmark.load_benchmark_inputs(unified_dir=unified)

    def test_simulated_training_points_use_only_visible_boundaries_and_stop_before_test(self) -> None:
        physical = pd.DataFrame(
            {
                "metric": ["tourist_visits", "tourist_visits"],
                "year": [2019, 2022],
                "value": [100.0, 133.1],
                "unit": ["u", "u"],
                "status": ["observed", "observed"],
                "source_ids": ["a", "b"],
                "quality_note": ["", ""],
                "is_observed": [True, True],
            }
        )

        augmented, audit = benchmark.augment_training_rows(
            physical,
            test_year=2024,
            scope="test_scope",
            fold_id="test_fold",
        )

        self.assertEqual(augmented["year"].tolist(), [2019, 2020, 2021, 2022, 2023])
        self.assertNotIn(2024, set(augmented["year"]))
        self.assertAlmostEqual(float(augmented.loc[augmented["year"] == 2020, "value"].iloc[0]), 110.0)
        self.assertAlmostEqual(float(augmented.loc[augmented["year"] == 2021, "value"].iloc[0]), 121.0)
        self.assertAlmostEqual(float(augmented.loc[augmented["year"] == 2023, "value"].iloc[0]), 146.41)
        self.assertEqual(
            audit.loc[audit["year"] == 2020, "method"].iloc[0],
            "log_linear_interpolation",
        )
        self.assertEqual(
            audit.loc[audit["year"] == 2023, "method"].iloc[0],
            "tail_median_annualized_log_growth",
        )
        self.assertTrue(audit.loc[audit["year"].isin([2020, 2021, 2023]), "is_simulated"].all())
        self.assertEqual(set(audit["known_through_year"]), {2022})


class MetricContractTest(unittest.TestCase):
    def test_macro_smape_is_equal_weighted_across_the_two_targets(self) -> None:
        predictions = pd.DataFrame(
            [
                {"branch": "b", "model": "exact", "metric": "tourist_visits", "year": 2024, "actual": 100.0, "prediction": 100.0},
                {"branch": "b", "model": "exact", "metric": "tourism_comprehensive_income", "year": 2024, "actual": 200.0, "prediction": 200.0},
                {"branch": "b", "model": "half", "metric": "tourist_visits", "year": 2024, "actual": 100.0, "prediction": 50.0},
                {"branch": "b", "model": "half", "metric": "tourism_comprehensive_income", "year": 2024, "actual": 200.0, "prediction": 100.0},
            ]
        )

        by_target, macro = benchmark.summarize_predictions(predictions)

        self.assertEqual(len(by_target), 4)
        exact = macro.loc[macro["model"] == "exact"].iloc[0]
        half = macro.loc[macro["model"] == "half"].iloc[0]
        self.assertEqual(int(exact["rank"]), 1)
        self.assertEqual(int(half["rank"]), 2)
        self.assertAlmostEqual(float(exact["macro_smape_percent"]), 0.0)
        self.assertAlmostEqual(float(half["macro_smape_percent"]), 200.0 / 3.0)
        self.assertTrue(np.isfinite(by_target[["mae", "rmse", "mape_percent", "smape_percent"]]).all().all())


class ApplicabilityContractTest(unittest.TestCase):
    def test_level_break_requires_a_post_2022_training_observation(self) -> None:
        primary_years = np.arange(2010, 2020)
        secondary_years = np.append(primary_years, 2023)

        primary = benchmark.assess_level_break(primary_years)
        secondary = benchmark.assess_level_break(secondary_years)

        self.assertEqual(primary["status"], "not_applicable")
        self.assertIn("post-2022", primary["reason"])
        self.assertEqual(secondary["status"], "applicable")

    def test_common_row_traditional_adapter_never_drops_physical_training_rows(self) -> None:
        physical = pd.DataFrame(
            {
                "metric": ["tourism_comprehensive_income"] * 3,
                "year": [2019, 2021, 2023],
                "value": [165.0, 110.0, 191.5],
            }
        )

        common = benchmark._traditional_effective_train(
            physical, "no_break_log_linear_common_rows"
        )
        native = benchmark._traditional_effective_train(physical, "no_break_log_linear")

        self.assertEqual(common["year"].tolist(), [2019, 2021, 2023])
        self.assertEqual(native["year"].tolist(), [2019, 2023])


class ModelSourceProvenanceTest(unittest.TestCase):
    def test_pinned_model_sources_are_accepted_and_one_byte_drift_is_rejected(self) -> None:
        provenance = benchmark.validate_model_source_provenance(ROOT)
        self.assertEqual(set(provenance["branch"]), {benchmark.TRADITIONAL_BRANCH, benchmark.ML_BRANCH})
        self.assertTrue(provenance["validated"].all())

        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            scripts = temp_root / "code/scripts"
            scripts.mkdir(parents=True)
            for filename in ("model_jizhou_tourism.py", "model_jizhou_tourism_ml.py"):
                shutil.copy2(ROOT / "code/scripts" / filename, scripts / filename)
            (scripts / "model_jizhou_tourism_ml.py").write_text(
                (scripts / "model_jizhou_tourism_ml.py").read_text(encoding="utf-8") + "\n# drift\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "source drift"):
                benchmark.validate_model_source_provenance(temp_root)


class ProblemAlignedComputationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = pd.read_csv(
            ROOT / "data/unified/canonical_official_annual_2010_2025.csv"
        )
        cls.observations = pd.read_csv(
            ROOT / "data/unified/benchmark_observations.csv"
        )
        (
            cls.final_training,
            cls.forecasts,
            cls.final_diagnostics,
            cls.ridge_parameters,
        ) = benchmark.build_problem2_outputs(cls.observations)

    def test_problem1_indicator_coverage_growth_and_status_boundaries(self) -> None:
        summary = benchmark.build_problem1_indicator_summary(self.canonical).set_index(
            "metric"
        )

        self.assertEqual(set(summary.index), set(benchmark.PROBLEM1_INDICATORS))
        self.assertEqual(int(summary.loc["tourist_visits", "nonmissing_count"]), 12)
        self.assertEqual(
            summary.loc["tourist_visits", "missing_years"], "2020;2021;2022;2025"
        )
        self.assertEqual(
            summary.loc["tourism_comprehensive_income", "missing_years"],
            "2016;2020;2022;2025",
        )
        self.assertEqual(int(summary.loc["jizhou_gdp", "nonmissing_count"]), 16)
        self.assertTrue(
            np.isnan(float(summary.loc["jizhou_gdp", "cagr_2010_2019_percent"]))
        )
        self.assertAlmostEqual(
            float(summary.loc["jizhou_gdp", "cagr_2010_2018_percent"]),
            7.4079009368,
            places=8,
        )
        self.assertAlmostEqual(
            float(summary.loc["jizhou_gdp", "cagr_2019_2025_percent"]),
            6.0229449942,
            places=8,
        )
        visitors = self.canonical.set_index("year")[
            "preferred_visitor_10k_persons"
        ]
        expected_cagr = ((visitors.loc[2019] / visitors.loc[2010]) ** (1 / 9) - 1) * 100
        self.assertAlmostEqual(
            float(summary.loc["tourist_visits", "cagr_2010_2019_percent"]),
            float(expected_cagr),
        )
        self.assertIn("scope break", summary.loc["jizhou_gdp", "status_boundary_note"])

    def test_problem1_simple_growth_has_full_inference_and_diagnostics(self) -> None:
        parameters, diagnostics = benchmark.build_problem1_simple_growth_outputs(
            self.canonical
        )

        self.assertEqual(set(parameters["model"]), {"pre_covid_exponential"})
        self.assertEqual(len(parameters), 4)
        self.assertTrue(
            np.isfinite(
                parameters[
                    [
                        "estimate",
                        "standard_error",
                        "t_value",
                        "p_value",
                        "ci95_lower",
                        "ci95_upper",
                    ]
                ]
            ).all().all()
        )
        self.assertTrue(
            parameters["estimate"].between(
                parameters["ci95_lower"], parameters["ci95_upper"]
            ).all()
        )
        self.assertEqual(len(diagnostics), 2)
        required = [
            "r_squared_log",
            "adjusted_r_squared_log",
            "rmse_original_units",
            "mape_percent",
            "aicc_log",
            "loocv_log_rmse",
            "durbin_watson",
            "jarque_bera_p",
        ]
        self.assertTrue(np.isfinite(diagnostics[required]).all().all())

        forecasts = benchmark.build_problem1_simple_growth_forecasts(self.canonical)
        self.assertEqual(len(forecasts), 10)
        self.assertEqual(set(forecasts["year"]), set(range(2026, 2031)))
        self.assertFalse(forecasts["uses_2025_as_training"].any())
        keyed = forecasts.set_index(["metric", "year"])["forecast"]
        self.assertAlmostEqual(
            float(keyed.loc[("tourist_visits", 2026)]), 7723.7, delta=0.1
        )
        self.assertAlmostEqual(
            float(keyed.loc[("tourist_visits", 2030)]), 13223.9, delta=0.1
        )
        self.assertAlmostEqual(
            float(keyed.loc[("tourism_comprehensive_income", 2026)]),
            572.5,
            delta=0.1,
        )
        self.assertAlmostEqual(
            float(keyed.loc[("tourism_comprehensive_income", 2030)]),
            1128.4,
            delta=0.1,
        )

    def test_canonical_cited_sources_have_verified_access_dates(self) -> None:
        sources = pd.read_csv(ROOT / "data/metadata/sources.csv")
        audit = benchmark.build_canonical_source_access_audit(
            self.canonical, sources
        )

        self.assertEqual(len(audit), 23)
        self.assertEqual(set(audit["accessed_date"]), {"2026-08-17"})
        self.assertTrue(audit["used_by_canonical"].all())

    def test_problem2_final_training_is_internal_gap_only_and_excludes_2025(self) -> None:
        expected_simulated = {
            "tourist_visits": {2020, 2021, 2022},
            "tourism_comprehensive_income": {2016, 2020, 2022},
        }
        self.assertNotIn("test_year", self.final_training.columns)
        for metric in benchmark.TARGETS:
            rows = self.final_training[self.final_training["metric"].eq(metric)]
            self.assertEqual(rows["year"].tolist(), list(range(2010, 2025)))
            self.assertEqual(len(rows), 15)
            self.assertEqual(int(rows["is_simulated"].sum()), 3)
            simulated = rows[rows["is_simulated"]]
            self.assertEqual(set(simulated["year"]), expected_simulated[metric])
            self.assertEqual(set(simulated["method"]), {"log_linear_interpolation"})
            self.assertTrue(simulated["boundary_left_year"].lt(simulated["year"]).all())
            self.assertTrue(simulated["boundary_right_year"].gt(simulated["year"]).all())
            self.assertTrue(simulated["boundary_right_year"].le(2024).all())
        self.assertFalse(self.final_training["uses_2025_as_training"].any())

    def test_problem2_2025_sentinel_cannot_change_training_or_forecasts(self) -> None:
        sentinel_rows = []
        for metric, unit in (
            ("tourist_visits", "10k_persons"),
            ("tourism_comprehensive_income", "100m_cny"),
        ):
            sentinel_rows.append(
                {
                    "metric": metric,
                    "year": 2025,
                    "value": 1.0e15,
                    "unit": unit,
                    "status": "observed_sentinel",
                    "source_ids": "test_sentinel",
                    "quality_note": "must be ignored",
                    "is_observed": True,
                }
            )
        injected = pd.concat(
            [self.observations, pd.DataFrame(sentinel_rows)], ignore_index=True
        )
        training, forecasts, diagnostics, parameters = benchmark.build_problem2_outputs(
            injected
        )
        pd.testing.assert_frame_equal(training, self.final_training)
        pd.testing.assert_frame_equal(forecasts, self.forecasts)
        pd.testing.assert_frame_equal(diagnostics, self.final_diagnostics)
        pd.testing.assert_frame_equal(parameters, self.ridge_parameters)

    def test_problem2_forecast_values_intervals_and_bootstrap_contract(self) -> None:
        self.assertEqual(set(self.forecasts["year"]), set(range(2026, 2031)))
        self.assertEqual(
            set(self.forecasts["model"]),
            {
                "raw_target_ridge_alpha_0.1",
                "no_break_log_linear_common_rows",
            },
        )
        self.assertTrue(self.forecasts["training_end_year"].eq(2024).all())
        self.assertFalse(self.forecasts["uses_2025_as_training"].any())
        self.assertTrue(
            self.forecasts["prediction_interval95_lower"]
            .le(self.forecasts["mean_ci95_lower"])
            .all()
        )
        self.assertTrue(
            self.forecasts["mean_ci95_lower"].le(self.forecasts["forecast"]).all()
        )
        self.assertTrue(
            self.forecasts["forecast"].le(self.forecasts["mean_ci95_upper"]).all()
        )
        self.assertTrue(
            self.forecasts["mean_ci95_upper"]
            .le(self.forecasts["prediction_interval95_upper"])
            .all()
        )
        ridge = self.forecasts[
            self.forecasts["model"].eq("raw_target_ridge_alpha_0.1")
        ]
        self.assertTrue(
            ridge["bootstrap_repetitions"].eq(
                benchmark.RIDGE_BOOTSTRAP_REPETITIONS
            ).all()
        )
        self.assertTrue(ridge["random_seed"].eq(benchmark.RANDOM_SEED).all())
        expected = {
            ("tourist_visits", "no_break_log_linear_common_rows", 2026): 3840.327121,
            ("tourist_visits", "raw_target_ridge_alpha_0.1", 2030): 3887.258656,
            (
                "tourism_comprehensive_income",
                "no_break_log_linear_common_rows",
                2030,
            ): 441.150844,
            (
                "tourism_comprehensive_income",
                "raw_target_ridge_alpha_0.1",
                2026,
            ): 241.158964,
        }
        keyed = self.forecasts.set_index(["metric", "model", "year"])["forecast"]
        for key, value in expected.items():
            self.assertAlmostEqual(float(keyed.loc[key]), value, places=5)
        self.assertEqual(len(self.ridge_parameters), 8)
        self.assertEqual(
            set(self.ridge_parameters["parameter"]),
            {"intercept", *benchmark.ml_model.FEATURE_NAMES},
        )
        self.assertTrue(
            self.ridge_parameters["estimate"].between(
                self.ridge_parameters["bootstrap_ci95_lower"],
                self.ridge_parameters["bootstrap_ci95_upper"],
            ).all()
        )

    def test_problem3_scenarios_and_oat_sensitivity_obey_identity(self) -> None:
        scenarios = benchmark.build_problem3_scenario_forecasts()
        sensitivity = benchmark.build_problem3_policy_sensitivity(scenarios)

        self.assertEqual(len(scenarios), 45)
        self.assertFalse(scenarios["anchor_is_observed"].any())
        self.assertEqual(
            set(scenarios["anchor_status"]),
            {"government_target_proxy_not_actual"},
        )
        pivot = scenarios.pivot_table(
            index=["scenario", "year"], columns="metric", values="value"
        )
        reconstructed_income = (
            pivot["tourist_visits"]
            * pivot["nominal_spend_per_visit"]
            / 10_000.0
        )
        np.testing.assert_allclose(
            reconstructed_income,
            pivot["tourism_comprehensive_income"],
            rtol=0.0,
            atol=1e-10,
        )
        baseline_2030 = pivot.loc[("baseline_policy_anchor", 2030)]
        self.assertAlmostEqual(float(baseline_2030["tourist_visits"]), 3548.874857, places=6)
        self.assertAlmostEqual(
            float(baseline_2030["tourism_comprehensive_income"]),
            339.414786,
            places=6,
        )
        self.assertEqual(len(sensitivity), 16)
        self.assertEqual(
            set(sensitivity["factor"]),
            {
                "source_market_growth",
                "new_format_spend_growth",
                "policy_coordination_multiplier",
                "external_shock",
            },
        )
        self.assertEqual(set(sensitivity["setting"]), {"low", "high"})
        self.assertFalse(sensitivity["historically_identified_causal_effect"].any())
        np.testing.assert_allclose(
            sensitivity["delta_2030"],
            sensitivity["scenario_2030"] - sensitivity["baseline_2030"],
            rtol=0.0,
            atol=1e-10,
        )


class GeneratedBenchmarkContractTest(unittest.TestCase):
    OUTPUT = ROOT / "outputs/unified_model_benchmark"
    ADAPTER_MODELS = {
        "no_break_log_linear_common_rows",
        "raw_target_ridge_alpha_0.1",
    }

    def test_rolling_stability_predictions_are_complete_and_stop_at_2023(self) -> None:
        predictions = pd.read_csv(self.OUTPUT / "stability_rolling_predictions.csv")
        self.assertLessEqual(int(predictions["test_year"].max()), 2023)
        self.assertNotIn(2024, set(predictions["test_year"]))
        counts = predictions.groupby(["branch", "model"]).size()
        self.assertTrue(counts.eq(12).all())
        self.assertNotIn("post_2022_level_break", set(predictions["model"]))

    def test_2024_is_an_execution_only_pseudo_holdout(self) -> None:
        holdout = pd.read_csv(self.OUTPUT / "final_holdout_2024_predictions.csv")
        applicability = pd.read_csv(self.OUTPUT / "model_applicability.csv")
        summary = json.loads((self.OUTPUT / "run_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(set(holdout["test_year"]), {2024})
        self.assertTrue(
            {
                "post_2022_level_break",
                "strict_evidence_level_break",
            }.isdisjoint(set(holdout["model"]))
        )
        all_breaks = applicability[
            applicability["model"].isin(
                ["post_2022_level_break", "strict_evidence_level_break"]
            )
        ]
        self.assertFalse(all_breaks.empty)
        self.assertTrue(
            all_breaks["status"].eq("not_executed_user_protocol").all()
        )
        self.assertTrue(summary["final_holdout_2024"]["pseudo_holdout"])
        self.assertFalse(
            summary["final_holdout_2024"]["prospectively_unseen_at_research_design_level"]
        )
        self.assertFalse(
            summary["headline"]["original_branch_winner_determined"]
        )

    def test_every_tuned_holdout_model_stops_training_at_2023(self) -> None:
        tuning = pd.read_csv(self.OUTPUT / "train_only_hyperparameters.csv")
        holdout_tuning = tuning[
            tuning["evaluation_scope"]
            == "user_simulated_augmentation_holdout_2024"
        ]
        self.assertFalse(holdout_tuning.empty)
        self.assertTrue(holdout_tuning["outer_train_end_year"].le(2023).all())
        stress = pd.read_csv(self.OUTPUT / "cross_regime_stress_predictions.csv")
        self.assertLess(int(stress["test_year"].max()), 2024)

    def test_simulated_points_are_fold_local_non_observations_before_each_test(self) -> None:
        points = pd.read_csv(self.OUTPUT / "simulated_training_points.csv")
        simulated = points[points["is_simulated"]]

        self.assertFalse(simulated.empty)
        self.assertTrue(points["year"].lt(points["test_year"]).all())
        self.assertTrue(points["known_through_year"].lt(points["test_year"]).all())
        self.assertTrue(simulated["is_observed"].eq(False).all())
        self.assertEqual(
            set(simulated["method"]),
            {
                "log_linear_interpolation",
                "tail_median_annualized_log_growth",
            },
        )
        interpolation = simulated[
            simulated["method"].eq("log_linear_interpolation")
        ]
        self.assertTrue(
            interpolation["boundary_right_year"]
            .le(interpolation["known_through_year"])
            .all()
        )

    def test_2024_augmented_training_contract_pairs_with_physical_evidence_and_test(self) -> None:
        augmented = pd.read_csv(self.OUTPUT / "primary_train_augmented.csv")
        evidence = pd.read_csv(ROOT / "data/unified/primary_train.csv")
        test = pd.read_csv(ROOT / "data/unified/primary_test.csv")

        self.assertEqual(set(test["year"]), {2024})
        for metric in benchmark.TARGETS:
            metric_rows = augmented[augmented["metric"].eq(metric)]
            self.assertEqual(metric_rows["year"].tolist(), list(range(2010, 2024)))
            self.assertEqual(set(metric_rows["known_through_year"]), {2023})
            self.assertEqual(set(metric_rows["test_year"]), {2024})
        physical = augmented[~augmented["is_simulated"]][
            ["metric", "year", "value"]
        ].sort_values(["metric", "year"], ignore_index=True)
        expected = evidence[["metric", "year", "value"]].sort_values(
            ["metric", "year"], ignore_index=True
        )
        pd.testing.assert_frame_equal(physical, expected, check_dtype=False)

    def test_common_row_adapters_use_identical_effective_rows_in_both_tracks(self) -> None:
        filenames = (
            "stability_rolling_predictions.csv",
            "stability_official_only_common_row_predictions.csv",
        )
        for filename in filenames:
            predictions = pd.read_csv(self.OUTPUT / filename)
            pair = predictions[predictions["model"].isin(self.ADAPTER_MODELS)]
            keys = ["fold_id", "metric", "test_year"]
            self.assertTrue(pair.groupby(keys).size().eq(2).all(), filename)
            self.assertTrue(pair["uses_all_augmented_training_rows"].all(), filename)
            self.assertTrue(
                pair["effective_train_n"].eq(pair["augmented_train_n"]).all(),
                filename,
            )
            for column in (
                "physical_train_n",
                "observed_train_n",
                "physical_canonical_train_n",
                "simulated_train_n",
                "augmented_train_n",
                "effective_train_n",
            ):
                self.assertTrue(
                    pair.groupby(keys)[column].nunique().eq(1).all(),
                    f"{filename}: {column}",
                )
        official = pd.read_csv(
            self.OUTPUT / "stability_official_only_common_row_predictions.csv"
        )
        official_pair = official[official["model"].isin(self.ADAPTER_MODELS)]
        self.assertTrue(official_pair["simulated_train_n"].eq(0).all())
        self.assertTrue(
            official_pair["effective_train_n"]
            .eq(official_pair["physical_train_n"])
            .all()
        )

    def test_adapter_tables_have_fresh_scope_and_original_representatives_are_not_ranked(self) -> None:
        simulated = pd.read_csv(
            self.OUTPUT / "stability_simulated_augmentation_adapter_comparison.csv"
        )
        official = pd.read_csv(
            self.OUTPUT / "stability_official_only_common_row_comparison.csv"
        )
        declared = pd.read_csv(
            self.OUTPUT / "stability_predeclared_representatives.csv"
        )

        self.assertEqual(set(simulated["model"]), self.ADAPTER_MODELS)
        self.assertEqual(set(official["model"]), self.ADAPTER_MODELS)
        self.assertEqual(
            set(simulated["ranking_scope"]),
            {"user_simulated_augmentation_fixed_adapters"},
        )
        self.assertEqual(
            set(official["ranking_scope"]),
            {"official_only_common_effective_rows_fixed_adapters"},
        )
        self.assertFalse(simulated["robust_winner"].any())
        self.assertFalse(official["robust_winner"].any())
        self.assertTrue(
            declared["jointly_rankable_original_declared_representatives"]
            .eq(False)
            .all()
        )
        traditional = declared[
            declared["branch"].eq(benchmark.TRADITIONAL_BRANCH)
        ].iloc[0]
        self.assertEqual(traditional["model"], "post_2022_level_break")
        self.assertEqual(
            traditional["rolling_execution_status"],
            "not_executed_user_protocol",
        )

    def test_model_source_pin_is_recorded_in_both_inventory_and_summary(self) -> None:
        provenance = pd.read_csv(self.OUTPUT / "model_source_provenance.csv")
        inventory = pd.read_csv(self.OUTPUT / "branch_implementation_inventory.csv")
        summary = json.loads(
            (self.OUTPUT / "run_summary.json").read_text(encoding="utf-8")
        )

        self.assertTrue(provenance["validated"].all())
        executable = inventory[inventory["executable_model_implementation"]]
        self.assertTrue(executable["model_source_validated"].all())
        self.assertTrue(
            all(item["validated"] for item in summary["model_source_provenance"])
        )

    def test_problem_aligned_outputs_and_report_embeds_are_present(self) -> None:
        required_csv = {
            "problem1_indicator_summary.csv",
            "problem_source_access_dates.csv",
            "problem1_simple_growth_parameters.csv",
            "problem1_simple_growth_diagnostics.csv",
            "problem1_simple_growth_forecasts_2026_2030.csv",
            "problem2_final_training_2010_2024.csv",
            "problem2_forecasts_2026_2030.csv",
            "problem2_final_model_diagnostics.csv",
            "problem2_ridge_standardized_parameters.csv",
            "problem3_scenario_forecasts_2026_2030.csv",
            "problem3_policy_sensitivity.csv",
        }
        self.assertTrue(
            all((self.OUTPUT / filename).is_file() for filename in required_csv)
        )
        report = (ROOT / "docs/unified_branch_model_comparison.md").read_text(
            encoding="utf-8"
        )
        for section in (
            "# C题：蓟州区旅游经济趋势预测与对策分析——统一分支模型比较报告",
            "## 题目要求覆盖矩阵",
            "## 指标、缩写与技术用语对照",
            "## 问题1：指标整理与简单增长模型",
            "## 问题2：模型评判与 2026—2030 预测",
            "## 问题3：政策锚定三情景与敏感性",
            "## 附录：统一分支模型回测与审计",
            "## 题目导向综合结论",
        ):
            self.assertIn(section, report)
        for relative_path in (
            "../outputs/unified_model_benchmark/q1_required_indicators.png",
            "../outputs/unified_model_benchmark/q2_model_judgement.png",
            "../outputs/unified_model_benchmark/q2_forecast_2026_2030.png",
            "../outputs/unified_model_benchmark/q3_scenarios_sensitivity.png",
        ):
            self.assertIn(relative_path, report)
        self.assertIn("不是题面直接指定的评价指标", report)
        self.assertIn("不能把“综合收入/GDP”解释为旅游增加值贡献率", report)
        self.assertIn("23 个唯一来源编号（代码字段 `source_id`）", report)
        self.assertIn("实际获取日期均为 2026-08-17", report)
        self.assertIn("作为2026—2030年的主点预测", report)
        self.assertIn("问题1疫情前简单模型在疫情后不再适合作为主预测", report)
        self.assertIn("target_scale/loocv_scale", report)
        self.assertIn("不是标准化弹性", report)
        for definition in (
            "平均绝对误差（Mean Absolute Error）",
            "对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）",
            "小样本修正赤池信息准则（Corrected Akaike Information Criterion）",
            "留一交叉验证（Leave-One-Out Cross-Validation）",
            "德宾—沃森统计量（Durbin–Watson Statistic）",
            "雅克—贝拉正态性检验（Jarque–Bera Test）",
            "跨疫情阶段压力测试",
        ):
            self.assertIn(definition, report)
        self.assertIn("| 预测指标 | 模型 | 测试点数 |", report)
        self.assertNotIn("| metric | model | n_test |", report)


class BenchmarkReproducibilityTest(unittest.TestCase):
    def test_temp_rerun_is_byte_reproducible(self) -> None:
        expected_output = ROOT / "outputs/unified_model_benchmark"
        expected_report = ROOT / "docs/unified_branch_model_comparison.md"
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            output = temp_root / "outputs"
            report = temp_root / "unified_branch_model_comparison.md"
            benchmark.run_benchmark(output_dir=output, report_path=report)

            expected_files = sorted(path.name for path in expected_output.iterdir())
            actual_files = sorted(path.name for path in output.iterdir())
            self.assertEqual(actual_files, expected_files)
            for filename in expected_files:
                self.assertEqual(
                    (output / filename).read_bytes(),
                    (expected_output / filename).read_bytes(),
                    filename,
                )
            self.assertEqual(report.read_bytes(), expected_report.read_bytes())


if __name__ == "__main__":
    unittest.main()
