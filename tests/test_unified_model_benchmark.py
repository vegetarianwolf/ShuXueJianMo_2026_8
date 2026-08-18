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
