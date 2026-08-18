from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))

import compare_branch_models as benchmark  # noqa: E402
import optimize_ridge_model as ridge  # noqa: E402


def physical_series(years: list[int], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": "tourist_visits",
            "year": years,
            "value": values,
            "unit": "10k_persons",
            "status": "observed",
            "source_ids": "test_fixture",
            "quality_note": "",
            "is_observed": True,
        }
    )


class RidgeTuningContractTest(unittest.TestCase):
    def test_alpha_is_lambda_and_search_is_one_dimensional(self) -> None:
        model = ridge.build_ridge_model(0.25)
        self.assertEqual(model.get_params()["model__alpha"], 0.25)

        result = ridge.tune_raw_ridge_lambda(
            physical_series(
                list(range(2010, 2017)),
                [100.0, 108.0, 117.0, 127.0, 138.0, 150.0, 163.0],
            ),
            candidate_lambdas=(1.0, 0.1, 0.001, 0.1),
        )

        self.assertEqual(result.candidate_summary["lambda"].tolist(), [0.001, 0.1, 1.0])
        np.testing.assert_array_equal(
            result.candidate_summary["lambda"],
            result.candidate_summary["alpha_code_parameter"],
        )
        np.testing.assert_allclose(
            ridge.POSITIVE_LAMBDA_GRID,
            np.logspace(-4, 3, 29),
            rtol=0.0,
            atol=0.0,
        )

    def test_insufficient_inner_validation_falls_back_to_baseline(self) -> None:
        result = ridge.tune_raw_ridge_lambda(
            physical_series(
                list(range(2010, 2016)),
                [100.0, 108.0, 117.0, 127.0, 138.0, 150.0],
            ),
            candidate_lambdas=(0.001, 1.0),
            min_inner_train_records=4,
            min_inner_validations=3,
            fallback_lambda=0.1,
        )

        self.assertEqual(result.inner_validation_count, 2)
        self.assertEqual(result.selected_lambda, 0.1)
        self.assertEqual(result.status, "fallback_insufficient_inner_validations")
        self.assertFalse(result.boundary_hit)
        self.assertTrue(result.candidate_summary["selected_lambda"].eq(0.1).all())

    def test_inner_prediction_cannot_see_validation_or_future_and_augmentation_is_fold_local(
        self,
    ) -> None:
        original = physical_series(
            [2010, 2012, 2014, 2017, 2020, 2023],
            [100.0, 121.0, 146.41, 180.0, 230.0, 290.0],
        )
        changed_unseen_rows = physical_series(
            [2010, 2012, 2014, 2017, 2020, 2023],
            [100.0, 121.0, 146.41, 18_000.0, 23_000.0, 29_000.0],
        )

        results = [
            ridge.tune_raw_ridge_lambda(
                frame,
                candidate_lambdas=(0.1,),
                simulate_missing_years=True,
                min_inner_train_records=3,
                min_inner_validations=1,
                scope="fold_local_leakage_test",
            )
            for frame in (original, changed_unseen_rows)
        ]
        validation_rows = [
            result.fold_scores.loc[
                result.fold_scores["validation_year"].eq(2017)
            ].iloc[0]
            for result in results
        ]

        self.assertNotEqual(validation_rows[0]["actual"], validation_rows[1]["actual"])
        self.assertAlmostEqual(
            float(validation_rows[0]["prediction"]),
            float(validation_rows[1]["prediction"]),
            places=12,
        )
        for result in results:
            self.assertTrue(
                result.fold_scores["inner_train_end_year"]
                .lt(result.fold_scores["validation_year"])
                .all()
            )

        visible_prefix = original.iloc[:3].copy()
        augmented, _ = benchmark.augment_training_rows(
            visible_prefix,
            test_year=2017,
            scope="expected_fold_local_training",
            fold_id="validation_2017",
        )
        expected_model = ridge.build_ridge_model(0.1)
        expected_model.fit(
            ridge.make_features(augmented["year"]),
            augmented["value"].to_numpy(dtype=float),
        )
        expected_prediction = max(
            0.0,
            float(expected_model.predict(ridge.make_features([2017]))[0]),
        )

        self.assertLess(int(augmented["year"].max()), 2017)
        self.assertEqual(
            int(validation_rows[0]["inner_effective_train_n"]), len(augmented)
        )
        self.assertAlmostEqual(
            float(validation_rows[0]["prediction"]), expected_prediction, places=12
        )


class RidgeOptimizationEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory(
            prefix="ridge_optimization_test_"
        )
        cls.temporary_root = Path(cls._temporary_directory.name)
        cls.output_dir = cls.temporary_root / "outputs"
        cls.report_path = cls.temporary_root / "ridge_report.md"
        cls.summary = ridge.run_optimization(
            output_dir=cls.output_dir,
            report_path=cls.report_path,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_official_outer_baseline_and_strict_nested_scores_are_reproduced(
        self,
    ) -> None:
        macro = pd.read_csv(self.output_dir / "official_outer_macro_metrics.csv")
        baseline = float(
            macro.loc[
                macro["model"].eq(ridge.BASELINE_MODEL), "macro_smape_percent"
            ].iloc[0]
        )
        tuned = float(
            macro.loc[
                macro["model"].eq(ridge.TUNED_MODEL), "macro_smape_percent"
            ].iloc[0]
        )

        self.assertEqual(f"{baseline:.10f}", "14.6611796538")
        self.assertEqual(f"{tuned:.10f}", "14.6158601569")

        deployment_macro = pd.read_csv(
            self.output_dir / "augmented_outer_macro_metrics.csv"
        )
        deployment_baseline = float(
            deployment_macro.loc[
                deployment_macro["model"].eq(ridge.BASELINE_MODEL),
                "macro_smape_percent",
            ].iloc[0]
        )
        deployment_tuned = float(
            deployment_macro.loc[
                deployment_macro["model"].eq(ridge.TUNED_MODEL),
                "macro_smape_percent",
            ].iloc[0]
        )
        self.assertEqual(f"{deployment_baseline:.10f}", "14.7445528549")
        self.assertEqual(f"{deployment_tuned:.10f}", "14.7196190584")

    def test_custom_report_image_links_resolve_to_generated_files(self) -> None:
        report_text = self.report_path.read_text(encoding="utf-8")
        image_targets = re.findall(r"!\[[^]]*\]\(([^)]+)\)", report_text)

        self.assertEqual(len(image_targets), 4)
        for target in image_targets:
            resolved = (self.report_path.parent / target).resolve()
            self.assertTrue(resolved.is_file(), f"missing report image: {resolved}")

    def test_final_selection_uses_pre_2024_data_and_selects_one_e_minus_four(
        self,
    ) -> None:
        selection = pd.read_csv(self.output_dir / "final_lambda_selection.csv")
        deployed = selection[
            selection["selected_for_forecast"]
            .astype(str)
            .str.lower()
            .eq("true")
        ]

        self.assertEqual(set(deployed["metric"]), set(ridge.TARGETS))
        self.assertEqual(len(deployed), len(ridge.TARGETS))
        self.assertTrue(
            deployed["selection_track"]
            .eq("fold_local_augmented_sensitivity")
            .all()
        )
        np.testing.assert_allclose(
            deployed["selected_lambda"],
            np.full(len(ridge.TARGETS), 1.0e-4),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            deployed["selected_lambda"], deployed["alpha_code_parameter"]
        )
        self.assertTrue(deployed["training_max_year"].lt(2024).all())
        self.assertFalse(
            deployed["uses_2024_for_selection"]
            .astype(str)
            .str.lower()
            .eq("true")
            .any()
        )

    def test_final_forecasts_exclude_2025_and_cover_2026_through_2030(self) -> None:
        forecasts = pd.read_csv(
            self.output_dir / "final_forecasts_2026_2030.csv"
        )
        final_training = pd.read_csv(
            self.output_dir / "final_training_2010_2024.csv"
        )

        expected_years = list(range(2026, 2031))
        self.assertEqual(sorted(forecasts["year"].unique()), expected_years)
        for _, group in forecasts.groupby(["metric", "model"], sort=True):
            self.assertEqual(group.sort_values("year")["year"].tolist(), expected_years)
        self.assertEqual(set(forecasts["metric"]), set(ridge.TARGETS))
        self.assertEqual(
            set(forecasts["model"]), {ridge.BASELINE_MODEL, ridge.TUNED_MODEL}
        )
        self.assertTrue(forecasts["training_end_year"].eq(2024).all())
        self.assertFalse(
            forecasts["uses_2025_as_training"]
            .astype(str)
            .str.lower()
            .eq("true")
            .any()
        )
        self.assertLessEqual(int(final_training["year"].max()), 2024)
        self.assertNotIn(2025, set(final_training["year"]))

    def test_alpha_point_one_forecasts_and_intervals_reproduce_old_csv(self) -> None:
        generated = pd.read_csv(
            self.output_dir / "final_forecasts_2026_2030.csv"
        )
        generated = generated[generated["model"].eq(ridge.BASELINE_MODEL)]
        reference = pd.read_csv(
            ROOT
            / "outputs/unified_model_benchmark/problem2_forecasts_2026_2030.csv"
        )
        reference = reference[reference["model"].eq(ridge.BASELINE_MODEL)]

        self.assertTrue(generated["lambda"].eq(0.1).all())
        self.assertTrue(generated["alpha_code_parameter"].eq(0.1).all())
        comparison = generated.merge(
            reference,
            on=["metric", "model", "year"],
            suffixes=("_generated", "_reference"),
            validate="one_to_one",
        )
        self.assertEqual(len(comparison), len(reference))
        for column in (
            "forecast",
            "mean_ci95_lower",
            "mean_ci95_upper",
            "prediction_interval95_lower",
            "prediction_interval95_upper",
        ):
            np.testing.assert_allclose(
                comparison[f"{column}_generated"],
                comparison[f"{column}_reference"],
                rtol=0.0,
                atol=1.0e-8,
            )


if __name__ == "__main__":
    unittest.main()
