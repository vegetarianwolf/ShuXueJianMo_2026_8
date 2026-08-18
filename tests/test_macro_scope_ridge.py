from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))

import compare_macro_scope_ridge as macro_ridge  # noqa: E402


class MacroHarmonizationTest(unittest.TestCase):
    def test_documented_bridge_is_applied_only_before_2019(self) -> None:
        canonical = pd.read_csv(
            ROOT / "data/unified/canonical_official_annual_2010_2025.csv"
        )

        result = macro_ridge.build_harmonized_macro_series(canonical)

        expected_factors = {
            "jizhou_gdp": 219.62 / (1.0 + 0.021) / 381.66,
            "jizhou_tertiary_value_added": 136.91 / (1.0 + 0.044) / 234.32,
        }
        for metric, expected_factor in expected_factors.items():
            metric_rows = result[result["metric"].eq(metric)].set_index("year")
            self.assertEqual(set(metric_rows.index), set(range(2010, 2026)))
            self.assertAlmostEqual(
                float(metric_rows.loc[2018, "bridge_factor"]), expected_factor
            )
            self.assertAlmostEqual(
                float(metric_rows.loc[2018, "harmonized_value"]),
                float(metric_rows.loc[2018, "original_value"]) * expected_factor,
            )
            self.assertTrue(metric_rows.loc[:2018, "bridge_applied"].all())
            self.assertFalse(metric_rows.loc[2019:, "bridge_applied"].any())
            np.testing.assert_allclose(
                metric_rows.loc[2019:, "harmonized_value"],
                metric_rows.loc[2019:, "original_value"],
                rtol=0.0,
                atol=0.0,
            )

        self.assertTrue(
            result["harmonization_warning"]
            .str.contains("approximate", case=False)
            .all()
        )
        self.assertTrue(
            result["harmonization_warning"]
            .str.contains("not an official current-price backcast", case=False)
            .all()
        )
        self.assertTrue(
            result["harmonization_warning"]
            .str.contains("vintage mismatch", case=False)
            .all()
        )

    def test_rolling_comparison_is_leakage_safe_and_repairs_the_break_fold(self) -> None:
        canonical = pd.read_csv(
            ROOT / "data/unified/canonical_official_annual_2010_2025.csv"
        )
        series = macro_ridge.build_harmonized_macro_series(canonical)

        predictions = macro_ridge.evaluate_rolling_models(series)

        self.assertEqual(
            set(predictions["data_scope"]),
            {"original_mixed", "harmonized_bridge"},
        )
        self.assertEqual(
            set(predictions["model"]),
            {"ridge_fixed_lambda_0.1", "ridge_nested_tuned"},
        )
        self.assertEqual(set(predictions["metric"]), set(macro_ridge.MACRO_SERIES))
        for _, group in predictions.groupby(
            ["data_scope", "model", "metric"], sort=True
        ):
            self.assertEqual(group.sort_values("year")["year"].tolist(), list(range(2015, 2024)))
        self.assertTrue(predictions["training_end_year"].lt(predictions["year"]).all())
        self.assertFalse(predictions["uses_test_in_training"].any())

        break_rows = predictions[predictions["year"].eq(2019)]
        break_scores = break_rows.pivot(
            index=["metric", "model"],
            columns="data_scope",
            values="point_smape_percent",
        )
        self.assertTrue(
            break_scores["harmonized_bridge"].lt(break_scores["original_mixed"]).all()
        )

    def test_rolling_summaries_compare_each_model_on_both_scopes(self) -> None:
        canonical = pd.read_csv(
            ROOT / "data/unified/canonical_official_annual_2010_2025.csv"
        )
        series = macro_ridge.build_harmonized_macro_series(canonical)
        predictions = macro_ridge.evaluate_rolling_models(series)

        metrics = macro_ridge.summarize_rolling_predictions(predictions)
        break_comparison = macro_ridge.build_break_fold_comparison(predictions)

        self.assertEqual(len(metrics), 8)
        self.assertTrue(metrics["n_test"].eq(9).all())
        self.assertTrue(metrics["test_years"].eq("2015;2016;2017;2018;2019;2020;2021;2022;2023").all())
        self.assertTrue(metrics["smape_percent"].ge(0.0).all())
        self.assertTrue(metrics["delta_smape_vs_original_same_model_pp"].notna().all())
        self.assertEqual(len(break_comparison), 8)
        self.assertTrue(break_comparison["year"].eq(2019).all())
        harmonized = break_comparison[
            break_comparison["data_scope"].eq("harmonized_bridge")
        ]
        self.assertTrue(harmonized["smape_improvement_vs_original_pp"].gt(0.0).all())

    def test_final_selection_holdout_and_forecast_keep_2025_out_of_training(self) -> None:
        canonical = pd.read_csv(
            ROOT / "data/unified/canonical_official_annual_2010_2025.csv"
        )
        series = macro_ridge.build_harmonized_macro_series(canonical)

        final = macro_ridge.build_final_evaluation(series)

        self.assertEqual(len(final.lambda_selection), 4)
        self.assertTrue(final.lambda_selection["training_max_year"].eq(2023).all())
        self.assertFalse(final.lambda_selection["uses_2024_for_selection"].any())
        self.assertTrue(final.lambda_selection["selected_lambda"].gt(0.0).all())

        self.assertEqual(len(final.holdout_2025), 8)
        self.assertTrue(final.holdout_2025["year"].eq(2025).all())
        self.assertTrue(final.holdout_2025["training_end_year"].eq(2024).all())
        self.assertFalse(final.holdout_2025["uses_2025_as_training"].any())
        actual_by_metric = final.holdout_2025.groupby("metric")["actual"].nunique()
        self.assertTrue(actual_by_metric.eq(1).all())

        self.assertEqual(len(final.forecasts), 40)
        self.assertEqual(set(final.forecasts["year"]), set(range(2026, 2031)))
        self.assertTrue(final.forecasts["training_end_year"].eq(2024).all())
        self.assertFalse(final.forecasts["uses_2025_as_training"].any())
        self.assertEqual(len(final.forecast_comparison), 20)
        self.assertTrue(
            final.forecast_comparison[
                "harmonized_minus_original_forecast"
            ].notna().all()
        )

    def test_run_experiment_writes_reproducible_delivery_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="macro_scope_ridge_test_") as temporary:
            output_dir = Path(temporary)

            summary = macro_ridge.run_experiment(output_dir=output_dir)

            expected = {
                "harmonized_macro_series.csv",
                "rolling_predictions.csv",
                "metrics_by_target.csv",
                "break_fold_comparison.csv",
                "final_lambda_selection.csv",
                "holdout_2025.csv",
                "forecasts_2026_2030.csv",
                "forecast_comparison.csv",
                "tourism_ridge_invariance.csv",
                "run_summary.json",
                "README.md",
            }
            self.assertEqual(set(summary["generated_files"]), expected)
            for filename in expected:
                self.assertTrue((output_dir / filename).is_file(), filename)

            invariance = pd.read_csv(output_dir / "tourism_ridge_invariance.csv")
            self.assertGreater(len(invariance), 0)
            self.assertTrue(invariance["macro_values_used_by_ridge"].eq(False).all())
            self.assertTrue(invariance["delta"].eq(0.0).all())

            persisted = json.loads(
                (output_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["generated_files"], summary["generated_files"])
            self.assertEqual(
                set(persisted["bridge_factors"]), set(macro_ridge.MACRO_SERIES)
            )
            readme = (output_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("approximate", readme.lower())
            self.assertIn("not an official current-price backcast", readme.lower())
            self.assertIn("vintage mismatch", readme.lower())
            self.assertIn("https://www.stats.gov.cn/sj/sjjd/202302/t20230202_1896273.html", readme)
            self.assertIn("W020211117618518239847.xls", readme)


if __name__ == "__main__":
    unittest.main()
