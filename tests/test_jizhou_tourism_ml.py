from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))

import model_jizhou_tourism_ml as ml_model  # noqa: E402


class FeatureConstructionTest(unittest.TestCase):
    def test_calendar_regime_features(self) -> None:
        features = ml_model.make_features([2019, 2021, 2024])
        np.testing.assert_allclose(features[:, 0], [0.9, 1.1, 1.4])
        np.testing.assert_array_equal(features[:, 1], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(features[:, 2], [0.0, 0.0, 1.0])

    def test_rolling_splits_never_use_future_rows(self) -> None:
        splits = ml_model.rolling_origin_splits(n_samples=12, min_train_size=5)
        self.assertEqual(len(splits), 7)
        for train, test in splits:
            self.assertLess(int(train.max()), int(test.min()))
            self.assertEqual(len(test), 1)


class GeneratedOutputTest(unittest.TestCase):
    def test_recommended_forecast_is_positive_and_enclosed(self) -> None:
        path = ROOT / "outputs/jizhou_tourism_ml/recommended_ml_forecasts_2025_2030.csv"
        frame = pd.read_csv(path)
        self.assertEqual(set(frame["year"]), set(range(2025, 2031)))
        self.assertEqual(len(frame), 12)
        self.assertTrue((frame["forecast"] > 0).all())
        self.assertTrue((frame["backtest_stress_lower"] < frame["forecast"]).all())
        self.assertTrue((frame["forecast"] < frame["backtest_stress_upper"]).all())

    def test_target_imputations_are_labelled_diagnostic_only(self) -> None:
        path = ROOT / "outputs/jizhou_tourism_ml/ml_missing_year_imputations.csv"
        frame = pd.read_csv(path)
        self.assertTrue(frame["use_restriction"].str.contains("do not overwrite").all())


if __name__ == "__main__":
    unittest.main()
