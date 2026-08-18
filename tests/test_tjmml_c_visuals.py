from __future__ import annotations

import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))

from render_tjmml_c_visuals import FIGURE_NAMES, render_problem_visuals  # noqa: E402


class ProblemAlignedVisualContractTest(unittest.TestCase):
    OUTPUT = ROOT / "outputs/unified_model_benchmark"
    CANONICAL = ROOT / "data/unified/canonical_official_annual_2010_2025.csv"
    INPUT_FILES = (
        "primary_train_augmented.csv",
        "stability_metrics_by_target.csv",
        "stability_official_only_common_row_metrics_by_target.csv",
        "final_holdout_2024_metrics_by_target.csv",
        "problem1_simple_growth_forecasts_2026_2030.csv",
        "problem2_forecasts_2026_2030.csv",
        "problem3_scenario_forecasts_2026_2030.csv",
        "problem3_policy_sensitivity.csv",
    )

    def test_checked_in_figures_are_readable_large_pngs(self) -> None:
        for name in FIGURE_NAMES:
            path = self.OUTPUT / name
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"), name)
            width, height = struct.unpack(">II", data[16:24])
            self.assertGreaterEqual(width, 1600, name)
            self.assertGreaterEqual(height, 700, name)

    def test_report_embeds_every_problem_figure_and_is_organized_by_q1_q2_q3(self) -> None:
        report = (ROOT / "docs/unified_branch_model_comparison.md").read_text(
            encoding="utf-8"
        )
        for name in FIGURE_NAMES:
            self.assertIn(f"../outputs/unified_model_benchmark/{name}", report)
        for heading in ("问题1", "问题2", "问题3"):
            self.assertIn(heading, report)

    def test_render_is_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = base / "first"
            second = base / "second"
            first.mkdir()
            second.mkdir()
            for name in self.INPUT_FILES:
                shutil.copy2(self.OUTPUT / name, first / name)
                shutil.copy2(self.OUTPUT / name, second / name)
            render_problem_visuals(first, self.CANONICAL)
            render_problem_visuals(second, self.CANONICAL)
            for name in FIGURE_NAMES:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
