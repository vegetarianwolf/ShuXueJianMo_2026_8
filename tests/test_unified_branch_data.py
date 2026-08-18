from __future__ import annotations

import csv
import hashlib
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_unified_branch_data", ROOT / "scripts/build_unified_branch_data.py"
)
assert SPEC and SPEC.loader
unified = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(unified)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class UnifiedBranchDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.output = Path(cls._temp_dir.name)
        cls.summary = unified.build_unified_data(ROOT, cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_canonical_copy_is_exactly_main_blob(self) -> None:
        expected = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{unified.PINNED_AUDITED_TIPS['main']}:{unified.CANONICAL_PATH}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        actual = (self.output / "canonical_official_annual_2010_2025.csv").read_bytes()
        self.assertEqual(actual, expected)
        self.assertEqual(hashlib.sha256(actual).hexdigest(), self.summary["canonical_sha256"])

    def test_fixed_2024_holdout_and_canonical_training_contract(self) -> None:
        train = read_csv(self.output / "primary_train.csv")
        test = read_csv(self.output / "primary_test.csv")

        self.assertEqual(len(train), 22)
        self.assertEqual(len(test), 2)
        self.assertTrue(all(int(row["year"]) <= 2023 for row in train))
        self.assertTrue(all(int(row["year"]) == 2024 for row in test))
        self.assertTrue(all(row["is_observed"] == "true" for row in test))
        self.assertTrue(all(row["status"].startswith("observed") for row in test))
        self.assertEqual(
            {(row["metric"], int(row["year"])) for row in test},
            {
                ("tourist_visits", 2024),
                ("tourism_comprehensive_income", 2024),
            },
        )
        self.assertEqual(
            [
                (row["metric"], row["year"])
                for row in train
                if row["is_observed"] == "false"
            ],
            [("tourism_comprehensive_income", "2010")],
        )

    def test_benchmark_pool_preserves_the_2019_stress_marker(self) -> None:
        observations = read_csv(self.output / "benchmark_observations.csv")
        self.assertEqual(len(observations), 24)
        self.assertTrue(all(row["cutoff_year"] == "2019" for row in observations))
        self.assertEqual(
            {row["split_id"] for row in observations}, {"chronological_cutoff_2019"}
        )

    def test_rolling_origin_folds_are_past_only_and_stop_before_holdout(self) -> None:
        rows = read_csv(self.output / "rolling_origin_folds.csv")
        folds: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            folds.setdefault(row["fold_id"], []).append(row)
        self.assertEqual(len(folds), 12)
        self.assertEqual(len(rows), 102)
        expected_test_years = {
            "tourist_visits": {2015, 2016, 2017, 2018, 2019, 2023},
            "tourism_comprehensive_income": {2015, 2017, 2018, 2019, 2021, 2023},
        }
        actual_test_years = {metric: set() for metric in expected_test_years}
        for fold_rows in folds.values():
            test_rows = [row for row in fold_rows if row["role"] == "test"]
            train_rows = [row for row in fold_rows if row["role"] == "train"]
            self.assertEqual(len(test_rows), 1)
            self.assertGreaterEqual(len(train_rows), 5)
            test_row = test_rows[0]
            test_year = int(test_row["year"])
            self.assertLessEqual(test_year, 2023)
            self.assertEqual(test_row["is_observed"], "true")
            self.assertTrue(all(int(row["year"]) < test_year for row in train_rows))
            actual_test_years[test_row["metric"]].add(test_year)
        self.assertEqual(actual_test_years, expected_test_years)

    def test_stress_split_crosses_pandemic_without_touching_holdout(self) -> None:
        train = read_csv(self.output / "stress_train.csv")
        test = read_csv(self.output / "stress_test.csv")
        self.assertEqual(len(train), 19)
        self.assertEqual(len(test), 3)
        self.assertTrue(all(int(row["year"]) <= 2019 for row in train))
        self.assertTrue(all(2020 <= int(row["year"]) <= 2023 for row in test))
        self.assertTrue(all(row["is_observed"] == "true" for row in test))

    def test_sensitivity_sidecar_cannot_be_a_benchmark_label(self) -> None:
        sidecar = read_csv(self.output / "sensitivity_imputations.csv")
        test = read_csv(self.output / "primary_test.csv")
        self.assertEqual(len(sidecar), 8)
        self.assertTrue(all(row["is_observed"] == "false" for row in sidecar))
        self.assertTrue(
            all(row["excluded_from_benchmark"] == "true" for row in sidecar)
        )
        test_keys = {(row["metric"], row["year"], row["value"]) for row in test}
        sidecar_keys = {
            (row["metric"], row["year"], row["value"]) for row in sidecar
        }
        self.assertTrue(test_keys.isdisjoint(sidecar_keys))

    def test_origin_111_workbook_contracts(self) -> None:
        gdp = read_csv(self.output / "tianjin_gdp_2010_2025.csv")
        tourism = read_csv(self.output / "tianjin_tourism_benchmark.csv")
        self.assertEqual([int(row["year"]) for row in gdp], list(range(2010, 2026)))
        self.assertEqual(gdp[0]["tianjin_gdp_100m_cny"], "6830.76")
        self.assertEqual(gdp[-1]["tianjin_gdp_100m_cny"], "18539.82")
        self.assertEqual([int(row["year"]) for row in tourism], [2020, 2021, 2023, 2024])
        self.assertEqual(tourism[0]["tianjin_domestic_visitors_10k_persons"], "14100")
        self.assertEqual(tourism[-1]["tianjin_domestic_tourism_income_100m_cny"], "2930.98")

    def test_inventory_audits_five_tips_with_content_hashes_and_decisions(self) -> None:
        inventory = read_csv(self.output / "branch_data_inventory.csv")
        self.assertEqual({row["audited_ref"] for row in inventory}, set(unified.AUDITED_REFS))
        self.assertTrue(all(len(row["commit_sha"]) == 40 for row in inventory))
        self.assertTrue(all(len(row["git_blob_oid"]) == 40 for row in inventory))
        self.assertTrue(all(len(row["sha256"]) == 64 for row in inventory))
        canonical = [
            row
            for row in inventory
            if row["audited_ref"] == "main"
            and row["asset_path"] == unified.CANONICAL_PATH
        ]
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["decision"], "canonical_target_truth")
        self.assertIn("sensitivity_only", {row["decision"] for row in inventory})
        self.assertIn("accepted_auxiliary_covariate", {row["decision"] for row in inventory})
        self.assertIn("accepted_benchmark_context", {row["decision"] for row in inventory})

    def test_generation_never_resolves_or_reads_live_branch_names(self) -> None:
        original_git = unified._git

        def reject_live_refs(
            repo_root: Path, args: list[str], *, text: bool = False
        ) -> bytes | str:
            for argument in args:
                for label in unified.AUDITED_REFS:
                    if argument == label or argument.startswith(f"{label}:"):
                        raise AssertionError(f"live ref leaked into git command: {argument}")
            return original_git(repo_root, args, text=text)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(unified, "_git", side_effect=reject_live_refs):
                summary = unified.build_unified_data(ROOT, Path(directory))
        self.assertEqual(
            summary["canonical_commit"], unified.PINNED_AUDITED_TIPS["main"]
        )

    def test_checked_in_outputs_are_reproducible(self) -> None:
        checked_in = ROOT / "data/unified"
        for generated in sorted(self.output.iterdir()):
            self.assertEqual(
                generated.read_bytes(),
                (checked_in / generated.name).read_bytes(),
                generated.name,
            )


if __name__ == "__main__":
    unittest.main()
