import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np

from consumerbr_resolution.baselines import evaluate_historical_baselines, score_company_split
from consumerbr_resolution.experiments.report_export import PREPARATION_REPORTS, export_reports
from consumerbr_resolution.experiments.reproducibility import (
    register_execution, sha256_file, source_files, validate_execution, write_json,
)


class CompanyBaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tables = self.root / "results/study/tables"
        self.predictions = self.root / "results/study/predictions"
        self.source = self.root / "features.parquet"
        for name in ("main.py", "pyproject.toml", "uv.lock"):
            (self.root / name).write_text("fixture\n")
        write_json(self.root / "logs/git_state.json", {
            "commit": "fixture", "branch": "experiment/text-metadata-ablation",
            "tags": [], "status": "", "files": source_files(self.root),
        })
        dates = (
            "2021-05-01", "2023-11-30", "2023-12-01", "2023-12-31",
            "2024-01-01", "2024-08-31", "2024-09-01", "2024-09-30",
            "2024-10-01", "2025-03-31", "2025-04-01", "2025-04-03",
        )
        companies = ("A", "B", "A", "A", "A", "UNKNOWN", "A", "A", "A", "B", "A", "A")
        self.rows = [
            (i, f"id-{i}", date, i % 2, f"text-{i}", companies[i])
            for i, date in enumerate(dates)
        ]
        self.save_source()
        register_execution(self.root, self.source, self.tables)

    def save_source(self):
        if self.source.exists():
            self.source.unlink()
        with duckdb.connect() as connection:
            connection.execute(
                "CREATE TABLE records(record_id INTEGER, complaint_id VARCHAR, "
                "opening_date DATE, target_resolved INTEGER, complaint_text VARCHAR, company VARCHAR)"
            )
            connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)", self.rows)
            escaped = str(self.source).replace("'", "''")
            connection.execute(f"COPY records TO '{escaped}' (FORMAT PARQUET)")

    def score(self):
        with duckdb.connect() as connection:
            return score_company_split(
                connection, self.source, self.tables / "split_membership.parquet", "validation", 100,
            )

    def test_smoothed_score_and_unknown_company(self):
        frame = self.score()
        np.testing.assert_allclose(frame["score"], [50 / 101, 0.5])
        self.assertEqual(frame["company_seen"].tolist(), [True, False])

    def test_future_labels_and_gap_do_not_affect_history(self):
        before = self.score()["score"].to_numpy()
        self.rows = [row if row[0] < 2 else (*row[:3], 1 - row[3], *row[4:]) for row in self.rows]
        self.save_source()
        np.testing.assert_array_equal(before, self.score()["score"].to_numpy())

    def test_changed_data_blocks_reuse(self):
        self.rows[8] = (*self.rows[8][:4], "changed text", self.rows[8][5])
        self.save_source()
        with self.assertRaisesRegex(RuntimeError, "inputs changed"):
            validate_execution(self.root, self.source, self.tables)

    def test_changed_code_requires_host_snapshot(self):
        (self.root / "main.py").write_text("changed\n")
        with self.assertRaisesRegex(RuntimeError, "Capture Git state"):
            validate_execution(self.root, self.source, self.tables)

    def test_registration_reuses_existing_manifest(self):
        path = self.tables / "execution_manifest.json"
        before = path.read_bytes()
        register_execution(self.root, self.source, self.tables)
        self.assertEqual(before, path.read_bytes())

    def test_validation_threshold_is_reused_on_test(self):
        result = evaluate_historical_baselines(
            self.root, self.source, self.tables, self.predictions,
        )
        with duckdb.connect() as connection:
            for split in ("validation", "test"):
                path = self.predictions / "company_historical_rate" / f"{split}.parquet"
                thresholds = connection.execute(
                    "SELECT DISTINCT threshold FROM read_parquet(?)", [str(path)],
                ).fetchall()
                self.assertEqual(thresholds, [(result["threshold"],)])
        with patch("consumerbr_resolution.baselines.score_company_split") as score:
            evaluate_historical_baselines(self.root, self.source, self.tables, self.predictions)
            score.assert_not_called()

    def test_corrupted_predictions_block_reuse(self):
        evaluate_historical_baselines(self.root, self.source, self.tables, self.predictions)
        path = self.predictions / "company_historical_rate/test.parquet"
        path.write_bytes(b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
            evaluate_historical_baselines(self.root, self.source, self.tables, self.predictions)

    def test_export_normalizes_reports_and_excludes_predictions(self):
        evaluate_historical_baselines(self.root, self.source, self.tables, self.predictions)
        for name in PREPARATION_REPORTS:
            path = self.tables / name
            if not path.exists():
                path.write_bytes(b"field,value\r\nfixture,1\r\n")
        destination = self.root / "reports/study"
        export_reports(self.root, self.source, self.tables, destination)
        manifest = json.loads((destination / "report_manifest.json").read_text())
        for name, digest in manifest["files"].items():
            self.assertEqual(sha256_file(destination / name), digest)
            self.assertNotIn(b"\r\n", (destination / name).read_bytes())
        self.assertEqual(list(destination.rglob("*.parquet")), [])
        self.assertIn("company_baseline_metrics.csv", manifest["files"])


if __name__ == "__main__":
    unittest.main()
