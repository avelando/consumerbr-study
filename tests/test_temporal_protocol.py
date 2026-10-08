import csv
import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from consumerbr_resolution.config import TEMPORAL_FOLDS
from consumerbr_resolution.experiments.temporal_design import (
    build_single_temporal_fold,
    get_temporal_windows,
    validate_temporal_folds,
)
from consumerbr_resolution.experiments.temporal_protocol import build_temporal_protocol


class TemporalDesignTests(unittest.TestCase):
    def test_gap_windows_and_partial_april_are_excluded(self):
        windows = get_temporal_windows(TEMPORAL_FOLDS[0], "2025-04-03")
        excluded = [item for item in windows if not item["included"]]
        self.assertEqual(
            [(item["start_date"], item["end_date"]) for item in excluded],
            [("2023-12-01", "2023-12-31"),
             ("2024-09-01", "2024-09-30"),
             ("2025-04-01", "2025-04-03")],
        )

    def test_contiguous_partitions_are_rejected(self):
        with self.assertRaises(ValueError):
            build_single_temporal_fold(
                "2021-05-01", "2023-12-31", "2024-01-01",
                "2024-08-31", "2024-10-01", "2025-03-31",
            )

    def test_partial_month_is_rejected(self):
        with self.assertRaises(ValueError):
            build_single_temporal_fold(
                "2021-05-01", "2023-11-30", "2024-01-02",
                "2024-08-31", "2024-10-01", "2025-03-31",
            )

    def test_multiple_folds_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_temporal_folds(TEMPORAL_FOLDS * 2)


class TemporalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "features.parquet"
        self.destination = self.root / "tables"
        dates = [
            "2021-05-01", "2023-11-30", "2023-12-01", "2023-12-31",
            "2024-01-01", "2024-08-31", "2024-09-01", "2024-09-30",
            "2024-10-01", "2025-03-31", "2025-04-01", "2025-04-03",
        ]
        self.rows = [
            (index, f"id-{index}", value, index % 2, f"text-{index}")
            for index, value in enumerate(dates)
        ]

    def save_source(self):
        with duckdb.connect() as connection:
            connection.execute(
                "CREATE TABLE records(record_id INTEGER, complaint_id VARCHAR, "
                "opening_date DATE, target_resolved INTEGER, complaint_text VARCHAR)"
            )
            connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", self.rows)
            escaped = str(self.source).replace("'", "''")
            connection.execute(f"COPY records TO '{escaped}' (FORMAT PARQUET)")

    def run_protocol(self):
        self.save_source()
        return build_temporal_protocol(self.source, self.destination)

    def test_boundary_assignments_and_outputs(self):
        summaries = self.run_protocol()
        self.assertEqual(sum(item["complaint_count"] for item in summaries), 12)
        with duckdb.connect() as connection:
            assigned = connection.execute(
                "SELECT split FROM read_parquet(?) ORDER BY record_id",
                [str(self.destination / "split_membership.parquet")],
            ).fetchall()
        self.assertEqual(
            [row[0] for row in assigned],
            ["train", "train", "gap_train_validation", "gap_train_validation",
             "validation", "validation", "gap_validation_test", "gap_validation_test",
             "test", "test", "after_test", "after_test"],
        )
        protocol = json.loads(
            (self.destination / "experimental_protocol.json").read_text()
        )
        self.assertEqual(protocol["seeds"], [42, 13, 101])
        self.assertTrue(protocol["prior_test_inspection"])

    def test_duplicate_record_ids_are_rejected(self):
        self.rows[1] = (0, *self.rows[1][1:])
        with self.assertRaisesRegex(RuntimeError, "unique_record_ids"):
            self.run_protocol()
        self.assertFalse((self.destination / "split_membership.parquet").exists())

    def test_cross_partition_complaint_ids_are_rejected(self):
        row = self.rows[4]
        self.rows[4] = (row[0], self.rows[0][1], *row[2:])
        with self.assertRaisesRegex(RuntimeError, "complaint_ids_do_not_cross"):
            self.run_protocol()

    def test_missing_class_is_rejected(self):
        row = self.rows[9]
        self.rows[9] = (*row[:3], 0, row[4])
        with self.assertRaisesRegex(RuntimeError, "both_classes_in_test"):
            self.run_protocol()

    def test_null_date_is_rejected(self):
        row = self.rows[2]
        self.rows[2] = (*row[:2], None, *row[3:])
        with self.assertRaisesRegex(RuntimeError, "valid_required_values"):
            self.run_protocol()

    def test_text_overlap_is_reported_without_deleting_records(self):
        row = self.rows[4]
        self.rows[4] = (*row[:4], self.rows[0][4])
        self.run_protocol()
        with (self.destination / "temporal_text_overlap.csv").open() as file:
            overlap = list(csv.DictReader(file))
        self.assertEqual(overlap[0]["shared_exact_texts"], "1")


if __name__ == "__main__":
    unittest.main()
