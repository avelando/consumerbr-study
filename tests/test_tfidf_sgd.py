import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import joblib
import numpy as np
import pandas as pd

from consumerbr_resolution.experiments.report_export import PREPARATION_REPORTS, export_reports
from consumerbr_resolution.experiments.reproducibility import register_execution, source_files, write_json
from consumerbr_resolution.modeling import tfidf_sgd
from consumerbr_resolution.modeling.tfidf import load_split


class TfidfSGDTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "features.parquet"
        self.tables = self.root / "results/study/tables"
        for name in ("main.py", "pyproject.toml", "uv.lock"):
            (self.root / name).write_text("fixture\n")
        rows = []
        for date, count, marker in (
            ("2022-01-01", 18, "trainingtoken"), ("2023-12-01", 2, "gaptoken"),
            ("2024-01-01", 4, "validationtoken"), ("2024-09-01", 2, "gaptoken"),
            ("2024-10-01", 4, "testtoken"), ("2025-04-01", 2, "aftertoken"),
        ):
            for index in range(count):
                label = index % 2
                text = ("refund accepted " if label else "request denied ") + marker
                rows.append((len(rows), f"id-{len(rows)}", date, label, text, "company"))
        with duckdb.connect() as connection:
            connection.execute(
                "CREATE TABLE records(record_id INTEGER, complaint_id VARCHAR, "
                "opening_date DATE, target_resolved INTEGER, complaint_text VARCHAR, company VARCHAR)"
            )
            connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)", rows)
            connection.execute(f"COPY records TO '{self.source}' (FORMAT PARQUET)")
        write_json(self.root / "logs/git_state.json", {
            "commit": "fixture", "branch": "fixture", "files": source_files(self.root),
        })
        register_execution(self.root, self.source, self.tables)

    def run_study(self):
        return tfidf_sgd.evaluate_tfidf_sgd(self.root, self.source, self.tables)

    def test_five_training_runs_and_train_only_vocabulary(self):
        events = []
        original = tfidf_sgd.train_candidate
        def train(*args):
            events.append("training")
            return original(*args)
        def load(*args):
            events.append(args[-1])
            return load_split(*args)
        with patch.object(tfidf_sgd, "train_candidate", side_effect=train), \
             patch.object(tfidf_sgd, "load_split", side_effect=load):
            result = self.run_study()
        self.assertEqual(events, ["train", "validation"] + ["training"] * 5 + ["test"])
        self.assertEqual(result["training_runs"], 5)
        self.assertEqual(result["seeds"], [42, 13, 101])
        vectorizer = joblib.load(self.root / "models/study/classical/tfidf_sgd/vectorizer.joblib")
        for token in ("validationtoken", "testtoken", "gaptoken", "aftertoken"):
            self.assertNotIn(token, vectorizer.vocabulary_)
        metrics = pd.read_csv(self.tables / "tfidf_sgd_metrics.csv")
        self.assertEqual(len(metrics), 6)
        self.assertEqual(metrics["alpha"].nunique(), 1)
        for seed in (42, 13, 101):
            pair = metrics[metrics.seed == seed]
            self.assertEqual(pair["threshold"].nunique(), 1)
        selection = pd.read_csv(self.tables / "tfidf_sgd_selection.csv")
        self.assertEqual(len(selection), 3)
        self.assertEqual(selection["selected"].sum(), 1)
        with patch.object(tfidf_sgd, "load_split") as loader:
            self.run_study()
            loader.assert_not_called()

    def test_same_seed_reproduces_training_and_threshold(self):
        self.run_study()
        directory = self.root / "models/study/classical/tfidf_sgd/candidate_00_seed_42"
        previous = json.loads((directory / "run.json").read_text())
        with duckdb.connect() as connection:
            membership = self.tables / "split_membership.parquet"
            train = load_split(connection, self.source, membership, "train")
            validation = load_split(connection, self.source, membership, "validation")
        vectorizer_path = directory.parent / "vectorizer.joblib"
        vectorizer = joblib.load(vectorizer_path)
        repeated = tfidf_sgd.train_candidate(
            vectorizer.transform(train.complaint_text), train.target_resolved.to_numpy(dtype=np.int8),
            vectorizer.transform(validation.complaint_text), validation.target_resolved.to_numpy(dtype=np.int8),
            previous["alpha"], 42, directory.parent / "repeat", self.root,
            previous["fingerprint"], vectorizer_path,
        )
        self.assertEqual(previous["history"], repeated["history"])
        self.assertEqual(previous["threshold"], repeated["threshold"])

    def test_completed_candidates_resume_after_interruption(self):
        original = tfidf_sgd.train_candidate
        counter = 0
        def interrupt(*args):
            nonlocal counter
            counter += 1
            if counter == 2:
                raise RuntimeError("interrupted")
            return original(*args)
        with patch.object(tfidf_sgd, "train_candidate", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_study()
        path = self.root / "models/study/classical/tfidf_sgd/candidate_00_seed_42/model.joblib"
        before = path.stat().st_mtime_ns
        self.run_study()
        self.assertEqual(path.stat().st_mtime_ns, before)

    def test_corrupted_models_block_reuse_and_export(self):
        self.run_study()
        path = self.root / "models/study/classical/tfidf_sgd/candidate_00_seed_42/model.joblib"
        path.write_bytes(b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
            self.run_study()
        with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
            export_reports(self.root, self.source, self.tables)

    def test_export_contains_aggregate_reports_only(self):
        self.run_study()
        for name in PREPARATION_REPORTS:
            path = self.tables / name
            if not path.exists():
                path.write_text("field,value\nfixture,1\n")
        destination = export_reports(self.root, self.source, self.tables)
        manifest = json.loads((destination / "report_manifest.json").read_text())
        self.assertEqual(manifest["stage"], "tfidf_sgd")
        self.assertTrue(set(tfidf_sgd.REPORTS).issubset(manifest["files"]))
        self.assertFalse(list(destination.rglob("*.parquet")))
        self.assertFalse(list(destination.rglob("*.joblib")))


if __name__ == "__main__":
    unittest.main()
