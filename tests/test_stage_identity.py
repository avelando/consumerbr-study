import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import test_tfidf_sgd as fixtures

from consumerbr_resolution import config
from consumerbr_resolution.baselines import evaluate_historical_baselines
from consumerbr_resolution.experiments.reproducibility import (
    register_execution, sha256_file, source_files, validate_execution, write_json,
)
from consumerbr_resolution.experiments.stage_identity import (
    STAGES, digest, migration_reference, stage_identities,
)
from consumerbr_resolution.modeling.tfidf_sgd import evaluate_tfidf_sgd


class StageIdentityTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.TfidfSGDTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.source, self.tables = fixture.root, fixture.source, fixture.tables
        self.models = self.root / "models/study"
        shutil.copytree(config.PROJECT_ROOT / "src", self.root / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        self.capture()
        self.manifest = register_execution(self.root, self.source, self.tables)

    def capture(self):
        write_json(self.root / "logs/git_state.json", {
            "commit": "fixture", "branch": "fixture", "files": source_files(self.root),
        })

    def complete(self):
        evaluate_historical_baselines(self.root, self.source, self.tables,
                                      self.tables.parent / "predictions")
        evaluate_tfidf_sgd(self.root, self.source, self.tables)

    def update(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(target.read_text() + text if target.exists() else text)
        self.capture()
        return register_execution(self.root, self.source, self.tables)

    def baseline_bytes(self):
        return {name: (self.tables / name).read_bytes() for name in (
            "company_baseline_metrics.csv", "company_baseline_run.json",
        )}

    def legacy_manifest(self, unknown=False):
        manifest = validate_execution(self.root, self.source, self.tables)
        identity = json.loads(json.dumps(manifest["identity"]))
        for record in migration_reference()["stages"].values():
            identity["files"].update(record["raw_files"])
        if unknown:
            identity["files"]["src/consumerbr_resolution/modeling/tfidf_sgd.py"] = "unknown"
        fingerprint = digest(identity)
        for record in manifest["stages"].values():
            record["artifact_fingerprint"] = fingerprint
        write_json(self.tables / "execution_manifest.json", manifest)
        self.complete()
        previous = {"fingerprint": fingerprint, "identity": identity, "git": manifest["git"]}
        write_json(self.tables / "execution_manifest.json", previous)
        return previous

    def test_analysis_changes_preserve_completed_models_and_predictions(self):
        self.complete()
        before = self.baseline_bytes()
        model = self.models / "classical/tfidf_sgd/candidate_00_seed_42/model.joblib"
        model_time, model_hash = model.stat().st_mtime_ns, sha256_file(model)
        current = self.update("src/consumerbr_resolution/evaluation/new_analysis.py", "VALUE = 1\n")
        self.assertNotEqual(self.manifest["fingerprint"], current["fingerprint"])
        self.assertEqual(before, self.baseline_bytes())
        self.assertEqual(model_time, model.stat().st_mtime_ns)
        self.assertEqual(model_hash, sha256_file(model))
        with patch("consumerbr_resolution.modeling.tfidf_sgd.load_split") as loader:
            evaluate_tfidf_sgd(self.root, self.source, self.tables)
            loader.assert_not_called()

    def test_new_tests_preserve_stage_identities(self):
        current = self.update("tests/test_new_analysis.py", "VALUE = 1\n")
        for stage in current["stages"]:
            self.assertEqual(self.manifest["stages"][stage], current["stages"][stage])

    def test_transformer_training_changes_preserve_classical_stages(self):
        current = self.update("src/consumerbr_resolution/modeling/bertimbau_finetuning.py",
                              "\nNEW_TRAINER_VERSION = 2\n")
        for stage in ("company_baseline", "tfidf_sgd", "bertimbau_tokens"):
            self.assertEqual(self.manifest["stages"][stage], current["stages"][stage])

    def test_sgd_changes_archive_only_sgd_artifacts(self):
        self.complete()
        before = self.baseline_bytes()
        current = self.update("src/consumerbr_resolution/modeling/tfidf_sgd.py", "\nVERSION = 2\n")
        self.assertEqual(before, self.baseline_bytes())
        self.assertFalse((self.tables / "tfidf_sgd_run.json").exists())
        self.assertFalse((self.models / "classical/tfidf_sgd").exists())
        self.assertNotEqual(self.manifest["stages"]["tfidf_sgd"]["dependency_fingerprint"],
                            current["stages"]["tfidf_sgd"]["dependency_fingerprint"])
        self.assertTrue(list((self.root / "results/archive").rglob("model.joblib")))
        with patch("consumerbr_resolution.baselines.score_company_split") as score:
            evaluate_historical_baselines(self.root, self.source, self.tables,
                                          self.tables.parent / "predictions")
            score.assert_not_called()

    def test_registering_a_new_training_stage_preserves_existing_stages(self):
        spec = {
            "files": ("modeling/bertimbau_finetuning.py",),
            "config": ("BERTIMBAU_MAX_LENGTH",), "protocol": (),
            "packages": ("torch", "transformers"), "reports": (),
            "upstream": ("bertimbau_assets", "bertimbau_tokens"),
        }
        with patch.dict(STAGES, {"bertimbau_text": spec}):
            current = register_execution(self.root, self.source, self.tables)
        for stage, before in self.manifest["stages"].items():
            self.assertEqual(before, current["stages"][stage])
        self.assertIn("bertimbau_text", current["stages"])

    def test_bert_length_change_does_not_invalidate_classical_stages(self):
        with patch.object(config, "BERTIMBAU_MAX_LENGTH", 512):
            current = register_execution(self.root, self.source, self.tables)
        for stage in ("company_baseline", "tfidf_sgd", "bertimbau_assets"):
            self.assertEqual(self.manifest["stages"][stage], current["stages"][stage])
        for stage in ("bertimbau_preflight", "bertimbau_tokens"):
            self.assertNotEqual(self.manifest["stages"][stage]["dependency_fingerprint"],
                                current["stages"][stage]["dependency_fingerprint"])

    def test_bootstrap_configuration_does_not_invalidate_training(self):
        with patch("consumerbr_resolution.experiments.temporal_protocol.BOOTSTRAP_REPLICATES", 3000):
            current = register_execution(self.root, self.source, self.tables)
        self.assertNotEqual(self.manifest["fingerprint"], current["fingerprint"])
        self.assertEqual(self.manifest["stages"], current["stages"])

    def test_changed_data_blocks_reuse_until_registration(self):
        before = self.manifest["stages"]
        self.source.write_bytes(self.source.read_bytes() + b" ")
        with self.assertRaisesRegex(RuntimeError, "inputs changed"):
            validate_execution(self.root, self.source, self.tables, stage="tfidf_sgd")
        identity = dict(self.manifest["identity"], source_sha256=sha256_file(self.source))
        after = stage_identities(self.root, identity)
        for stage in ("company_baseline", "tfidf_sgd", "bertimbau_tokens"):
            self.assertNotEqual(before[stage]["dependency_fingerprint"],
                                after[stage]["dependency_fingerprint"])
        for stage in ("bertimbau_assets", "bertimbau_preflight"):
            self.assertEqual(before[stage]["dependency_fingerprint"],
                             after[stage]["dependency_fingerprint"])

    def test_missing_snapshot_blocks_registration_before_archiving(self):
        evaluate_historical_baselines(self.root, self.source, self.tables,
                                      self.tables.parent / "predictions")
        before = self.baseline_bytes()
        (self.root / "main.py").write_text("changed\n")
        with self.assertRaisesRegex(RuntimeError, "Capture Git state"):
            register_execution(self.root, self.source, self.tables)
        self.assertEqual(before, self.baseline_bytes())

    def test_corrupted_artifacts_block_registration_without_moving_models(self):
        self.complete()
        model = self.models / "classical/tfidf_sgd/candidate_00_seed_42/model.joblib"
        model.write_bytes(b"corrupted")
        target = self.root / "tests/test_new.py"
        target.parent.mkdir(exist_ok=True)
        target.write_text("VALUE = 1\n")
        self.capture()
        with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
            register_execution(self.root, self.source, self.tables)
        self.assertEqual(model.read_bytes(), b"corrupted")
        self.assertTrue((self.tables / "tfidf_sgd_run.json").exists())

    def test_legacy_migration_preserves_artifacts_and_their_original_identity(self):
        previous = self.legacy_manifest()
        before = self.baseline_bytes()
        model = self.models / "classical/tfidf_sgd/candidate_00_seed_42/model.joblib"
        model_time = model.stat().st_mtime_ns
        current = register_execution(self.root, self.source, self.tables)
        self.assertEqual(current["schema_version"], 2)
        self.assertEqual(before, self.baseline_bytes())
        self.assertEqual(model_time, model.stat().st_mtime_ns)
        for stage in ("company_baseline", "tfidf_sgd"):
            self.assertEqual(current["stages"][stage]["artifact_fingerprint"], previous["fingerprint"])
            self.assertEqual(current["stages"][stage]["origin_execution_fingerprint"], previous["fingerprint"])
        with patch("consumerbr_resolution.modeling.tfidf_sgd.load_split") as loader:
            evaluate_tfidf_sgd(self.root, self.source, self.tables)
            loader.assert_not_called()

    def test_unknown_legacy_sgd_source_preserves_the_company_baseline(self):
        previous = self.legacy_manifest(unknown=True)
        before = self.baseline_bytes()
        old_tag = previous["fingerprint"]
        current = register_execution(self.root, self.source, self.tables)
        self.assertEqual(before, self.baseline_bytes())
        self.assertFalse((self.tables / "tfidf_sgd_run.json").exists())
        self.assertEqual(current["stages"]["company_baseline"]["artifact_fingerprint"], old_tag)

    def test_paths_outside_the_project_are_rejected(self):
        evaluate_historical_baselines(self.root, self.source, self.tables,
                                      self.tables.parent / "predictions")
        path = self.tables / "company_baseline_run.json"
        record = json.loads(path.read_text())
        record["artifacts"]["../../../../outside.parquet"] = "unknown"
        write_json(path, record)
        with self.assertRaisesRegex(RuntimeError, "escapes the project"):
            register_execution(self.root, self.source, self.tables)


if __name__ == "__main__":
    unittest.main()
