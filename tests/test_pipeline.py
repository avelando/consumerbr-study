import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from consumerbr_resolution.pipeline import STAGES, run_all, run_stage_by_command


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("host_pipeline_runner", ROOT / "scripts/run_pipeline.py")
runner = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec.loader.exec_module(runner)
finally:
    sys.path.pop(0)


class PipelineTests(unittest.TestCase):
    def test_all_stages_run_once_in_order(self):
        with patch("consumerbr_resolution.pipeline.execute_stage") as execute:
            run_all()
        self.assertEqual(execute.call_args_list, [
            unittest.mock.call(index, stage) for index, stage in enumerate(STAGES, start=1)
        ])

    def test_first_failure_stops_remaining_stages(self):
        with patch("consumerbr_resolution.pipeline.execute_stage") as execute:
            execute.side_effect = [None, RuntimeError("failed")]
            with self.assertRaisesRegex(RuntimeError, "failed"):
                run_all()
        self.assertEqual(execute.call_count, 2)

    def test_individual_stage_keeps_its_position(self):
        with patch("consumerbr_resolution.pipeline.execute_stage") as execute:
            run_stage_by_command("temporal-protocol")
        execute.assert_called_once_with(10, STAGES[9])

    def test_commands_are_unique(self):
        commands = [stage.command for stage in STAGES]
        self.assertEqual(len(commands), len(set(commands)))

    def test_changed_execution_preserves_results_and_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = root / "results/study/tables"
            models = root / "models/study"
            tables.mkdir(parents=True)
            models.mkdir(parents=True)
            previous = {"fingerprint": "old-fingerprint"}
            (tables / "execution_manifest.json").write_text(json.dumps(previous))
            (tables / "metrics.csv").write_text("original metrics\n")
            (models / "weights.bin").write_bytes(b"original model")
            with patch.object(runner, "execution_identity", return_value=({}, "new-fingerprint", {})):
                destination = runner.preserve_previous_execution(root, root / "source", tables, models)
            self.assertEqual((destination / "results/tables/metrics.csv").read_text(), "original metrics\n")
            self.assertEqual((destination / "models/weights.bin").read_bytes(), b"original model")
            self.assertFalse(tables.parent.exists())
            self.assertFalse(models.exists())

    def test_matching_execution_is_preserved_in_place(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = root / "results/study/tables"
            tables.mkdir(parents=True)
            manifest = tables / "execution_manifest.json"
            manifest.write_text(json.dumps({"fingerprint": "same"}))
            with patch.object(runner, "execution_identity", return_value=({}, "same", {})):
                destination = runner.preserve_previous_execution(
                    root, root / "source", tables, root / "models/study",
                )
            self.assertIsNone(destination)
            self.assertTrue(manifest.exists())
            self.assertFalse((root / "results/archive").exists())


if __name__ == "__main__":
    unittest.main()
