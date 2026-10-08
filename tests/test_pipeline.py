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

    def test_host_runner_captures_state_and_runs_all(self):
        events = []
        with patch.object(runner, "capture_git_state", side_effect=lambda: events.append("capture")), \
             patch.object(runner.subprocess, "run", side_effect=lambda *args, **kwargs: events.append("run")) as run:
            runner.main()
        self.assertEqual(events, ["capture", "run"])
        run.assert_called_once_with(
            [sys.executable, str(runner.ROOT / "main.py"), "all"], cwd=runner.ROOT, check=True,
        )


if __name__ == "__main__":
    unittest.main()
