import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_git_state import capture_git_state


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consumerbr_resolution.config import FEATURE_BASE_PATH, MODELS_DIR, TABLES_DIR
from consumerbr_resolution.experiments.reproducibility import execution_identity, write_json


def preserve_previous_execution(root, source, tables, models):
    root, source, tables, models = map(Path, (root, source, tables, models))
    path = tables / "execution_manifest.json"
    if not path.is_file():
        return None
    previous = json.loads(path.read_text(encoding="utf-8"))
    _, fingerprint, _ = execution_identity(root, source, tables)
    if fingerprint == previous["fingerprint"]:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / "results/archive" / tables.parent.name / (
        previous["fingerprint"][:12] + "-" + timestamp
    )
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "archive_reason.json", {
        "previous_fingerprint": previous["fingerprint"],
        "current_fingerprint": fingerprint,
        "reason": "Execution inputs changed; preserve previous artifacts before rerunning.",
    })
    shutil.move(str(tables.parent), str(destination / "results"))
    if models.exists():
        shutil.move(str(models), str(destination / "models"))
    print(f"Previous execution preserved: {destination}", flush=True)
    return destination


def main():
    capture_git_state()
    preserve_previous_execution(ROOT, FEATURE_BASE_PATH, TABLES_DIR, MODELS_DIR)
    subprocess.run([sys.executable, str(ROOT / "main.py"), "all"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
