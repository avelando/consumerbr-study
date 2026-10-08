import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def capture_git_state():
    paths = [ROOT / "main.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for name in ("src", "scripts", "tests"):
        paths.extend((ROOT / name).rglob("*.py"))
    files = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    state = {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tags": git("tag", "--points-at", "HEAD").splitlines(),
        "status": git("status", "--porcelain"),
        "files": files,
    }
    destination = ROOT / "logs/git_state.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Git state captured: {destination}")


if __name__ == "__main__":
    capture_git_state()
