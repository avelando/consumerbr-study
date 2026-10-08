import subprocess
import sys
from pathlib import Path

from capture_git_state import capture_git_state


ROOT = Path(__file__).resolve().parents[1]


def main():
    capture_git_state()
    subprocess.run([sys.executable, str(ROOT / "main.py"), "all"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
