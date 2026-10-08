import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from consumerbr_resolution.config import (
    BERTIMBAU_MAX_LENGTH, BERTIMBAU_MODEL_NAME, BERTIMBAU_REVISION,
    FEATURE_BASE_PATH, PROJECT_ROOT, TABLES_DIR,
)
from consumerbr_resolution.experiments.temporal_protocol import build_temporal_protocol


PACKAGE_NAMES = (
    "duckdb", "huggingface-hub", "joblib", "numpy", "pandas", "pyarrow",
    "requests", "scikit-learn", "scipy", "torch", "transformers", "tqdm",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def source_files(root):
    root = Path(root)
    paths = [root / "main.py", root / "pyproject.toml", root / "uv.lock"]
    for name in ("src", "scripts", "tests"):
        paths.extend((root / name).rglob("*.py"))
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(paths)}


def execution_identity(root, source, tables):
    root, source, tables = Path(root), Path(source), Path(tables)
    provenance = json.loads((root / "logs/git_state.json").read_text(encoding="utf-8"))
    files = source_files(root)
    if provenance["files"] != files:
        raise RuntimeError("Source files changed. Capture Git state again on the host.")
    protocol = json.loads((tables / "experimental_protocol.json").read_text(encoding="utf-8"))
    for key in ("source_path", "source_size_bytes", "source_mtime_ns"):
        protocol.pop(key, None)
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    identity = {
        "source_sha256": sha256_file(source),
        "membership_sha256": sha256_file(tables / "split_membership.parquet"),
        "protocol": protocol,
        "files": files,
        "python": platform.python_version(),
        "packages": packages,
        "transformer": {"name": BERTIMBAU_MODEL_NAME, "revision": BERTIMBAU_REVISION,
                        "max_length": BERTIMBAU_MAX_LENGTH},
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return identity, fingerprint, provenance


def validate_execution(root=None, source=None, tables=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    source = Path(source) if source is not None else FEATURE_BASE_PATH
    tables = Path(tables) if tables is not None else TABLES_DIR
    manifest = json.loads((tables / "execution_manifest.json").read_text(encoding="utf-8"))
    identity, fingerprint, _ = execution_identity(root, source, tables)
    if manifest["fingerprint"] != fingerprint or manifest["identity"] != identity:
        raise RuntimeError("Execution inputs changed. Existing results cannot be reused.")
    return manifest


def register_execution(root=None, source=None, tables=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    source = Path(source) if source is not None else FEATURE_BASE_PATH
    tables = Path(tables) if tables is not None else TABLES_DIR
    if not (root / "logs/git_state.json").is_file():
        raise FileNotFoundError("Run python3 scripts/capture_git_state.py on the host first.")
    destination = tables / "execution_manifest.json"
    if destination.exists():
        manifest = validate_execution(root, source, tables)
        print(f"Execution manifest verified: {destination}")
        return manifest
    build_temporal_protocol(source, tables)
    identity, fingerprint, provenance = execution_identity(root, source, tables)
    manifest = {"fingerprint": fingerprint, "identity": identity, "git": provenance}
    write_json(destination, manifest)
    print(f"Execution registered: {fingerprint}")
    return manifest
