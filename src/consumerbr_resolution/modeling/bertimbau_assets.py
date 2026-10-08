import json
from pathlib import Path

from consumerbr_resolution.config import (
    BERTIMBAU_MODEL_NAME, BERTIMBAU_PRETRAINED_DIR, BERTIMBAU_REVISION, PROJECT_ROOT, TABLES_DIR,
)
from consumerbr_resolution.experiments.reproducibility import sha256_file, validate_execution, write_json


REPORT_NAME = "bertimbau_assets.json"
TOKENIZER_FILES = ("config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json")


def asset_directory(root):
    return Path(root) / BERTIMBAU_PRETRAINED_DIR.relative_to(PROJECT_ROOT)


def read_assets(root):
    directory = asset_directory(root)
    record = json.loads((directory.parent / "pretrained_run.json").read_text(encoding="utf-8"))
    if record["model"] != BERTIMBAU_MODEL_NAME or record["revision"] != BERTIMBAU_REVISION:
        raise RuntimeError("Pretrained assets do not match the configured model revision.")
    required = [directory / name for name in ("config.json", "vocab.txt", record["weight_file"])]
    if not all(path.relative_to(root).as_posix() in record["artifacts"] for path in required):
        raise RuntimeError("Pretrained asset manifest is incomplete.")
    for name, digest in record["artifacts"].items():
        path = Path(root) / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Pretrained assets changed: {name}")
    return directory, record


def prepare_bertimbau_assets(root=None, source=None, tables=None, downloader=None, repo_files=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    tables = Path(tables) if tables is not None else TABLES_DIR
    manifest = validate_execution(root, source, tables, stage="bertimbau_assets")
    directory = asset_directory(root)
    marker = directory.parent / "pretrained_run.json"
    if not marker.exists():
        if downloader is None or repo_files is None:
            from huggingface_hub import list_repo_files, snapshot_download
            downloader = downloader or snapshot_download
            if repo_files is None:
                repo_files = list_repo_files(BERTIMBAU_MODEL_NAME, revision=BERTIMBAU_REVISION)
        weight = next((name for name in ("model.safetensors", "pytorch_model.bin")
                       if name in repo_files), None)
        if weight is None or not {"config.json", "vocab.txt"}.issubset(repo_files):
            raise RuntimeError("The pinned BERTimbau revision lacks the required assets.")
        allowed = [name for name in (*TOKENIZER_FILES, weight) if name in repo_files]
        directory.mkdir(parents=True, exist_ok=True)
        downloader(repo_id=BERTIMBAU_MODEL_NAME, revision=BERTIMBAU_REVISION,
                   local_dir=directory, allow_patterns=allowed)
        write_json(marker, {
            "model": BERTIMBAU_MODEL_NAME, "revision": BERTIMBAU_REVISION, "weight_file": weight,
            "artifacts": {path.relative_to(root).as_posix(): sha256_file(path)
                          for path in (directory / name for name in allowed)},
        })
    directory, record = read_assets(root)
    report = {"fingerprint": manifest["fingerprint"], **record}
    write_json(tables / REPORT_NAME, report)
    print(f"Pinned BERTimbau assets verified: {directory}")
    return report
