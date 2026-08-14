import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import (
    PackageNotFoundError,
    version,
)

import torch

from consumerbr_resolution.config import (
    ALBERTINA_MODEL_NAME,
    ALBERTINA_REVISION,
    BERTIMBAU_MODEL_NAME,
    BERTIMBAU_REVISION,
    EXPERIMENT_SEEDS,
    FEATURE_BASE_PATH,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PRIMARY_EXPERIMENT_SEED,
    RANDOM_SEED,
    TABLES_DIR,
    TEMPORAL_FOLDS,
    create_project_directories,
)


OFFICIAL_RUN_MANIFEST_PATH = (
    TABLES_DIR
    / "official_run_manifest.json"
)


PACKAGE_NAMES = (
    "catboost",
    "duckdb",
    "huggingface-hub",
    "joblib",
    "numpy",
    "pandas",
    "pyarrow",
    "requests",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
    "tqdm",
)


def run_git(
    *arguments,
    required=True,
):
    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if (
        required
        and result.returncode != 0
    ):
        raise RuntimeError(
            result.stderr.strip()
            or "Git command failed."
        )

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def directory_has_files(path):
    return any(
        candidate.is_file()
        for candidate
        in path.rglob("*")
    )


def get_package_versions():
    versions = {}

    for package_name in (
        PACKAGE_NAMES
    ):
        try:
            versions[
                package_name
            ] = version(
                package_name
            )
        except PackageNotFoundError:
            versions[
                package_name
            ] = None

    return versions


def get_git_state():
    return {
        "commit": run_git(
            "rev-parse",
            "HEAD",
        ),
        "branch": run_git(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ),
        "tag": run_git(
            "describe",
            "--tags",
            "--exact-match",
            "HEAD",
            required=False,
        ),
        "status": run_git(
            "status",
            "--porcelain",
        ),
    }


def build_manifest(
    git_state,
):
    cuda_available = (
        torch.cuda.is_available()
    )

    gpu = None

    if cuda_available:
        properties = (
            torch.cuda
            .get_device_properties(0)
        )

        gpu = {
            "name": (
                torch.cuda
                .get_device_name(0)
            ),
            "total_memory_bytes": int(
                properties.total_memory
            ),
        }

    return {
        "git": {
            "commit": (
                git_state[
                    "commit"
                ]
            ),
            "branch": (
                git_state[
                    "branch"
                ]
            ),
            "tag": (
                git_state["tag"]
            ),
        },
        "runtime": {
            "python": (
                platform
                .python_version()
            ),
            "platform": (
                platform.platform()
            ),
            "cuda_available": (
                cuda_available
            ),
            "torch_cuda_version": (
                torch.version.cuda
            ),
            "gpu": gpu,
        },
        "packages": (
            get_package_versions()
        ),
        "dataset": {
            "feature_base_path": str(
                FEATURE_BASE_PATH
            ),
            "feature_base_sha256": (
                sha256_file(
                    FEATURE_BASE_PATH
                )
            ),
        },
        "randomness": {
            "random_seed": (
                RANDOM_SEED
            ),
            "primary_experiment_seed": (
                PRIMARY_EXPERIMENT_SEED
            ),
            "experiment_seeds": list(
                EXPERIMENT_SEEDS
            ),
        },
        "transformers": {
            "bertimbau": {
                "model_name": (
                    BERTIMBAU_MODEL_NAME
                ),
                "revision": (
                    BERTIMBAU_REVISION
                ),
            },
            "albertina": {
                "model_name": (
                    ALBERTINA_MODEL_NAME
                ),
                "revision": (
                    ALBERTINA_REVISION
                ),
            },
        },
        "temporal_folds": list(
            TEMPORAL_FOLDS
        ),
    }


def validate_or_create_official_run_manifest():
    create_project_directories()

    official_mode = (
        os.environ.get(
            "CONSUMERBR_OFFICIAL_RUN",
            "0",
        )
        == "1"
    )

    git_state = get_git_state()

    if (
        OFFICIAL_RUN_MANIFEST_PATH
        .exists()
    ):
        with (
            OFFICIAL_RUN_MANIFEST_PATH
            .open(
                "r",
                encoding="utf-8",
            )
        ) as file:
            existing = json.load(
                file
            )

        if (
            existing[
                "git"
            ]["commit"]
            != git_state[
                "commit"
            ]
        ):
            raise RuntimeError(
                "Existing official-run "
                "manifest belongs to a "
                "different commit."
            )

        if (
            existing[
                "git"
            ]["tag"]
            != git_state[
                "tag"
            ]
        ):
            raise RuntimeError(
                "Existing official-run "
                "manifest belongs to a "
                "different tag."
            )

        if (
            existing[
                "temporal_folds"
            ]
            != list(
                TEMPORAL_FOLDS
            )
        ):
            raise RuntimeError(
                "Existing official-run "
                "manifest uses a different "
                "temporal protocol."
            )

        print(
            "Official-run manifest "
            "matches the current "
            "experiment state."
        )

        return

    if official_mode:
        if (
            git_state["branch"]
            != "main"
        ):
            raise RuntimeError(
                "Official execution must "
                "start from the main branch."
            )

        if git_state["status"]:
            raise RuntimeError(
                "Official execution requires "
                "a clean Git working tree."
            )

        if not git_state["tag"]:
            raise RuntimeError(
                "Official execution requires "
                "HEAD to have an exact Git tag."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the "
                "official transformer "
                "evaluation."
            )

        stale_directories = [
            path
            for path
            in (
                MODELS_DIR,
                METRICS_DIR,
                PREDICTIONS_DIR,
            )
            if directory_has_files(
                path
            )
        ]

        if stale_directories:
            raise RuntimeError(
                "Official execution must "
                "start without pre-existing "
                "model, metric, or prediction "
                "files: "
                + ", ".join(
                    str(path)
                    for path
                    in stale_directories
                )
            )

    manifest = build_manifest(
        git_state
    )

    temporary_path = (
        OFFICIAL_RUN_MANIFEST_PATH
        .with_suffix(
            ".json.part"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(
        OFFICIAL_RUN_MANIFEST_PATH
    )

    print(
        "Official-run reproducibility "
        "manifest created."
    )

    print(
        f"Saved to: "
        f"{OFFICIAL_RUN_MANIFEST_PATH}"
    )