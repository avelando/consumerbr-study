import csv
import time

from huggingface_hub import snapshot_download

from consumerbr_resolution.config import (
    ALBERTINA_MODEL_NAME,
    ALBERTINA_PRETRAINED_DIR,
    ALBERTINA_REVISION,
    TABLES_DIR,
    create_project_directories,
)


ALBERTINA_ASSETS_PATH = (
    TABLES_DIR / "albertina_assets.csv"
)


ASSET_FIELDS = [
    "model_name",
    "revision",
    "local_path",
    "download_seconds",
]


def prepare_albertina_assets():
    create_project_directories()

    if (
        ALBERTINA_ASSETS_PATH.exists()
        and ALBERTINA_PRETRAINED_DIR.exists()
    ):
        print(
            "Albertina pretrained assets already exist."
        )
        return

    print(
        "Preparing Albertina pretrained assets"
    )

    print(
        f"Model: {ALBERTINA_MODEL_NAME}"
    )

    print(
        f"Revision: {ALBERTINA_REVISION}"
    )

    print(
        f"Destination: {ALBERTINA_PRETRAINED_DIR}"
    )

    start_time = time.perf_counter()

    snapshot_download(
        repo_id=ALBERTINA_MODEL_NAME,
        revision=ALBERTINA_REVISION,
        local_dir=ALBERTINA_PRETRAINED_DIR,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )

    download_seconds = (
        time.perf_counter()
        - start_time
    )

    with ALBERTINA_ASSETS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=ASSET_FIELDS,
        )

        writer.writeheader()

        writer.writerow(
            {
                "model_name": (
                    ALBERTINA_MODEL_NAME
                ),
                "revision": (
                    ALBERTINA_REVISION
                ),
                "local_path": str(
                    ALBERTINA_PRETRAINED_DIR
                ),
                "download_seconds": (
                    download_seconds
                ),
            }
        )

    print()
    print(
        "Albertina pretrained assets completed."
    )

    print(
        f"Saved to: "
        f"{ALBERTINA_PRETRAINED_DIR}"
    )

    print(
        f"Manifest: "
        f"{ALBERTINA_ASSETS_PATH}"
    )