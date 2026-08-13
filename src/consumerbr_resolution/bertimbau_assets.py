import csv
import time

from huggingface_hub import snapshot_download

from consumerbr_resolution.config import (
    BERTIMBAU_MODEL_NAME,
    BERTIMBAU_PRETRAINED_DIR,
    BERTIMBAU_REVISION,
    TABLES_DIR,
    create_project_directories,
)


BERTIMBAU_ASSETS_PATH = (
    TABLES_DIR / "bertimbau_assets.csv"
)


ASSET_FIELDS = [
    "model_name",
    "revision",
    "local_path",
    "download_seconds",
]


def prepare_bertimbau_assets():
    create_project_directories()

    if (
        BERTIMBAU_ASSETS_PATH.exists()
        and BERTIMBAU_PRETRAINED_DIR.exists()
    ):
        print(
            "BERTimbau pretrained assets already exist."
        )
        return

    print(
        "Preparing BERTimbau pretrained assets"
    )

    print(
        f"Model: {BERTIMBAU_MODEL_NAME}"
    )

    print(
        f"Revision: {BERTIMBAU_REVISION}"
    )

    print(
        f"Destination: {BERTIMBAU_PRETRAINED_DIR}"
    )

    start_time = time.perf_counter()

    snapshot_download(
        repo_id=BERTIMBAU_MODEL_NAME,
        revision=BERTIMBAU_REVISION,
        local_dir=BERTIMBAU_PRETRAINED_DIR,
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "vocab.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )

    download_seconds = (
        time.perf_counter()
        - start_time
    )

    with BERTIMBAU_ASSETS_PATH.open(
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
                    BERTIMBAU_MODEL_NAME
                ),
                "revision": (
                    BERTIMBAU_REVISION
                ),
                "local_path": str(
                    BERTIMBAU_PRETRAINED_DIR
                ),
                "download_seconds": (
                    download_seconds
                ),
            }
        )

    print()
    print(
        "BERTimbau pretrained assets completed."
    )

    print(
        f"Saved to: "
        f"{BERTIMBAU_PRETRAINED_DIR}"
    )

    print(
        f"Manifest: "
        f"{BERTIMBAU_ASSETS_PATH}"
    )