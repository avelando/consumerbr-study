import hashlib
import json

import requests
from tqdm import tqdm

from consumerbr_resolution.config import (
    CORPUS_ARCHIVE_NAME,
    CORPUS_ARCHIVE_PATH,
    DOWNLOAD_CHUNK_SIZE,
    REQUEST_TIMEOUT,
    ZENODO_API_URL,
    ZENODO_RECORD_ID,
    create_project_directories,
)


CORPUS_DOWNLOAD_MANIFEST_PATH = (
    CORPUS_ARCHIVE_PATH
    .with_suffix(
        ".download.json"
    )
)


def get_corpus_file_metadata():
    response = requests.get(
        ZENODO_API_URL,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    record = response.json()

    for file_info in (
        record["files"]
    ):
        if (
            file_info["key"]
            == CORPUS_ARCHIVE_NAME
        ):
            return {
                "url": (
                    file_info[
                        "links"
                    ]["self"]
                ),
                "checksum": (
                    file_info[
                        "checksum"
                    ]
                ),
                "size": int(
                    file_info["size"]
                ),
            }

    raise FileNotFoundError(
        f"{CORPUS_ARCHIVE_NAME} "
        "was not found in the "
        "Zenodo record."
    )


def calculate_checksum(
    path,
    checksum_specification,
):
    (
        algorithm,
        expected_digest,
    ) = (
        checksum_specification
        .split(
            ":",
            1,
        )
    )

    digest = hashlib.new(
        algorithm
    )

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                DOWNLOAD_CHUNK_SIZE
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return (
        digest.hexdigest(),
        expected_digest,
    )


def validate_download(
    path,
    metadata,
):
    observed_size = (
        path.stat().st_size
    )

    if (
        observed_size
        != int(
            metadata["size"]
        )
    ):
        raise RuntimeError(
            "Corpus archive size "
            "does not match Zenodo "
            "metadata."
        )

    (
        observed_digest,
        expected_digest,
    ) = calculate_checksum(
        path=path,
        checksum_specification=(
            metadata[
                "checksum"
            ]
        ),
    )

    if (
        observed_digest.lower()
        != expected_digest.lower()
    ):
        raise RuntimeError(
            "Corpus archive checksum "
            "does not match Zenodo "
            "metadata."
        )


def write_download_manifest(
    metadata,
):
    temporary_path = (
        CORPUS_DOWNLOAD_MANIFEST_PATH
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
            {
                "zenodo_record_id": (
                    ZENODO_RECORD_ID
                ),
                "archive_name": (
                    CORPUS_ARCHIVE_NAME
                ),
                "checksum": (
                    metadata[
                        "checksum"
                    ]
                ),
                "size": int(
                    metadata[
                        "size"
                    ]
                ),
            },
            file,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(
        CORPUS_DOWNLOAD_MANIFEST_PATH
    )


def load_download_manifest():
    if not (
        CORPUS_DOWNLOAD_MANIFEST_PATH
        .exists()
    ):
        return None

    with (
        CORPUS_DOWNLOAD_MANIFEST_PATH
        .open(
            "r",
            encoding="utf-8",
        )
    ) as file:
        manifest = json.load(
            file
        )

    return {
        "checksum": (
            manifest[
                "checksum"
            ]
        ),
        "size": int(
            manifest[
                "size"
            ]
        ),
    }


def download_corpus():
    create_project_directories()

    if CORPUS_ARCHIVE_PATH.exists():
        metadata = (
            load_download_manifest()
        )

        if metadata is None:
            metadata = (
                get_corpus_file_metadata()
            )

            validate_download(
                CORPUS_ARCHIVE_PATH,
                metadata,
            )

            write_download_manifest(
                metadata
            )

        else:
            validate_download(
                CORPUS_ARCHIVE_PATH,
                metadata,
            )

        print(
            "Corpus archive already "
            "exists and passed "
            "integrity validation: "
            f"{CORPUS_ARCHIVE_PATH}"
        )

        return

    metadata = (
        get_corpus_file_metadata()
    )

    temporary_path = (
        CORPUS_ARCHIVE_PATH
        .with_suffix(
            CORPUS_ARCHIVE_PATH.suffix
            + ".part"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    print(
        f"Downloading "
        f"{CORPUS_ARCHIVE_NAME}"
    )

    print(
        f"Destination: "
        f"{CORPUS_ARCHIVE_PATH}"
    )

    with requests.get(
        metadata["url"],
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()

        total_size = int(
            response.headers.get(
                "content-length",
                metadata["size"],
            )
        )

        with temporary_path.open(
            "wb"
        ) as file:
            with tqdm(
                total=(
                    total_size
                    or None
                ),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                for chunk in (
                    response
                    .iter_content(
                        chunk_size=(
                            DOWNLOAD_CHUNK_SIZE
                        )
                    )
                ):
                    if chunk:
                        file.write(
                            chunk
                        )

                        progress.update(
                            len(chunk)
                        )

    validate_download(
        temporary_path,
        metadata,
    )

    temporary_path.replace(
        CORPUS_ARCHIVE_PATH
    )

    write_download_manifest(
        metadata
    )

    print(
        "Download completed "
        "and validated."
    )

    print(
        f"Saved to: "
        f"{CORPUS_ARCHIVE_PATH}"
    )

    print(
        f"Manifest: "
        f"{CORPUS_DOWNLOAD_MANIFEST_PATH}"
    )