import requests
from tqdm import tqdm

from consumerbr_resolution.config import (
    CORPUS_ARCHIVE_NAME,
    CORPUS_ARCHIVE_PATH,
    DOWNLOAD_CHUNK_SIZE,
    REQUEST_TIMEOUT,
    ZENODO_API_URL,
    create_project_directories,
)


def get_corpus_download_url():
    response = requests.get(
        ZENODO_API_URL,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    record = response.json()

    for file_info in record["files"]:
        if file_info["key"] == CORPUS_ARCHIVE_NAME:
            return file_info["links"]["self"]

    raise FileNotFoundError(
        f"{CORPUS_ARCHIVE_NAME} was not found in the Zenodo record."
    )


def download_corpus():
    create_project_directories()

    if CORPUS_ARCHIVE_PATH.exists():
        print(f"Corpus archive already exists: {CORPUS_ARCHIVE_PATH}")
        return

    download_url = get_corpus_download_url()
    temporary_path = CORPUS_ARCHIVE_PATH.with_suffix(
        CORPUS_ARCHIVE_PATH.suffix + ".part"
    )

    print(f"Downloading {CORPUS_ARCHIVE_NAME}")
    print(f"Destination: {CORPUS_ARCHIVE_PATH}")

    with requests.get(
        download_url,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with temporary_path.open("wb") as file:
            with tqdm(
                total=total_size or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                for chunk in response.iter_content(
                    chunk_size=DOWNLOAD_CHUNK_SIZE
                ):
                    if chunk:
                        file.write(chunk)
                        progress.update(len(chunk))

    temporary_path.replace(CORPUS_ARCHIVE_PATH)

    print("Download completed.")
    print(f"Saved to: {CORPUS_ARCHIVE_PATH}")