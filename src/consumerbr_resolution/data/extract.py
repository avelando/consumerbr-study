from pathlib import Path
import zipfile

from consumerbr_resolution.config import (
    CORPUS_ARCHIVE_PATH,
    CORPUS_CSV_NAME,
    CORPUS_CSV_PATH,
    RAW_DATA_DIR,
    create_project_directories,
)


def extract_corpus():
    create_project_directories()

    if CORPUS_CSV_PATH.exists():
        print(f"Corpus CSV already exists: {CORPUS_CSV_PATH}")
        return

    print(f"Extracting {CORPUS_ARCHIVE_PATH.name}")
    print(f"Destination: {CORPUS_CSV_PATH}")

    with zipfile.ZipFile(CORPUS_ARCHIVE_PATH, "r") as archive:
        csv_member = next(
            member
            for member in archive.namelist()
            if Path(member).name == CORPUS_CSV_NAME
        )

        extracted_path = Path(
            archive.extract(
                csv_member,
                RAW_DATA_DIR,
            )
        )

    if extracted_path != CORPUS_CSV_PATH:
        extracted_path.replace(CORPUS_CSV_PATH)

    print("Extraction completed.")
    print(f"Saved to: {CORPUS_CSV_PATH}")