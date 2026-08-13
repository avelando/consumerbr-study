import duckdb

from consumerbr_resolution.config import (
    CORPUS_CSV_PATH,
    CORPUS_PARQUET_PATH,
    create_project_directories,
)


def convert_corpus_to_parquet():
    create_project_directories()

    if CORPUS_PARQUET_PATH.exists():
        print(f"Corpus Parquet already exists: {CORPUS_PARQUET_PATH}")
        return

    temporary_path = CORPUS_PARQUET_PATH.with_suffix(".parquet.part")

    if temporary_path.exists():
        temporary_path.unlink()

    source_path = str(CORPUS_CSV_PATH).replace("'", "''")
    target_path = str(temporary_path).replace("'", "''")

    print("Converting ConsumerBR CSV to Parquet")
    print(f"Source: {CORPUS_CSV_PATH}")
    print(f"Destination: {CORPUS_PARQUET_PATH}")

    connection = duckdb.connect()

    try:
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_csv(
                    '{source_path}',
                    header = true,
                    all_varchar = true
                )
            )
            TO '{target_path}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )
    finally:
        connection.close()

    temporary_path.replace(CORPUS_PARQUET_PATH)

    print("Conversion completed.")
    print(f"Saved to: {CORPUS_PARQUET_PATH}")