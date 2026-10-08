import duckdb

from consumerbr_resolution.config import (
    CORPUS_PARQUET_PATH,
    MODELING_BASE_PATH,
    RESOLVED_STATUS,
    UNRESOLVED_STATUS,
    create_project_directories,
)


def build_modeling_base():
    create_project_directories()

    if MODELING_BASE_PATH.exists():
        print(f"Modeling base already exists: {MODELING_BASE_PATH}")
        return

    temporary_path = MODELING_BASE_PATH.with_suffix(".parquet.part")

    if temporary_path.exists():
        temporary_path.unlink()

    source_path = str(CORPUS_PARQUET_PATH).replace("'", "''")
    target_path = str(temporary_path).replace("'", "''")

    print("Building binary modeling base")
    print(f"Source: {CORPUS_PARQUET_PATH}")
    print(f"Destination: {MODELING_BASE_PATH}")

    connection = duckdb.connect()

    try:
        connection.execute(
            f"""
            COPY (
                SELECT
                    ID,
                    id_reclamacao,
                    empresa,
                    reclamacao_labeled,
                    data_abertura,
                    localizacao,
                    CASE
                        WHEN status = '{RESOLVED_STATUS}' THEN 1
                        WHEN status = '{UNRESOLVED_STATUS}' THEN 0
                    END AS target_resolved
                FROM read_parquet('{source_path}')
                WHERE status IN (
                    '{RESOLVED_STATUS}',
                    '{UNRESOLVED_STATUS}'
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

    temporary_path.replace(MODELING_BASE_PATH)

    print("Modeling base completed.")
    print(f"Saved to: {MODELING_BASE_PATH}")