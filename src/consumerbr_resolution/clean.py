import duckdb

from consumerbr_resolution.config import (
    CLEAN_BASE_PATH,
    MIN_TEXT_CHARS,
    MIN_TEXT_WORDS,
    MODELING_BASE_PATH,
    create_project_directories,
)


def clean_modeling_base():
    create_project_directories()

    if CLEAN_BASE_PATH.exists():
        print(f"Clean modeling base already exists: {CLEAN_BASE_PATH}")
        return

    temporary_path = CLEAN_BASE_PATH.with_suffix(".parquet.part")

    if temporary_path.exists():
        temporary_path.unlink()

    source_path = str(MODELING_BASE_PATH).replace("'", "''")
    target_path = str(temporary_path).replace("'", "''")

    print("Cleaning modeling base")
    print(f"Source: {MODELING_BASE_PATH}")
    print(f"Destination: {CLEAN_BASE_PATH}")

    connection = duckdb.connect()

    try:
        connection.execute(
            f"""
            COPY (
                WITH prepared AS (
                    SELECT
                        ID AS record_id,
                        id_reclamacao AS complaint_id,
                        TRIM(
                            REGEXP_REPLACE(
                                COALESCE(empresa, ''),
                                '[[:space:]]+',
                                ' ',
                                'g'
                            )
                        ) AS company,
                        TRIM(
                            REGEXP_REPLACE(
                                COALESCE(reclamacao_labeled, ''),
                                '[[:space:]]+',
                                ' ',
                                'g'
                            )
                        ) AS complaint_text,
                        localizacao AS location,
                        COALESCE(
                            TRY_CAST(data_abertura AS DATE),
                            CAST(
                                TRY_STRPTIME(
                                    data_abertura,
                                    '%d/%m/%Y'
                                )
                                AS DATE
                            )
                        ) AS opening_date,
                        CAST(target_resolved AS INTEGER) AS target_resolved
                    FROM read_parquet('{source_path}')
                ),
                measured AS (
                    SELECT
                        *,
                        LENGTH(complaint_text) AS text_char_count,
                        ARRAY_LENGTH(
                            STRING_SPLIT(complaint_text, ' ')
                        ) AS text_word_count
                    FROM prepared
                )
                SELECT
                    record_id,
                    complaint_id,
                    company,
                    complaint_text,
                    location,
                    opening_date,
                    target_resolved
                FROM measured
                WHERE
                    opening_date IS NOT NULL
                    AND text_char_count >= {MIN_TEXT_CHARS}
                    AND text_word_count >= {MIN_TEXT_WORDS}
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

    temporary_path.replace(CLEAN_BASE_PATH)

    print("Cleaning completed.")
    print(f"Saved to: {CLEAN_BASE_PATH}")