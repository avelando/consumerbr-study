import duckdb

from consumerbr_resolution.config import (
    CLEAN_BASE_PATH,
    FEATURE_BASE_PATH,
    VALID_UFS,
    create_project_directories,
)


def build_feature_base():
    create_project_directories()

    if FEATURE_BASE_PATH.exists():
        print(f"Feature base already exists: {FEATURE_BASE_PATH}")
        return

    temporary_path = FEATURE_BASE_PATH.with_suffix(".parquet.part")

    if temporary_path.exists():
        temporary_path.unlink()

    source_path = str(CLEAN_BASE_PATH).replace("'", "''")
    target_path = str(temporary_path).replace("'", "''")

    valid_ufs = ", ".join(f"'{uf}'" for uf in VALID_UFS)

    print("Building deterministic pre-response features")
    print(f"Source: {CLEAN_BASE_PATH}")
    print(f"Destination: {FEATURE_BASE_PATH}")

    connection = duckdb.connect()

    try:
        connection.execute(
            f"""
            COPY (
                WITH base AS (
                    SELECT
                        record_id,
                        complaint_id,
                        company,
                        complaint_text,
                        location,
                        opening_date,
                        target_resolved,
                        UPPER(
                            REGEXP_EXTRACT(
                                COALESCE(location, ''),
                                '([A-Z]{{2}})[[:space:]]*$',
                                1
                            )
                        ) AS extracted_uf
                    FROM read_parquet('{source_path}')
                )
                SELECT
                    record_id,
                    complaint_id,
                    company,
                    complaint_text,
                    location,
                    CASE
                        WHEN extracted_uf IN ({valid_ufs})
                            THEN extracted_uf
                        ELSE 'UNKNOWN'
                    END AS uf,
                    opening_date,
                    target_resolved,
                    LENGTH(complaint_text) AS text_char_count,
                    ARRAY_LENGTH(
                        STRING_SPLIT(complaint_text, ' ')
                    ) AS text_word_count,
                    LN(
                        1 + LENGTH(complaint_text)
                    ) AS log_text_char_count,
                    LN(
                        1 + ARRAY_LENGTH(
                            STRING_SPLIT(complaint_text, ' ')
                        )
                    ) AS log_text_word_count,
                    LENGTH(complaint_text)
                        - LENGTH(
                            REPLACE(
                                complaint_text,
                                '!',
                                ''
                            )
                        ) AS exclamation_count,
                    LENGTH(complaint_text)
                        - LENGTH(
                            REPLACE(
                                complaint_text,
                                '?',
                                ''
                            )
                        ) AS question_count,
                    ARRAY_LENGTH(
                        REGEXP_EXTRACT_ALL(
                            complaint_text,
                            '\\\\[[A-ZÀ-Ü_]+\\\\]'
                        )
                    ) AS anonymization_marker_count,
                    CASE
                        WHEN POSITION('!' IN complaint_text) > 0
                            THEN 1
                        ELSE 0
                    END AS has_exclamation,
                    CASE
                        WHEN POSITION('?' IN complaint_text) > 0
                            THEN 1
                        ELSE 0
                    END AS has_question,
                    CASE
                        WHEN ARRAY_LENGTH(
                            REGEXP_EXTRACT_ALL(
                                complaint_text,
                                '\\\\[[A-ZÀ-Ü_]+\\\\]'
                            )
                        ) > 0
                            THEN 1
                        ELSE 0
                    END AS has_anonymization_marker,
                    EXTRACT(
                        MONTH FROM opening_date
                    ) AS opening_month,
                    EXTRACT(
                        DOW FROM opening_date
                    ) AS opening_weekday,
                    SIN(
                        2 * PI()
                        * EXTRACT(MONTH FROM opening_date)
                        / 12.0
                    ) AS opening_month_sin,
                    COS(
                        2 * PI()
                        * EXTRACT(MONTH FROM opening_date)
                        / 12.0
                    ) AS opening_month_cos,
                    SIN(
                        2 * PI()
                        * EXTRACT(DOW FROM opening_date)
                        / 7.0
                    ) AS opening_weekday_sin,
                    COS(
                        2 * PI()
                        * EXTRACT(DOW FROM opening_date)
                        / 7.0
                    ) AS opening_weekday_cos
                FROM base
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

    temporary_path.replace(FEATURE_BASE_PATH)

    print("Feature construction completed.")
    print(f"Saved to: {FEATURE_BASE_PATH}")