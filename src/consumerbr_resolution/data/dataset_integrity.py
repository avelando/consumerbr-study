import csv

import duckdb

from consumerbr_resolution.config import (
    FEATURE_BASE_PATH,
    MIN_TEXT_CHARS,
    MIN_TEXT_WORDS,
    TABLES_DIR,
    create_project_directories,
)


DATASET_INTEGRITY_AUDIT_PATH = (
    TABLES_DIR
    / "dataset_integrity_audit.csv"
)


REQUIRED_FEATURE_COLUMNS = {
    "record_id",
    "complaint_id",
    "company",
    "complaint_text",
    "location",
    "uf",
    "opening_date",
    "target_resolved",
    "text_char_count",
    "text_word_count",
    "log_text_char_count",
    "log_text_word_count",
    "exclamation_count",
    "question_count",
    "anonymization_marker_count",
    "has_exclamation",
    "has_question",
    "has_anonymization_marker",
    "opening_month",
    "opening_weekday",
    "opening_month_sin",
    "opening_month_cos",
    "opening_weekday_sin",
    "opening_weekday_cos",
}


def validate_dataset_integrity():
    create_project_directories()

    if not FEATURE_BASE_PATH.exists():
        raise FileNotFoundError(
            "Feature base was not found: "
            f"{FEATURE_BASE_PATH}"
        )

    print(
        "Validating experimental "
        "dataset integrity"
    )

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    connection = duckdb.connect()

    try:
        columns = {
            row[0]
            for row
            in connection.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet(
                    '{source_path}'
                )
                """
            ).fetchall()
        }

        missing_columns = sorted(
            REQUIRED_FEATURE_COLUMNS
            - columns
        )

        row = connection.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(
                    DISTINCT record_id
                ),
                SUM(
                    CASE
                        WHEN record_id
                            IS NULL
                            THEN 1
                        ELSE 0
                    END
                ),
                SUM(
                    CASE
                        WHEN complaint_id
                            IS NULL
                            THEN 1
                        ELSE 0
                    END
                ),
                SUM(
                    CASE
                        WHEN complaint_text
                            IS NULL
                            THEN 1
                        ELSE 0
                    END
                ),
                SUM(
                    CASE
                        WHEN opening_date
                            IS NULL
                            THEN 1
                        ELSE 0
                    END
                ),
                SUM(
                    CASE
                        WHEN target_resolved
                            IS NULL
                            THEN 1
                        ELSE 0
                    END
                ),
                SUM(
                    CASE
                        WHEN target_resolved
                            NOT IN (0, 1)
                            THEN 1
                        ELSE 0
                    END
                ),
                SUM(
                    CASE
                        WHEN text_char_count
                            < {MIN_TEXT_CHARS}
                            THEN 1
                        ELSE 0
                    END
                ),
                SUM(
                    CASE
                        WHEN text_word_count
                            < {MIN_TEXT_WORDS}
                            THEN 1
                        ELSE 0
                    END
                )
            FROM read_parquet(
                '{source_path}'
            )
            """
        ).fetchone()

        conflicting_complaints = (
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT
                        complaint_id
                    FROM read_parquet(
                        '{source_path}'
                    )
                    GROUP BY
                        complaint_id
                    HAVING
                        COUNT(
                            DISTINCT
                            opening_date
                        ) > 1
                        OR COUNT(
                            DISTINCT
                            target_resolved
                        ) > 1
                )
                """
            ).fetchone()[0]
        )

    finally:
        connection.close()

    row_count = int(row[0])

    unique_record_count = int(
        row[1]
    )

    audit_rows = [
        {
            "criterion": (
                "required_feature_columns_present"
            ),
            "value": (
                ",".join(
                    missing_columns
                )
                if missing_columns
                else "none"
            ),
            "passed": (
                len(missing_columns)
                == 0
            ),
        },
        {
            "criterion": (
                "dataset_is_not_empty"
            ),
            "value": row_count,
            "passed": (
                row_count > 0
            ),
        },
        {
            "criterion": (
                "record_id_is_unique"
            ),
            "value": (
                f"{unique_record_count}/"
                f"{row_count}"
            ),
            "passed": (
                unique_record_count
                == row_count
            ),
        },
        {
            "criterion": (
                "record_id_has_no_nulls"
            ),
            "value": int(row[2]),
            "passed": (
                int(row[2]) == 0
            ),
        },
        {
            "criterion": (
                "complaint_id_has_no_nulls"
            ),
            "value": int(row[3]),
            "passed": (
                int(row[3]) == 0
            ),
        },
        {
            "criterion": (
                "complaint_text_has_no_nulls"
            ),
            "value": int(row[4]),
            "passed": (
                int(row[4]) == 0
            ),
        },
        {
            "criterion": (
                "opening_date_has_no_nulls"
            ),
            "value": int(row[5]),
            "passed": (
                int(row[5]) == 0
            ),
        },
        {
            "criterion": (
                "target_has_no_nulls"
            ),
            "value": int(row[6]),
            "passed": (
                int(row[6]) == 0
            ),
        },
        {
            "criterion": (
                "target_is_binary"
            ),
            "value": int(row[7]),
            "passed": (
                int(row[7]) == 0
            ),
        },
        {
            "criterion": (
                "minimum_text_characters_respected"
            ),
            "value": int(row[8]),
            "passed": (
                int(row[8]) == 0
            ),
        },
        {
            "criterion": (
                "minimum_text_words_respected"
            ),
            "value": int(row[9]),
            "passed": (
                int(row[9]) == 0
            ),
        },
        {
            "criterion": (
                "complaint_ids_have_consistent_date_and_target"
            ),
            "value": int(
                conflicting_complaints
            ),
            "passed": (
                int(
                    conflicting_complaints
                )
                == 0
            ),
        },
    ]

    temporary_path = (
        DATASET_INTEGRITY_AUDIT_PATH
        .with_suffix(
            ".csv.part"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "criterion",
                "value",
                "passed",
            ],
        )

        writer.writeheader()
        writer.writerows(
            audit_rows
        )

    temporary_path.replace(
        DATASET_INTEGRITY_AUDIT_PATH
    )

    failed = [
        row
        for row
        in audit_rows
        if not bool(
            row["passed"]
        )
    ]

    if failed:
        failed_names = ", ".join(
            row["criterion"]
            for row
            in failed
        )

        raise RuntimeError(
            "Dataset integrity validation "
            "failed: "
            f"{failed_names}."
        )

    print(
        "Dataset integrity "
        "validation completed."
    )

    print(
        f"Saved to: "
        f"{DATASET_INTEGRITY_AUDIT_PATH}"
    )