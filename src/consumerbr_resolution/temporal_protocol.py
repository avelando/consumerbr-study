import csv

import duckdb

from consumerbr_resolution.config import (
    FEATURE_BASE_PATH,
    TABLES_DIR,
    TEMPORAL_FOLDS,
    create_project_directories,
)

from consumerbr_resolution.config import (
    EXPECTED_CORPUS_OBSERVATION_END,
    FEATURE_BASE_PATH,
    TABLES_DIR,
    TEMPORAL_FIRST_VALIDATION_START,
    TEMPORAL_FOLDS,
    TEMPORAL_STEP_MONTHS,
    TEMPORAL_TEST_MONTHS,
    TEMPORAL_TRAIN_START,
    TEMPORAL_VALIDATION_MONTHS,
    TUNING_VALIDATION_END,
    create_project_directories,
)

from consumerbr_resolution.temporal_design import (
    generate_temporal_folds,
    generate_test_window_candidates,
)


TEMPORAL_PROTOCOL_PATH = TABLES_DIR / "temporal_protocol.csv"
TEMPORAL_FOLD_SUMMARY_PATH = TABLES_DIR / "temporal_fold_summary.csv"
TEMPORAL_PROTOCOL_AUDIT_PATH = (
    TABLES_DIR
    / "temporal_protocol_audit.csv"
)

TEMPORAL_TEST_WINDOW_ELIGIBILITY_PATH = (
    TABLES_DIR
    / "temporal_test_window_eligibility.csv"
)

def build_temporal_protocol():
    create_project_directories()

    output_paths = [
        TEMPORAL_PROTOCOL_PATH,
        TEMPORAL_FOLD_SUMMARY_PATH,
    ]

    if all(path.exists() for path in output_paths):
        print("Temporal protocol already exists.")
        return

    print("Building temporal evaluation protocol")
    print(f"Source: {FEATURE_BASE_PATH}")
    print(f"Destination: {TABLES_DIR}")

    with TEMPORAL_PROTOCOL_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fold",
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
            ],
        )

        writer.writeheader()

        for fold in TEMPORAL_FOLDS:
            writer.writerow(
                {
                    "fold": fold["fold"],
                    "train_start": "2021-05-01",
                    "train_end": fold["train_end"],
                    "validation_start": fold["validation_start"],
                    "validation_end": fold["validation_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                }
            )

    source_path = str(FEATURE_BASE_PATH).replace("'", "''")
    summary_path = str(
        TEMPORAL_FOLD_SUMMARY_PATH
    ).replace("'", "''")

    queries = []

    for fold in TEMPORAL_FOLDS:
        fold_number = fold["fold"]
        train_end = fold["train_end"]
        validation_start = fold["validation_start"]
        validation_end = fold["validation_end"]
        test_start = fold["test_start"]
        test_end = fold["test_end"]

        queries.append(
            f"""
            SELECT
                {fold_number} AS fold,
                'train' AS split,
                COUNT(*) AS complaint_count,
                AVG(target_resolved) AS resolution_rate,
                MIN(opening_date) AS first_opening_date,
                MAX(opening_date) AS last_opening_date
            FROM read_parquet('{source_path}')
            WHERE opening_date <= DATE '{train_end}'
            """
        )

        queries.append(
            f"""
            SELECT
                {fold_number} AS fold,
                'validation' AS split,
                COUNT(*) AS complaint_count,
                AVG(target_resolved) AS resolution_rate,
                MIN(opening_date) AS first_opening_date,
                MAX(opening_date) AS last_opening_date
            FROM read_parquet('{source_path}')
            WHERE opening_date BETWEEN
                DATE '{validation_start}'
                AND DATE '{validation_end}'
            """
        )

        queries.append(
            f"""
            SELECT
                {fold_number} AS fold,
                'test' AS split,
                COUNT(*) AS complaint_count,
                AVG(target_resolved) AS resolution_rate,
                MIN(opening_date) AS first_opening_date,
                MAX(opening_date) AS last_opening_date
            FROM read_parquet('{source_path}')
            WHERE opening_date BETWEEN
                DATE '{test_start}'
                AND DATE '{test_end}'
            """
        )

    union_query = "\nUNION ALL\n".join(queries)

    connection = duckdb.connect()

    try:
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM (
                    {union_query}
                )
                ORDER BY
                    fold,
                    CASE split
                        WHEN 'train' THEN 1
                        WHEN 'validation' THEN 2
                        WHEN 'test' THEN 3
                    END
            )
            TO '{summary_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )
    finally:
        connection.close()

    print("Temporal protocol completed.")
    print(f"Saved to: {TEMPORAL_PROTOCOL_PATH}")
    print(f"Saved to: {TEMPORAL_FOLD_SUMMARY_PATH}")