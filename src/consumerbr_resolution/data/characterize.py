import duckdb

from consumerbr_resolution.config import (
    FEATURE_BASE_PATH,
    TABLES_DIR,
    create_project_directories,
)


DATASET_OVERVIEW_PATH = TABLES_DIR / "dataset_overview.csv"
CLASS_DISTRIBUTION_PATH = TABLES_DIR / "class_distribution.csv"
MONTHLY_DISTRIBUTION_PATH = TABLES_DIR / "monthly_distribution.csv"
COMPANY_DISTRIBUTION_PATH = TABLES_DIR / "company_distribution.csv"
UF_DISTRIBUTION_PATH = TABLES_DIR / "uf_distribution.csv"
FEATURE_SUMMARY_PATH = TABLES_DIR / "feature_summary.csv"


def characterize_dataset():
    create_project_directories()

    output_paths = [
        DATASET_OVERVIEW_PATH,
        CLASS_DISTRIBUTION_PATH,
        MONTHLY_DISTRIBUTION_PATH,
        COMPANY_DISTRIBUTION_PATH,
        UF_DISTRIBUTION_PATH,
        FEATURE_SUMMARY_PATH,
    ]

    if all(path.exists() for path in output_paths):
        print("Dataset characterization already exists.")
        return

    source_path = str(FEATURE_BASE_PATH).replace("'", "''")

    overview_path = str(DATASET_OVERVIEW_PATH).replace("'", "''")
    class_path = str(CLASS_DISTRIBUTION_PATH).replace("'", "''")
    monthly_path = str(MONTHLY_DISTRIBUTION_PATH).replace("'", "''")
    company_path = str(COMPANY_DISTRIBUTION_PATH).replace("'", "''")
    uf_path = str(UF_DISTRIBUTION_PATH).replace("'", "''")
    feature_path = str(FEATURE_SUMMARY_PATH).replace("'", "''")

    print("Characterizing experimental dataset")
    print(f"Source: {FEATURE_BASE_PATH}")
    print(f"Destination: {TABLES_DIR}")

    connection = duckdb.connect()

    try:
        connection.execute(
            f"""
            COPY (
                SELECT
                    COUNT(*) AS complaint_count,
                    COUNT(DISTINCT complaint_id) AS unique_complaint_count,
                    COUNT(DISTINCT company) AS company_count,
                    MIN(opening_date) AS first_opening_date,
                    MAX(opening_date) AS last_opening_date,
                    AVG(target_resolved) AS resolution_rate,
                    SUM(
                        CASE
                            WHEN uf = 'UNKNOWN' THEN 1
                            ELSE 0
                        END
                    ) AS unknown_uf_count
                FROM read_parquet('{source_path}')
            )
            TO '{overview_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )

        connection.execute(
            f"""
            COPY (
                SELECT
                    target_resolved,
                    CASE
                        WHEN target_resolved = 1 THEN 'resolved'
                        ELSE 'unresolved'
                    END AS outcome,
                    COUNT(*) AS complaint_count,
                    COUNT(*) * 1.0
                        / SUM(COUNT(*)) OVER () AS proportion
                FROM read_parquet('{source_path}')
                GROUP BY target_resolved
                ORDER BY target_resolved
            )
            TO '{class_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )

        connection.execute(
            f"""
            COPY (
                SELECT
                    DATE_TRUNC(
                        'month',
                        opening_date
                    ) AS month,
                    COUNT(*) AS complaint_count,
                    AVG(target_resolved) AS resolution_rate
                FROM read_parquet('{source_path}')
                GROUP BY 1
                ORDER BY 1
            )
            TO '{monthly_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )

        connection.execute(
            f"""
            COPY (
                SELECT
                    company,
                    COUNT(*) AS complaint_count,
                    AVG(target_resolved) AS resolution_rate,
                    MIN(opening_date) AS first_opening_date,
                    MAX(opening_date) AS last_opening_date
                FROM read_parquet('{source_path}')
                GROUP BY company
                ORDER BY complaint_count DESC
            )
            TO '{company_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )

        connection.execute(
            f"""
            COPY (
                SELECT
                    uf,
                    COUNT(*) AS complaint_count,
                    AVG(target_resolved) AS resolution_rate
                FROM read_parquet('{source_path}')
                GROUP BY uf
                ORDER BY complaint_count DESC
            )
            TO '{uf_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )

        connection.execute(
            f"""
            COPY (
                SELECT
                    AVG(text_char_count) AS mean_text_char_count,
                    MEDIAN(text_char_count) AS median_text_char_count,
                    QUANTILE_CONT(
                        text_char_count,
                        0.90
                    ) AS p90_text_char_count,
                    QUANTILE_CONT(
                        text_char_count,
                        0.95
                    ) AS p95_text_char_count,
                    AVG(text_word_count) AS mean_text_word_count,
                    MEDIAN(text_word_count) AS median_text_word_count,
                    QUANTILE_CONT(
                        text_word_count,
                        0.90
                    ) AS p90_text_word_count,
                    QUANTILE_CONT(
                        text_word_count,
                        0.95
                    ) AS p95_text_word_count,
                    AVG(exclamation_count) AS mean_exclamation_count,
                    AVG(question_count) AS mean_question_count,
                    AVG(
                        anonymization_marker_count
                    ) AS mean_anonymization_marker_count,
                    MAX(
                        anonymization_marker_count
                    ) AS max_anonymization_marker_count,
                    AVG(
                        has_anonymization_marker
                    ) AS anonymization_marker_rate
                FROM read_parquet('{source_path}')
            )
            TO '{feature_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )
    finally:
        connection.close()

    print("Dataset characterization completed.")
    print(f"Saved to: {TABLES_DIR}")