import duckdb

from consumerbr_resolution.config import (
    CORPUS_PARQUET_PATH,
    RESOLVED_STATUS,
    TABLES_DIR,
    UNRESOLVED_STATUS,
    create_project_directories,
)


STATUS_DISTRIBUTION_PATH = (
    TABLES_DIR / "outcome_status_distribution.csv"
)
OBSERVATION_SUMMARY_PATH = (
    TABLES_DIR / "outcome_observation_summary.csv"
)
OBSERVATION_MONTHLY_PATH = (
    TABLES_DIR / "outcome_observation_monthly.csv"
)
OBSERVATION_COMPANY_PATH = (
    TABLES_DIR / "outcome_observation_company.csv"
)
OBSERVATION_TEXT_PATH = (
    TABLES_DIR / "outcome_observation_text.csv"
)


def analyze_outcome_observation():
    create_project_directories()

    output_paths = [
        STATUS_DISTRIBUTION_PATH,
        OBSERVATION_SUMMARY_PATH,
        OBSERVATION_MONTHLY_PATH,
        OBSERVATION_COMPANY_PATH,
        OBSERVATION_TEXT_PATH,
    ]

    if all(path.exists() for path in output_paths):
        print("Outcome observation analysis already exists.")
        return

    source_path = str(CORPUS_PARQUET_PATH).replace("'", "''")

    status_path = str(
        STATUS_DISTRIBUTION_PATH
    ).replace("'", "''")
    summary_path = str(
        OBSERVATION_SUMMARY_PATH
    ).replace("'", "''")
    monthly_path = str(
        OBSERVATION_MONTHLY_PATH
    ).replace("'", "''")
    company_path = str(
        OBSERVATION_COMPANY_PATH
    ).replace("'", "''")
    text_path = str(
        OBSERVATION_TEXT_PATH
    ).replace("'", "''")

    print("Analyzing outcome observation patterns")
    print(f"Source: {CORPUS_PARQUET_PATH}")
    print(f"Destination: {TABLES_DIR}")

    connection = duckdb.connect()

    try:
        connection.execute(
            f"""
            COPY (
                SELECT
                    COALESCE(
                        status,
                        'MISSING'
                    ) AS status,
                    COUNT(*) AS complaint_count,
                    COUNT(*) * 1.0
                        / SUM(COUNT(*)) OVER () AS proportion
                FROM read_parquet('{source_path}')
                GROUP BY status
                ORDER BY complaint_count DESC
            )
            TO '{status_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )

        connection.execute(
            f"""
            COPY (
                WITH base AS (
                    SELECT
                        CASE
                            WHEN status IN (
                                '{RESOLVED_STATUS}',
                                '{UNRESOLVED_STATUS}'
                            )
                                THEN 1
                            ELSE 0
                        END AS observed_outcome
                    FROM read_parquet('{source_path}')
                )
                SELECT
                    observed_outcome,
                    CASE
                        WHEN observed_outcome = 1
                            THEN 'observed'
                        ELSE 'not_observed'
                    END AS outcome_group,
                    COUNT(*) AS complaint_count,
                    COUNT(*) * 1.0
                        / SUM(COUNT(*)) OVER () AS proportion
                FROM base
                GROUP BY observed_outcome
                ORDER BY observed_outcome DESC
            )
            TO '{summary_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )

        connection.execute(
            f"""
            COPY (
                WITH base AS (
                    SELECT
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
                        CASE
                            WHEN status IN (
                                '{RESOLVED_STATUS}',
                                '{UNRESOLVED_STATUS}'
                            )
                                THEN 1
                            ELSE 0
                        END AS observed_outcome
                    FROM read_parquet('{source_path}')
                )
                SELECT
                    DATE_TRUNC(
                        'month',
                        opening_date
                    ) AS month,
                    COUNT(*) AS complaint_count,
                    SUM(
                        observed_outcome
                    ) AS observed_outcome_count,
                    AVG(
                        observed_outcome
                    ) AS observed_outcome_rate
                FROM base
                WHERE opening_date IS NOT NULL
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
                    COALESCE(
                        NULLIF(
                            TRIM(empresa),
                            ''
                        ),
                        'UNKNOWN'
                    ) AS company,
                    COUNT(*) AS complaint_count,
                    SUM(
                        CASE
                            WHEN status IN (
                                '{RESOLVED_STATUS}',
                                '{UNRESOLVED_STATUS}'
                            )
                                THEN 1
                            ELSE 0
                        END
                    ) AS observed_outcome_count,
                    AVG(
                        CASE
                            WHEN status IN (
                                '{RESOLVED_STATUS}',
                                '{UNRESOLVED_STATUS}'
                            )
                                THEN 1
                            ELSE 0
                        END
                    ) AS observed_outcome_rate
                FROM read_parquet('{source_path}')
                GROUP BY 1
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
                WITH base AS (
                    SELECT
                        CASE
                            WHEN status IN (
                                '{RESOLVED_STATUS}',
                                '{UNRESOLVED_STATUS}'
                            )
                                THEN 1
                            ELSE 0
                        END AS observed_outcome,
                        LENGTH(
                            TRIM(
                                REGEXP_REPLACE(
                                    COALESCE(
                                        reclamacao_labeled,
                                        ''
                                    ),
                                    '[[:space:]]+',
                                    ' ',
                                    'g'
                                )
                            )
                        ) AS text_char_count
                    FROM read_parquet('{source_path}')
                )
                SELECT
                    observed_outcome,
                    CASE
                        WHEN observed_outcome = 1
                            THEN 'observed'
                        ELSE 'not_observed'
                    END AS outcome_group,
                    COUNT(*) AS complaint_count,
                    AVG(
                        text_char_count
                    ) AS mean_text_char_count,
                    MEDIAN(
                        text_char_count
                    ) AS median_text_char_count,
                    QUANTILE_CONT(
                        text_char_count,
                        0.90
                    ) AS p90_text_char_count
                FROM base
                GROUP BY observed_outcome
                ORDER BY observed_outcome DESC
            )
            TO '{text_path}'
            (
                FORMAT CSV,
                HEADER TRUE
            )
            """
        )
    finally:
        connection.close()

    print("Outcome observation analysis completed.")
    print(f"Saved to: {TABLES_DIR}")