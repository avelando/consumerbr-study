import csv

import duckdb

from consumerbr_resolution.config import (
    COMPANY_HISTORY_DIR,
    FEATURE_BASE_PATH,
    TABLES_DIR,
    TEMPORAL_FOLDS,
    create_project_directories,
)


COMPANY_HISTORY_SUMMARY_PATH = (
    TABLES_DIR / "company_history_feature_summary.csv"
)


SUMMARY_FIELDS = [
    "fold",
    "split",
    "complaint_count",
    "company_seen_rate",
    "mean_company_history_count",
    "mean_company_history_rate",
    "mean_global_history_rate",
]


def get_split_path(
    fold_number,
    split,
):
    return (
        COMPANY_HISTORY_DIR
        / f"fold_{fold_number:02d}_{split}.parquet"
    )


def write_train_history(
    connection,
    source_path,
    output_path,
    train_end,
):
    temporary_path = output_path.with_suffix(
        ".parquet.part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    target_path = str(
        temporary_path
    ).replace("'", "''")

    connection.execute(
        f"""
        COPY (
            WITH base AS (
                SELECT
                    record_id,
                    complaint_id,
                    company,
                    opening_date,
                    target_resolved
                FROM read_parquet('{source_path}')
                WHERE opening_date <= DATE '{train_end}'
            ),
            company_daily AS (
                SELECT
                    company,
                    opening_date,
                    COUNT(*) AS daily_count,
                    SUM(target_resolved) AS daily_resolved
                FROM base
                GROUP BY
                    company,
                    opening_date
            ),
            company_history AS (
                SELECT
                    company,
                    opening_date,
                    COALESCE(
                        SUM(daily_count) OVER (
                            PARTITION BY company
                            ORDER BY opening_date
                            ROWS BETWEEN
                                UNBOUNDED PRECEDING
                                AND 1 PRECEDING
                        ),
                        0
                    ) AS prior_company_count,
                    COALESCE(
                        SUM(daily_resolved) OVER (
                            PARTITION BY company
                            ORDER BY opening_date
                            ROWS BETWEEN
                                UNBOUNDED PRECEDING
                                AND 1 PRECEDING
                        ),
                        0
                    ) AS prior_company_resolved
                FROM company_daily
            ),
            global_daily AS (
                SELECT
                    opening_date,
                    COUNT(*) AS daily_count,
                    SUM(target_resolved) AS daily_resolved
                FROM base
                GROUP BY opening_date
            ),
            global_history AS (
                SELECT
                    opening_date,
                    COALESCE(
                        SUM(daily_count) OVER (
                            ORDER BY opening_date
                            ROWS BETWEEN
                                UNBOUNDED PRECEDING
                                AND 1 PRECEDING
                        ),
                        0
                    ) AS prior_global_count,
                    COALESCE(
                        SUM(daily_resolved) OVER (
                            ORDER BY opening_date
                            ROWS BETWEEN
                                UNBOUNDED PRECEDING
                                AND 1 PRECEDING
                        ),
                        0
                    ) AS prior_global_resolved
                FROM global_daily
            )
            SELECT
                base.record_id,
                base.complaint_id,
                base.opening_date,
                base.target_resolved,
                CAST(
                    company_history.prior_company_count
                    AS BIGINT
                ) AS company_history_count,
                LN(
                    1
                    + company_history.prior_company_count
                ) AS log_company_history_count,
                CASE
                    WHEN
                        company_history.prior_company_count > 0
                        THEN
                            company_history.prior_company_resolved
                            * 1.0
                            / company_history.prior_company_count
                    WHEN
                        global_history.prior_global_count > 0
                        THEN
                            global_history.prior_global_resolved
                            * 1.0
                            / global_history.prior_global_count
                    ELSE 0.5
                END AS company_history_rate,
                CASE
                    WHEN
                        company_history.prior_company_count > 0
                        THEN 1
                    ELSE 0
                END AS company_seen_before,
                CASE
                    WHEN
                        global_history.prior_global_count > 0
                        THEN
                            global_history.prior_global_resolved
                            * 1.0
                            / global_history.prior_global_count
                    ELSE 0.5
                END AS global_history_rate
            FROM base
            JOIN company_history
                ON
                    base.company = company_history.company
                    AND base.opening_date
                        = company_history.opening_date
            JOIN global_history
                ON
                    base.opening_date
                        = global_history.opening_date
            ORDER BY
                base.opening_date,
                base.complaint_id,
                base.record_id
        )
        TO '{target_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    temporary_path.replace(
        output_path
    )


def write_future_history(
    connection,
    source_path,
    output_path,
    train_end,
    split_start,
    split_end,
):
    temporary_path = output_path.with_suffix(
        ".parquet.part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    target_path = str(
        temporary_path
    ).replace("'", "''")

    connection.execute(
        f"""
        COPY (
            WITH company_history AS (
                SELECT
                    company,
                    COUNT(*) AS company_history_count,
                    AVG(target_resolved)
                        AS company_history_rate
                FROM read_parquet('{source_path}')
                WHERE opening_date <= DATE '{train_end}'
                GROUP BY company
            ),
            global_history AS (
                SELECT
                    COUNT(*) AS global_history_count,
                    AVG(target_resolved)
                        AS global_history_rate
                FROM read_parquet('{source_path}')
                WHERE opening_date <= DATE '{train_end}'
            )
            SELECT
                data.record_id,
                data.complaint_id,
                data.opening_date,
                data.target_resolved,
                CAST(
                    COALESCE(
                        company_history.company_history_count,
                        0
                    )
                    AS BIGINT
                ) AS company_history_count,
                LN(
                    1
                    + COALESCE(
                        company_history.company_history_count,
                        0
                    )
                ) AS log_company_history_count,
                COALESCE(
                    company_history.company_history_rate,
                    global_history.global_history_rate
                ) AS company_history_rate,
                CASE
                    WHEN
                        company_history.company_history_count
                        IS NULL
                        THEN 0
                    ELSE 1
                END AS company_seen_before,
                global_history.global_history_rate
                    AS global_history_rate
            FROM read_parquet('{source_path}')
                AS data
            CROSS JOIN global_history
            LEFT JOIN company_history
                ON
                    data.company
                    = company_history.company
            WHERE
                data.opening_date
                BETWEEN
                    DATE '{split_start}'
                    AND DATE '{split_end}'
            ORDER BY
                data.opening_date,
                data.complaint_id,
                data.record_id
        )
        TO '{target_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    temporary_path.replace(
        output_path
    )


def build_summary(
    connection,
):
    rows = []

    for fold in TEMPORAL_FOLDS:
        fold_number = fold["fold"]

        for split in (
            "train",
            "validation",
            "test",
        ):
            path = get_split_path(
                fold_number,
                split,
            )

            source_path = str(
                path
            ).replace("'", "''")

            result = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS complaint_count,
                    AVG(company_seen_before)
                        AS company_seen_rate,
                    AVG(company_history_count)
                        AS mean_company_history_count,
                    AVG(company_history_rate)
                        AS mean_company_history_rate,
                    AVG(global_history_rate)
                        AS mean_global_history_rate
                FROM read_parquet('{source_path}')
                """
            ).fetchone()

            rows.append(
                {
                    "fold": fold_number,
                    "split": split,
                    "complaint_count": int(
                        result[0]
                    ),
                    "company_seen_rate": float(
                        result[1]
                    ),
                    "mean_company_history_count": float(
                        result[2]
                    ),
                    "mean_company_history_rate": float(
                        result[3]
                    ),
                    "mean_global_history_rate": float(
                        result[4]
                    ),
                }
            )

    with COMPANY_HISTORY_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


def build_company_history_features():
    create_project_directories()

    fold_paths = []

    for fold in TEMPORAL_FOLDS:
        fold_number = fold["fold"]

        fold_paths.extend(
            [
                get_split_path(
                    fold_number,
                    "train",
                ),
                get_split_path(
                    fold_number,
                    "validation",
                ),
                get_split_path(
                    fold_number,
                    "test",
                ),
            ]
        )

    if (
        COMPANY_HISTORY_SUMMARY_PATH.exists()
        and all(
            path.exists()
            for path in fold_paths
        )
    ):
        print(
            "Company history features already exist."
        )
        return

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    print(
        "Building causal company history features"
    )
    print(
        f"Source: {FEATURE_BASE_PATH}"
    )
    print(
        f"Destination: {COMPANY_HISTORY_DIR}"
    )

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            train_path = get_split_path(
                fold_number,
                "train",
            )

            validation_path = get_split_path(
                fold_number,
                "validation",
            )

            test_path = get_split_path(
                fold_number,
                "test",
            )

            outputs = [
                train_path,
                validation_path,
                test_path,
            ]

            if all(
                path.exists()
                for path in outputs
            ):
                print()
                print(
                    f"Company history fold "
                    f"{fold_number} already exists."
                )
                continue

            for path in outputs:
                if path.exists():
                    path.unlink()

            print()
            print(
                f"Building company history "
                f"fold {fold_number}"
            )

            write_train_history(
                connection=connection,
                source_path=source_path,
                output_path=train_path,
                train_end=fold["train_end"],
            )

            write_future_history(
                connection=connection,
                source_path=source_path,
                output_path=validation_path,
                train_end=fold["train_end"],
                split_start=fold[
                    "validation_start"
                ],
                split_end=fold[
                    "validation_end"
                ],
            )

            write_future_history(
                connection=connection,
                source_path=source_path,
                output_path=test_path,
                train_end=fold["train_end"],
                split_start=fold[
                    "test_start"
                ],
                split_end=fold[
                    "test_end"
                ],
            )

            print(
                f"Fold {fold_number} completed."
            )

        build_summary(
            connection=connection,
        )

    finally:
        connection.close()

    print()
    print(
        "Company history feature construction completed."
    )
    print(
        f"Saved to: "
        f"{COMPANY_HISTORY_SUMMARY_PATH}"
    )