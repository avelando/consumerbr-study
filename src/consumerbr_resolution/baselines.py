import csv

import duckdb
import numpy as np

from consumerbr_resolution.config import (
    FEATURE_BASE_PATH,
    METRICS_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
    TEMPORAL_FOLDS,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)


BASELINE_METRICS_PATH = (
    METRICS_DIR
    / "historical_baseline_metrics.csv"
)

BASELINE_COVERAGE_PATH = (
    TABLES_DIR
    / "historical_baseline_coverage.csv"
)

BASELINE_PREDICTIONS_DIR = (
    PREDICTIONS_DIR
    / "historical_baselines"
)


METRIC_FIELDS = [
    "fold",
    "split",
    "model",
    "threshold_source",
    "threshold",
    "accuracy",
    "balanced_accuracy",
    "precision_resolved",
    "recall_resolved",
    "f1_resolved",
    "precision_unresolved",
    "recall_unresolved",
    "f1_unresolved",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
]


COVERAGE_FIELDS = [
    "fold",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "train_company_count",
    "train_resolution_rate",
    "validation_count",
    "validation_company_seen_rate",
    "test_count",
    "test_company_seen_rate",
    "company_threshold",
    "validation_company_macro_f1",
    "test_company_macro_f1",
]


def load_split_scores(
    connection,
    source_path,
    train_end,
    split_start,
    split_end,
):
    result = connection.execute(
        f"""
        WITH company_history AS (
            SELECT
                company,
                AVG(
                    target_resolved
                ) AS company_rate
            FROM read_parquet(
                '{source_path}'
            )
            WHERE opening_date <= DATE '{train_end}'
            GROUP BY company
        ),
        global_history AS (
            SELECT
                AVG(
                    target_resolved
                ) AS global_rate
            FROM read_parquet(
                '{source_path}'
            )
            WHERE opening_date <= DATE '{train_end}'
        )
        SELECT
            CAST(
                data.target_resolved
                AS TINYINT
            ) AS target_resolved,
            global_history.global_rate
                AS global_score,
            COALESCE(
                company_history.company_rate,
                global_history.global_rate
            ) AS company_score,
            CASE
                WHEN company_history.company
                    IS NULL
                    THEN 0
                ELSE 1
            END AS company_seen
        FROM read_parquet(
            '{source_path}'
        ) AS data
        CROSS JOIN global_history
        LEFT JOIN company_history
            ON data.company
                = company_history.company
        WHERE data.opening_date
            BETWEEN DATE '{split_start}'
            AND DATE '{split_end}'
        ORDER BY
            data.opening_date,
            data.complaint_id,
            data.record_id
        """
    ).fetchnumpy()

    return {
        "target": np.asarray(
            result["target_resolved"],
            dtype=np.int8,
        ),
        "global_score": np.asarray(
            result["global_score"],
            dtype=np.float64,
        ),
        "company_score": np.asarray(
            result["company_score"],
            dtype=np.float64,
        ),
        "company_seen": np.asarray(
            result["company_seen"],
            dtype=np.int8,
        ),
    }


def write_test_predictions(
    connection,
    source_path,
    fold,
    company_threshold,
    output_path,
):
    train_end = fold["train_end"]
    test_start = fold["test_start"]
    test_end = fold["test_end"]

    target_path = str(
        output_path
    ).replace("'", "''")

    threshold = format(
        company_threshold,
        ".17g",
    )

    connection.execute(
        f"""
        COPY (
            WITH company_history AS (
                SELECT
                    company,
                    AVG(
                        target_resolved
                    ) AS company_rate
                FROM read_parquet(
                    '{source_path}'
                )
                WHERE opening_date
                    <= DATE '{train_end}'
                GROUP BY company
            ),
            global_history AS (
                SELECT
                    AVG(
                        target_resolved
                    ) AS global_rate
                FROM read_parquet(
                    '{source_path}'
                )
                WHERE opening_date
                    <= DATE '{train_end}'
            )
            SELECT
                data.record_id,
                data.complaint_id,
                data.opening_date,
                data.company,
                data.target_resolved,
                global_history.global_rate
                    AS global_score,
                CAST(
                    CASE
                        WHEN
                            global_history.global_rate
                            >= 0.5
                            THEN 1
                        ELSE 0
                    END
                    AS INTEGER
                ) AS global_prediction,
                COALESCE(
                    company_history.company_rate,
                    global_history.global_rate
                ) AS company_score,
                CAST(
                    CASE
                        WHEN COALESCE(
                            company_history.company_rate,
                            global_history.global_rate
                        ) >= {threshold}
                            THEN 1
                        ELSE 0
                    END
                    AS INTEGER
                ) AS company_prediction,
                CASE
                    WHEN company_history.company
                        IS NULL
                        THEN 0
                    ELSE 1
                END AS company_seen
            FROM read_parquet(
                '{source_path}'
            ) AS data
            CROSS JOIN global_history
            LEFT JOIN company_history
                ON data.company
                    = company_history.company
            WHERE data.opening_date
                BETWEEN DATE '{test_start}'
                AND DATE '{test_end}'
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


def evaluate_historical_baselines():
    create_project_directories()

    BASELINE_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_paths = [
        BASELINE_PREDICTIONS_DIR
        / f"fold_{fold['fold']:02d}.parquet"
        for fold in TEMPORAL_FOLDS
    ]

    output_paths = [
        BASELINE_METRICS_PATH,
        BASELINE_COVERAGE_PATH,
        *prediction_paths,
    ]

    if all(
        path.exists()
        for path in output_paths
    ):
        print(
            "Historical baseline evaluation "
            "already exists."
        )
        return

    for path in output_paths:
        if path.exists():
            path.unlink()

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    print("Evaluating historical baselines")
    print(f"Source: {FEATURE_BASE_PATH}")
    print(f"Metrics: {BASELINE_METRICS_PATH}")
    print(
        f"Predictions: "
        f"{BASELINE_PREDICTIONS_DIR}"
    )

    metrics_rows = []
    coverage_rows = []

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            print()
            print(
                f"Evaluating fold "
                f"{fold_number}"
            )

            validation = load_split_scores(
                connection=connection,
                source_path=source_path,
                train_end=fold["train_end"],
                split_start=fold[
                    "validation_start"
                ],
                split_end=fold[
                    "validation_end"
                ],
            )

            test = load_split_scores(
                connection=connection,
                source_path=source_path,
                train_end=fold["train_end"],
                split_start=fold[
                    "test_start"
                ],
                split_end=fold[
                    "test_end"
                ],
            )

            company_threshold, _ = (
                find_best_macro_f1_threshold(
                    validation["target"],
                    validation["company_score"],
                )
            )

            validation_global_metrics = (
                calculate_binary_metrics(
                    validation["target"],
                    validation["global_score"],
                    0.5,
                )
            )

            test_global_metrics = (
                calculate_binary_metrics(
                    test["target"],
                    test["global_score"],
                    0.5,
                )
            )

            validation_company_metrics = (
                calculate_binary_metrics(
                    validation["target"],
                    validation["company_score"],
                    company_threshold,
                )
            )

            test_company_metrics = (
                calculate_binary_metrics(
                    test["target"],
                    test["company_score"],
                    company_threshold,
                )
            )

            metrics_rows.append(
                {
                    "fold": fold_number,
                    "split": "validation",
                    "model": "global_prior",
                    "threshold_source": "fixed",
                    **validation_global_metrics,
                }
            )

            metrics_rows.append(
                {
                    "fold": fold_number,
                    "split": "test",
                    "model": "global_prior",
                    "threshold_source": "fixed",
                    **test_global_metrics,
                }
            )

            metrics_rows.append(
                {
                    "fold": fold_number,
                    "split": "validation",
                    "model": (
                        "company_historical_rate"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    **validation_company_metrics,
                }
            )

            metrics_rows.append(
                {
                    "fold": fold_number,
                    "split": "test",
                    "model": (
                        "company_historical_rate"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    **test_company_metrics,
                }
            )

            train_summary = connection.execute(
                f"""
                SELECT
                    COUNT(
                        DISTINCT company
                    ) AS company_count,
                    AVG(
                        target_resolved
                    ) AS resolution_rate
                FROM read_parquet(
                    '{source_path}'
                )
                WHERE opening_date
                    <= DATE '{fold["train_end"]}'
                """
            ).fetchone()

            coverage_rows.append(
                {
                    "fold": fold_number,
                    "train_end": fold[
                        "train_end"
                    ],
                    "validation_start": fold[
                        "validation_start"
                    ],
                    "validation_end": fold[
                        "validation_end"
                    ],
                    "test_start": fold[
                        "test_start"
                    ],
                    "test_end": fold[
                        "test_end"
                    ],
                    "train_company_count": int(
                        train_summary[0]
                    ),
                    "train_resolution_rate": (
                        float(
                            train_summary[1]
                        )
                    ),
                    "validation_count": int(
                        len(
                            validation["target"]
                        )
                    ),
                    "validation_company_seen_rate": (
                        float(
                            validation[
                                "company_seen"
                            ].mean()
                        )
                    ),
                    "test_count": int(
                        len(
                            test["target"]
                        )
                    ),
                    "test_company_seen_rate": (
                        float(
                            test[
                                "company_seen"
                            ].mean()
                        )
                    ),
                    "company_threshold": (
                        company_threshold
                    ),
                    "validation_company_macro_f1": (
                        validation_company_metrics[
                            "macro_f1"
                        ]
                    ),
                    "test_company_macro_f1": (
                        test_company_metrics[
                            "macro_f1"
                        ]
                    ),
                }
            )

            prediction_path = (
                BASELINE_PREDICTIONS_DIR
                / (
                    f"fold_"
                    f"{fold_number:02d}.parquet"
                )
            )

            write_test_predictions(
                connection=connection,
                source_path=source_path,
                fold=fold,
                company_threshold=(
                    company_threshold
                ),
                output_path=prediction_path,
            )

            print(
                "Company baseline "
                f"validation Macro-F1: "
                f"{validation_company_metrics['macro_f1']:.4f}"
            )

            print(
                "Company baseline "
                f"test Macro-F1: "
                f"{test_company_metrics['macro_f1']:.4f}"
            )

    finally:
        connection.close()

    with BASELINE_METRICS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METRIC_FIELDS,
        )

        writer.writeheader()
        writer.writerows(metrics_rows)

    with BASELINE_COVERAGE_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=COVERAGE_FIELDS,
        )

        writer.writeheader()
        writer.writerows(coverage_rows)

    print()
    print(
        "Historical baseline evaluation "
        "completed."
    )
    print(
        f"Saved to: "
        f"{BASELINE_METRICS_PATH}"
    )
    print(
        f"Saved to: "
        f"{BASELINE_COVERAGE_PATH}"
    )