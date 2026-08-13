import csv

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from consumerbr_resolution.analysis_registry import (
    MODEL_PREDICTION_SPECS,
    get_prediction_path,
)
from consumerbr_resolution.config import (
    ANALYSIS_DIR,
    COMPANY_MIN_FREQUENCY,
    FEATURE_BASE_PATH,
    TEMPORAL_FOLDS,
    create_project_directories,
)


COMPANY_GENERALIZATION_PATH = (
    ANALYSIS_DIR
    / "company_generalization_metrics.csv"
)

MONTHLY_MODEL_METRICS_PATH = (
    ANALYSIS_DIR
    / "monthly_test_metrics.csv"
)


GENERALIZATION_FIELDS = [
    "fold",
    "model",
    "company_segment",
    "complaint_count",
    "resolved_rate",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
]


MONTHLY_FIELDS = [
    "fold",
    "model",
    "month",
    "complaint_count",
    "resolved_rate",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
]


def calculate_metrics(
    targets,
    scores,
    predictions,
):
    targets = np.asarray(
        targets,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int8,
    )

    unique_classes = np.unique(
        targets
    )

    if len(unique_classes) == 2:
        roc_auc = float(
            roc_auc_score(
                targets,
                scores,
            )
        )
    else:
        roc_auc = float("nan")

    return {
        "complaint_count": int(
            len(targets)
        ),
        "resolved_rate": float(
            targets.mean()
        ),
        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                targets,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc,
        "pr_auc": float(
            average_precision_score(
                targets,
                scores,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                targets,
                scores,
            )
        ),
    }


def load_model_fold(
    connection,
    specification,
    fold,
):
    fold_number = fold["fold"]

    prediction_path = (
        get_prediction_path(
            specification=(
                specification
            ),
            fold_number=fold_number,
        )
    )

    prediction_source = str(
        prediction_path
    ).replace("'", "''")

    feature_source = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    score_column = specification[
        "score_column"
    ]

    prediction_column = specification[
        "prediction_column"
    ]

    train_end = fold["train_end"]

    return connection.execute(
        f"""
        WITH company_counts AS (
            SELECT
                company,
                COUNT(*)
                    AS train_company_count
            FROM read_parquet(
                '{feature_source}'
            )
            WHERE
                opening_date
                <= DATE '{train_end}'
            GROUP BY company
        )
        SELECT
            prediction.record_id,
            prediction.complaint_id,
            prediction.opening_date,
            CAST(
                prediction.target_resolved
                AS TINYINT
            ) AS target_resolved,
            CAST(
                prediction.{score_column}
                AS DOUBLE
            ) AS score,
            CAST(
                prediction.{prediction_column}
                AS TINYINT
            ) AS prediction,
            COALESCE(
                company_counts.train_company_count,
                0
            ) AS train_company_count
        FROM read_parquet(
            '{prediction_source}'
        ) AS prediction
        JOIN read_parquet(
            '{feature_source}'
        ) AS data
            ON
                prediction.record_id
                = data.record_id
        LEFT JOIN company_counts
            ON
                data.company
                = company_counts.company
        ORDER BY
            prediction.record_id
        """
    ).fetchdf()


def assign_company_segments(
    frame,
):
    counts = frame[
        "train_company_count"
    ].to_numpy()

    frame = frame.copy()

    frame[
        "company_segment"
    ] = np.select(
        [
            counts == 0,
            counts
            < COMPANY_MIN_FREQUENCY,
        ],
        [
            "unseen",
            "rare",
        ],
        default="frequent",
    )

    return frame


def analyze_company_segments(
    frame,
    fold_number,
    model_name,
):
    rows = []

    groups = [
        (
            "all",
            frame,
        )
    ]

    for segment in (
        "frequent",
        "rare",
        "unseen",
    ):
        groups.append(
            (
                segment,
                frame[
                    frame[
                        "company_segment"
                    ]
                    == segment
                ],
            )
        )

    for segment, group in groups:
        if len(group) == 0:
            continue

        metrics = calculate_metrics(
            targets=group[
                "target_resolved"
            ],
            scores=group["score"],
            predictions=group[
                "prediction"
            ],
        )

        rows.append(
            {
                "fold": fold_number,
                "model": model_name,
                "company_segment": (
                    segment
                ),
                **metrics,
            }
        )

    return rows


def analyze_months(
    frame,
    fold_number,
    model_name,
):
    frame = frame.copy()

    frame["month"] = (
        pd.to_datetime(
            frame["opening_date"]
        )
        .dt.to_period("M")
        .astype(str)
    )

    rows = []

    for month, group in frame.groupby(
        "month",
        sort=True,
    ):
        metrics = calculate_metrics(
            targets=group[
                "target_resolved"
            ],
            scores=group["score"],
            predictions=group[
                "prediction"
            ],
        )

        rows.append(
            {
                "fold": fold_number,
                "model": model_name,
                "month": month,
                **metrics,
            }
        )

    return rows


def write_rows(
    path,
    fieldnames,
    rows,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def analyze_generalization():
    create_project_directories()

    output_paths = [
        COMPANY_GENERALIZATION_PATH,
        MONTHLY_MODEL_METRICS_PATH,
    ]

    if all(
        path.exists()
        for path in output_paths
    ):
        print(
            "Generalization analysis already exists."
        )
        return

    for path in output_paths:
        if path.exists():
            path.unlink()

    print(
        "Analyzing temporal and company generalization"
    )

    company_rows = []
    monthly_rows = []

    connection = duckdb.connect()

    try:
        for specification in (
            MODEL_PREDICTION_SPECS
        ):
            model_name = specification[
                "model"
            ]

            print()
            print(
                f"Analyzing model: "
                f"{model_name}"
            )

            for fold in TEMPORAL_FOLDS:
                fold_number = fold[
                    "fold"
                ]

                frame = load_model_fold(
                    connection=connection,
                    specification=(
                        specification
                    ),
                    fold=fold,
                )

                frame = (
                    assign_company_segments(
                        frame
                    )
                )

                company_rows.extend(
                    analyze_company_segments(
                        frame=frame,
                        fold_number=(
                            fold_number
                        ),
                        model_name=(
                            model_name
                        ),
                    )
                )

                monthly_rows.extend(
                    analyze_months(
                        frame=frame,
                        fold_number=(
                            fold_number
                        ),
                        model_name=(
                            model_name
                        ),
                    )
                )

                print(
                    f"Fold {fold_number} "
                    f"completed."
                )

    finally:
        connection.close()

    write_rows(
        path=COMPANY_GENERALIZATION_PATH,
        fieldnames=(
            GENERALIZATION_FIELDS
        ),
        rows=company_rows,
    )

    write_rows(
        path=MONTHLY_MODEL_METRICS_PATH,
        fieldnames=MONTHLY_FIELDS,
        rows=monthly_rows,
    )

    print()
    print(
        "Generalization analysis completed."
    )

    print(
        f"Saved to: "
        f"{COMPANY_GENERALIZATION_PATH}"
    )

    print(
        f"Saved to: "
        f"{MONTHLY_MODEL_METRICS_PATH}"
    )