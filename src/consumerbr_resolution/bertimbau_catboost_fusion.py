import csv
import time

import duckdb
import numpy as np
import pandas as pd

from consumerbr_resolution.config import (
    LATE_FUSION_WEIGHTS,
    METRICS_DIR,
    PREDICTIONS_DIR,
    TEMPORAL_FOLDS,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)


BERTIMBAU_PREDICTIONS_DIR = (
    PREDICTIONS_DIR / "bertimbau"
)

CATBOOST_PREDICTIONS_DIR = (
    PREDICTIONS_DIR / "catboost"
)

FUSION_PREDICTIONS_DIR = (
    PREDICTIONS_DIR
    / "bertimbau_catboost_fusion"
)

FUSION_METRICS_DIR = (
    METRICS_DIR
    / "bertimbau_catboost_fusion"
)

FUSION_METRICS_PATH = (
    METRICS_DIR
    / "bertimbau_catboost_fusion_metrics.csv"
)


METRIC_FIELDS = [
    "fold",
    "split",
    "model",
    "threshold_source",
    "threshold",
    "bertimbau_weight",
    "catboost_weight",
    "tuning_seconds",
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


def get_prediction_path(
    directory,
    fold_number,
    split,
):
    return (
        directory
        / (
            f"fold_{fold_number:02d}"
            f"_{split}.parquet"
        )
    )


def load_predictions(
    connection,
    bertimbau_path,
    catboost_path,
):
    bertimbau_source = str(
        bertimbau_path
    ).replace("'", "''")

    catboost_source = str(
        catboost_path
    ).replace("'", "''")

    return connection.execute(
        f"""
        SELECT
            bertimbau.record_id,
            bertimbau.complaint_id,
            bertimbau.opening_date,
            CAST(
                bertimbau.target_resolved
                AS TINYINT
            ) AS target_resolved,
            CAST(
                bertimbau.score
                AS DOUBLE
            ) AS bertimbau_score,
            CAST(
                catboost.score
                AS DOUBLE
            ) AS catboost_score
        FROM read_parquet(
            '{bertimbau_source}'
        ) AS bertimbau
        JOIN read_parquet(
            '{catboost_source}'
        ) AS catboost
            ON
                bertimbau.record_id
                = catboost.record_id
        ORDER BY
            bertimbau.record_id
        """
    ).fetchdf()


def combine_scores(
    bertimbau_scores,
    catboost_scores,
    bertimbau_weight,
):
    catboost_weight = (
        1.0
        - bertimbau_weight
    )

    return (
        bertimbau_weight
        * bertimbau_scores
        + catboost_weight
        * catboost_scores
    )


def select_fusion_parameters(
    targets,
    bertimbau_scores,
    catboost_scores,
):
    best_result = None

    for weight in LATE_FUSION_WEIGHTS:
        scores = combine_scores(
            bertimbau_scores=(
                bertimbau_scores
            ),
            catboost_scores=(
                catboost_scores
            ),
            bertimbau_weight=weight,
        )

        (
            threshold,
            macro_f1,
        ) = (
            find_best_macro_f1_threshold(
                targets,
                scores,
            )
        )

        candidate = {
            "bertimbau_weight": (
                float(weight)
            ),
            "catboost_weight": (
                float(1.0 - weight)
            ),
            "threshold": (
                float(threshold)
            ),
            "macro_f1": (
                float(macro_f1)
            ),
        }

        if best_result is None:
            best_result = candidate
            continue

        candidate_key = (
            candidate["macro_f1"],
            -abs(
                candidate[
                    "bertimbau_weight"
                ]
                - 0.5
            ),
            -abs(
                candidate[
                    "threshold"
                ]
                - 0.5
            ),
        )

        best_key = (
            best_result["macro_f1"],
            -abs(
                best_result[
                    "bertimbau_weight"
                ]
                - 0.5
            ),
            -abs(
                best_result[
                    "threshold"
                ]
                - 0.5
            ),
        )

        if candidate_key > best_key:
            best_result = candidate

    return best_result


def write_predictions(
    path,
    frame,
    scores,
    threshold,
):
    temporary_path = (
        path.with_suffix(
            ".parquet.part"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    predictions = (
        scores >= threshold
    ).astype(np.int8)

    output = pd.DataFrame(
        {
            "record_id": (
                frame["record_id"]
            ),
            "complaint_id": (
                frame["complaint_id"]
            ),
            "opening_date": (
                frame["opening_date"]
            ),
            "target_resolved": (
                frame["target_resolved"]
                .astype(np.int8)
            ),
            "score": scores,
            "prediction": predictions,
        }
    )

    output.to_parquet(
        temporary_path,
        index=False,
        compression="zstd",
    )

    temporary_path.replace(
        path
    )


def write_fold_metrics(
    path,
    rows,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METRIC_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


def rebuild_aggregate_metrics():
    rows = []

    for fold in TEMPORAL_FOLDS:
        path = (
            FUSION_METRICS_DIR
            / f"fold_{fold['fold']:02d}.csv"
        )

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(
                file
            )

            rows.extend(reader)

    with FUSION_METRICS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METRIC_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


def evaluate_bertimbau_catboost_fusion():
    create_project_directories()

    FUSION_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FUSION_METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Evaluating BERTimbau + CatBoost late fusion"
    )

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            validation_output_path = (
                get_prediction_path(
                    directory=(
                        FUSION_PREDICTIONS_DIR
                    ),
                    fold_number=fold_number,
                    split="validation",
                )
            )

            test_output_path = (
                get_prediction_path(
                    directory=(
                        FUSION_PREDICTIONS_DIR
                    ),
                    fold_number=fold_number,
                    split="test",
                )
            )

            metrics_path = (
                FUSION_METRICS_DIR
                / f"fold_{fold_number:02d}.csv"
            )

            outputs = [
                validation_output_path,
                test_output_path,
                metrics_path,
            ]

            if all(
                path.exists()
                for path in outputs
            ):
                print()
                print(
                    f"Fold {fold_number} "
                    f"already exists."
                )
                continue

            for path in outputs:
                if path.exists():
                    path.unlink()

            print()
            print(
                f"Evaluating fusion fold "
                f"{fold_number}"
            )

            validation = load_predictions(
                connection=connection,
                bertimbau_path=(
                    get_prediction_path(
                        directory=(
                            BERTIMBAU_PREDICTIONS_DIR
                        ),
                        fold_number=(
                            fold_number
                        ),
                        split="validation",
                    )
                ),
                catboost_path=(
                    get_prediction_path(
                        directory=(
                            CATBOOST_PREDICTIONS_DIR
                        ),
                        fold_number=(
                            fold_number
                        ),
                        split="validation",
                    )
                ),
            )

            validation_targets = (
                validation[
                    "target_resolved"
                ].to_numpy(
                    dtype=np.int8
                )
            )

            validation_bertimbau = (
                validation[
                    "bertimbau_score"
                ].to_numpy(
                    dtype=np.float64
                )
            )

            validation_catboost = (
                validation[
                    "catboost_score"
                ].to_numpy(
                    dtype=np.float64
                )
            )

            tuning_start = (
                time.perf_counter()
            )

            parameters = (
                select_fusion_parameters(
                    targets=(
                        validation_targets
                    ),
                    bertimbau_scores=(
                        validation_bertimbau
                    ),
                    catboost_scores=(
                        validation_catboost
                    ),
                )
            )

            tuning_seconds = (
                time.perf_counter()
                - tuning_start
            )

            validation_scores = (
                combine_scores(
                    bertimbau_scores=(
                        validation_bertimbau
                    ),
                    catboost_scores=(
                        validation_catboost
                    ),
                    bertimbau_weight=(
                        parameters[
                            "bertimbau_weight"
                        ]
                    ),
                )
            )

            validation_metrics = (
                calculate_binary_metrics(
                    validation_targets,
                    validation_scores,
                    parameters[
                        "threshold"
                    ],
                )
            )

            test = load_predictions(
                connection=connection,
                bertimbau_path=(
                    get_prediction_path(
                        directory=(
                            BERTIMBAU_PREDICTIONS_DIR
                        ),
                        fold_number=(
                            fold_number
                        ),
                        split="test",
                    )
                ),
                catboost_path=(
                    get_prediction_path(
                        directory=(
                            CATBOOST_PREDICTIONS_DIR
                        ),
                        fold_number=(
                            fold_number
                        ),
                        split="test",
                    )
                ),
            )

            test_targets = (
                test[
                    "target_resolved"
                ].to_numpy(
                    dtype=np.int8
                )
            )

            test_scores = combine_scores(
                bertimbau_scores=(
                    test[
                        "bertimbau_score"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ),
                catboost_scores=(
                    test[
                        "catboost_score"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ),
                bertimbau_weight=(
                    parameters[
                        "bertimbau_weight"
                    ]
                ),
            )

            test_metrics = (
                calculate_binary_metrics(
                    test_targets,
                    test_scores,
                    parameters[
                        "threshold"
                    ],
                )
            )

            write_predictions(
                path=(
                    validation_output_path
                ),
                frame=validation,
                scores=validation_scores,
                threshold=(
                    parameters[
                        "threshold"
                    ]
                ),
            )

            write_predictions(
                path=test_output_path,
                frame=test,
                scores=test_scores,
                threshold=(
                    parameters[
                        "threshold"
                    ]
                ),
            )

            rows = [
                {
                    "fold": fold_number,
                    "split": "validation",
                    "model": (
                        "bertimbau_catboost_fusion"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "bertimbau_weight": (
                        parameters[
                            "bertimbau_weight"
                        ]
                    ),
                    "catboost_weight": (
                        parameters[
                            "catboost_weight"
                        ]
                    ),
                    "tuning_seconds": (
                        tuning_seconds
                    ),
                    **validation_metrics,
                },
                {
                    "fold": fold_number,
                    "split": "test",
                    "model": (
                        "bertimbau_catboost_fusion"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "bertimbau_weight": (
                        parameters[
                            "bertimbau_weight"
                        ]
                    ),
                    "catboost_weight": (
                        parameters[
                            "catboost_weight"
                        ]
                    ),
                    "tuning_seconds": (
                        tuning_seconds
                    ),
                    **test_metrics,
                },
            ]

            write_fold_metrics(
                path=metrics_path,
                rows=rows,
            )

            print(
                "BERTimbau weight: "
                f"{parameters['bertimbau_weight']:.2f}"
            )

            print(
                "CatBoost weight: "
                f"{parameters['catboost_weight']:.2f}"
            )

            print(
                "Threshold: "
                f"{parameters['threshold']:.6f}"
            )

            print(
                "Validation Macro-F1: "
                f"{validation_metrics['macro_f1']:.4f}"
            )

            print(
                "Test Macro-F1: "
                f"{test_metrics['macro_f1']:.4f}"
            )

    finally:
        connection.close()

    rebuild_aggregate_metrics()

    print()
    print(
        "BERTimbau + CatBoost fusion completed."
    )

    print(
        f"Saved to: "
        f"{FUSION_METRICS_PATH}"
    )