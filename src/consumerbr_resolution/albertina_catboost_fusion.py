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


ALBERTINA_PREDICTIONS_DIR = (
    PREDICTIONS_DIR / "albertina"
)

CATBOOST_PREDICTIONS_DIR = (
    PREDICTIONS_DIR / "catboost"
)

FUSION_PREDICTIONS_DIR = (
    PREDICTIONS_DIR
    / "albertina_catboost_fusion"
)

FUSION_METRICS_DIR = (
    METRICS_DIR
    / "albertina_catboost_fusion"
)

FUSION_METRICS_PATH = (
    METRICS_DIR
    / "albertina_catboost_fusion_metrics.csv"
)


METRIC_FIELDS = [
    "fold",
    "split",
    "model",
    "threshold_source",
    "threshold",
    "albertina_weight",
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
    albertina_path,
    catboost_path,
):
    albertina_source = str(
        albertina_path
    ).replace("'", "''")

    catboost_source = str(
        catboost_path
    ).replace("'", "''")

    return connection.execute(
        f"""
        SELECT
            albertina.record_id,
            albertina.complaint_id,
            albertina.opening_date,
            CAST(
                albertina.target_resolved
                AS TINYINT
            ) AS target_resolved,
            CAST(
                albertina.score
                AS DOUBLE
            ) AS albertina_score,
            CAST(
                catboost.score
                AS DOUBLE
            ) AS catboost_score
        FROM read_parquet(
            '{albertina_source}'
        ) AS albertina
        JOIN read_parquet(
            '{catboost_source}'
        ) AS catboost
            ON
                albertina.record_id
                = catboost.record_id
        ORDER BY
            albertina.record_id
        """
    ).fetchdf()


def combine_scores(
    albertina_scores,
    catboost_scores,
    albertina_weight,
):
    catboost_weight = (
        1.0
        - albertina_weight
    )

    return (
        albertina_weight
        * albertina_scores
        + catboost_weight
        * catboost_scores
    )


def select_fusion_parameters(
    targets,
    albertina_scores,
    catboost_scores,
):
    best_result = None

    for weight in LATE_FUSION_WEIGHTS:
        scores = combine_scores(
            albertina_scores=(
                albertina_scores
            ),
            catboost_scores=(
                catboost_scores
            ),
            albertina_weight=weight,
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
            "albertina_weight": (
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
                    "albertina_weight"
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
                    "albertina_weight"
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


def evaluate_albertina_catboost_fusion():
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
        "Evaluating Albertina + CatBoost late fusion"
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
                albertina_path=(
                    get_prediction_path(
                        directory=(
                            ALBERTINA_PREDICTIONS_DIR
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

            validation_albertina = (
                validation[
                    "albertina_score"
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
                    albertina_scores=(
                        validation_albertina
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
                    albertina_scores=(
                        validation_albertina
                    ),
                    catboost_scores=(
                        validation_catboost
                    ),
                    albertina_weight=(
                        parameters[
                            "albertina_weight"
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
                albertina_path=(
                    get_prediction_path(
                        directory=(
                            ALBERTINA_PREDICTIONS_DIR
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
                albertina_scores=(
                    test[
                        "albertina_score"
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
                albertina_weight=(
                    parameters[
                        "albertina_weight"
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
                        "albertina_catboost_fusion"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "albertina_weight": (
                        parameters[
                            "albertina_weight"
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
                        "albertina_catboost_fusion"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "albertina_weight": (
                        parameters[
                            "albertina_weight"
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
                "Albertina weight: "
                f"{parameters['albertina_weight']:.2f}"
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
        "Albertina + CatBoost fusion completed."
    )

    print(
        f"Saved to: "
        f"{FUSION_METRICS_PATH}"
    )