import csv
import time

import duckdb
import joblib
import numpy as np
from sklearn.naive_bayes import ComplementNB
from tqdm import tqdm

from consumerbr_resolution.config import (
    COMPLEMENT_NB_ALPHA,
    FEATURE_BASE_PATH,
    METRICS_DIR,
    PREDICTIONS_DIR,
    SGD_BATCH_SIZE,
    TEMPORAL_FOLDS,
    TFIDF_COMPLEMENT_NB_MODELS_DIR,
    TFIDF_MODELS_DIR,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)
from consumerbr_resolution.tfidf_sgd import (
    get_document_count,
    iter_batches,
    score_split,
    write_predictions,
)


METRICS_DIRECTORY = (
    METRICS_DIR / "tfidf_complement_nb"
)

AGGREGATE_METRICS_PATH = (
    METRICS_DIR
    / "tfidf_complement_nb_metrics.csv"
)

PREDICTION_DIRECTORY = (
    PREDICTIONS_DIR
    / "tfidf_complement_nb"
)


METRIC_FIELDS = [
    "fold",
    "split",
    "model",
    "threshold_source",
    "threshold",
    "alpha",
    "training_seconds",
    "scoring_seconds",
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


def create_classifier():
    return ComplementNB(
        alpha=COMPLEMENT_NB_ALPHA,
    )


def train_model(
    connection,
    source_path,
    train_end,
    vectorizer,
    model,
    document_count,
):
    classes = np.asarray(
        [0, 1],
        dtype=np.int8,
    )

    first_batch = True

    progress = tqdm(
        total=document_count,
        desc="Training",
        unit="docs",
        dynamic_ncols=True,
    )

    start_time = time.perf_counter()

    try:
        for rows in iter_batches(
            connection=connection,
            source_path=source_path,
            end_date=train_end,
        ):
            texts = [
                row[0]
                for row in rows
            ]

            targets = np.asarray(
                [
                    row[1]
                    for row in rows
                ],
                dtype=np.int8,
            )

            features = vectorizer.transform(
                texts
            )

            if first_batch:
                model.partial_fit(
                    features,
                    targets,
                    classes=classes,
                )

                first_batch = False
            else:
                model.partial_fit(
                    features,
                    targets,
                )

            progress.update(
                len(rows)
            )

    finally:
        progress.close()

    return (
        time.perf_counter()
        - start_time
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
            METRICS_DIRECTORY
            / f"fold_{fold['fold']:02d}.csv"
        )

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)
            rows.extend(reader)

    with AGGREGATE_METRICS_PATH.open(
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


def evaluate_tfidf_complement_nb():
    create_project_directories()

    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    print(
        "Evaluating TF-IDF + ComplementNB"
    )

    print(
        f"Source: {FEATURE_BASE_PATH}"
    )

    print(
        f"Models: "
        f"{TFIDF_COMPLEMENT_NB_MODELS_DIR}"
    )

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            model_path = (
                TFIDF_COMPLEMENT_NB_MODELS_DIR
                / f"fold_{fold_number:02d}.joblib"
            )

            prediction_path = (
                PREDICTION_DIRECTORY
                / f"fold_{fold_number:02d}.parquet"
            )

            metrics_path = (
                METRICS_DIRECTORY
                / f"fold_{fold_number:02d}.csv"
            )

            outputs = [
                model_path,
                prediction_path,
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
                f"Evaluating fold "
                f"{fold_number}"
            )

            vectorizer = joblib.load(
                TFIDF_MODELS_DIR
                / f"fold_{fold_number:02d}.joblib"
            )

            model = create_classifier()

            document_count = (
                get_document_count(
                    connection=connection,
                    source_path=source_path,
                    end_date=fold[
                        "train_end"
                    ],
                )
            )

            training_seconds = train_model(
                connection=connection,
                source_path=source_path,
                train_end=fold[
                    "train_end"
                ],
                vectorizer=vectorizer,
                model=model,
                document_count=(
                    document_count
                ),
            )

            validation = score_split(
                connection=connection,
                source_path=source_path,
                start_date=fold[
                    "validation_start"
                ],
                end_date=fold[
                    "validation_end"
                ],
                vectorizer=vectorizer,
                model=model,
            )

            (
                threshold,
                validation_macro_f1,
            ) = (
                find_best_macro_f1_threshold(
                    validation["target"],
                    validation["score"],
                )
            )

            validation_metrics = (
                calculate_binary_metrics(
                    validation["target"],
                    validation["score"],
                    threshold,
                )
            )

            test = score_split(
                connection=connection,
                source_path=source_path,
                start_date=fold[
                    "test_start"
                ],
                end_date=fold[
                    "test_end"
                ],
                vectorizer=vectorizer,
                model=model,
                collect_metadata=True,
            )

            test_metrics = (
                calculate_binary_metrics(
                    test["target"],
                    test["score"],
                    threshold,
                )
            )

            rows = [
                {
                    "fold": fold_number,
                    "split": "validation",
                    "model": (
                        "tfidf_complement_nb"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "alpha": (
                        COMPLEMENT_NB_ALPHA
                    ),
                    "training_seconds": (
                        training_seconds
                    ),
                    "scoring_seconds": (
                        validation[
                            "scoring_seconds"
                        ]
                    ),
                    **validation_metrics,
                },
                {
                    "fold": fold_number,
                    "split": "test",
                    "model": (
                        "tfidf_complement_nb"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "alpha": (
                        COMPLEMENT_NB_ALPHA
                    ),
                    "training_seconds": (
                        training_seconds
                    ),
                    "scoring_seconds": (
                        test[
                            "scoring_seconds"
                        ]
                    ),
                    **test_metrics,
                },
            ]

            temporary_model_path = (
                model_path.with_suffix(
                    ".joblib.part"
                )
            )

            if temporary_model_path.exists():
                temporary_model_path.unlink()

            joblib.dump(
                model,
                temporary_model_path,
                compress=3,
            )

            temporary_model_path.replace(
                model_path
            )

            write_predictions(
                connection=connection,
                prediction_path=(
                    prediction_path
                ),
                result=test,
                threshold=threshold,
            )

            write_fold_metrics(
                path=metrics_path,
                rows=rows,
            )

            print(
                f"Threshold: "
                f"{threshold:.6f}"
            )

            print(
                "Validation Macro-F1: "
                f"{validation_macro_f1:.4f}"
            )

            print(
                "Test Macro-F1: "
                f"{test_metrics['macro_f1']:.4f}"
            )

            print(
                f"Training time: "
                f"{training_seconds:.2f} seconds"
            )

    finally:
        connection.close()

    rebuild_aggregate_metrics()

    print()
    print(
        "TF-IDF + ComplementNB "
        "evaluation completed."
    )

    print(
        f"Saved to: "
        f"{AGGREGATE_METRICS_PATH}"
    )