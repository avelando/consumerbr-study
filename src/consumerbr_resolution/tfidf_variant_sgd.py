import csv
import time

import duckdb
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from consumerbr_resolution.config import (
    FEATURE_BASE_PATH,
    METRICS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    SGD_BATCH_SIZE,
    SGD_EPOCHS,
    SGD_LOSS,
    SGD_PENALTY,
    TEMPORAL_FOLDS,
    TFIDF_CHAR_MODELS_DIR,
    TFIDF_CHAR_SGD_MODELS_DIR,
    TFIDF_MODELS_DIR,
    TFIDF_WORD_CHAR_SGD_MODELS_DIR,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)
from consumerbr_resolution.hyperparameter_selection import (
    get_selected_sgd_alpha,
)


METRIC_FIELDS = [
    "fold",
    "split",
    "model",
    "threshold_source",
    "threshold",
    "epochs",
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


VARIANT_SPECS = {
    "char": {
        "model": "tfidf_char_sgd",
        "model_dir": (
            TFIDF_CHAR_SGD_MODELS_DIR
        ),
        "metrics_dir": (
            METRICS_DIR / "tfidf_char_sgd"
        ),
        "metrics_path": (
            METRICS_DIR
            / "tfidf_char_sgd_metrics.csv"
        ),
        "predictions_dir": (
            PREDICTIONS_DIR
            / "tfidf_char_sgd"
        ),
    },
    "word_char": {
        "model": "tfidf_word_char_sgd",
        "model_dir": (
            TFIDF_WORD_CHAR_SGD_MODELS_DIR
        ),
        "metrics_dir": (
            METRICS_DIR
            / "tfidf_word_char_sgd"
        ),
        "metrics_path": (
            METRICS_DIR
            / "tfidf_word_char_sgd_metrics.csv"
        ),
        "predictions_dir": (
            PREDICTIONS_DIR
            / "tfidf_word_char_sgd"
        ),
    },
}


def get_document_count(
    connection,
    source_path,
    start_date=None,
    end_date=None,
):
    conditions = []

    if start_date is not None:
        conditions.append(
            f"opening_date >= DATE '{start_date}'"
        )

    if end_date is not None:
        conditions.append(
            f"opening_date <= DATE '{end_date}'"
        )

    where_clause = " AND ".join(
        conditions
    )

    result = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{source_path}')
        WHERE {where_clause}
        """
    ).fetchone()

    return int(result[0])


def iter_batches(
    connection,
    source_path,
    start_date=None,
    end_date=None,
    include_metadata=False,
):
    conditions = []

    if start_date is not None:
        conditions.append(
            f"opening_date >= DATE '{start_date}'"
        )

    if end_date is not None:
        conditions.append(
            f"opening_date <= DATE '{end_date}'"
        )

    where_clause = " AND ".join(
        conditions
    )

    if include_metadata:
        columns = """
            record_id,
            complaint_id,
            opening_date,
            complaint_text,
            target_resolved
        """
    else:
        columns = """
            complaint_text,
            target_resolved
        """

    cursor = connection.execute(
        f"""
        SELECT
            {columns}
        FROM read_parquet('{source_path}')
        WHERE {where_clause}
        """
    )

    while True:
        rows = cursor.fetchmany(
            SGD_BATCH_SIZE
        )

        if not rows:
            break

        yield rows


def create_classifier(alpha):
    return SGDClassifier(
        loss=SGD_LOSS,
        penalty=SGD_PENALTY,
        alpha=alpha,
        random_state=RANDOM_SEED,
    )


def transform_texts(
    texts,
    word_vectorizer,
    char_vectorizer,
    variant,
):
    char_features = (
        char_vectorizer.transform(
            texts
        )
    )

    if variant == "char":
        return char_features

    word_features = (
        word_vectorizer.transform(
            texts
        )
    )

    return hstack(
        [
            word_features,
            char_features,
        ],
        format="csr",
    ).astype(
        np.float32,
        copy=False,
    )


def train_model(
    connection,
    source_path,
    train_end,
    word_vectorizer,
    char_vectorizer,
    model,
    variant,
    train_document_count,
):
    classes = np.array(
        [0, 1],
        dtype=np.int8,
    )

    first_batch = True
    start_time = time.perf_counter()

    for epoch in range(
        1,
        SGD_EPOCHS + 1,
    ):
        progress = tqdm(
            total=train_document_count,
            desc=(
                f"Epoch {epoch}/"
                f"{SGD_EPOCHS}"
            ),
            unit="docs",
            dynamic_ncols=True,
        )

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

                features = transform_texts(
                    texts=texts,
                    word_vectorizer=(
                        word_vectorizer
                    ),
                    char_vectorizer=(
                        char_vectorizer
                    ),
                    variant=variant,
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


def score_split(
    connection,
    source_path,
    start_date,
    end_date,
    word_vectorizer,
    char_vectorizer,
    model,
    variant,
    collect_metadata=False,
):
    targets = []
    scores = []

    record_ids = []
    complaint_ids = []
    opening_dates = []

    document_count = get_document_count(
        connection=connection,
        source_path=source_path,
        start_date=start_date,
        end_date=end_date,
    )

    progress = tqdm(
        total=document_count,
        desc="Scoring",
        unit="docs",
        dynamic_ncols=True,
    )

    start_time = time.perf_counter()

    try:
        for rows in iter_batches(
            connection=connection,
            source_path=source_path,
            start_date=start_date,
            end_date=end_date,
            include_metadata=(
                collect_metadata
            ),
        ):
            if collect_metadata:
                texts = [
                    row[3]
                    for row in rows
                ]

                batch_targets = np.asarray(
                    [
                        row[4]
                        for row in rows
                    ],
                    dtype=np.int8,
                )

                record_ids.extend(
                    row[0]
                    for row in rows
                )

                complaint_ids.extend(
                    row[1]
                    for row in rows
                )

                opening_dates.extend(
                    row[2]
                    for row in rows
                )
            else:
                texts = [
                    row[0]
                    for row in rows
                ]

                batch_targets = np.asarray(
                    [
                        row[1]
                        for row in rows
                    ],
                    dtype=np.int8,
                )

            features = transform_texts(
                texts=texts,
                word_vectorizer=(
                    word_vectorizer
                ),
                char_vectorizer=(
                    char_vectorizer
                ),
                variant=variant,
            )

            batch_scores = (
                model.predict_proba(
                    features
                )[:, 1]
            )

            targets.append(
                batch_targets
            )

            scores.append(
                batch_scores
            )

            progress.update(
                len(rows)
            )

    finally:
        progress.close()

    result = {
        "target": np.concatenate(
            targets
        ),
        "score": np.concatenate(
            scores
        ),
        "scoring_seconds": (
            time.perf_counter()
            - start_time
        ),
    }

    if collect_metadata:
        result["record_id"] = (
            record_ids
        )
        result["complaint_id"] = (
            complaint_ids
        )
        result["opening_date"] = (
            opening_dates
        )

    return result


def write_predictions(
    connection,
    prediction_path,
    result,
    threshold,
):
    temporary_csv_path = (
        prediction_path.with_suffix(
            ".csv.part"
        )
    )

    temporary_parquet_path = (
        prediction_path.with_suffix(
            ".parquet.part"
        )
    )

    if temporary_csv_path.exists():
        temporary_csv_path.unlink()

    if temporary_parquet_path.exists():
        temporary_parquet_path.unlink()

    predictions = (
        result["score"] >= threshold
    ).astype(np.int8)

    with temporary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "record_id",
                "complaint_id",
                "opening_date",
                "target_resolved",
                "score",
                "prediction",
            ]
        )

        for index in range(
            len(result["target"])
        ):
            writer.writerow(
                [
                    result[
                        "record_id"
                    ][index],
                    result[
                        "complaint_id"
                    ][index],
                    result[
                        "opening_date"
                    ][index],
                    int(
                        result[
                            "target"
                        ][index]
                    ),
                    float(
                        result[
                            "score"
                        ][index]
                    ),
                    int(
                        predictions[index]
                    ),
                ]
            )

    csv_path = str(
        temporary_csv_path
    ).replace("'", "''")

    parquet_path = str(
        temporary_parquet_path
    ).replace("'", "''")

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM read_csv_auto(
                '{csv_path}',
                header=true
            )
        )
        TO '{parquet_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    temporary_csv_path.unlink()

    temporary_parquet_path.replace(
        prediction_path
    )


def write_fold_metrics(
    fold_metrics_path,
    rows,
):
    with fold_metrics_path.open(
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


def rebuild_aggregate_metrics(
    metrics_dir,
    metrics_path,
):
    rows = []

    for fold in TEMPORAL_FOLDS:
        fold_number = fold["fold"]

        fold_metrics_path = (
            metrics_dir
            / f"fold_{fold_number:02d}.csv"
        )

        with fold_metrics_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)
            rows.extend(reader)

    with metrics_path.open(
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


def evaluate_variant(
    variant,
):
    create_project_directories()

    specification = VARIANT_SPECS[
        variant
    ]

    metrics_dir = specification[
        "metrics_dir"
    ]
    predictions_dir = specification[
        "predictions_dir"
    ]
    model_dir = specification[
        "model_dir"
    ]
    model_name = specification[
        "model"
    ]
    metrics_path = specification[
        "metrics_path"
    ]

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    alpha = get_selected_sgd_alpha()

    print(
        f"Evaluating {model_name}"
    )
    print(f"Source: {FEATURE_BASE_PATH}")
    print(f"Models: {model_dir}")
    print(
        f"Predictions: "
        f"{predictions_dir}"
    )

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            model_path = (
                model_dir
                / f"fold_{fold_number:02d}.joblib"
            )

            prediction_path = (
                predictions_dir
                / f"fold_{fold_number:02d}.parquet"
            )

            fold_metrics_path = (
                metrics_dir
                / f"fold_{fold_number:02d}.csv"
            )

            fold_outputs = [
                model_path,
                prediction_path,
                fold_metrics_path,
            ]

            if all(
                path.exists()
                for path in fold_outputs
            ):
                print()
                print(
                    f"Fold {fold_number} "
                    f"already exists."
                )
                continue

            for path in fold_outputs:
                if path.exists():
                    path.unlink()

            print()
            print(
                f"Evaluating fold "
                f"{fold_number}"
            )

            char_vectorizer = joblib.load(
                TFIDF_CHAR_MODELS_DIR
                / f"fold_{fold_number:02d}.joblib"
            )

            word_vectorizer = None

            if variant == "word_char":
                word_vectorizer = (
                    joblib.load(
                        TFIDF_MODELS_DIR
                        / f"fold_{fold_number:02d}.joblib"
                    )
                )

            model = create_classifier(alpha)

            train_document_count = (
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
                word_vectorizer=(
                    word_vectorizer
                ),
                char_vectorizer=(
                    char_vectorizer
                ),
                model=model,
                variant=variant,
                train_document_count=(
                    train_document_count
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
                word_vectorizer=(
                    word_vectorizer
                ),
                char_vectorizer=(
                    char_vectorizer
                ),
                model=model,
                variant=variant,
            )

            (
                threshold,
                validation_best_macro_f1,
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
                word_vectorizer=(
                    word_vectorizer
                ),
                char_vectorizer=(
                    char_vectorizer
                ),
                model=model,
                variant=variant,
                collect_metadata=True,
            )

            test_metrics = (
                calculate_binary_metrics(
                    test["target"],
                    test["score"],
                    threshold,
                )
            )

            metric_rows = [
                {
                    "fold": fold_number,
                    "split": "validation",
                    "model": model_name,
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "epochs": SGD_EPOCHS,
                    "alpha": alpha,
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
                    "model": model_name,
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "epochs": SGD_EPOCHS,
                    "alpha": SGD_ALPHA,
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
                fold_metrics_path=(
                    fold_metrics_path
                ),
                rows=metric_rows,
            )

            print(
                f"Threshold: "
                f"{threshold:.6f}"
            )

            print(
                "Validation Macro-F1: "
                f"{validation_best_macro_f1:.4f}"
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

    rebuild_aggregate_metrics(
        metrics_dir=metrics_dir,
        metrics_path=metrics_path,
    )

    print()
    print(
        f"{model_name} evaluation "
        f"completed."
    )
    print(
        f"Saved to: "
        f"{metrics_path}"
    )


def evaluate_tfidf_char_sgd():
    evaluate_variant(
        variant="char"
    )


def evaluate_tfidf_word_char_sgd():
    evaluate_variant(
        variant="word_char"
    )