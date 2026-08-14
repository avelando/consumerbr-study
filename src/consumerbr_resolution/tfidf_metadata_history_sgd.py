import csv
import time

import duckdb
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from consumerbr_resolution.company_history import (
    get_split_path,
)
from consumerbr_resolution.config import (
    FEATURE_BASE_PATH,
    METADATA_MODELS_DIR,
    METADATA_NUMERIC_FEATURES,
    METRICS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    SGD_BATCH_SIZE,
    SGD_EPOCHS,
    SGD_LOSS,
    SGD_PENALTY,
    TEMPORAL_FOLDS,
    TFIDF_METADATA_HISTORY_SGD_MODELS_DIR,
    TFIDF_MODELS_DIR,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)
from consumerbr_resolution.metadata import (
    transform_metadata_rows,
)
from consumerbr_resolution.hyperparameter_selection import (
    get_selected_sgd_alpha,
)


METRICS_DIRECTORY = (
    METRICS_DIR
    / "tfidf_metadata_history_sgd"
)

AGGREGATE_METRICS_PATH = (
    METRICS_DIR
    / "tfidf_metadata_history_sgd_metrics.csv"
)

PREDICTION_DIRECTORY = (
    PREDICTIONS_DIR
    / "tfidf_metadata_history_sgd"
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


def get_document_count(
    connection,
    history_path,
):
    path = str(
        history_path
    ).replace("'", "''")

    result = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{path}')
        """
    ).fetchone()

    return int(result[0])


def iter_batches(
    connection,
    source_path,
    history_path,
    include_identifiers=False,
):
    numeric_columns = ", ".join(
        f"data.{feature}"
        for feature in METADATA_NUMERIC_FEATURES
    )

    history_source = str(
        history_path
    ).replace("'", "''")

    if include_identifiers:
        columns = f"""
            data.complaint_text,
            data.company,
            data.uf,
            {numeric_columns},
            history.company_history_rate,
            history.log_company_history_count,
            history.company_seen_before,
            history.global_history_rate,
            data.record_id,
            data.complaint_id,
            data.opening_date,
            data.target_resolved
        """
    else:
        columns = f"""
            data.complaint_text,
            data.company,
            data.uf,
            {numeric_columns},
            history.company_history_rate,
            history.log_company_history_count,
            history.company_seen_before,
            history.global_history_rate,
            data.target_resolved
        """

    cursor = connection.execute(
        f"""
        SELECT
            {columns}
        FROM read_parquet('{source_path}')
            AS data
        JOIN read_parquet('{history_source}')
            AS history
            ON data.record_id = history.record_id
        ORDER BY
            data.opening_date,
            data.complaint_id,
            data.record_id
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


def transform_rows(
    rows,
    vectorizer,
    preprocessor,
):
    texts = [
        row[0]
        for row in rows
    ]

    text_features = vectorizer.transform(
        texts
    )

    metadata_end = (
        1
        + 2
        + len(
            METADATA_NUMERIC_FEATURES
        )
    )

    metadata_rows = [
        row[1:metadata_end]
        for row in rows
    ]

    metadata_features = (
        transform_metadata_rows(
            rows=metadata_rows,
            preprocessor=preprocessor,
            include_company=True,
        )
    )

    history_start = metadata_end

    history_features = np.asarray(
        [
            row[
                history_start:
                history_start + 4
            ]
            for row in rows
        ],
        dtype=np.float32,
    )

    history_sparse = csr_matrix(
        history_features,
        dtype=np.float32,
    )

    return hstack(
        [
            text_features,
            metadata_features,
            history_sparse,
        ],
        format="csr",
        dtype=np.float32,
    )


def get_target_index(
    include_identifiers,
):
    index = (
        1
        + 2
        + len(
            METADATA_NUMERIC_FEATURES
        )
        + 4
    )

    if include_identifiers:
        return index + 3

    return index


def train_model(
    connection,
    source_path,
    history_path,
    vectorizer,
    preprocessor,
    model,
    document_count,
):
    classes = np.asarray(
        [0, 1],
        dtype=np.int8,
    )

    first_batch = True

    target_index = get_target_index(
        include_identifiers=False
    )

    start_time = time.perf_counter()

    for epoch in range(
        1,
        SGD_EPOCHS + 1,
    ):
        progress = tqdm(
            total=document_count,
            desc=f"Epoch {epoch}/{SGD_EPOCHS}",
            unit="docs",
            dynamic_ncols=True,
        )

        try:
            for rows in iter_batches(
                connection=connection,
                source_path=source_path,
                history_path=history_path,
            ):
                features = transform_rows(
                    rows=rows,
                    vectorizer=vectorizer,
                    preprocessor=preprocessor,
                )

                targets = np.asarray(
                    [
                        row[target_index]
                        for row in rows
                    ],
                    dtype=np.int8,
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
    history_path,
    vectorizer,
    preprocessor,
    model,
    collect_identifiers=False,
):
    targets = []
    scores = []

    record_ids = []
    complaint_ids = []
    opening_dates = []

    document_count = get_document_count(
        connection=connection,
        history_path=history_path,
    )

    progress = tqdm(
        total=document_count,
        desc="Scoring",
        unit="docs",
        dynamic_ncols=True,
    )

    target_index = get_target_index(
        include_identifiers=collect_identifiers
    )

    identifier_start = (
        1
        + 2
        + len(
            METADATA_NUMERIC_FEATURES
        )
        + 4
    )

    start_time = time.perf_counter()

    try:
        for rows in iter_batches(
            connection=connection,
            source_path=source_path,
            history_path=history_path,
            include_identifiers=(
                collect_identifiers
            ),
        ):
            features = transform_rows(
                rows=rows,
                vectorizer=vectorizer,
                preprocessor=preprocessor,
            )

            batch_targets = np.asarray(
                [
                    row[target_index]
                    for row in rows
                ],
                dtype=np.int8,
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

            if collect_identifiers:
                record_ids.extend(
                    row[identifier_start]
                    for row in rows
                )

                complaint_ids.extend(
                    row[
                        identifier_start + 1
                    ]
                    for row in rows
                )

                opening_dates.extend(
                    row[
                        identifier_start + 2
                    ]
                    for row in rows
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

    if collect_identifiers:
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
    csv_path = prediction_path.with_suffix(
        ".csv.part"
    )

    parquet_path = prediction_path.with_suffix(
        ".parquet.part"
    )

    if csv_path.exists():
        csv_path.unlink()

    if parquet_path.exists():
        parquet_path.unlink()

    predictions = (
        result["score"] >= threshold
    ).astype(np.int8)

    with csv_path.open(
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
                    result["record_id"][index],
                    result["complaint_id"][index],
                    result["opening_date"][index],
                    int(
                        result["target"][index]
                    ),
                    float(
                        result["score"][index]
                    ),
                    int(
                        predictions[index]
                    ),
                ]
            )

    source_csv = str(
        csv_path
    ).replace("'", "''")

    target_parquet = str(
        parquet_path
    ).replace("'", "''")

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM read_csv_auto(
                '{source_csv}',
                header=true
            )
        )
        TO '{target_parquet}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    csv_path.unlink()

    parquet_path.replace(
        prediction_path
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


def evaluate_tfidf_metadata_history_sgd():
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

    alpha = get_selected_sgd_alpha()

    print(
        "Evaluating TF-IDF + metadata + "
        "company history with SGD"
    )

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            model_path = (
                TFIDF_METADATA_HISTORY_SGD_MODELS_DIR
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

            preprocessor = joblib.load(
                METADATA_MODELS_DIR
                / f"fold_{fold_number:02d}.joblib"
            )

            train_history_path = (
                get_split_path(
                    fold_number,
                    "train",
                )
            )

            validation_history_path = (
                get_split_path(
                    fold_number,
                    "validation",
                )
            )

            test_history_path = (
                get_split_path(
                    fold_number,
                    "test",
                )
            )

            train_document_count = (
                get_document_count(
                    connection=connection,
                    history_path=(
                        train_history_path
                    ),
                )
            )

            model = create_classifier(alpha)

            training_seconds = train_model(
                connection=connection,
                source_path=source_path,
                history_path=(
                    train_history_path
                ),
                vectorizer=vectorizer,
                preprocessor=preprocessor,
                model=model,
                document_count=(
                    train_document_count
                ),
            )

            validation = score_split(
                connection=connection,
                source_path=source_path,
                history_path=(
                    validation_history_path
                ),
                vectorizer=vectorizer,
                preprocessor=preprocessor,
                model=model,
            )

            threshold, validation_macro_f1 = (
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
                history_path=(
                    test_history_path
                ),
                vectorizer=vectorizer,
                preprocessor=preprocessor,
                model=model,
                collect_identifiers=True,
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
                        "tfidf_metadata_history_sgd"
                    ),
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
                    "model": (
                        "tfidf_metadata_history_sgd"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "epochs": SGD_EPOCHS,
                    "alpha": alpha,
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
                f"Validation Macro-F1: "
                f"{validation_macro_f1:.4f}"
            )

            print(
                f"Test Macro-F1: "
                f"{test_metrics['macro_f1']:.4f}"
            )

    finally:
        connection.close()

    rebuild_aggregate_metrics()

    print()
    print(
        "TF-IDF + metadata + company history "
        "evaluation completed."
    )

    print(
        f"Saved to: "
        f"{AGGREGATE_METRICS_PATH}"
    )