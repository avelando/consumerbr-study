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
    METADATA_MODELS_DIR,
    METADATA_NUMERIC_FEATURES,
    METRICS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    SGD_ALPHA,
    SGD_BATCH_SIZE,
    SGD_EPOCHS,
    SGD_LOSS,
    SGD_PENALTY,
    TEMPORAL_FOLDS,
    TFIDF_METADATA_SGD_MODELS_DIR,
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


TFIDF_METADATA_SGD_METRICS_DIR = (
    METRICS_DIR / "tfidf_metadata_sgd"
)

TFIDF_METADATA_SGD_METRICS_PATH = (
    METRICS_DIR / "tfidf_metadata_sgd_metrics.csv"
)

TFIDF_METADATA_SGD_PREDICTIONS_DIR = (
    PREDICTIONS_DIR / "tfidf_metadata_sgd"
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

    where_clause = " AND ".join(conditions)

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
    include_identifiers=False,
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

    where_clause = " AND ".join(conditions)

    numeric_columns = ", ".join(
        METADATA_NUMERIC_FEATURES
    )

    if include_identifiers:
        select_columns = f"""
            complaint_text,
            company,
            uf,
            {numeric_columns},
            record_id,
            complaint_id,
            opening_date,
            target_resolved
        """
    else:
        select_columns = f"""
            complaint_text,
            company,
            uf,
            {numeric_columns},
            target_resolved
        """

    cursor = connection.execute(
        f"""
        SELECT
            {select_columns}
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


def create_classifier():
    return SGDClassifier(
        loss=SGD_LOSS,
        penalty=SGD_PENALTY,
        alpha=SGD_ALPHA,
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

    metadata_rows = [
        row[1:]
        for row in rows
    ]

    metadata_features = (
        transform_metadata_rows(
            rows=metadata_rows,
            preprocessor=preprocessor,
            include_company=True,
        )
    )

    return hstack(
        [
            text_features,
            metadata_features,
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
    )

    if include_identifiers:
        return index + 3

    return index


def train_model(
    connection,
    source_path,
    train_end,
    vectorizer,
    preprocessor,
    model,
    train_document_count,
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
            total=train_document_count,
            desc=f"Epoch {epoch}/{SGD_EPOCHS}",
            unit="docs",
            dynamic_ncols=True,
        )

        try:
            for rows in iter_batches(
                connection=connection,
                source_path=source_path,
                end_date=train_end,
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
    start_date,
    end_date,
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

    target_index = get_target_index(
        include_identifiers=collect_identifiers
    )

    identifier_start = (
        1
        + 2
        + len(
            METADATA_NUMERIC_FEATURES
        )
    )

    start_time = time.perf_counter()

    try:
        for rows in iter_batches(
            connection=connection,
            source_path=source_path,
            start_date=start_date,
            end_date=end_date,
            include_identifiers=collect_identifiers,
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

    source_path = str(
        csv_path
    ).replace("'", "''")

    target_path = str(
        parquet_path
    ).replace("'", "''")

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM read_csv_auto(
                '{source_path}',
                header=true
            )
        )
        TO '{target_path}'
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
        fold_number = fold["fold"]

        path = (
            TFIDF_METADATA_SGD_METRICS_DIR
            / f"fold_{fold_number:02d}.csv"
        )

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            rows.extend(reader)

    with TFIDF_METADATA_SGD_METRICS_PATH.open(
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


def evaluate_tfidf_metadata_sgd():
    create_project_directories()

    TFIDF_METADATA_SGD_METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TFIDF_METADATA_SGD_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    print("Evaluating TF-IDF + metadata with SGD")
    print(f"Source: {FEATURE_BASE_PATH}")
    print(
        f"Models: "
        f"{TFIDF_METADATA_SGD_MODELS_DIR}"
    )

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            model_path = (
                TFIDF_METADATA_SGD_MODELS_DIR
                / f"fold_{fold_number:02d}.joblib"
            )

            prediction_path = (
                TFIDF_METADATA_SGD_PREDICTIONS_DIR
                / f"fold_{fold_number:02d}.parquet"
            )

            metrics_path = (
                TFIDF_METADATA_SGD_METRICS_DIR
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

            model = create_classifier()

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
                vectorizer=vectorizer,
                preprocessor=preprocessor,
                model=model,
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
                vectorizer=vectorizer,
                preprocessor=preprocessor,
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
                        "tfidf_metadata_sgd"
                    ),
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "epochs": SGD_EPOCHS,
                    "alpha": SGD_ALPHA,
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
                        "tfidf_metadata_sgd"
                    ),
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
                prediction_path=prediction_path,
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
        "TF-IDF + metadata SGD "
        "evaluation completed."
    )

    print(
        f"Saved to: "
        f"{TFIDF_METADATA_SGD_METRICS_PATH}"
    )