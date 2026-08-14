import csv
import gc
import time

import duckdb
import numpy as np
import pandas as pd
from catboost import (
    CatBoostClassifier,
    Pool,
)

from consumerbr_resolution.company_history import (
    get_split_path,
)
from consumerbr_resolution.config import (
    CATBOOST_DEPTH,
    CATBOOST_DEVICES,
    CATBOOST_EARLY_STOPPING_ROUNDS,
    CATBOOST_GPU_RAM_PART,
    CATBOOST_ITERATIONS,
    CATBOOST_L2_LEAF_REG,
    CATBOOST_LEARNING_RATE,
    CATBOOST_MODELS_DIR,
    CATBOOST_TASK_TYPE,
    FEATURE_BASE_PATH,
    METADATA_NUMERIC_FEATURES,
    METRICS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    TEMPORAL_FOLDS,
    create_project_directories,
)
from consumerbr_resolution.hyperparameter_selection import (
    get_selected_catboost_hyperparameters,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)


METRICS_DIRECTORY = (
    METRICS_DIR / "catboost"
)

AGGREGATE_METRICS_PATH = (
    METRICS_DIR / "catboost_metrics.csv"
)

PREDICTION_DIRECTORY = (
    PREDICTIONS_DIR / "catboost"
)


HISTORY_FEATURES = (
    "company_history_rate",
    "log_company_history_count",
    "company_seen_before",
    "global_history_rate",
)


CATEGORICAL_FEATURES = [
    "company",
    "uf",
]


FEATURE_COLUMNS = [
    "company",
    "uf",
    *METADATA_NUMERIC_FEATURES,
    *HISTORY_FEATURES,
]


METRIC_FIELDS = [
    "fold",
    "split",
    "model",
    "threshold_source",
    "threshold",
    "best_iteration",
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
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


def load_split(
    connection,
    source_path,
    history_path,
    include_identifiers=False,
):
    history_source = str(
        history_path
    ).replace("'", "''")

    numeric_columns = ", ".join(
        (
            f"CAST(data.{feature} AS FLOAT) "
            f"AS {feature}"
        )
        for feature
        in METADATA_NUMERIC_FEATURES
    )

    history_columns = ", ".join(
        (
            f"CAST(history.{feature} AS FLOAT) "
            f"AS {feature}"
        )
        for feature
        in HISTORY_FEATURES
    )

    if include_identifiers:
        identifier_columns = """
            data.record_id,
            data.complaint_id,
            data.opening_date,
        """
    else:
        identifier_columns = ""

    frame = connection.execute(
        f"""
        SELECT
            COALESCE(
                NULLIF(data.company, ''),
                'UNKNOWN'
            ) AS company,
            COALESCE(
                NULLIF(data.uf, ''),
                'UNKNOWN'
            ) AS uf,
            {numeric_columns},
            {history_columns},
            {identifier_columns}
            CAST(
                data.target_resolved
                AS TINYINT
            ) AS target_resolved
        FROM read_parquet('{source_path}')
            AS data
        JOIN read_parquet('{history_source}')
            AS history
            ON data.record_id
                = history.record_id
        ORDER BY
            data.opening_date,
            data.complaint_id,
            data.record_id
        """
    ).fetchdf()

    frame["company"] = (
        frame["company"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    frame["uf"] = (
        frame["uf"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    return frame


def create_pool(
    frame,
):
    return Pool(
        data=frame[
            FEATURE_COLUMNS
        ],
        label=frame[
            "target_resolved"
        ].to_numpy(
            dtype=np.int8
        ),
        cat_features=(
            CATEGORICAL_FEATURES
        ),
    )


def create_prediction_pool(
    frame,
):
    return Pool(
        data=frame[
            FEATURE_COLUMNS
        ],
        cat_features=(
            CATEGORICAL_FEATURES
        ),
    )


def create_classifier(
    parameters=None,
    random_seed=RANDOM_SEED,
):
    if parameters is None:
        parameters = (
            get_selected_catboost_hyperparameters()
        )

    return CatBoostClassifier(
        iterations=int(
            parameters.get(
                "iterations",
                CATBOOST_ITERATIONS,
            )
        ),
        learning_rate=float(
            parameters.get(
                "learning_rate",
                CATBOOST_LEARNING_RATE,
            )
        ),
        depth=int(
            parameters.get(
                "depth",
                CATBOOST_DEPTH,
            )
        ),
        l2_leaf_reg=float(
            parameters.get(
                "l2_leaf_reg",
                CATBOOST_L2_LEAF_REG,
            )
        ),
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=random_seed,
        task_type=CATBOOST_TASK_TYPE,
        devices=CATBOOST_DEVICES,
        gpu_ram_part=(
            CATBOOST_GPU_RAM_PART
        ),
        allow_writing_files=False,
        verbose=100,
    )


def write_predictions(
    connection,
    prediction_path,
    frame,
    scores,
    threshold,
):
    temporary_path = (
        prediction_path.with_suffix(
            ".parquet.part"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    predictions = (
        scores >= threshold
    ).astype(np.int8)

    prediction_frame = pd.DataFrame(
        {
            "record_id": frame[
                "record_id"
            ],
            "complaint_id": frame[
                "complaint_id"
            ],
            "opening_date": frame[
                "opening_date"
            ],
            "target_resolved": frame[
                "target_resolved"
            ].astype(np.int8),
            "score": scores,
            "prediction": predictions,
        }
    )

    target_path = str(
        temporary_path
    ).replace("'", "''")

    connection.register(
        "catboost_prediction_frame",
        prediction_frame,
    )

    try:
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM catboost_prediction_frame
            )
            TO '{target_path}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )
    finally:
        connection.unregister(
            "catboost_prediction_frame"
        )

    temporary_path.replace(
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


def evaluate_catboost():
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

    parameters = (
        get_selected_catboost_hyperparameters()
    )

    print(
        "Evaluating CatBoost tabular model"
    )

    print(
        f"Source: {FEATURE_BASE_PATH}"
    )

    print(
        f"Models: {CATBOOST_MODELS_DIR}"
    )

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            model_path = (
                CATBOOST_MODELS_DIR
                / f"fold_{fold_number:02d}.cbm"
            )

            validation_prediction_path = (
                PREDICTION_DIRECTORY
                / (
                    f"fold_{fold_number:02d}"
                    "_validation.parquet"
                )
            )

            test_prediction_path = (
                PREDICTION_DIRECTORY
                / (
                    f"fold_{fold_number:02d}"
                    "_test.parquet"
                )
            )

            metrics_path = (
                METRICS_DIRECTORY
                / f"fold_{fold_number:02d}.csv"
            )

            outputs = [
                model_path,
                validation_prediction_path,
                test_prediction_path,
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

            train_frame = load_split(
                connection=connection,
                source_path=source_path,
                history_path=get_split_path(
                    fold_number,
                    "train",
                ),
            )

            validation_frame = (
                load_split(
                    connection=connection,
                    source_path=source_path,
                    history_path=get_split_path(
                        fold_number,
                        "validation",
                    ),
                    include_identifiers=True,
                )
            )

            train_pool = create_pool(
                train_frame
            )

            validation_pool = (
                create_pool(
                    validation_frame
                )
            )

            model = create_classifier(
                parameters=parameters
            )

            start_time = (
                time.perf_counter()
            )

            model.fit(
                train_pool,
                eval_set=validation_pool,
                use_best_model=True,
                early_stopping_rounds=(
                    CATBOOST_EARLY_STOPPING_ROUNDS
                ),
            )

            training_seconds = (
                time.perf_counter()
                - start_time
            )

            scoring_start = (
                time.perf_counter()
            )

            validation_scores = (
                model.predict_proba(
                    validation_pool
                )[:, 1]
            )

            validation_scoring_seconds = (
                time.perf_counter()
                - scoring_start
            )

            validation_targets = (
                validation_frame[
                    "target_resolved"
                ].to_numpy(
                    dtype=np.int8
                )
            )

            (
                threshold,
                validation_macro_f1,
            ) = (
                find_best_macro_f1_threshold(
                    validation_targets,
                    validation_scores,
                )
            )

            validation_metrics = (
                calculate_binary_metrics(
                    validation_targets,
                    validation_scores,
                    threshold,
                )
            )

            del train_pool
            del train_frame
            del validation_pool

            gc.collect()

            test_frame = load_split(
                connection=connection,
                source_path=source_path,
                history_path=get_split_path(
                    fold_number,
                    "test",
                ),
                include_identifiers=True,
            )

            test_pool = (
                create_prediction_pool(
                    test_frame
                )
            )

            scoring_start = (
                time.perf_counter()
            )

            test_scores = (
                model.predict_proba(
                    test_pool
                )[:, 1]
            )

            test_scoring_seconds = (
                time.perf_counter()
                - scoring_start
            )

            test_targets = (
                test_frame[
                    "target_resolved"
                ].to_numpy(
                    dtype=np.int8
                )
            )

            test_metrics = (
                calculate_binary_metrics(
                    test_targets,
                    test_scores,
                    threshold,
                )
            )

            best_iteration = int(
                model.get_best_iteration()
            )

            rows = [
                {
                    "fold": fold_number,
                    "split": "validation",
                    "model": "catboost",
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "best_iteration": (
                        best_iteration
                    ),
                    "iterations": (
                        parameters["iterations"]
                    ),
                    "learning_rate": (
                        parameters["learning_rate"]
                    ),
                    "depth": parameters["depth"],
                    "l2_leaf_reg": (
                        parameters["l2_leaf_reg"]
                    ),
                    "training_seconds": (
                        training_seconds
                    ),
                    "scoring_seconds": (
                        validation_scoring_seconds
                    ),
                    **validation_metrics,
                },
                {
                    "fold": fold_number,
                    "split": "test",
                    "model": "catboost",
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    "best_iteration": (
                        best_iteration
                    ),
                    "iterations": (
                        parameters["iterations"]
                    ),
                    "learning_rate": (
                        parameters["learning_rate"]
                    ),
                    "depth": parameters["depth"],
                    "l2_leaf_reg": (
                        parameters["l2_leaf_reg"]
                    ),
                    "training_seconds": (
                        training_seconds
                    ),
                    "scoring_seconds": (
                        test_scoring_seconds
                    ),
                    **test_metrics,
                },
            ]

            temporary_model_path = (
                model_path.with_suffix(
                    ".cbm.part"
                )
            )

            if temporary_model_path.exists():
                temporary_model_path.unlink()

            model.save_model(
                str(
                    temporary_model_path
                ),
                format="cbm",
            )

            temporary_model_path.replace(
                model_path
            )

            write_predictions(
                connection=connection,
                prediction_path=(
                    validation_prediction_path
                ),
                frame=validation_frame,
                scores=validation_scores,
                threshold=threshold,
            )

            write_predictions(
                connection=connection,
                prediction_path=(
                    test_prediction_path
                ),
                frame=test_frame,
                scores=test_scores,
                threshold=threshold,
            )

            write_fold_metrics(
                path=metrics_path,
                rows=rows,
            )

            print(
                f"Best iteration: "
                f"{best_iteration}"
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

            print(
                f"Training time: "
                f"{training_seconds:.2f} seconds"
            )

            del validation_frame
            del test_frame
            del test_pool
            del model

            gc.collect()

    finally:
        connection.close()

    rebuild_aggregate_metrics()

    print()
    print(
        "CatBoost evaluation completed."
    )

    print(
        f"Saved to: "
        f"{AGGREGATE_METRICS_PATH}"
    )