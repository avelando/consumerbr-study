import csv
import gc
import json
import time
from itertools import product

import duckdb
import numpy as np

from consumerbr_resolution.company_history import (
    write_future_history,
    write_train_history,
)
from consumerbr_resolution.config import (
    CATBOOST_DEPTH_CANDIDATES,
    CATBOOST_EARLY_STOPPING_ROUNDS,
    CATBOOST_ITERATIONS,
    CATBOOST_L2_LEAF_REG_CANDIDATES,
    CATBOOST_LEARNING_RATE_CANDIDATES,
    CATBOOST_TUNING_RESULTS_PATH,
    CATBOOST_TUNING_TRAIN_HISTORY_PATH,
    CATBOOST_TUNING_VALIDATION_HISTORY_PATH,
    FEATURE_BASE_PATH,
    SELECTED_HYPERPARAMETERS_PATH,
    TUNING_TRAIN_END,
    TUNING_VALIDATION_END,
    TUNING_VALIDATION_START,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)
from consumerbr_resolution.tabular_catboost import (
    create_classifier,
    create_pool,
    load_split,
)


RESULT_FIELDS = [
    "depth",
    "learning_rate",
    "l2_leaf_reg",
    "iterations",
    "best_iteration",
    "validation_threshold",
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_f1",
    "validation_roc_auc",
    "validation_pr_auc",
    "validation_brier_score",
    "training_seconds",
    "scoring_seconds",
]


def build_tuning_history(
    connection,
    source_path,
):
    if not CATBOOST_TUNING_TRAIN_HISTORY_PATH.exists():
        write_train_history(
            connection=connection,
            source_path=source_path,
            output_path=(
                CATBOOST_TUNING_TRAIN_HISTORY_PATH
            ),
            train_end=TUNING_TRAIN_END,
        )

    if not CATBOOST_TUNING_VALIDATION_HISTORY_PATH.exists():
        write_future_history(
            connection=connection,
            source_path=source_path,
            output_path=(
                CATBOOST_TUNING_VALIDATION_HISTORY_PATH
            ),
            train_end=TUNING_TRAIN_END,
            split_start=(
                TUNING_VALIDATION_START
            ),
            split_end=(
                TUNING_VALIDATION_END
            ),
        )


def update_selected_hyperparameters(
    selected_row,
):
    if not SELECTED_HYPERPARAMETERS_PATH.exists():
        raise FileNotFoundError(
            "Selected classical hyperparameters were not found. "
            "Run the classical tuning stage first."
        )

    with SELECTED_HYPERPARAMETERS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        selected = json.load(file)

    selected["catboost"] = {
        "iterations": int(
            selected_row["iterations"]
        ),
        "learning_rate": float(
            selected_row["learning_rate"]
        ),
        "depth": int(
            selected_row["depth"]
        ),
        "l2_leaf_reg": float(
            selected_row["l2_leaf_reg"]
        ),
        "tuning_best_iteration": int(
            selected_row["best_iteration"]
        ),
    }

    temporary_path = (
        SELECTED_HYPERPARAMETERS_PATH
        .with_suffix(
            ".json.part"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected,
            file,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(
        SELECTED_HYPERPARAMETERS_PATH
    )


def selection_key(row):
    return (
        row["validation_macro_f1"],
        row["validation_roc_auc"],
        -row["validation_brier_score"],
        -row["depth"],
        -row["learning_rate"],
        row["l2_leaf_reg"],
    )


def tune_catboost_hyperparameters():
    create_project_directories()

    if (
        CATBOOST_TUNING_RESULTS_PATH.exists()
        and SELECTED_HYPERPARAMETERS_PATH.exists()
    ):
        with SELECTED_HYPERPARAMETERS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            selected = json.load(file)

        if "catboost" in selected:
            print(
                "CatBoost tuning already exists."
            )
            return

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    connection = duckdb.connect()

    try:
        build_tuning_history(
            connection=connection,
            source_path=source_path,
        )

        train_frame = load_split(
            connection=connection,
            source_path=source_path,
            history_path=(
                CATBOOST_TUNING_TRAIN_HISTORY_PATH
            ),
        )

        validation_frame = load_split(
            connection=connection,
            source_path=source_path,
            history_path=(
                CATBOOST_TUNING_VALIDATION_HISTORY_PATH
            ),
        )

    finally:
        connection.close()

    train_pool = create_pool(
        train_frame
    )

    validation_pool = create_pool(
        validation_frame
    )

    validation_targets = validation_frame[
        "target_resolved"
    ].to_numpy(
        dtype=np.int8
    )

    rows = []

    candidates = product(
        CATBOOST_DEPTH_CANDIDATES,
        CATBOOST_LEARNING_RATE_CANDIDATES,
        CATBOOST_L2_LEAF_REG_CANDIDATES,
    )

    try:
        for (
            depth,
            learning_rate,
            l2_leaf_reg,
        ) in candidates:
            parameters = {
                "iterations": (
                    CATBOOST_ITERATIONS
                ),
                "learning_rate": (
                    learning_rate
                ),
                "depth": depth,
                "l2_leaf_reg": (
                    l2_leaf_reg
                ),
            }

            print()
            print(
                "Tuning CatBoost: "
                f"depth={depth}, "
                f"learning_rate={learning_rate}, "
                f"l2_leaf_reg={l2_leaf_reg}"
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

            scoring_seconds = (
                time.perf_counter()
                - scoring_start
            )

            threshold, _ = (
                find_best_macro_f1_threshold(
                    validation_targets,
                    validation_scores,
                )
            )

            metrics = (
                calculate_binary_metrics(
                    validation_targets,
                    validation_scores,
                    threshold,
                )
            )

            rows.append(
                {
                    "depth": int(depth),
                    "learning_rate": float(
                        learning_rate
                    ),
                    "l2_leaf_reg": float(
                        l2_leaf_reg
                    ),
                    "iterations": int(
                        CATBOOST_ITERATIONS
                    ),
                    "best_iteration": int(
                        model.get_best_iteration()
                    ),
                    "validation_threshold": (
                        threshold
                    ),
                    "validation_accuracy": (
                        metrics["accuracy"]
                    ),
                    "validation_balanced_accuracy": (
                        metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "validation_macro_f1": (
                        metrics["macro_f1"]
                    ),
                    "validation_roc_auc": (
                        metrics["roc_auc"]
                    ),
                    "validation_pr_auc": (
                        metrics["pr_auc"]
                    ),
                    "validation_brier_score": (
                        metrics["brier_score"]
                    ),
                    "training_seconds": (
                        training_seconds
                    ),
                    "scoring_seconds": (
                        scoring_seconds
                    ),
                }
            )

            del model
            gc.collect()

    finally:
        del train_pool
        del validation_pool
        del train_frame
        del validation_frame
        gc.collect()

    selected_row = max(
        rows,
        key=selection_key,
    )

    with CATBOOST_TUNING_RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    update_selected_hyperparameters(
        selected_row=selected_row
    )

    print()
    print(
        "CatBoost tuning completed."
    )

    print(
        "Selected parameters: "
        f"depth={selected_row['depth']}, "
        f"learning_rate="
        f"{selected_row['learning_rate']}, "
        f"l2_leaf_reg="
        f"{selected_row['l2_leaf_reg']}"
    )

    print(
        f"Saved to: "
        f"{CATBOOST_TUNING_RESULTS_PATH}"
    )