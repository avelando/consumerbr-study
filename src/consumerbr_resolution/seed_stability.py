import csv
import gc
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from consumerbr_resolution import (
    albertina_finetuning,
    bertimbau_finetuning,
)
from consumerbr_resolution.company_history import (
    get_split_path,
)
from consumerbr_resolution.config import (
    ALBERTINA_GRADIENT_CHECKPOINTING,
    ALBERTINA_PRETRAINED_DIR,
    ALBERTINA_TOKEN_CACHE_PATH,
    BERTIMBAU_GRADIENT_CHECKPOINTING,
    BERTIMBAU_PRETRAINED_DIR,
    BERTIMBAU_TOKEN_CACHE_PATH,
    CATBOOST_EARLY_STOPPING_ROUNDS,
    EXPERIMENT_SEEDS,
    FEATURE_BASE_PATH,
    METRICS_DIR,
    PRIMARY_EXPERIMENT_SEED,
    SEED_STABILITY_BY_SEED_PATH,
    SEED_STABILITY_DIR,
    SEED_STABILITY_METRICS_PATH,
    SEED_STABILITY_SUMMARY_PATH,
    TEMPORAL_FOLDS,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)
from consumerbr_resolution.hyperparameter_selection import (
    get_selected_albertina_hyperparameters,
    get_selected_bertimbau_hyperparameters,
    get_selected_catboost_hyperparameters,
)
from consumerbr_resolution.tabular_catboost import (
    create_classifier as create_catboost_classifier,
    create_pool as create_catboost_pool,
    create_prediction_pool as create_catboost_prediction_pool,
    load_split as load_catboost_split,
)


CORE_METRIC_COLUMNS = [
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


METRIC_FIELDS = [
    "model",
    "seed",
    "fold",
    "split",
    "threshold_source",
    "threshold",
    "best_iteration",
    "iterations",
    "depth",
    "learning_rate",
    "l2_leaf_reg",
    "epochs",
    "weight_decay",
    "training_seconds",
    "scoring_seconds",
    *CORE_METRIC_COLUMNS,
]


PRIMARY_METRIC_PATHS = {
    "catboost": (
        METRICS_DIR / "catboost_metrics.csv"
    ),
    "bertimbau": (
        METRICS_DIR / "bertimbau_metrics.csv"
    ),
    "albertina": (
        METRICS_DIR / "albertina_metrics.csv"
    ),
}


def get_run_metrics_path(
    model_name,
    seed,
    fold_number,
):
    directory = (
        SEED_STABILITY_DIR
        / model_name
        / f"seed_{seed}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / f"fold_{fold_number:02d}.csv"
    )


def write_run_metrics(
    path,
    rows,
):
    temporary_path = (
        path.with_suffix(".csv.part")
    )

    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open(
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

    temporary_path.replace(path)


def make_metric_row(
    model_name,
    seed,
    fold_number,
    split,
    threshold,
    training_seconds,
    scoring_seconds,
    metrics,
    best_iteration="",
    iterations="",
    depth="",
    learning_rate="",
    l2_leaf_reg="",
    epochs="",
    weight_decay="",
):
    return {
        "model": model_name,
        "seed": seed,
        "fold": fold_number,
        "split": split,
        "threshold_source": (
            "validation_macro_f1"
        ),
        "threshold": threshold,
        "best_iteration": best_iteration,
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "training_seconds": training_seconds,
        "scoring_seconds": scoring_seconds,
        **metrics,
    }


def evaluate_catboost_seed_stability():
    parameters = (
        get_selected_catboost_hyperparameters()
    )

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    seeds = [
        seed
        for seed in EXPERIMENT_SEEDS
        if seed != PRIMARY_EXPERIMENT_SEED
    ]

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            pending_seeds = [
                seed
                for seed in seeds
                if not get_run_metrics_path(
                    "catboost",
                    seed,
                    fold_number,
                ).exists()
            ]

            if not pending_seeds:
                continue

            train_frame = load_catboost_split(
                connection=connection,
                source_path=source_path,
                history_path=get_split_path(
                    fold_number,
                    "train",
                ),
            )

            validation_frame = (
                load_catboost_split(
                    connection=connection,
                    source_path=source_path,
                    history_path=get_split_path(
                        fold_number,
                        "validation",
                    ),
                    include_identifiers=True,
                )
            )

            test_frame = load_catboost_split(
                connection=connection,
                source_path=source_path,
                history_path=get_split_path(
                    fold_number,
                    "test",
                ),
                include_identifiers=True,
            )

            train_pool = (
                create_catboost_pool(
                    train_frame
                )
            )

            validation_pool = (
                create_catboost_pool(
                    validation_frame
                )
            )

            test_pool = (
                create_catboost_prediction_pool(
                    test_frame
                )
            )

            validation_targets = (
                validation_frame[
                    "target_resolved"
                ].to_numpy(
                    dtype=np.int8
                )
            )

            test_targets = (
                test_frame[
                    "target_resolved"
                ].to_numpy(
                    dtype=np.int8
                )
            )

            for seed in pending_seeds:
                print(
                    f"CatBoost seed={seed}, "
                    f"fold={fold_number}"
                )

                model = (
                    create_catboost_classifier(
                        parameters=parameters,
                        random_seed=seed,
                    )
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

                start_time = (
                    time.perf_counter()
                )

                validation_scores = (
                    model.predict_proba(
                        validation_pool
                    )[:, 1]
                )

                validation_scoring_seconds = (
                    time.perf_counter()
                    - start_time
                )

                threshold, _ = (
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

                start_time = (
                    time.perf_counter()
                )

                test_scores = (
                    model.predict_proba(
                        test_pool
                    )[:, 1]
                )

                test_scoring_seconds = (
                    time.perf_counter()
                    - start_time
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
                    make_metric_row(
                        "catboost",
                        seed,
                        fold_number,
                        "validation",
                        threshold,
                        training_seconds,
                        validation_scoring_seconds,
                        validation_metrics,
                        best_iteration=best_iteration,
                        iterations=(
                            parameters["iterations"]
                        ),
                        depth=parameters["depth"],
                        learning_rate=(
                            parameters[
                                "learning_rate"
                            ]
                        ),
                        l2_leaf_reg=(
                            parameters[
                                "l2_leaf_reg"
                            ]
                        ),
                    ),
                    make_metric_row(
                        "catboost",
                        seed,
                        fold_number,
                        "test",
                        threshold,
                        training_seconds,
                        test_scoring_seconds,
                        test_metrics,
                        best_iteration=best_iteration,
                        iterations=(
                            parameters["iterations"]
                        ),
                        depth=parameters["depth"],
                        learning_rate=(
                            parameters[
                                "learning_rate"
                            ]
                        ),
                        l2_leaf_reg=(
                            parameters[
                                "l2_leaf_reg"
                            ]
                        ),
                    ),
                ]

                write_run_metrics(
                    get_run_metrics_path(
                        "catboost",
                        seed,
                        fold_number,
                    ),
                    rows,
                )

                del model
                gc.collect()

            del train_pool
            del validation_pool
            del test_pool
            del train_frame
            del validation_frame
            del test_frame

            gc.collect()

    finally:
        connection.close()


def evaluate_transformer_seed_stability(
    model_name,
    pretrained_dir,
    token_cache_path,
    gradient_checkpointing,
    finetuning_module,
    hyperparameters,
    tokenizer_kwargs=None,
    model_kwargs=None,
):
    seeds = [
        seed
        for seed in EXPERIMENT_SEEDS
        if seed != PRIMARY_EXPERIMENT_SEED
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            pretrained_dir,
            local_files_only=True,
            **(tokenizer_kwargs or {}),
        )
    )

    token_dataset = ds.dataset(
        token_cache_path,
        format="parquet",
    )

    source_path = str(
        token_cache_path
    ).replace("'", "''")

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            pending_seeds = [
                seed
                for seed in seeds
                if not get_run_metrics_path(
                    model_name,
                    seed,
                    fold_number,
                ).exists()
            ]

            if not pending_seeds:
                continue

            train_document_count = (
                finetuning_module.get_document_count(
                    connection=connection,
                    source_path=source_path,
                    end_date=fold[
                        "train_end"
                    ],
                )
            )

            for seed in pending_seeds:
                print(
                    f"{model_name} seed={seed}, "
                    f"fold={fold_number}"
                )

                finetuning_module.set_random_seed(
                    seed
                )

                model = (
                    AutoModelForSequenceClassification
                    .from_pretrained(
                        pretrained_dir,
                        num_labels=2,
                        local_files_only=True,
                        **(model_kwargs or {}),
                    )
                )

                if gradient_checkpointing:
                    model.gradient_checkpointing_enable()

                model.to(device)

                training_seconds = (
                    finetuning_module.train_model(
                        model=model,
                        dataset=token_dataset,
                        train_end=fold[
                            "train_end"
                        ],
                        tokenizer=tokenizer,
                        device=device,
                        train_document_count=(
                            train_document_count
                        ),
                        epochs=(
                            hyperparameters[
                                "epochs"
                            ]
                        ),
                        learning_rate=(
                            hyperparameters[
                                "learning_rate"
                            ]
                        ),
                        weight_decay=(
                            hyperparameters[
                                "weight_decay"
                            ]
                        ),
                    )
                )

                validation = (
                    finetuning_module.score_split(
                        model=model,
                        dataset=token_dataset,
                        start_date=fold[
                            "validation_start"
                        ],
                        end_date=fold[
                            "validation_end"
                        ],
                        tokenizer=tokenizer,
                        device=device,
                        include_identifiers=False,
                    )
                )

                threshold, _ = (
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

                test = (
                    finetuning_module.score_split(
                        model=model,
                        dataset=token_dataset,
                        start_date=fold[
                            "test_start"
                        ],
                        end_date=fold[
                            "test_end"
                        ],
                        tokenizer=tokenizer,
                        device=device,
                        include_identifiers=False,
                    )
                )

                test_metrics = (
                    calculate_binary_metrics(
                        test["target"],
                        test["score"],
                        threshold,
                    )
                )

                rows = [
                    make_metric_row(
                        model_name,
                        seed,
                        fold_number,
                        "validation",
                        threshold,
                        training_seconds,
                        validation[
                            "scoring_seconds"
                        ],
                        validation_metrics,
                        epochs=(
                            hyperparameters[
                                "epochs"
                            ]
                        ),
                        learning_rate=(
                            hyperparameters[
                                "learning_rate"
                            ]
                        ),
                        weight_decay=(
                            hyperparameters[
                                "weight_decay"
                            ]
                        ),
                    ),
                    make_metric_row(
                        model_name,
                        seed,
                        fold_number,
                        "test",
                        threshold,
                        training_seconds,
                        test[
                            "scoring_seconds"
                        ],
                        test_metrics,
                        epochs=(
                            hyperparameters[
                                "epochs"
                            ]
                        ),
                        learning_rate=(
                            hyperparameters[
                                "learning_rate"
                            ]
                        ),
                        weight_decay=(
                            hyperparameters[
                                "weight_decay"
                            ]
                        ),
                    ),
                ]

                write_run_metrics(
                    get_run_metrics_path(
                        model_name,
                        seed,
                        fold_number,
                    ),
                    rows,
                )

                del model
                del validation
                del test

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        connection.close()


def read_primary_seed_metrics():
    rows = []

    for model_name, path in (
        PRIMARY_METRIC_PATHS.items()
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Primary metrics not found: {path}"
            )

        frame = pd.read_csv(path)

        for _, source_row in (
            frame.iterrows()
        ):
            row = {
                field: ""
                for field in METRIC_FIELDS
            }

            row["model"] = model_name
            row["seed"] = (
                PRIMARY_EXPERIMENT_SEED
            )

            for field in METRIC_FIELDS:
                if field in source_row.index:
                    row[field] = (
                        source_row[field]
                    )

            rows.append(row)

    return rows


def read_secondary_seed_metrics():
    rows = []

    for model_name in (
        "catboost",
        "bertimbau",
        "albertina",
    ):
        for seed in EXPERIMENT_SEEDS:
            if (
                seed
                == PRIMARY_EXPERIMENT_SEED
            ):
                continue

            for fold in TEMPORAL_FOLDS:
                path = get_run_metrics_path(
                    model_name,
                    seed,
                    fold["fold"],
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Seed metrics not found: "
                        f"{path}"
                    )

                with path.open(
                    "r",
                    newline="",
                    encoding="utf-8",
                ) as file:
                    rows.extend(
                        csv.DictReader(file)
                    )

    return rows


def summarize_seed_stability():
    frame = pd.DataFrame(
        [
            *read_primary_seed_metrics(),
            *read_secondary_seed_metrics(),
        ]
    )

    numeric_columns = [
        "seed",
        "fold",
        "threshold",
        *CORE_METRIC_COLUMNS,
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame[
        METRIC_FIELDS
    ].sort_values(
        [
            "model",
            "seed",
            "fold",
            "split",
        ]
    )

    frame.to_csv(
        SEED_STABILITY_METRICS_PATH,
        index=False,
    )

    test_frame = frame[
        frame["split"] == "test"
    ].copy()

    summary_metrics = [
        "accuracy",
        "balanced_accuracy",
        "f1_resolved",
        "f1_unresolved",
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]

    by_seed = (
        test_frame
        .groupby(
            [
                "model",
                "seed",
            ],
            as_index=False,
        )[summary_metrics]
        .mean()
    )

    by_seed.to_csv(
        SEED_STABILITY_BY_SEED_PATH,
        index=False,
    )

    rows = []

    for model_name, group in (
        by_seed.groupby("model")
    ):
        row = {
            "model": model_name,
            "seed_count": len(group),
        }

        for metric in summary_metrics:
            values = (
                group[metric]
                .to_numpy(
                    dtype=np.float64
                )
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            row[
                f"{metric}_std"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            row[
                f"{metric}_min"
            ] = float(
                np.min(values)
            )

            row[
                f"{metric}_max"
            ] = float(
                np.max(values)
            )

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        SEED_STABILITY_SUMMARY_PATH,
        index=False,
    )


def evaluate_seed_stability():
    create_project_directories()

    print(
        "Evaluating seed stability"
    )

    evaluate_catboost_seed_stability()

    evaluate_transformer_seed_stability(
        model_name="bertimbau",
        pretrained_dir=(
            BERTIMBAU_PRETRAINED_DIR
        ),
        token_cache_path=(
            BERTIMBAU_TOKEN_CACHE_PATH
        ),
        gradient_checkpointing=(
            BERTIMBAU_GRADIENT_CHECKPOINTING
        ),
        finetuning_module=(
            bertimbau_finetuning
        ),
        hyperparameters=(
            get_selected_bertimbau_hyperparameters()
        ),
        tokenizer_kwargs={
            "do_lower_case": False,
        },
    )

    evaluate_transformer_seed_stability(
        model_name="albertina",
        pretrained_dir=(
            ALBERTINA_PRETRAINED_DIR
        ),
        token_cache_path=(
            ALBERTINA_TOKEN_CACHE_PATH
        ),
        gradient_checkpointing=(
            ALBERTINA_GRADIENT_CHECKPOINTING
        ),
        finetuning_module=(
            albertina_finetuning
        ),
        hyperparameters=(
            get_selected_albertina_hyperparameters()
        ),
        model_kwargs={
            "dtype": torch.float32,
        },
    )

    summarize_seed_stability()

    print(
        "Seed stability evaluation completed."
    )