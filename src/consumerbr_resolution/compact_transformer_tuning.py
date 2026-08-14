import csv
import gc
import json
from itertools import product

import duckdb
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
from consumerbr_resolution.config import (
    ALBERTINA_GRADIENT_CHECKPOINTING,
    ALBERTINA_LEARNING_RATE_CANDIDATES,
    ALBERTINA_PRETRAINED_DIR,
    ALBERTINA_TOKEN_CACHE_PATH,
    ALBERTINA_TRAIN_BATCH_SIZE,
    ALBERTINA_GRADIENT_ACCUMULATION_STEPS,
    ALBERTINA_TUNING_RESULTS_PATH,
    ALBERTINA_WEIGHT_DECAY,
    BERTIMBAU_GRADIENT_CHECKPOINTING,
    BERTIMBAU_LEARNING_RATE_CANDIDATES,
    BERTIMBAU_PRETRAINED_DIR,
    BERTIMBAU_TOKEN_CACHE_PATH,
    BERTIMBAU_TRAIN_BATCH_SIZE,
    BERTIMBAU_GRADIENT_ACCUMULATION_STEPS,
    BERTIMBAU_TUNING_RESULTS_PATH,
    BERTIMBAU_WEIGHT_DECAY,
    RANDOM_SEED,
    SELECTED_HYPERPARAMETERS_PATH,
    TRANSFORMER_EPOCH_CANDIDATES,
    TUNING_TRAIN_END,
    TUNING_VALIDATION_END,
    TUNING_VALIDATION_START,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)


RESULT_FIELDS = [
    "model",
    "epochs",
    "learning_rate",
    "weight_decay",
    "validation_threshold",
    "validation_macro_f1",
    "validation_roc_auc",
    "validation_pr_auc",
    "validation_brier_score",
    "training_seconds",
    "scoring_seconds",
]


def update_selected_hyperparameters(
    model_name,
    selected_row,
):
    if not SELECTED_HYPERPARAMETERS_PATH.exists():
        raise FileNotFoundError(
            "Selected hyperparameters were not found."
        )

    with SELECTED_HYPERPARAMETERS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        selected = json.load(file)

    selected[model_name] = {
        "epochs": int(
            selected_row["epochs"]
        ),
        "learning_rate": float(
            selected_row["learning_rate"]
        ),
        "weight_decay": float(
            selected_row["weight_decay"]
        ),
    }

    temporary_path = (
        SELECTED_HYPERPARAMETERS_PATH
        .with_suffix(".json.part")
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
        -row["epochs"],
        -row["learning_rate"],
    )


def tune_transformer(
    model_name,
    pretrained_dir,
    token_cache_path,
    tuning_results_path,
    learning_rate_candidates,
    weight_decay,
    train_batch_size,
    gradient_accumulation_steps,
    gradient_checkpointing,
    finetuning_module,
    tokenizer_kwargs=None,
    model_kwargs=None,
):
    create_project_directories()

    if (
        tuning_results_path.exists()
        and SELECTED_HYPERPARAMETERS_PATH.exists()
    ):
        with SELECTED_HYPERPARAMETERS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            selected = json.load(file)

        if model_name in selected:
            print(
                f"{model_name} tuning already exists."
            )
            return

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_dir,
        local_files_only=True,
        **(tokenizer_kwargs or {}),
    )

    dataset = ds.dataset(
        token_cache_path,
        format="parquet",
    )

    source_path = str(
        token_cache_path
    ).replace("'", "''")

    connection = duckdb.connect()

    try:
        train_document_count = (
            finetuning_module.get_document_count(
                connection=connection,
                source_path=source_path,
                end_date=TUNING_TRAIN_END,
            )
        )

        rows = []

        for (
            epochs,
            learning_rate,
        ) in product(
            TRANSFORMER_EPOCH_CANDIDATES,
            learning_rate_candidates,
        ):
            print()
            print(
                f"Tuning {model_name}: "
                f"epochs={epochs}, "
                f"learning_rate={learning_rate}"
            )

            finetuning_module.set_random_seed(
                RANDOM_SEED
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
                    dataset=dataset,
                    train_end=TUNING_TRAIN_END,
                    tokenizer=tokenizer,
                    device=device,
                    train_document_count=(
                        train_document_count
                    ),
                    train_batch_size=(
                        train_batch_size
                    ),
                    gradient_accumulation_steps=(
                        gradient_accumulation_steps
                    ),
                    epochs=epochs,
                    learning_rate=(
                        learning_rate
                    ),
                    weight_decay=weight_decay,
                )
            )

            validation = (
                finetuning_module.score_split(
                    model=model,
                    dataset=dataset,
                    start_date=(
                        TUNING_VALIDATION_START
                    ),
                    end_date=(
                        TUNING_VALIDATION_END
                    ),
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

            metrics = (
                calculate_binary_metrics(
                    validation["target"],
                    validation["score"],
                    threshold,
                )
            )

            rows.append(
                {
                    "model": model_name,
                    "epochs": int(epochs),
                    "learning_rate": float(
                        learning_rate
                    ),
                    "weight_decay": float(
                        weight_decay
                    ),
                    "validation_threshold": (
                        threshold
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
                        validation[
                            "scoring_seconds"
                        ]
                    ),
                }
            )

            del model
            del validation

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        connection.close()

    selected_row = max(
        rows,
        key=selection_key,
    )

    with tuning_results_path.open(
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
        model_name=model_name,
        selected_row=selected_row,
    )

    print()
    print(
        f"{model_name} tuning completed."
    )

    print(
        "Selected: "
        f"epochs={selected_row['epochs']}, "
        f"learning_rate="
        f"{selected_row['learning_rate']}"
    )


def tune_bertimbau_hyperparameters():
    tune_transformer(
        model_name="bertimbau",
        pretrained_dir=(
            BERTIMBAU_PRETRAINED_DIR
        ),
        token_cache_path=(
            BERTIMBAU_TOKEN_CACHE_PATH
        ),
        tuning_results_path=(
            BERTIMBAU_TUNING_RESULTS_PATH
        ),
        learning_rate_candidates=(
            BERTIMBAU_LEARNING_RATE_CANDIDATES
        ),
        weight_decay=(
            BERTIMBAU_WEIGHT_DECAY
        ),
        train_batch_size=(
            BERTIMBAU_TRAIN_BATCH_SIZE
        ),
        gradient_accumulation_steps=(
            BERTIMBAU_GRADIENT_ACCUMULATION_STEPS
        ),
        gradient_checkpointing=(
            BERTIMBAU_GRADIENT_CHECKPOINTING
        ),
        finetuning_module=(
            bertimbau_finetuning
        ),
        tokenizer_kwargs={
            "do_lower_case": False,
        },
    )


def tune_albertina_hyperparameters():
    tune_transformer(
        model_name="albertina",
        pretrained_dir=(
            ALBERTINA_PRETRAINED_DIR
        ),
        token_cache_path=(
            ALBERTINA_TOKEN_CACHE_PATH
        ),
        tuning_results_path=(
            ALBERTINA_TUNING_RESULTS_PATH
        ),
        learning_rate_candidates=(
            ALBERTINA_LEARNING_RATE_CANDIDATES
        ),
        weight_decay=(
            ALBERTINA_WEIGHT_DECAY
        ),
        train_batch_size=(
            ALBERTINA_TRAIN_BATCH_SIZE
        ),
        gradient_accumulation_steps=(
            ALBERTINA_GRADIENT_ACCUMULATION_STEPS
        ),
        gradient_checkpointing=(
            ALBERTINA_GRADIENT_CHECKPOINTING
        ),
        finetuning_module=(
            albertina_finetuning
        ),
        model_kwargs={
            "dtype": torch.float32,
        },
    )