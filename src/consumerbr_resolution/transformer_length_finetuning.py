import csv
import gc
import shutil

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
    ALBERTINA_EPOCHS,
    ALBERTINA_EVAL_BATCH_SIZE,
    ALBERTINA_GRADIENT_ACCUMULATION_STEPS,
    ALBERTINA_GRADIENT_CHECKPOINTING,
    ALBERTINA_HEAD_TAIL_FINETUNED_DIR,
    ALBERTINA_HEAD_TAIL_TOKEN_CACHE_PATH,
    ALBERTINA_LEARNING_RATE,
    ALBERTINA_LONG_EVAL_BATCH_SIZE,
    ALBERTINA_LONG_FINETUNED_DIR,
    ALBERTINA_LONG_GRADIENT_ACCUMULATION_STEPS,
    ALBERTINA_LONG_MAX_LENGTH,
    ALBERTINA_LONG_TOKEN_CACHE_PATH,
    ALBERTINA_LONG_TRAIN_BATCH_SIZE,
    ALBERTINA_MAX_LENGTH,
    ALBERTINA_PRETRAINED_DIR,
    ALBERTINA_TRAIN_BATCH_SIZE,
    BERTIMBAU_EPOCHS,
    BERTIMBAU_EVAL_BATCH_SIZE,
    BERTIMBAU_GRADIENT_ACCUMULATION_STEPS,
    BERTIMBAU_GRADIENT_CHECKPOINTING,
    BERTIMBAU_HEAD_TAIL_FINETUNED_DIR,
    BERTIMBAU_HEAD_TAIL_TOKEN_CACHE_PATH,
    BERTIMBAU_LEARNING_RATE,
    BERTIMBAU_LONG_EVAL_BATCH_SIZE,
    BERTIMBAU_LONG_FINETUNED_DIR,
    BERTIMBAU_LONG_GRADIENT_ACCUMULATION_STEPS,
    BERTIMBAU_LONG_MAX_LENGTH,
    BERTIMBAU_LONG_TOKEN_CACHE_PATH,
    BERTIMBAU_LONG_TRAIN_BATCH_SIZE,
    BERTIMBAU_MAX_LENGTH,
    BERTIMBAU_PRETRAINED_DIR,
    BERTIMBAU_TRAIN_BATCH_SIZE,
    METRICS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    TEMPORAL_FOLDS,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)


METRIC_FIELDS = [
    "fold",
    "split",
    "model",
    "token_strategy",
    "max_length",
    "threshold_source",
    "threshold",
    "epochs",
    "train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
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


def write_fold_metrics(
    metrics_path,
    rows,
):
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


def rebuild_aggregate_metrics(
    metrics_dir,
    metrics_path,
):
    rows = []

    for fold in TEMPORAL_FOLDS:
        fold_path = (
            metrics_dir
            / f"fold_{fold['fold']:02d}.csv"
        )

        with fold_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            rows.extend(
                csv.DictReader(file)
            )

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


def evaluate_transformer_length_variant(
    model_name,
    token_strategy,
    max_length,
    token_cache_path,
    pretrained_dir,
    finetuned_dir,
    train_batch_size,
    eval_batch_size,
    gradient_accumulation_steps,
    epochs,
    learning_rate,
    gradient_checkpointing,
    finetuning_module,
    tokenizer_kwargs=None,
    model_kwargs=None,
):
    create_project_directories()

    metrics_dir = (
        METRICS_DIR / model_name
    )

    metrics_path = (
        METRICS_DIR
        / f"{model_name}_metrics.csv"
    )

    predictions_dir = (
        PREDICTIONS_DIR / model_name
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    finetuned_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Evaluating {model_name} temporal fine-tuning"
    )
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_dir,
        local_files_only=True,
        **(tokenizer_kwargs or {}),
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

            model_path = (
                finetuned_dir
                / f"fold_{fold_number:02d}"
            )

            validation_prediction_path = (
                predictions_dir
                / (
                    f"fold_{fold_number:02d}"
                    "_validation.parquet"
                )
            )

            test_prediction_path = (
                predictions_dir
                / (
                    f"fold_{fold_number:02d}"
                    "_test.parquet"
                )
            )

            fold_metrics_path = (
                metrics_dir
                / f"fold_{fold_number:02d}.csv"
            )

            outputs = [
                model_path,
                validation_prediction_path,
                test_prediction_path,
                fold_metrics_path,
            ]

            if all(
                path.exists()
                for path in outputs
            ):
                print()
                print(
                    f"Fold {fold_number} already exists."
                )
                continue

            if model_path.exists():
                shutil.rmtree(model_path)

            if validation_prediction_path.exists():
                validation_prediction_path.unlink()

            if test_prediction_path.exists():
                test_prediction_path.unlink()

            if fold_metrics_path.exists():
                fold_metrics_path.unlink()

            finetuning_module.set_random_seed(
                RANDOM_SEED + fold_number
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

            train_document_count = (
                finetuning_module.get_document_count(
                    connection=connection,
                    source_path=source_path,
                    end_date=fold["train_end"],
                )
            )

            training_seconds = (
                finetuning_module.train_model(
                    model=model,
                    dataset=token_dataset,
                    train_end=fold["train_end"],
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
                    include_identifiers=True,
                    eval_batch_size=(
                        eval_batch_size
                    ),
                )
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

            test = finetuning_module.score_split(
                model=model,
                dataset=token_dataset,
                start_date=fold["test_start"],
                end_date=fold["test_end"],
                tokenizer=tokenizer,
                device=device,
                include_identifiers=True,
                eval_batch_size=eval_batch_size,
            )

            test_metrics = (
                calculate_binary_metrics(
                    test["target"],
                    test["score"],
                    threshold,
                )
            )

            temporary_model_path = (
                finetuned_dir
                / f"fold_{fold_number:02d}.part"
            )

            if temporary_model_path.exists():
                shutil.rmtree(
                    temporary_model_path
                )

            model.save_pretrained(
                temporary_model_path
            )

            temporary_model_path.rename(
                model_path
            )

            finetuning_module.write_predictions(
                prediction_path=(
                    validation_prediction_path
                ),
                result=validation,
                threshold=threshold,
            )

            finetuning_module.write_predictions(
                prediction_path=(
                    test_prediction_path
                ),
                result=test,
                threshold=threshold,
            )

            common = {
                "fold": fold_number,
                "model": model_name,
                "token_strategy": token_strategy,
                "max_length": max_length,
                "threshold_source": (
                    "validation_macro_f1"
                ),
                "epochs": epochs,
                "train_batch_size": (
                    train_batch_size
                ),
                "gradient_accumulation_steps": (
                    gradient_accumulation_steps
                ),
                "learning_rate": learning_rate,
                "training_seconds": (
                    training_seconds
                ),
            }

            rows = [
                {
                    **common,
                    "split": "validation",
                    "scoring_seconds": (
                        validation[
                            "scoring_seconds"
                        ]
                    ),
                    **validation_metrics,
                },
                {
                    **common,
                    "split": "test",
                    "scoring_seconds": (
                        test["scoring_seconds"]
                    ),
                    **test_metrics,
                },
            ]

            write_fold_metrics(
                metrics_path=fold_metrics_path,
                rows=rows,
            )

            print(
                f"Threshold: {threshold:.6f}"
            )

            print(
                "Validation Macro-F1: "
                f"{validation_macro_f1:.4f}"
            )

            print(
                "Test Macro-F1: "
                f"{test_metrics['macro_f1']:.4f}"
            )

            del model
            del validation
            del test

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        connection.close()

    rebuild_aggregate_metrics(
        metrics_dir=metrics_dir,
        metrics_path=metrics_path,
    )

    print()
    print(
        f"{model_name} evaluation completed."
    )


def evaluate_bertimbau_head_tail_256():
    evaluate_transformer_length_variant(
        model_name="bertimbau_head_tail_256",
        token_strategy="head_tail",
        max_length=BERTIMBAU_MAX_LENGTH,
        token_cache_path=(
            BERTIMBAU_HEAD_TAIL_TOKEN_CACHE_PATH
        ),
        pretrained_dir=BERTIMBAU_PRETRAINED_DIR,
        finetuned_dir=(
            BERTIMBAU_HEAD_TAIL_FINETUNED_DIR
        ),
        train_batch_size=(
            BERTIMBAU_TRAIN_BATCH_SIZE
        ),
        eval_batch_size=(
            BERTIMBAU_EVAL_BATCH_SIZE
        ),
        gradient_accumulation_steps=(
            BERTIMBAU_GRADIENT_ACCUMULATION_STEPS
        ),
        epochs=BERTIMBAU_EPOCHS,
        learning_rate=BERTIMBAU_LEARNING_RATE,
        gradient_checkpointing=(
            BERTIMBAU_GRADIENT_CHECKPOINTING
        ),
        finetuning_module=bertimbau_finetuning,
        tokenizer_kwargs={
            "do_lower_case": False,
        },
    )


def evaluate_bertimbau_head_512():
    evaluate_transformer_length_variant(
        model_name="bertimbau_head_512",
        token_strategy="head",
        max_length=BERTIMBAU_LONG_MAX_LENGTH,
        token_cache_path=(
            BERTIMBAU_LONG_TOKEN_CACHE_PATH
        ),
        pretrained_dir=BERTIMBAU_PRETRAINED_DIR,
        finetuned_dir=(
            BERTIMBAU_LONG_FINETUNED_DIR
        ),
        train_batch_size=(
            BERTIMBAU_LONG_TRAIN_BATCH_SIZE
        ),
        eval_batch_size=(
            BERTIMBAU_LONG_EVAL_BATCH_SIZE
        ),
        gradient_accumulation_steps=(
            BERTIMBAU_LONG_GRADIENT_ACCUMULATION_STEPS
        ),
        epochs=BERTIMBAU_EPOCHS,
        learning_rate=BERTIMBAU_LEARNING_RATE,
        gradient_checkpointing=(
            BERTIMBAU_GRADIENT_CHECKPOINTING
        ),
        finetuning_module=bertimbau_finetuning,
        tokenizer_kwargs={
            "do_lower_case": False,
        },
    )


def evaluate_albertina_head_tail_256():
    evaluate_transformer_length_variant(
        model_name="albertina_head_tail_256",
        token_strategy="head_tail",
        max_length=ALBERTINA_MAX_LENGTH,
        token_cache_path=(
            ALBERTINA_HEAD_TAIL_TOKEN_CACHE_PATH
        ),
        pretrained_dir=ALBERTINA_PRETRAINED_DIR,
        finetuned_dir=(
            ALBERTINA_HEAD_TAIL_FINETUNED_DIR
        ),
        train_batch_size=(
            ALBERTINA_TRAIN_BATCH_SIZE
        ),
        eval_batch_size=(
            ALBERTINA_EVAL_BATCH_SIZE
        ),
        gradient_accumulation_steps=(
            ALBERTINA_GRADIENT_ACCUMULATION_STEPS
        ),
        epochs=ALBERTINA_EPOCHS,
        learning_rate=ALBERTINA_LEARNING_RATE,
        gradient_checkpointing=(
            ALBERTINA_GRADIENT_CHECKPOINTING
        ),
        finetuning_module=albertina_finetuning,
        model_kwargs={
            "dtype": torch.float32,
        },
    )


def evaluate_albertina_head_512():
    evaluate_transformer_length_variant(
        model_name="albertina_head_512",
        token_strategy="head",
        max_length=ALBERTINA_LONG_MAX_LENGTH,
        token_cache_path=(
            ALBERTINA_LONG_TOKEN_CACHE_PATH
        ),
        pretrained_dir=ALBERTINA_PRETRAINED_DIR,
        finetuned_dir=(
            ALBERTINA_LONG_FINETUNED_DIR
        ),
        train_batch_size=(
            ALBERTINA_LONG_TRAIN_BATCH_SIZE
        ),
        eval_batch_size=(
            ALBERTINA_LONG_EVAL_BATCH_SIZE
        ),
        gradient_accumulation_steps=(
            ALBERTINA_LONG_GRADIENT_ACCUMULATION_STEPS
        ),
        epochs=ALBERTINA_EPOCHS,
        learning_rate=ALBERTINA_LEARNING_RATE,
        gradient_checkpointing=(
            ALBERTINA_GRADIENT_CHECKPOINTING
        ),
        finetuning_module=albertina_finetuning,
        model_kwargs={
            "dtype": torch.float32,
        },
    )