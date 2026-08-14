import csv
import gc
import math
import random
import shutil
import time
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from consumerbr_resolution.config import (
    ALBERTINA_ARROW_BATCH_SIZE,
    ALBERTINA_EPOCHS,
    ALBERTINA_EVAL_BATCH_SIZE,
    ALBERTINA_FINETUNED_DIR,
    ALBERTINA_GRADIENT_ACCUMULATION_STEPS,
    ALBERTINA_GRADIENT_CHECKPOINTING,
    ALBERTINA_LEARNING_RATE,
    ALBERTINA_MAX_GRAD_NORM,
    ALBERTINA_PRETRAINED_DIR,
    ALBERTINA_TOKEN_CACHE_PATH,
    ALBERTINA_TRAIN_BATCH_SIZE,
    ALBERTINA_USE_AMP,
    ALBERTINA_WARMUP_RATIO,
    ALBERTINA_WEIGHT_DECAY,
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
from consumerbr_resolution.hyperparameter_selection import (
    get_selected_albertina_hyperparameters,
)


ALBERTINA_METRICS_DIR = (
    METRICS_DIR / "albertina"
)

ALBERTINA_METRICS_PATH = (
    METRICS_DIR / "albertina_metrics.csv"
)

ALBERTINA_PREDICTIONS_DIR = (
    PREDICTIONS_DIR / "albertina"
)


METRIC_FIELDS = [
    "fold",
    "split",
    "model",
    "threshold_source",
    "threshold",
    "epochs",
    "train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
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


def set_random_seed(
    seed,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_filter(
    start_date=None,
    end_date=None,
):
    expression = None

    if start_date is not None:
        value = pa.scalar(
            date.fromisoformat(start_date),
            type=pa.date32(),
        )

        expression = (
            ds.field("opening_date") >= value
        )

    if end_date is not None:
        value = pa.scalar(
            date.fromisoformat(end_date),
            type=pa.date32(),
        )

        condition = (
            ds.field("opening_date") <= value
        )

        if expression is None:
            expression = condition
        else:
            expression = (
                expression & condition
            )

    return expression


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

    where_clause = ""

    if conditions:
        where_clause = (
            "WHERE "
            + " AND ".join(conditions)
        )

    result = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{source_path}')
        {where_clause}
        """
    ).fetchone()

    return int(result[0])


def iter_arrow_rows(
    dataset,
    start_date,
    end_date,
    include_identifiers,
):
    columns = [
        "target_resolved",
        "input_ids",
    ]

    if include_identifiers:
        columns = [
            "record_id",
            "complaint_id",
            "opening_date",
            *columns,
        ]

    scanner = dataset.scanner(
        columns=columns,
        filter=build_filter(
            start_date=start_date,
            end_date=end_date,
        ),
        batch_size=ALBERTINA_ARROW_BATCH_SIZE,
        use_threads=True,
    )

    for record_batch in (
        scanner.to_batches()
    ):
        values = record_batch.to_pydict()

        row_count = (
            record_batch.num_rows
        )

        for index in range(
            row_count
        ):
            if include_identifiers:
                yield {
                    "record_id": (
                        values[
                            "record_id"
                        ][index]
                    ),
                    "complaint_id": (
                        values[
                            "complaint_id"
                        ][index]
                    ),
                    "opening_date": (
                        values[
                            "opening_date"
                        ][index]
                    ),
                    "target_resolved": (
                        values[
                            "target_resolved"
                        ][index]
                    ),
                    "input_ids": (
                        values[
                            "input_ids"
                        ][index]
                    ),
                }
            else:
                yield {
                    "target_resolved": (
                        values[
                            "target_resolved"
                        ][index]
                    ),
                    "input_ids": (
                        values[
                            "input_ids"
                        ][index]
                    ),
                }


def iter_batches(
    dataset,
    start_date,
    end_date,
    batch_size,
    include_identifiers=False,
):
    batch = []

    for row in iter_arrow_rows(
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        include_identifiers=(
            include_identifiers
        ),
    ):
        batch.append(row)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def collate_batch(
    rows,
    pad_token_id,
    device,
):
    max_length = max(
        len(row["input_ids"])
        for row in rows
    )

    batch_size = len(rows)

    input_ids = torch.full(
        (
            batch_size,
            max_length,
        ),
        pad_token_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (
            batch_size,
            max_length,
        ),
        dtype=torch.long,
    )

    labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    for index, row in enumerate(
        rows
    ):
        token_ids = torch.tensor(
            row["input_ids"],
            dtype=torch.long,
        )

        length = token_ids.shape[0]

        input_ids[
            index,
            :length,
        ] = token_ids

        attention_mask[
            index,
            :length,
        ] = 1

        labels[index] = int(
            row["target_resolved"]
        )

    return (
        input_ids.to(
            device,
            non_blocking=True,
        ),
        attention_mask.to(
            device,
            non_blocking=True,
        ),
        labels.to(
            device,
            non_blocking=True,
        ),
    )


def create_scheduler(
    optimizer,
    total_steps,
):
    warmup_steps = int(
        total_steps
        * ALBERTINA_WARMUP_RATIO
    )

    def learning_rate_lambda(
        current_step,
    ):
        if (
            warmup_steps > 0
            and current_step
            < warmup_steps
        ):
            return (
                current_step
                / max(
                    1,
                    warmup_steps,
                )
            )

        remaining_steps = (
            total_steps
            - current_step
        )

        decay_steps = (
            total_steps
            - warmup_steps
        )

        return max(
            0.0,
            remaining_steps
            / max(
                1,
                decay_steps,
            ),
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_lambda,
    )


def train_model(
    model,
    dataset,
    train_end,
    tokenizer,
    device,
    train_document_count,
    train_batch_size=None,
    gradient_accumulation_steps=None,
    epochs=None,
    learning_rate=None,
    weight_decay=None,
):
    if train_batch_size is None:
        train_batch_size = (
            ALBERTINA_TRAIN_BATCH_SIZE
        )

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = (
            ALBERTINA_GRADIENT_ACCUMULATION_STEPS
        )

    if epochs is None:
        epochs = ALBERTINA_EPOCHS

    if learning_rate is None:
        learning_rate = (
            ALBERTINA_LEARNING_RATE
        )

    if weight_decay is None:
        weight_decay = (
            ALBERTINA_WEIGHT_DECAY
        )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    batches_per_epoch = math.ceil(
        train_document_count
        / train_batch_size
    )

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch
        / gradient_accumulation_steps
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * epochs
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=(
            total_optimizer_steps
        ),
    )

    use_amp = (
        ALBERTINA_USE_AMP
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    processed = 0
    optimizer_step = 0

    start_time = time.perf_counter()

    for epoch in range(
        1,
        epochs + 1,
    ):
        print()
        print(
            f"Epoch {epoch}/"
            f"{ALBERTINA_EPOCHS}"
        )

        batch_index = 0

        for rows in iter_batches(
            dataset=dataset,
            start_date=None,
            end_date=train_end,
            batch_size=train_batch_size,
        ):
            batch_index += 1

            (
                input_ids,
                attention_mask,
                labels,
            ) = collate_batch(
                rows=rows,
                pad_token_id=(
                    tokenizer.pad_token_id
                ),
                device=device,
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model(
                    input_ids=input_ids,
                    attention_mask=(
                        attention_mask
                    ),
                    labels=labels,
                )

                loss = (
                    output.loss
                    / gradient_accumulation_steps
                )

            scaler.scale(
                loss
            ).backward()

            should_step = (
                batch_index
                % gradient_accumulation_steps
                == 0
                or batch_index
                == batches_per_epoch
            )

            if should_step:
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    ALBERTINA_MAX_GRAD_NORM,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                scheduler.step()

                optimizer_step += 1

            processed += len(rows)

            if (
                processed % 10_000
                < len(rows)
            ):
                print(
                    f"\rProcessed: "
                    f"{processed}",
                    end="",
                    flush=True,
                )

        print()

    return (
        time.perf_counter()
        - start_time
    )


def score_split(
    model,
    dataset,
    start_date,
    end_date,
    tokenizer,
    device,
    include_identifiers=False,
    eval_batch_size=None,
):
    if eval_batch_size is None:
        eval_batch_size = (
            ALBERTINA_EVAL_BATCH_SIZE
        )
    targets = []
    scores = []

    record_ids = []
    complaint_ids = []
    opening_dates = []

    use_amp = (
        ALBERTINA_USE_AMP
        and device.type == "cuda"
    )

    model.eval()

    start_time = time.perf_counter()

    with torch.no_grad():
        for rows in iter_batches(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            batch_size=eval_batch_size,
            include_identifiers=(
                include_identifiers
            ),
        ):
            (
                input_ids,
                attention_mask,
                labels,
            ) = collate_batch(
                rows=rows,
                pad_token_id=(
                    tokenizer.pad_token_id
                ),
                device=device,
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model(
                    input_ids=input_ids,
                    attention_mask=(
                        attention_mask
                    ),
                )

            probabilities = (
                torch.softmax(
                    output.logits,
                    dim=1,
                )[:, 1]
                .float()
                .cpu()
                .numpy()
            )

            targets.append(
                labels.cpu().numpy()
            )

            scores.append(
                probabilities
            )

            if include_identifiers:
                record_ids.extend(
                    row["record_id"]
                    for row in rows
                )

                complaint_ids.extend(
                    row["complaint_id"]
                    for row in rows
                )

                opening_dates.extend(
                    row["opening_date"]
                    for row in rows
                )

    model.train()

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

    if include_identifiers:
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
    prediction_path,
    result,
    threshold,
):
    predictions = (
        result["score"] >= threshold
    ).astype(np.int8)

    frame = pd.DataFrame(
        {
            "record_id": (
                result["record_id"]
            ),
            "complaint_id": (
                result["complaint_id"]
            ),
            "opening_date": (
                result["opening_date"]
            ),
            "target_resolved": (
                result["target"]
            ),
            "score": (
                result["score"]
            ),
            "prediction": predictions,
        }
    )

    temporary_path = (
        prediction_path.with_suffix(
            ".parquet.part"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    frame.to_parquet(
        temporary_path,
        index=False,
        compression="zstd",
    )

    temporary_path.replace(
        prediction_path
    )


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


def rebuild_aggregate_metrics():
    rows = []

    for fold in TEMPORAL_FOLDS:
        path = (
            ALBERTINA_METRICS_DIR
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

    with ALBERTINA_METRICS_PATH.open(
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


def evaluate_albertina():
    create_project_directories()

    hyperparameters = (
        get_selected_albertina_hyperparameters()
    )

    ALBERTINA_METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ALBERTINA_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_random_seed(
        RANDOM_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Evaluating Albertina temporal fine-tuning"
    )

    print(
        f"Device: {device}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            ALBERTINA_PRETRAINED_DIR,
            local_files_only=True,
        )
    )

    token_dataset = ds.dataset(
        ALBERTINA_TOKEN_CACHE_PATH,
        format="parquet",
    )

    token_cache_path = str(
        ALBERTINA_TOKEN_CACHE_PATH
    ).replace("'", "''")

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            model_path = (
                ALBERTINA_FINETUNED_DIR
                / f"fold_{fold_number:02d}"
            )

            validation_prediction_path = (
                ALBERTINA_PREDICTIONS_DIR
                / (
                    f"fold_{fold_number:02d}"
                    "_validation.parquet"
                )
            )

            test_prediction_path = (
                ALBERTINA_PREDICTIONS_DIR
                / (
                    f"fold_{fold_number:02d}"
                    "_test.parquet"
                )
            )

            metrics_path = (
                ALBERTINA_METRICS_DIR
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

            if model_path.exists():
                shutil.rmtree(
                    model_path
                )

            if validation_prediction_path.exists():
                validation_prediction_path.unlink()

            if test_prediction_path.exists():
                test_prediction_path.unlink()

            if metrics_path.exists():
                metrics_path.unlink()

            print()
            print(
                f"Evaluating fold "
                f"{fold_number}"
            )

            set_random_seed(
                RANDOM_SEED
                + fold_number
            )

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    ALBERTINA_PRETRAINED_DIR,
                    num_labels=2,
                    dtype=torch.float32,
                    local_files_only=True,
                )
            )

            if (
                ALBERTINA_GRADIENT_CHECKPOINTING
            ):
                model.gradient_checkpointing_enable()

            model.to(device)

            train_document_count = (
                get_document_count(
                    connection=connection,
                    source_path=(
                        token_cache_path
                    ),
                    end_date=fold[
                        "train_end"
                    ],
                )
            )

            training_seconds = train_model(
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
            )

            validation = score_split(
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
                include_identifiers=True,
            )

            test_metrics = (
                calculate_binary_metrics(
                    test["target"],
                    test["score"],
                    threshold,
                )
            )

            temporary_model_path = (
                ALBERTINA_FINETUNED_DIR
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

            write_predictions(
                prediction_path=(
                    validation_prediction_path
                ),
                result=validation,
                threshold=threshold,
            )

            write_predictions(
                prediction_path=(
                    test_prediction_path
                ),
                result=test,
                threshold=threshold,
            )

            rows = [
                {
                    "fold": fold_number,
                    "split": "validation",
                    "model": "albertina",
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    epochs=(
                        hyperparameters["epochs"]
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
                    "train_batch_size": (
                        ALBERTINA_TRAIN_BATCH_SIZE
                    ),
                    "gradient_accumulation_steps": (
                        ALBERTINA_GRADIENT_ACCUMULATION_STEPS
                    ),
                    "learning_rate": (
                        ALBERTINA_LEARNING_RATE
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
                    "model": "albertina",
                    "threshold_source": (
                        "validation_macro_f1"
                    ),
                    epochs=(
                        hyperparameters["epochs"]
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
                    "train_batch_size": (
                        ALBERTINA_TRAIN_BATCH_SIZE
                    ),
                    "gradient_accumulation_steps": (
                        ALBERTINA_GRADIENT_ACCUMULATION_STEPS
                    ),
                    "learning_rate": (
                        ALBERTINA_LEARNING_RATE
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

            write_fold_metrics(
                metrics_path=metrics_path,
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

            print(
                f"Training time: "
                f"{training_seconds:.2f} seconds"
            )

            del model
            del validation
            del test

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        connection.close()

    rebuild_aggregate_metrics()

    print()
    print(
        "Albertina evaluation completed."
    )

    print(
        f"Saved to: "
        f"{ALBERTINA_METRICS_PATH}"
    )