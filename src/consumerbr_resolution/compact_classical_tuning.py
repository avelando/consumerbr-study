import csv
import json

import duckdb
import joblib
from sklearn.linear_model import (
    SGDClassifier,
)
from sklearn.naive_bayes import (
    ComplementNB,
)

from consumerbr_resolution.config import (
    CLASSICAL_TUNING_RESULTS_PATH,
    COMPLEMENT_NB_ALPHA_CANDIDATES,
    FEATURE_BASE_PATH,
    RANDOM_SEED,
    SELECTED_HYPERPARAMETERS_PATH,
    SGD_ALPHA_CANDIDATES,
    SGD_LOSS,
    SGD_PENALTY,
    TUNING_CHAR_TFIDF_PATH,
    TUNING_TRAIN_END,
    TUNING_VALIDATION_END,
    TUNING_VALIDATION_START,
    TUNING_WORD_TFIDF_PATH,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)
from consumerbr_resolution.tfidf import (
    create_tfidf_vectorizer,
    get_train_document_count,
    iter_training_texts,
)
from consumerbr_resolution.tfidf_char import (
    create_tfidf_char_vectorizer,
)
from consumerbr_resolution.tfidf_complement_nb import (
    train_model as train_complement_nb,
)
from consumerbr_resolution.tfidf_sgd import (
    score_split as score_word_split,
)
from consumerbr_resolution.tfidf_variant_sgd import (
    score_split as score_variant_split,
    train_model as train_variant_sgd,
)


RESULT_FIELDS = [
    "family",
    "candidate",
    "validation_threshold",
    "validation_macro_f1",
    "validation_roc_auc",
    "validation_pr_auc",
    "validation_brier_score",
]


def evaluate_candidate(
    targets,
    scores,
):
    threshold, _ = (
        find_best_macro_f1_threshold(
            targets,
            scores,
        )
    )

    metrics = calculate_binary_metrics(
        targets,
        scores,
        threshold,
    )

    return threshold, metrics


def fit_tuning_vectorizers(
    connection,
    source_path,
):
    train_document_count = (
        get_train_document_count(
            connection=connection,
            source_path=source_path,
            train_end=TUNING_TRAIN_END,
        )
    )

    word_vectorizer = (
        create_tfidf_vectorizer()
    )

    word_vectorizer.fit(
        iter_training_texts(
            connection=connection,
            source_path=source_path,
            train_end=TUNING_TRAIN_END,
        )
    )

    char_vectorizer = (
        create_tfidf_char_vectorizer()
    )

    char_vectorizer.fit(
        iter_training_texts(
            connection=connection,
            source_path=source_path,
            train_end=TUNING_TRAIN_END,
        )
    )

    joblib.dump(
        word_vectorizer,
        TUNING_WORD_TFIDF_PATH,
        compress=3,
    )

    joblib.dump(
        char_vectorizer,
        TUNING_CHAR_TFIDF_PATH,
        compress=3,
    )

    return (
        word_vectorizer,
        char_vectorizer,
        train_document_count,
    )


def tune_sgd(
    connection,
    source_path,
    word_vectorizer,
    char_vectorizer,
    train_document_count,
):
    rows = []

    for alpha in SGD_ALPHA_CANDIDATES:
        model = SGDClassifier(
            loss=SGD_LOSS,
            penalty=SGD_PENALTY,
            alpha=alpha,
            random_state=RANDOM_SEED,
        )

        train_variant_sgd(
            connection=connection,
            source_path=source_path,
            train_end=TUNING_TRAIN_END,
            word_vectorizer=word_vectorizer,
            char_vectorizer=char_vectorizer,
            model=model,
            variant="word_char",
            train_document_count=(
                train_document_count
            ),
        )

        validation = (
            score_variant_split(
                connection=connection,
                source_path=source_path,
                start_date=(
                    TUNING_VALIDATION_START
                ),
                end_date=(
                    TUNING_VALIDATION_END
                ),
                word_vectorizer=(
                    word_vectorizer
                ),
                char_vectorizer=(
                    char_vectorizer
                ),
                model=model,
                variant="word_char",
            )
        )

        threshold, metrics = (
            evaluate_candidate(
                validation["target"],
                validation["score"],
            )
        )

        rows.append(
            {
                "family": "sgd",
                "candidate": float(alpha),
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
            }
        )

    best = max(
        rows,
        key=lambda row: (
            row[
                "validation_macro_f1"
            ],
            row[
                "validation_roc_auc"
            ],
            -row["candidate"],
        ),
    )

    return (
        rows,
        float(best["candidate"]),
    )


def tune_complement_nb(
    connection,
    source_path,
    word_vectorizer,
    train_document_count,
):
    rows = []

    for alpha in (
        COMPLEMENT_NB_ALPHA_CANDIDATES
    ):
        model = ComplementNB(
            alpha=alpha,
        )

        train_complement_nb(
            connection=connection,
            source_path=source_path,
            train_end=TUNING_TRAIN_END,
            vectorizer=word_vectorizer,
            model=model,
            document_count=(
                train_document_count
            ),
        )

        validation = score_word_split(
            connection=connection,
            source_path=source_path,
            start_date=(
                TUNING_VALIDATION_START
            ),
            end_date=(
                TUNING_VALIDATION_END
            ),
            vectorizer=word_vectorizer,
            model=model,
        )

        threshold, metrics = (
            evaluate_candidate(
                validation["target"],
                validation["score"],
            )
        )

        rows.append(
            {
                "family": (
                    "complement_nb"
                ),
                "candidate": float(alpha),
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
            }
        )

    best = max(
        rows,
        key=lambda row: (
            row[
                "validation_macro_f1"
            ],
            row[
                "validation_roc_auc"
            ],
            -row["candidate"],
        ),
    )

    return (
        rows,
        float(best["candidate"]),
    )


def tune_classical_text_hyperparameters():
    create_project_directories()

    outputs = [
        CLASSICAL_TUNING_RESULTS_PATH,
        SELECTED_HYPERPARAMETERS_PATH,
        TUNING_WORD_TFIDF_PATH,
        TUNING_CHAR_TFIDF_PATH,
    ]

    if all(
        path.exists()
        for path in outputs
    ):
        print(
            "Classical text tuning "
            "already exists."
        )
        return

    for path in outputs:
        if path.exists():
            path.unlink()

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    connection = duckdb.connect()

    try:
        (
            word_vectorizer,
            char_vectorizer,
            train_document_count,
        ) = fit_tuning_vectorizers(
            connection=connection,
            source_path=source_path,
        )

        (
            sgd_rows,
            selected_sgd_alpha,
        ) = tune_sgd(
            connection=connection,
            source_path=source_path,
            word_vectorizer=(
                word_vectorizer
            ),
            char_vectorizer=(
                char_vectorizer
            ),
            train_document_count=(
                train_document_count
            ),
        )

        (
            nb_rows,
            selected_nb_alpha,
        ) = tune_complement_nb(
            connection=connection,
            source_path=source_path,
            word_vectorizer=(
                word_vectorizer
            ),
            train_document_count=(
                train_document_count
            ),
        )

    finally:
        connection.close()

    rows = [
        *sgd_rows,
        *nb_rows,
    ]

    with (
        CLASSICAL_TUNING_RESULTS_PATH
        .open(
            "w",
            newline="",
            encoding="utf-8",
        )
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    selected = {
        "protocol": {
            "train_end": (
                TUNING_TRAIN_END
            ),
            "validation_start": (
                TUNING_VALIDATION_START
            ),
            "validation_end": (
                TUNING_VALIDATION_END
            ),
            "selection_metric": (
                "macro_f1"
            ),
        },
        "sgd": {
            "alpha": (
                selected_sgd_alpha
            ),
        },
        "complement_nb": {
            "alpha": (
                selected_nb_alpha
            ),
        },
    }

    with (
        SELECTED_HYPERPARAMETERS_PATH
        .open(
            "w",
            encoding="utf-8",
        )
    ) as file:
        json.dump(
            selected,
            file,
            indent=2,
            sort_keys=True,
        )

    print(
        "Classical text tuning completed."
    )

    print(
        f"Selected SGD alpha: "
        f"{selected_sgd_alpha}"
    )

    print(
        f"Selected ComplementNB alpha: "
        f"{selected_nb_alpha}"
    )