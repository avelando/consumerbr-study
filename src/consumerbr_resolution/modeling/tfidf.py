import csv
import time

import duckdb
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from consumerbr_resolution.config import (
    FEATURE_BASE_PATH,
    TABLES_DIR,
    TEMPORAL_FOLDS,
    TFIDF_BATCH_SIZE,
    TFIDF_LOWERCASE,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_MODELS_DIR,
    TFIDF_NGRAM_RANGE,
    TFIDF_STRIP_ACCENTS,
    TFIDF_SUBLINEAR_TF,
    create_project_directories,
)


TFIDF_SUMMARY_PATH = TABLES_DIR / "tfidf_vectorizer_summary.csv"


SUMMARY_FIELDS = [
    "fold",
    "train_end",
    "train_document_count",
    "vocabulary_size",
    "ngram_min",
    "ngram_max",
    "min_df",
    "max_df",
    "max_features",
    "fit_seconds",
]


def get_train_document_count(
    connection,
    source_path,
    train_end,
):
    result = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{source_path}')
        WHERE opening_date <= DATE '{train_end}'
        """
    ).fetchone()

    return int(result[0])


def iter_training_texts(
    connection,
    source_path,
    train_end,
):
    cursor = connection.execute(
        f"""
        SELECT complaint_text
        FROM read_parquet('{source_path}')
        WHERE opening_date <= DATE '{train_end}'
        """
    )

    while True:
        rows = cursor.fetchmany(TFIDF_BATCH_SIZE)

        if not rows:
            break

        for row in rows:
            yield row[0]


def create_tfidf_vectorizer():
    return TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        strip_accents=TFIDF_STRIP_ACCENTS,
        lowercase=TFIDF_LOWERCASE,
        dtype=np.float32,
    )


def fit_tfidf_vectorizers():
    create_project_directories()

    model_paths = [
        TFIDF_MODELS_DIR
        / f"fold_{fold['fold']:02d}.joblib"
        for fold in TEMPORAL_FOLDS
    ]

    output_paths = [
        TFIDF_SUMMARY_PATH,
        *model_paths,
    ]

    if all(path.exists() for path in output_paths):
        print("TF-IDF vectorizers already exist.")
        return

    for path in output_paths:
        if path.exists():
            path.unlink()

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    print("Fitting fold-specific TF-IDF vectorizers")
    print(f"Source: {FEATURE_BASE_PATH}")
    print(f"Models: {TFIDF_MODELS_DIR}")

    summary_rows = []

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]
            train_end = fold["train_end"]

            print()
            print(
                f"Fitting TF-IDF for fold "
                f"{fold_number}"
            )

            train_document_count = (
                get_train_document_count(
                    connection=connection,
                    source_path=source_path,
                    train_end=train_end,
                )
            )

            vectorizer = create_tfidf_vectorizer()

            progress = tqdm(
                iter_training_texts(
                    connection=connection,
                    source_path=source_path,
                    train_end=train_end,
                ),
                total=train_document_count,
                desc=f"Fold {fold_number}",
                unit="docs",
                dynamic_ncols=True,
            )

            start_time = time.perf_counter()

            try:
                vectorizer.fit(progress)
            finally:
                progress.close()

            fit_seconds = (
                time.perf_counter()
                - start_time
            )

            vocabulary_size = len(
                vectorizer.vocabulary_
            )

            model_path = (
                TFIDF_MODELS_DIR
                / f"fold_{fold_number:02d}.joblib"
            )

            temporary_path = model_path.with_suffix(
                ".joblib.part"
            )

            if temporary_path.exists():
                temporary_path.unlink()

            joblib.dump(
                vectorizer,
                temporary_path,
                compress=3,
            )

            temporary_path.replace(model_path)

            summary_rows.append(
                {
                    "fold": fold_number,
                    "train_end": train_end,
                    "train_document_count": (
                        train_document_count
                    ),
                    "vocabulary_size": (
                        vocabulary_size
                    ),
                    "ngram_min": (
                        TFIDF_NGRAM_RANGE[0]
                    ),
                    "ngram_max": (
                        TFIDF_NGRAM_RANGE[1]
                    ),
                    "min_df": TFIDF_MIN_DF,
                    "max_df": TFIDF_MAX_DF,
                    "max_features": (
                        TFIDF_MAX_FEATURES
                    ),
                    "fit_seconds": (
                        fit_seconds
                    ),
                }
            )

            print(
                f"Vocabulary size: "
                f"{vocabulary_size}"
            )

            print(
                f"Fit time: "
                f"{fit_seconds:.2f} seconds"
            )

            print(
                f"Saved to: {model_path}"
            )

    finally:
        connection.close()

    with TFIDF_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("TF-IDF vectorizer fitting completed.")
    print(f"Saved to: {TFIDF_SUMMARY_PATH}")