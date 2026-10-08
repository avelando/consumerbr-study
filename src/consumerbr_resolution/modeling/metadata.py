import csv
import time

import duckdb
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder

from consumerbr_resolution.config import (
    COMPANY_MIN_FREQUENCY,
    FEATURE_BASE_PATH,
    METADATA_MODELS_DIR,
    METADATA_NUMERIC_FEATURES,
    RARE_COMPANY_LABEL,
    TABLES_DIR,
    TEMPORAL_FOLDS,
    VALID_UFS,
    create_project_directories,
)


METADATA_SUMMARY_PATH = TABLES_DIR / "metadata_preprocessor_summary.csv"


SUMMARY_FIELDS = [
    "fold",
    "train_end",
    "train_document_count",
    "numeric_feature_count",
    "frequent_company_count",
    "company_feature_count",
    "uf_feature_count",
    "total_feature_count_with_company",
    "total_feature_count_without_company",
    "fit_seconds",
]


def get_numeric_statistics(
    connection,
    source_path,
    train_end,
):
    expressions = []

    for feature in METADATA_NUMERIC_FEATURES:
        expressions.append(
            f"AVG({feature}) AS {feature}_mean"
        )
        expressions.append(
            f"STDDEV_POP({feature}) AS {feature}_std"
        )

    result = connection.execute(
        f"""
        SELECT
            {", ".join(expressions)}
        FROM read_parquet('{source_path}')
        WHERE opening_date <= DATE '{train_end}'
        """
    ).fetchone()

    means = []
    standard_deviations = []

    for index in range(
        len(METADATA_NUMERIC_FEATURES)
    ):
        mean_value = result[index * 2]
        std_value = result[index * 2 + 1]

        means.append(
            float(mean_value)
        )

        if std_value is None or std_value == 0:
            standard_deviations.append(1.0)
        else:
            standard_deviations.append(
                float(std_value)
            )

    return (
        np.asarray(
            means,
            dtype=np.float32,
        ),
        np.asarray(
            standard_deviations,
            dtype=np.float32,
        ),
    )


def get_frequent_companies(
    connection,
    source_path,
    train_end,
):
    rows = connection.execute(
        f"""
        SELECT
            company
        FROM read_parquet('{source_path}')
        WHERE
            opening_date <= DATE '{train_end}'
            AND company IS NOT NULL
            AND company <> ''
        GROUP BY company
        HAVING COUNT(*) >= {COMPANY_MIN_FREQUENCY}
        ORDER BY company
        """
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]


def create_company_encoder(
    company_categories,
):
    encoder = OneHotEncoder(
        categories=[
            np.asarray(
                company_categories,
                dtype=object,
            )
        ],
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )

    encoder.fit(
        np.asarray(
            [[RARE_COMPANY_LABEL]],
            dtype=object,
        )
    )

    return encoder


def create_uf_encoder():
    categories = [
        *VALID_UFS,
        "UNKNOWN",
    ]

    encoder = OneHotEncoder(
        categories=[
            np.asarray(
                categories,
                dtype=object,
            )
        ],
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )

    encoder.fit(
        np.asarray(
            [["UNKNOWN"]],
            dtype=object,
        )
    )

    return encoder


def transform_metadata_rows(
    rows,
    preprocessor,
    include_company,
):
    numeric_start = 2
    numeric_end = (
        numeric_start
        + len(METADATA_NUMERIC_FEATURES)
    )

    numeric_matrix = np.asarray(
        [
            row[numeric_start:numeric_end]
            for row in rows
        ],
        dtype=np.float32,
    )

    numeric_matrix = (
        numeric_matrix
        - preprocessor["numeric_mean"]
    ) / preprocessor["numeric_std"]

    numeric_sparse = csr_matrix(
        numeric_matrix,
        dtype=np.float32,
    )

    companies = []

    frequent_companies = preprocessor[
        "frequent_company_set"
    ]

    for row in rows:
        company = row[0]

        if company in frequent_companies:
            companies.append(company)
        else:
            companies.append(
                RARE_COMPANY_LABEL
            )

    company_matrix = (
        preprocessor["company_encoder"]
        .transform(
            np.asarray(
                companies,
                dtype=object,
            ).reshape(-1, 1)
        )
    )

    known_ufs = preprocessor[
        "known_uf_set"
    ]

    ufs = []

    for row in rows:
        uf = row[1]

        if uf in known_ufs:
            ufs.append(uf)
        else:
            ufs.append("UNKNOWN")

    uf_matrix = (
        preprocessor["uf_encoder"]
        .transform(
            np.asarray(
                ufs,
                dtype=object,
            ).reshape(-1, 1)
        )
    )

    blocks = [
        numeric_sparse,
    ]

    if include_company:
        blocks.append(
            company_matrix
        )

    blocks.append(
        uf_matrix
    )

    return hstack(
        blocks,
        format="csr",
        dtype=np.float32,
    )


def fit_metadata_preprocessors():
    create_project_directories()

    model_paths = [
        METADATA_MODELS_DIR
        / f"fold_{fold['fold']:02d}.joblib"
        for fold in TEMPORAL_FOLDS
    ]

    output_paths = [
        METADATA_SUMMARY_PATH,
        *model_paths,
    ]

    if all(
        path.exists()
        for path in output_paths
    ):
        print(
            "Metadata preprocessors already exist."
        )
        return

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    print(
        "Fitting fold-specific metadata preprocessors"
    )
    print(
        f"Source: {FEATURE_BASE_PATH}"
    )
    print(
        f"Models: {METADATA_MODELS_DIR}"
    )

    summary_rows = {}

    if METADATA_SUMMARY_PATH.exists():
        with METADATA_SUMMARY_PATH.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                summary_rows[
                    int(row["fold"])
                ] = row

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            model_path = (
                METADATA_MODELS_DIR
                / f"fold_{fold_number:02d}.joblib"
            )

            if (
                model_path.exists()
                and fold_number in summary_rows
            ):
                print()
                print(
                    f"Metadata preprocessor fold "
                    f"{fold_number} already exists."
                )
                continue

            if model_path.exists():
                model_path.unlink()

            print()
            print(
                f"Fitting metadata preprocessor "
                f"for fold {fold_number}"
            )

            start_time = (
                time.perf_counter()
            )

            train_end = fold["train_end"]

            train_document_count = (
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet(
                        '{source_path}'
                    )
                    WHERE opening_date
                        <= DATE '{train_end}'
                    """
                ).fetchone()[0]
            )

            (
                numeric_mean,
                numeric_std,
            ) = get_numeric_statistics(
                connection=connection,
                source_path=source_path,
                train_end=train_end,
            )

            frequent_companies = (
                get_frequent_companies(
                    connection=connection,
                    source_path=source_path,
                    train_end=train_end,
                )
            )

            company_categories = [
                *frequent_companies,
                RARE_COMPANY_LABEL,
            ]

            company_encoder = (
                create_company_encoder(
                    company_categories
                )
            )

            uf_encoder = (
                create_uf_encoder()
            )

            preprocessor = {
                "fold": fold_number,
                "train_end": train_end,
                "numeric_features": (
                    METADATA_NUMERIC_FEATURES
                ),
                "numeric_mean": (
                    numeric_mean
                ),
                "numeric_std": (
                    numeric_std
                ),
                "frequent_companies": (
                    frequent_companies
                ),
                "frequent_company_set": (
                    set(
                        frequent_companies
                    )
                ),
                "known_uf_set": set(
                    [
                        *VALID_UFS,
                        "UNKNOWN",
                    ]
                ),
                "company_encoder": (
                    company_encoder
                ),
                "uf_encoder": (
                    uf_encoder
                ),
            }

            temporary_path = (
                model_path.with_suffix(
                    ".joblib.part"
                )
            )

            if temporary_path.exists():
                temporary_path.unlink()

            joblib.dump(
                preprocessor,
                temporary_path,
                compress=3,
            )

            temporary_path.replace(
                model_path
            )

            company_feature_count = (
                len(
                    company_encoder.categories_[0]
                )
            )

            uf_feature_count = len(
                uf_encoder.categories_[0]
            )

            numeric_feature_count = len(
                METADATA_NUMERIC_FEATURES
            )

            fit_seconds = (
                time.perf_counter()
                - start_time
            )

            summary_rows[fold_number] = {
                "fold": fold_number,
                "train_end": train_end,
                "train_document_count": (
                    int(
                        train_document_count
                    )
                ),
                "numeric_feature_count": (
                    numeric_feature_count
                ),
                "frequent_company_count": (
                    len(
                        frequent_companies
                    )
                ),
                "company_feature_count": (
                    company_feature_count
                ),
                "uf_feature_count": (
                    uf_feature_count
                ),
                "total_feature_count_with_company": (
                    numeric_feature_count
                    + company_feature_count
                    + uf_feature_count
                ),
                "total_feature_count_without_company": (
                    numeric_feature_count
                    + uf_feature_count
                ),
                "fit_seconds": (
                    fit_seconds
                ),
            }

            print(
                f"Frequent companies: "
                f"{len(frequent_companies)}"
            )

            print(
                "Features with company: "
                f"{numeric_feature_count + company_feature_count + uf_feature_count}"
            )

            print(
                "Features without company: "
                f"{numeric_feature_count + uf_feature_count}"
            )

            print(
                f"Saved to: {model_path}"
            )

    finally:
        connection.close()

    ordered_rows = [
        summary_rows[fold["fold"]]
        for fold in TEMPORAL_FOLDS
    ]

    print()
    print(
        "Metadata preprocessor fitting completed."
    )
    print(
        f"Saved to: "
        f"{METADATA_SUMMARY_PATH}"
    )