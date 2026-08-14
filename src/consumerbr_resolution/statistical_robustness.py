import csv
from itertools import combinations

import duckdb
import numpy as np

from consumerbr_resolution.analysis_registry import (
    MODEL_PREDICTION_SPECS,
    get_prediction_path,
)
from consumerbr_resolution.config import (
    ANALYSIS_DIR,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_REPLICATES,
    FEATURE_BASE_PATH,
    PERMUTATION_REPLICATES,
    RANDOM_SEED,
    TEMPORAL_FOLDS,
    create_project_directories,
)


MODEL_BOOTSTRAP_PATH = (
    ANALYSIS_DIR
    / "model_macro_f1_bootstrap.csv"
)

PAIRWISE_BOOTSTRAP_PATH = (
    ANALYSIS_DIR
    / "pairwise_macro_f1_bootstrap.csv"
)


MODEL_FIELDS = [
    "model",
    "fold_count",
    "bootstrap_replicates",
    "confidence_level",
    "observed_mean_macro_f1",
    "bootstrap_mean_macro_f1",
    "ci_lower",
    "ci_upper",
]


PAIRWISE_FIELDS = [
    "model_a",
    "model_b",
    "fold_count",
    "bootstrap_replicates",
    "permutation_replicates",
    "confidence_level",
    "observed_mean_macro_f1_a",
    "observed_mean_macro_f1_b",
    "observed_delta",
    "bootstrap_mean_delta",
    "ci_lower",
    "ci_upper",
    "p_value",
    "p_value_holm",
    "significant_holm_0_05",
    "model_a_better_folds",
    "model_b_better_folds",
    "tied_folds",
]


def safe_f1(
    numerator,
    denominator,
):
    numerator = np.asarray(
        numerator,
        dtype=np.float64,
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float64,
    )

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            denominator,
            dtype=np.float64,
        ),
        where=(
            denominator != 0
        ),
    )


def macro_f1_from_confusion(
    true_negative,
    false_positive,
    false_negative,
    true_positive,
):
    positive_f1 = safe_f1(
        2.0 * true_positive,
        (
            2.0 * true_positive
            + false_positive
            + false_negative
        ),
    )

    negative_f1 = safe_f1(
        2.0 * true_negative,
        (
            2.0 * true_negative
            + false_positive
            + false_negative
        ),
    )

    return (
        positive_f1
        + negative_f1
    ) / 2.0


def macro_f1_from_predictions(
    targets,
    predictions,
):
    targets = np.asarray(
        targets,
        dtype=np.int8,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int8,
    )

    true_negative = np.sum(
        (targets == 0)
        & (predictions == 0)
    )

    false_positive = np.sum(
        (targets == 0)
        & (predictions == 1)
    )

    false_negative = np.sum(
        (targets == 1)
        & (predictions == 0)
    )

    true_positive = np.sum(
        (targets == 1)
        & (predictions == 1)
    )

    return float(
        macro_f1_from_confusion(
            true_negative=(
                true_negative
            ),
            false_positive=(
                false_positive
            ),
            false_negative=(
                false_negative
            ),
            true_positive=(
                true_positive
            ),
        )
    )


def load_fold_predictions(
    connection,
    fold,
):
    fold_number = fold["fold"]

    feature_source = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    select_columns = [
        "data.record_id",
        (
            "CAST("
            "data.target_resolved "
            "AS TINYINT"
            ") AS target_resolved"
        ),
    ]

    joins = []

    for index, specification in enumerate(
        MODEL_PREDICTION_SPECS
    ):
        alias = f"prediction_{index}"

        prediction_path = (
            get_prediction_path(
                specification=(
                    specification
                ),
                fold_number=(
                    fold_number
                ),
            )
        )

        prediction_source = str(
            prediction_path
        ).replace("'", "''")

        prediction_column = (
            specification[
                "prediction_column"
            ]
        )

        model_name = specification[
            "model"
        ]

        select_columns.append(
            (
                f"CAST("
                f"{alias}."
                f"{prediction_column} "
                f"AS TINYINT"
                f") AS {model_name}"
            )
        )

        joins.append(
            f"""
            JOIN read_parquet(
                '{prediction_source}'
            ) AS {alias}
                ON
                    data.record_id
                    = {alias}.record_id
            """
        )

    return connection.execute(
        f"""
        SELECT
            {", ".join(select_columns)}
        FROM read_parquet(
            '{feature_source}'
        ) AS data
        {" ".join(joins)}
        WHERE
            data.opening_date
            BETWEEN
                DATE '{fold["test_start"]}'
                AND DATE '{fold["test_end"]}'
        ORDER BY
            data.record_id
        """
    ).fetchnumpy()


def bootstrap_single_model_fold(
    targets,
    predictions,
    rng,
):
    categories = (
        targets.astype(np.int64)
        * 2
        + predictions.astype(
            np.int64
        )
    )

    counts = np.bincount(
        categories,
        minlength=4,
    )

    total = int(
        counts.sum()
    )

    probabilities = (
        counts
        / total
    )

    draws = rng.multinomial(
        total,
        probabilities,
        size=BOOTSTRAP_REPLICATES,
    )

    true_negative = draws[:, 0]
    false_positive = draws[:, 1]
    false_negative = draws[:, 2]
    true_positive = draws[:, 3]

    return macro_f1_from_confusion(
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


def bootstrap_pair_fold(
    targets,
    predictions_a,
    predictions_b,
    rng,
):
    categories = (
        targets.astype(np.int64)
        * 4
        + predictions_a.astype(
            np.int64
        )
        * 2
        + predictions_b.astype(
            np.int64
        )
    )

    counts = np.bincount(
        categories,
        minlength=8,
    )

    total = int(
        counts.sum()
    )

    probabilities = (
        counts
        / total
    )

    draws = rng.multinomial(
        total,
        probabilities,
        size=BOOTSTRAP_REPLICATES,
    )

    true_negative_a = (
        draws[:, 0]
        + draws[:, 1]
    )

    false_positive_a = (
        draws[:, 2]
        + draws[:, 3]
    )

    false_negative_a = (
        draws[:, 4]
        + draws[:, 5]
    )

    true_positive_a = (
        draws[:, 6]
        + draws[:, 7]
    )

    true_negative_b = (
        draws[:, 0]
        + draws[:, 2]
    )

    false_positive_b = (
        draws[:, 1]
        + draws[:, 3]
    )

    false_negative_b = (
        draws[:, 4]
        + draws[:, 6]
    )

    true_positive_b = (
        draws[:, 5]
        + draws[:, 7]
    )

    macro_f1_a = (
        macro_f1_from_confusion(
            true_negative=(
                true_negative_a
            ),
            false_positive=(
                false_positive_a
            ),
            false_negative=(
                false_negative_a
            ),
            true_positive=(
                true_positive_a
            ),
        )
    )

    macro_f1_b = (
        macro_f1_from_confusion(
            true_negative=(
                true_negative_b
            ),
            false_positive=(
                false_positive_b
            ),
            false_negative=(
                false_negative_b
            ),
            true_positive=(
                true_positive_b
            ),
        )
    )

    return (
        macro_f1_a,
        macro_f1_b,
    )


def get_confidence_interval(
    values,
):
    alpha = (
        1.0
        - BOOTSTRAP_CONFIDENCE_LEVEL
    ) / 2.0

    lower, upper = np.quantile(
        values,
        [
            alpha,
            1.0 - alpha,
        ],
    )

    return (
        float(lower),
        float(upper),
    )


def permutation_pair_fold(
    targets,
    predictions_a,
    predictions_b,
    rng,
):
    categories = (
        targets.astype(np.int64)
        * 4
        + predictions_a.astype(
            np.int64
        )
        * 2
        + predictions_b.astype(
            np.int64
        )
    )

    counts = np.bincount(
        categories,
        minlength=8,
    )

    discordant_negative = int(
        counts[1] + counts[2]
    )

    discordant_positive = int(
        counts[5] + counts[6]
    )

    a_positive_negative = rng.binomial(
        discordant_negative,
        0.5,
        size=PERMUTATION_REPLICATES,
    )

    a_positive_positive = rng.binomial(
        discordant_positive,
        0.5,
        size=PERMUTATION_REPLICATES,
    )

    a_negative_negative = (
        discordant_negative
        - a_positive_negative
    )

    a_negative_positive = (
        discordant_positive
        - a_positive_positive
    )

    true_negative_a = (
        counts[0]
        + a_negative_negative
    )

    false_positive_a = (
        counts[3]
        + a_positive_negative
    )

    false_negative_a = (
        counts[4]
        + a_negative_positive
    )

    true_positive_a = (
        counts[7]
        + a_positive_positive
    )

    true_negative_b = (
        counts[0]
        + a_positive_negative
    )

    false_positive_b = (
        counts[3]
        + a_negative_negative
    )

    false_negative_b = (
        counts[4]
        + a_positive_positive
    )

    true_positive_b = (
        counts[7]
        + a_negative_positive
    )

    macro_f1_a = (
        macro_f1_from_confusion(
            true_negative=(
                true_negative_a
            ),
            false_positive=(
                false_positive_a
            ),
            false_negative=(
                false_negative_a
            ),
            true_positive=(
                true_positive_a
            ),
        )
    )

    macro_f1_b = (
        macro_f1_from_confusion(
            true_negative=(
                true_negative_b
            ),
            false_positive=(
                false_positive_b
            ),
            false_negative=(
                false_negative_b
            ),
            true_positive=(
                true_positive_b
            ),
        )
    )

    return (
        macro_f1_a
        - macro_f1_b
    )


def calculate_permutation_p_value(
    observed_delta,
    permutation_deltas,
):
    return float(
        (
            np.count_nonzero(
                np.abs(
                    permutation_deltas
                )
                >= abs(
                    observed_delta
                )
            )
            + 1
        )
        / (
            len(
                permutation_deltas
            )
            + 1
        )
    )


def apply_holm_correction(
    rows,
):
    count = len(rows)

    order = sorted(
        range(count),
        key=lambda index: (
            rows[index][
                "p_value"
            ]
        ),
    )

    previous = 0.0

    for rank, index in enumerate(
        order
    ):
        factor = (
            count
            - rank
        )

        adjusted = min(
            1.0,
            factor
            * rows[index][
                "p_value"
            ],
        )

        adjusted = max(
            previous,
            adjusted,
        )

        rows[index][
            "p_value_holm"
        ] = adjusted

        rows[index][
            "significant_holm_0_05"
        ] = int(
            adjusted < 0.05
        )

        previous = adjusted


def analyze_single_models(
    fold_data,
):
    rows = []

    model_names = [
        specification["model"]
        for specification
        in MODEL_PREDICTION_SPECS
    ]

    fold_count = len(
        TEMPORAL_FOLDS
    )

    for model_index, model_name in enumerate(
        model_names
    ):
        rng = np.random.default_rng(
            RANDOM_SEED
            + 10_000
            + model_index
        )

        observed_values = []

        bootstrap_sum = np.zeros(
            BOOTSTRAP_REPLICATES,
            dtype=np.float64,
        )

        for fold in TEMPORAL_FOLDS:
            fold_number = fold[
                "fold"
            ]

            data = fold_data[
                fold_number
            ]

            targets = np.asarray(
                data[
                    "target_resolved"
                ],
                dtype=np.int8,
            )

            predictions = np.asarray(
                data[
                    model_name
                ],
                dtype=np.int8,
            )

            observed_values.append(
                macro_f1_from_predictions(
                    targets=targets,
                    predictions=(
                        predictions
                    ),
                )
            )

            bootstrap_sum += (
                bootstrap_single_model_fold(
                    targets=targets,
                    predictions=(
                        predictions
                    ),
                    rng=rng,
                )
            )

        bootstrap_mean = (
            bootstrap_sum
            / fold_count
        )

        (
            ci_lower,
            ci_upper,
        ) = get_confidence_interval(
            bootstrap_mean
        )

        rows.append(
            {
                "model": model_name,
                "fold_count": (
                    fold_count
                ),
                "bootstrap_replicates": (
                    BOOTSTRAP_REPLICATES
                ),
                "confidence_level": (
                    BOOTSTRAP_CONFIDENCE_LEVEL
                ),
                "observed_mean_macro_f1": float(
                    np.mean(
                        observed_values
                    )
                ),
                "bootstrap_mean_macro_f1": float(
                    bootstrap_mean.mean()
                ),
                "ci_lower": (
                    ci_lower
                ),
                "ci_upper": (
                    ci_upper
                ),
            }
        )

    return rows


def analyze_pairwise_models(
    fold_data,
):
    rows = []

    model_names = [
        specification["model"]
        for specification
        in MODEL_PREDICTION_SPECS
    ]

    fold_count = len(
        TEMPORAL_FOLDS
    )

    for pair_index, (
        model_a,
        model_b,
    ) in enumerate(
        combinations(
            model_names,
            2,
        )
    ):
        rng = np.random.default_rng(
            RANDOM_SEED
            + 100_000
            + pair_index
        )

        permutation_rng = (
            np.random.default_rng(
                RANDOM_SEED
                + 200_000
                + pair_index
            )
        )

        observed_a = []
        observed_b = []

        bootstrap_a_sum = np.zeros(
            BOOTSTRAP_REPLICATES,
            dtype=np.float64,
        )

        bootstrap_b_sum = np.zeros(
            BOOTSTRAP_REPLICATES,
            dtype=np.float64,
        )

        permutation_delta_sum = np.zeros(
            PERMUTATION_REPLICATES,
            dtype=np.float64,
        )

        a_better_folds = 0
        b_better_folds = 0
        tied_folds = 0

        for fold in TEMPORAL_FOLDS:
            fold_number = fold[
                "fold"
            ]

            data = fold_data[
                fold_number
            ]

            targets = np.asarray(
                data[
                    "target_resolved"
                ],
                dtype=np.int8,
            )

            predictions_a = np.asarray(
                data[model_a],
                dtype=np.int8,
            )

            predictions_b = np.asarray(
                data[model_b],
                dtype=np.int8,
            )

            fold_macro_a = (
                macro_f1_from_predictions(
                    targets=targets,
                    predictions=(
                        predictions_a
                    ),
                )
            )

            fold_macro_b = (
                macro_f1_from_predictions(
                    targets=targets,
                    predictions=(
                        predictions_b
                    ),
                )
            )

            observed_a.append(
                fold_macro_a
            )

            observed_b.append(
                fold_macro_b
            )

            if (
                fold_macro_a
                > fold_macro_b
            ):
                a_better_folds += 1
            elif (
                fold_macro_b
                > fold_macro_a
            ):
                b_better_folds += 1
            else:
                tied_folds += 1

            (
                bootstrap_a,
                bootstrap_b,
            ) = bootstrap_pair_fold(
                targets=targets,
                predictions_a=(
                    predictions_a
                ),
                predictions_b=(
                    predictions_b
                ),
                rng=rng,
            )

            bootstrap_a_sum += (
                bootstrap_a
            )

            bootstrap_b_sum += (
                bootstrap_b
            )

            permutation_delta_sum += (
                permutation_pair_fold(
                    targets=targets,
                    predictions_a=(
                        predictions_a
                    ),
                    predictions_b=(
                        predictions_b
                    ),
                    rng=(
                        permutation_rng
                    ),
                )
            )

        bootstrap_a_mean = (
            bootstrap_a_sum
            / fold_count
        )

        bootstrap_b_mean = (
            bootstrap_b_sum
            / fold_count
        )

        bootstrap_delta = (
            bootstrap_a_mean
            - bootstrap_b_mean
        )

        observed_mean_a = float(
            np.mean(
                observed_a
            )
        )

        observed_mean_b = float(
            np.mean(
                observed_b
            )
        )

        observed_delta = (
            observed_mean_a
            - observed_mean_b
        )

        (
            ci_lower,
            ci_upper,
        ) = get_confidence_interval(
            bootstrap_delta
        )

        permutation_delta = (
            permutation_delta_sum
            / fold_count
        )

        p_value = (
            calculate_permutation_p_value(
                observed_delta=(
                    observed_delta
                ),
                permutation_deltas=(
                    permutation_delta
                ),
            )
        )

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "fold_count": (
                    fold_count
                ),
                "bootstrap_replicates": (
                    BOOTSTRAP_REPLICATES
                ),
                "permutation_replicates": (
                    PERMUTATION_REPLICATES
                ),
                "confidence_level": (
                    BOOTSTRAP_CONFIDENCE_LEVEL
                ),
                "observed_mean_macro_f1_a": (
                    observed_mean_a
                ),
                "observed_mean_macro_f1_b": (
                    observed_mean_b
                ),
                "observed_delta": (
                    observed_delta
                ),
                "bootstrap_mean_delta": float(
                    bootstrap_delta.mean()
                ),
                "ci_lower": (
                    ci_lower
                ),
                "ci_upper": (
                    ci_upper
                ),
                "p_value": (
                    p_value
                ),
                "p_value_holm": (
                    None
                ),
                "significant_holm_0_05": (
                    None
                ),
                "model_a_better_folds": (
                    a_better_folds
                ),
                "model_b_better_folds": (
                    b_better_folds
                ),
                "tied_folds": (
                    tied_folds
                ),
            }
        )

    apply_holm_correction(
        rows
    )

    return rows


def write_rows(
    path,
    fieldnames,
    rows,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def analyze_statistical_robustness():
    create_project_directories()

    output_paths = [
        MODEL_BOOTSTRAP_PATH,
        PAIRWISE_BOOTSTRAP_PATH,
    ]

    if all(
        path.exists()
        for path in output_paths
    ):
        print(
            "Statistical robustness analysis already exists."
        )
        return

    for path in output_paths:
        if path.exists():
            path.unlink()

    print(
        "Analyzing statistical robustness"
    )

    fold_data = {}

    connection = duckdb.connect()

    try:
        for fold in TEMPORAL_FOLDS:
            fold_number = fold[
                "fold"
            ]

            print(
                f"Loading fold "
                f"{fold_number}"
            )

            fold_data[
                fold_number
            ] = load_fold_predictions(
                connection=connection,
                fold=fold,
            )

    finally:
        connection.close()

    print()
    print(
        "Calculating model confidence intervals"
    )

    model_rows = (
        analyze_single_models(
            fold_data=fold_data
        )
    )

    print(
        "Calculating paired model comparisons"
    )

    pairwise_rows = (
        analyze_pairwise_models(
            fold_data=fold_data
        )
    )

    write_rows(
        path=MODEL_BOOTSTRAP_PATH,
        fieldnames=MODEL_FIELDS,
        rows=model_rows,
    )

    write_rows(
        path=PAIRWISE_BOOTSTRAP_PATH,
        fieldnames=PAIRWISE_FIELDS,
        rows=pairwise_rows,
    )

    print()
    print(
        "Statistical robustness analysis completed."
    )

    print(
        f"Saved to: "
        f"{MODEL_BOOTSTRAP_PATH}"
    )

    print(
        f"Saved to: "
        f"{PAIRWISE_BOOTSTRAP_PATH}"
    )