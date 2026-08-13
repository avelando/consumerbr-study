import csv

import duckdb
import numpy as np
from sklearn.metrics import (
    brier_score_loss,
)

from consumerbr_resolution.analysis_registry import (
    MODEL_PREDICTION_SPECS,
    get_prediction_path,
)
from consumerbr_resolution.config import (
    ANALYSIS_DIR,
    CALIBRATION_BIN_COUNT,
    RISK_RANKING_FRACTIONS,
    TEMPORAL_FOLDS,
    create_project_directories,
)


RISK_RANKING_PATH = (
    ANALYSIS_DIR
    / "risk_ranking_metrics.csv"
)

CALIBRATION_SUMMARY_PATH = (
    ANALYSIS_DIR
    / "calibration_summary.csv"
)

CALIBRATION_BINS_PATH = (
    ANALYSIS_DIR
    / "calibration_bins.csv"
)


RISK_FIELDS = [
    "fold",
    "model",
    "fraction",
    "selected_count",
    "unresolved_count",
    "base_unresolved_rate",
    "precision_at_k",
    "recall_at_k",
    "lift_at_k",
]


CALIBRATION_SUMMARY_FIELDS = [
    "fold",
    "model",
    "complaint_count",
    "resolved_rate",
    "mean_predicted_probability",
    "brier_score",
    "expected_calibration_error",
]


CALIBRATION_BIN_FIELDS = [
    "fold",
    "model",
    "bin",
    "lower_bound",
    "upper_bound",
    "complaint_count",
    "mean_predicted_probability",
    "observed_resolved_rate",
    "absolute_gap",
]


def load_predictions(
    connection,
    specification,
    fold_number,
):
    path = get_prediction_path(
        specification=specification,
        fold_number=fold_number,
    )

    source_path = str(
        path
    ).replace("'", "''")

    score_column = specification[
        "score_column"
    ]

    return connection.execute(
        f"""
        SELECT
            CAST(
                target_resolved
                AS TINYINT
            ) AS target_resolved,
            CAST(
                {score_column}
                AS DOUBLE
            ) AS score
        FROM read_parquet(
            '{source_path}'
        )
        """
    ).fetchnumpy()


def calculate_risk_ranking(
    targets,
    scores,
    fold_number,
    model_name,
):
    unresolved = (
        1
        - targets
    ).astype(np.int8)

    risks = (
        1.0
        - scores
    )

    order = np.argsort(
        -risks,
        kind="mergesort",
    )

    unresolved_total = int(
        unresolved.sum()
    )

    base_rate = float(
        unresolved.mean()
    )

    rows = []

    for fraction in (
        RISK_RANKING_FRACTIONS
    ):
        selected_count = max(
            1,
            int(
                np.ceil(
                    len(targets)
                    * fraction
                )
            ),
        )

        selected_indices = order[
            :selected_count
        ]

        selected_unresolved = int(
            unresolved[
                selected_indices
            ].sum()
        )

        precision = (
            selected_unresolved
            / selected_count
        )

        if unresolved_total > 0:
            recall = (
                selected_unresolved
                / unresolved_total
            )
        else:
            recall = 0.0

        if base_rate > 0:
            lift = (
                precision
                / base_rate
            )
        else:
            lift = 0.0

        rows.append(
            {
                "fold": fold_number,
                "model": model_name,
                "fraction": (
                    fraction
                ),
                "selected_count": (
                    selected_count
                ),
                "unresolved_count": (
                    selected_unresolved
                ),
                "base_unresolved_rate": (
                    base_rate
                ),
                "precision_at_k": (
                    precision
                ),
                "recall_at_k": recall,
                "lift_at_k": lift,
            }
        )

    return rows


def calculate_calibration(
    targets,
    scores,
    fold_number,
    model_name,
):
    bin_indices = np.minimum(
        (
            scores
            * CALIBRATION_BIN_COUNT
        ).astype(np.int64),
        CALIBRATION_BIN_COUNT - 1,
    )

    bin_indices = np.maximum(
        bin_indices,
        0,
    )

    bin_rows = []

    expected_calibration_error = 0.0

    total_count = len(targets)

    for bin_number in range(
        CALIBRATION_BIN_COUNT
    ):
        mask = (
            bin_indices
            == bin_number
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        bin_scores = scores[
            mask
        ]

        bin_targets = targets[
            mask
        ]

        mean_probability = float(
            bin_scores.mean()
        )

        observed_rate = float(
            bin_targets.mean()
        )

        gap = abs(
            mean_probability
            - observed_rate
        )

        expected_calibration_error += (
            count
            / total_count
            * gap
        )

        lower_bound = (
            bin_number
            / CALIBRATION_BIN_COUNT
        )

        upper_bound = (
            (
                bin_number + 1
            )
            / CALIBRATION_BIN_COUNT
        )

        bin_rows.append(
            {
                "fold": fold_number,
                "model": model_name,
                "bin": bin_number + 1,
                "lower_bound": (
                    lower_bound
                ),
                "upper_bound": (
                    upper_bound
                ),
                "complaint_count": (
                    count
                ),
                "mean_predicted_probability": (
                    mean_probability
                ),
                "observed_resolved_rate": (
                    observed_rate
                ),
                "absolute_gap": gap,
            }
        )

    summary = {
        "fold": fold_number,
        "model": model_name,
        "complaint_count": (
            total_count
        ),
        "resolved_rate": float(
            targets.mean()
        ),
        "mean_predicted_probability": float(
            scores.mean()
        ),
        "brier_score": float(
            brier_score_loss(
                targets,
                scores,
            )
        ),
        "expected_calibration_error": float(
            expected_calibration_error
        ),
    }

    return summary, bin_rows


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


def analyze_risk_and_calibration():
    create_project_directories()

    output_paths = [
        RISK_RANKING_PATH,
        CALIBRATION_SUMMARY_PATH,
        CALIBRATION_BINS_PATH,
    ]

    if all(
        path.exists()
        for path in output_paths
    ):
        print(
            "Risk and calibration analysis already exists."
        )
        return

    for path in output_paths:
        if path.exists():
            path.unlink()

    print(
        "Analyzing risk ranking and calibration"
    )

    risk_rows = []
    calibration_summary_rows = []
    calibration_bin_rows = []

    connection = duckdb.connect()

    try:
        for specification in (
            MODEL_PREDICTION_SPECS
        ):
            model_name = specification[
                "model"
            ]

            print()
            print(
                f"Analyzing model: "
                f"{model_name}"
            )

            for fold in TEMPORAL_FOLDS:
                fold_number = fold[
                    "fold"
                ]

                data = load_predictions(
                    connection=connection,
                    specification=(
                        specification
                    ),
                    fold_number=(
                        fold_number
                    ),
                )

                targets = np.asarray(
                    data[
                        "target_resolved"
                    ],
                    dtype=np.int8,
                )

                scores = np.asarray(
                    data["score"],
                    dtype=np.float64,
                )

                if (
                    model_name
                    != "global_prior"
                ):
                    risk_rows.extend(
                        calculate_risk_ranking(
                            targets=targets,
                            scores=scores,
                            fold_number=(
                                fold_number
                            ),
                            model_name=(
                                model_name
                            ),
                        )
                    )

                (
                    calibration_summary,
                    calibration_bins,
                ) = calculate_calibration(
                    targets=targets,
                    scores=scores,
                    fold_number=(
                        fold_number
                    ),
                    model_name=(
                        model_name
                    ),
                )

                calibration_summary_rows.append(
                    calibration_summary
                )

                calibration_bin_rows.extend(
                    calibration_bins
                )

                print(
                    f"Fold {fold_number} "
                    f"completed."
                )

    finally:
        connection.close()

    write_rows(
        path=RISK_RANKING_PATH,
        fieldnames=RISK_FIELDS,
        rows=risk_rows,
    )

    write_rows(
        path=CALIBRATION_SUMMARY_PATH,
        fieldnames=(
            CALIBRATION_SUMMARY_FIELDS
        ),
        rows=(
            calibration_summary_rows
        ),
    )

    write_rows(
        path=CALIBRATION_BINS_PATH,
        fieldnames=(
            CALIBRATION_BIN_FIELDS
        ),
        rows=calibration_bin_rows,
    )

    print()
    print(
        "Risk and calibration analysis completed."
    )

    print(
        f"Saved to: "
        f"{RISK_RANKING_PATH}"
    )

    print(
        f"Saved to: "
        f"{CALIBRATION_SUMMARY_PATH}"
    )

    print(
        f"Saved to: "
        f"{CALIBRATION_BINS_PATH}"
    )