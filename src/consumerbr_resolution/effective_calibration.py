import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from consumerbr_resolution.config import (
    EFFECTIVE_CALIBRATION_BINS_PATH,
    EFFECTIVE_CALIBRATION_DIR,
    EFFECTIVE_CALIBRATION_FOLD_SUMMARY_PATH,
    EFFECTIVE_CALIBRATION_MODEL_SUMMARY_PATH,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    TEMPORAL_FOLDS,
    create_project_directories,
)
from consumerbr_resolution.evaluation import (
    calculate_binary_metrics,
    find_best_macro_f1_threshold,
)
from consumerbr_resolution.risk_calibration_analysis import (
    calculate_calibration,
)


CALIBRATION_EPSILON = 1e-6


CALIBRATION_MODEL_SPECS = (
    {
        "model": "catboost",
        "directory": (
            PREDICTIONS_DIR / "catboost"
        ),
    },
    {
        "model": "bertimbau",
        "directory": (
            PREDICTIONS_DIR / "bertimbau"
        ),
    },
    {
        "model": "bertimbau_head_tail_256",
        "directory": (
            PREDICTIONS_DIR
            / "bertimbau_head_tail_256"
        ),
    },
    {
        "model": "bertimbau_head_512",
        "directory": (
            PREDICTIONS_DIR
            / "bertimbau_head_512"
        ),
    },
    {
        "model": "albertina",
        "directory": (
            PREDICTIONS_DIR / "albertina"
        ),
    },
    {
        "model": "albertina_head_tail_256",
        "directory": (
            PREDICTIONS_DIR
            / "albertina_head_tail_256"
        ),
    },
    {
        "model": "albertina_head_512",
        "directory": (
            PREDICTIONS_DIR
            / "albertina_head_512"
        ),
    },
    {
        "model": "bertimbau_catboost_fusion",
        "directory": (
            PREDICTIONS_DIR
            / "bertimbau_catboost_fusion"
        ),
    },
    {
        "model": "albertina_catboost_fusion",
        "directory": (
            PREDICTIONS_DIR
            / "albertina_catboost_fusion"
        ),
    },
)


FOLD_SUMMARY_FIELDS = [
    "fold",
    "model",
    "method",
    "calibration_slope",
    "calibration_intercept",
    "raw_threshold",
    "calibrated_threshold",
    "raw_brier_score",
    "calibrated_brier_score",
    "brier_improvement",
    "raw_expected_calibration_error",
    "calibrated_expected_calibration_error",
    "ece_improvement",
    "raw_macro_f1",
    "calibrated_macro_f1",
    "raw_roc_auc",
    "calibrated_roc_auc",
    "raw_pr_auc",
    "calibrated_pr_auc",
]


BIN_FIELDS = [
    "fold",
    "model",
    "calibration_variant",
    "bin",
    "lower_bound",
    "upper_bound",
    "complaint_count",
    "mean_predicted_probability",
    "observed_resolved_rate",
    "absolute_gap",
]


MODEL_SUMMARY_FIELDS = [
    "model",
    "method",
    "fold_count",
    "raw_brier_score_mean",
    "calibrated_brier_score_mean",
    "brier_improvement_mean",
    "brier_improved_fold_count",
    "raw_ece_mean",
    "calibrated_ece_mean",
    "ece_improvement_mean",
    "ece_improved_fold_count",
    "raw_macro_f1_mean",
    "calibrated_macro_f1_mean",
]


def get_prediction_path(
    directory,
    fold_number,
    split,
):
    return (
        directory
        / (
            f"fold_{fold_number:02d}"
            f"_{split}.parquet"
        )
    )


def get_calibrated_prediction_path(
    model_name,
    fold_number,
    split,
):
    directory = (
        EFFECTIVE_CALIBRATION_DIR
        / "predictions"
        / model_name
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / (
            f"fold_{fold_number:02d}"
            f"_{split}.parquet"
        )
    )


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    frame = pd.read_parquet(path)

    required_columns = {
        "record_id",
        "complaint_id",
        "opening_date",
        "target_resolved",
        "score",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing_columns)}"
        )

    return frame


def scores_to_logits(scores):
    clipped_scores = np.clip(
        np.asarray(
            scores,
            dtype=np.float64,
        ),
        CALIBRATION_EPSILON,
        1.0 - CALIBRATION_EPSILON,
    )

    return np.log(
        clipped_scores
        / (1.0 - clipped_scores)
    ).reshape(-1, 1)


def fit_platt_calibrator(
    targets,
    scores,
):
    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    calibrator.fit(
        scores_to_logits(scores),
        targets,
    )

    slope = float(
        calibrator.coef_[0, 0]
    )

    if slope <= 0.0:
        raise ValueError(
            "Platt calibration produced "
            "a non-positive slope."
        )

    return calibrator


def apply_platt_calibrator(
    calibrator,
    scores,
):
    return calibrator.predict_proba(
        scores_to_logits(scores)
    )[:, 1]


def write_predictions(
    path,
    frame,
    calibrated_scores,
    calibrated_threshold,
):
    output = frame[
        [
            "record_id",
            "complaint_id",
            "opening_date",
            "target_resolved",
        ]
    ].copy()

    output["raw_score"] = (
        frame["score"].to_numpy(
            dtype=np.float64
        )
    )

    output["calibrated_score"] = (
        calibrated_scores
    )

    output["calibrated_prediction"] = (
        calibrated_scores
        >= calibrated_threshold
    ).astype(np.int8)

    temporary_path = (
        path.with_suffix(".parquet.part")
    )

    if temporary_path.exists():
        temporary_path.unlink()

    output.to_parquet(
        temporary_path,
        index=False,
        compression="zstd",
    )

    temporary_path.replace(path)


def calibration_bin_rows(
    rows,
    variant,
):
    output = []

    for row in rows:
        output.append(
            {
                "fold": row["fold"],
                "model": row["model"],
                "calibration_variant": variant,
                "bin": row["bin"],
                "lower_bound": (
                    row["lower_bound"]
                ),
                "upper_bound": (
                    row["upper_bound"]
                ),
                "complaint_count": (
                    row["complaint_count"]
                ),
                "mean_predicted_probability": (
                    row[
                        "mean_predicted_probability"
                    ]
                ),
                "observed_resolved_rate": (
                    row[
                        "observed_resolved_rate"
                    ]
                ),
                "absolute_gap": (
                    row["absolute_gap"]
                ),
            }
        )

    return output


def write_csv(
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


def summarize_models(fold_rows):
    frame = pd.DataFrame(fold_rows)

    rows = []

    for model_name, group in (
        frame.groupby("model")
    ):
        rows.append(
            {
                "model": model_name,
                "method": "platt_logit",
                "fold_count": len(group),
                "raw_brier_score_mean": float(
                    group[
                        "raw_brier_score"
                    ].mean()
                ),
                "calibrated_brier_score_mean": float(
                    group[
                        "calibrated_brier_score"
                    ].mean()
                ),
                "brier_improvement_mean": float(
                    group[
                        "brier_improvement"
                    ].mean()
                ),
                "brier_improved_fold_count": int(
                    (
                        group[
                            "brier_improvement"
                        ]
                        > 0
                    ).sum()
                ),
                "raw_ece_mean": float(
                    group[
                        "raw_expected_calibration_error"
                    ].mean()
                ),
                "calibrated_ece_mean": float(
                    group[
                        "calibrated_expected_calibration_error"
                    ].mean()
                ),
                "ece_improvement_mean": float(
                    group[
                        "ece_improvement"
                    ].mean()
                ),
                "ece_improved_fold_count": int(
                    (
                        group[
                            "ece_improvement"
                        ]
                        > 0
                    ).sum()
                ),
                "raw_macro_f1_mean": float(
                    group[
                        "raw_macro_f1"
                    ].mean()
                ),
                "calibrated_macro_f1_mean": float(
                    group[
                        "calibrated_macro_f1"
                    ].mean()
                ),
            }
        )

    return rows


def calibrate_main_model_probabilities():
    create_project_directories()

    output_paths = [
        EFFECTIVE_CALIBRATION_FOLD_SUMMARY_PATH,
        EFFECTIVE_CALIBRATION_MODEL_SUMMARY_PATH,
        EFFECTIVE_CALIBRATION_BINS_PATH,
    ]

    if all(
        path.exists()
        for path in output_paths
    ):
        print(
            "Effective calibration already exists."
        )
        return

    for path in output_paths:
        if path.exists():
            path.unlink()

    print(
        "Calibrating main-model probabilities"
    )

    fold_rows = []
    bin_rows = []

    for specification in (
        CALIBRATION_MODEL_SPECS
    ):
        model_name = (
            specification["model"]
        )

        directory = (
            specification["directory"]
        )

        print()
        print(
            f"Calibrating model: "
            f"{model_name}"
        )

        for fold in TEMPORAL_FOLDS:
            fold_number = fold["fold"]

            validation = load_predictions(
                get_prediction_path(
                    directory=directory,
                    fold_number=fold_number,
                    split="validation",
                )
            )

            test = load_predictions(
                get_prediction_path(
                    directory=directory,
                    fold_number=fold_number,
                    split="test",
                )
            )

            validation_targets = (
                validation[
                    "target_resolved"
                ].to_numpy(
                    dtype=np.int8
                )
            )

            validation_scores = (
                validation["score"].to_numpy(
                    dtype=np.float64
                )
            )

            test_targets = (
                test[
                    "target_resolved"
                ].to_numpy(
                    dtype=np.int8
                )
            )

            test_scores = (
                test["score"].to_numpy(
                    dtype=np.float64
                )
            )

            calibrator = (
                fit_platt_calibrator(
                    targets=(
                        validation_targets
                    ),
                    scores=(
                        validation_scores
                    ),
                )
            )

            calibrated_validation_scores = (
                apply_platt_calibrator(
                    calibrator=calibrator,
                    scores=validation_scores,
                )
            )

            calibrated_test_scores = (
                apply_platt_calibrator(
                    calibrator=calibrator,
                    scores=test_scores,
                )
            )

            raw_threshold, _ = (
                find_best_macro_f1_threshold(
                    validation_targets,
                    validation_scores,
                )
            )

            calibrated_threshold = float(
                apply_platt_calibrator(
                    calibrator=calibrator,
                    scores=np.asarray(
                        [raw_threshold],
                        dtype=np.float64,
                    ),
                )[0]
            )

            raw_metrics = (
                calculate_binary_metrics(
                    test_targets,
                    test_scores,
                    raw_threshold,
                )
            )

            calibrated_metrics = (
                calculate_binary_metrics(
                    test_targets,
                    calibrated_test_scores,
                    calibrated_threshold,
                )
            )

            (
                raw_calibration,
                raw_bins,
            ) = calculate_calibration(
                targets=test_targets,
                scores=test_scores,
                fold_number=fold_number,
                model_name=model_name,
            )

            (
                calibrated_calibration,
                calibrated_bins,
            ) = calculate_calibration(
                targets=test_targets,
                scores=(
                    calibrated_test_scores
                ),
                fold_number=fold_number,
                model_name=model_name,
            )

            slope = float(
                calibrator.coef_[0, 0]
            )

            intercept = float(
                calibrator.intercept_[0]
            )

            fold_rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "method": "platt_logit",
                    "calibration_slope": slope,
                    "calibration_intercept": (
                        intercept
                    ),
                    "raw_threshold": (
                        raw_threshold
                    ),
                    "calibrated_threshold": (
                        calibrated_threshold
                    ),
                    "raw_brier_score": (
                        raw_calibration[
                            "brier_score"
                        ]
                    ),
                    "calibrated_brier_score": (
                        calibrated_calibration[
                            "brier_score"
                        ]
                    ),
                    "brier_improvement": (
                        raw_calibration[
                            "brier_score"
                        ]
                        - calibrated_calibration[
                            "brier_score"
                        ]
                    ),
                    "raw_expected_calibration_error": (
                        raw_calibration[
                            "expected_calibration_error"
                        ]
                    ),
                    "calibrated_expected_calibration_error": (
                        calibrated_calibration[
                            "expected_calibration_error"
                        ]
                    ),
                    "ece_improvement": (
                        raw_calibration[
                            "expected_calibration_error"
                        ]
                        - calibrated_calibration[
                            "expected_calibration_error"
                        ]
                    ),
                    "raw_macro_f1": (
                        raw_metrics[
                            "macro_f1"
                        ]
                    ),
                    "calibrated_macro_f1": (
                        calibrated_metrics[
                            "macro_f1"
                        ]
                    ),
                    "raw_roc_auc": (
                        raw_metrics["roc_auc"]
                    ),
                    "calibrated_roc_auc": (
                        calibrated_metrics[
                            "roc_auc"
                        ]
                    ),
                    "raw_pr_auc": (
                        raw_metrics["pr_auc"]
                    ),
                    "calibrated_pr_auc": (
                        calibrated_metrics[
                            "pr_auc"
                        ]
                    ),
                }
            )

            bin_rows.extend(
                calibration_bin_rows(
                    raw_bins,
                    "raw",
                )
            )

            bin_rows.extend(
                calibration_bin_rows(
                    calibrated_bins,
                    "platt_logit",
                )
            )

            write_predictions(
                path=(
                    get_calibrated_prediction_path(
                        model_name=model_name,
                        fold_number=fold_number,
                        split="validation",
                    )
                ),
                frame=validation,
                calibrated_scores=(
                    calibrated_validation_scores
                ),
                calibrated_threshold=(
                    calibrated_threshold
                ),
            )

            write_predictions(
                path=(
                    get_calibrated_prediction_path(
                        model_name=model_name,
                        fold_number=fold_number,
                        split="test",
                    )
                ),
                frame=test,
                calibrated_scores=(
                    calibrated_test_scores
                ),
                calibrated_threshold=(
                    calibrated_threshold
                ),
            )

            print(
                f"Fold {fold_number}: "
                f"Brier "
                f"{raw_calibration['brier_score']:.6f} -> "
                f"{calibrated_calibration['brier_score']:.6f}, "
                f"ECE "
                f"{raw_calibration['expected_calibration_error']:.6f} -> "
                f"{calibrated_calibration['expected_calibration_error']:.6f}"
            )

    model_rows = summarize_models(
        fold_rows
    )

    write_csv(
        EFFECTIVE_CALIBRATION_FOLD_SUMMARY_PATH,
        FOLD_SUMMARY_FIELDS,
        fold_rows,
    )

    write_csv(
        EFFECTIVE_CALIBRATION_BINS_PATH,
        BIN_FIELDS,
        bin_rows,
    )

    write_csv(
        EFFECTIVE_CALIBRATION_MODEL_SUMMARY_PATH,
        MODEL_SUMMARY_FIELDS,
        model_rows,
    )

    print()
    print(
        "Effective probability calibration "
        "completed."
    )