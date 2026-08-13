import numpy as np
import pandas as pd

from consumerbr_resolution.analysis_registry import (
    MODEL_PREDICTION_SPECS,
)
from consumerbr_resolution.config import (
    FINAL_RESULTS_DIR,
    METRICS_DIR,
    create_project_directories,
)
from consumerbr_resolution.generalization_analysis import (
    COMPANY_GENERALIZATION_PATH,
    MONTHLY_MODEL_METRICS_PATH,
)
from consumerbr_resolution.risk_calibration_analysis import (
    CALIBRATION_SUMMARY_PATH,
    RISK_RANKING_PATH,
)
from consumerbr_resolution.statistical_robustness import (
    MODEL_BOOTSTRAP_PATH,
    PAIRWISE_BOOTSTRAP_PATH,
)


FINAL_FOLD_METRICS_PATH = (
    FINAL_RESULTS_DIR
    / "final_fold_test_metrics.csv"
)

FINAL_MODEL_SUMMARY_PATH = (
    FINAL_RESULTS_DIR
    / "final_model_summary.csv"
)

FINAL_EFFICIENCY_SUMMARY_PATH = (
    FINAL_RESULTS_DIR
    / "final_efficiency_summary.csv"
)

FINAL_ABLATION_SUMMARY_PATH = (
    FINAL_RESULTS_DIR
    / "final_ablation_summary.csv"
)

FINAL_COMPANY_GENERALIZATION_PATH = (
    FINAL_RESULTS_DIR
    / "final_company_generalization_summary.csv"
)

FINAL_TEMPORAL_STABILITY_PATH = (
    FINAL_RESULTS_DIR
    / "final_temporal_stability_summary.csv"
)

FINAL_RISK_RANKING_PATH = (
    FINAL_RESULTS_DIR
    / "final_risk_ranking_summary.csv"
)

FINAL_CALIBRATION_PATH = (
    FINAL_RESULTS_DIR
    / "final_calibration_summary.csv"
)

FINAL_SIGNIFICANT_PAIRWISE_PATH = (
    FINAL_RESULTS_DIR
    / "final_significant_pairwise_comparisons.csv"
)


METRIC_SOURCE_PATHS = (
    METRICS_DIR
    / "historical_baseline_metrics.csv",
    METRICS_DIR
    / "tfidf_sgd_metrics.csv",
    METRICS_DIR
    / "metadata_sgd_metrics.csv",
    METRICS_DIR
    / "tfidf_metadata_sgd_metrics.csv",
    METRICS_DIR
    / "tfidf_metadata_history_sgd_metrics.csv",
    METRICS_DIR
    / "tfidf_complement_nb_metrics.csv",
    METRICS_DIR
    / "catboost_metrics.csv",
    METRICS_DIR
    / "bertimbau_metrics.csv",
    METRICS_DIR
    / "bertimbau_catboost_fusion_metrics.csv",
)


STANDARD_FOLD_COLUMNS = [
    "fold",
    "model",
    "threshold",
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
    "training_seconds",
    "scoring_seconds",
    "tuning_seconds",
    "best_iteration",
]


SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1_resolved",
    "f1_unresolved",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
)


ABLATION_COMPARISONS = (
    (
        "company_identity",
        "metadata_without_company",
        "metadata_with_company",
    ),
    (
        "text_over_metadata",
        "metadata_with_company",
        "tfidf_metadata_sgd",
    ),
    (
        "metadata_over_text",
        "tfidf_sgd",
        "tfidf_metadata_sgd",
    ),
    (
        "company_history",
        "tfidf_metadata_sgd",
        "tfidf_metadata_history_sgd",
    ),
    (
        "tabular_fusion",
        "bertimbau",
        "bertimbau_catboost_fusion",
    ),
)


def get_model_order():
    return {
        specification["model"]: index
        for index, specification
        in enumerate(
            MODEL_PREDICTION_SPECS
        )
    }


def load_fold_metrics():
    frames = []

    for path in METRIC_SOURCE_PATHS:
        frame = pd.read_csv(path)

        frame = frame[
            frame["split"] == "test"
        ].copy()

        for column in STANDARD_FOLD_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan

        frames.append(
            frame[
                STANDARD_FOLD_COLUMNS
            ]
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    model_order = get_model_order()

    combined["model_order"] = (
        combined["model"]
        .map(model_order)
        .fillna(len(model_order))
    )

    combined = combined.sort_values(
        [
            "model_order",
            "fold",
        ]
    ).drop(
        columns=["model_order"]
    )

    return combined


def summarize_models(
    fold_metrics,
):
    rows = []

    bootstrap = pd.read_csv(
        MODEL_BOOTSTRAP_PATH
    ).set_index("model")

    model_order = get_model_order()

    for model_name, group in (
        fold_metrics.groupby(
            "model",
            sort=False,
        )
    ):
        row = {
            "model": model_name,
            "fold_count": int(
                group["fold"].nunique()
            ),
        }

        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                row[
                    f"mean_{metric}"
                ] = np.nan
                row[
                    f"sd_{metric}"
                ] = np.nan
                continue

            row[
                f"mean_{metric}"
            ] = float(
                values.mean()
            )

            row[
                f"sd_{metric}"
            ] = float(
                values.std(ddof=1)
            ) if len(values) > 1 else 0.0

        macro_values = pd.to_numeric(
            group["macro_f1"],
            errors="coerce",
        ).dropna()

        row["min_macro_f1"] = float(
            macro_values.min()
        )

        row["max_macro_f1"] = float(
            macro_values.max()
        )

        if model_name in bootstrap.index:
            row[
                "macro_f1_ci_lower"
            ] = float(
                bootstrap.loc[
                    model_name,
                    "ci_lower",
                ]
            )

            row[
                "macro_f1_ci_upper"
            ] = float(
                bootstrap.loc[
                    model_name,
                    "ci_upper",
                ]
            )
        else:
            row[
                "macro_f1_ci_lower"
            ] = np.nan

            row[
                "macro_f1_ci_upper"
            ] = np.nan

        row["model_order"] = (
            model_order.get(
                model_name,
                len(model_order),
            )
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "model_order",
        ]
    ).drop(
        columns=["model_order"]
    )

    return result


def summarize_efficiency(
    fold_metrics,
):
    rows = []

    model_order = get_model_order()

    for model_name, group in (
        fold_metrics.groupby(
            "model",
            sort=False,
        )
    ):
        row = {
            "model": model_name,
            "fold_count": int(
                group["fold"].nunique()
            ),
        }

        for column in (
            "training_seconds",
            "scoring_seconds",
            "tuning_seconds",
        ):
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                row[
                    f"mean_{column}"
                ] = np.nan

                row[
                    f"total_{column}"
                ] = np.nan
            else:
                row[
                    f"mean_{column}"
                ] = float(
                    values.mean()
                )

                row[
                    f"total_{column}"
                ] = float(
                    values.sum()
                )

        row["model_order"] = (
            model_order.get(
                model_name,
                len(model_order),
            )
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["model_order"]
    ).drop(
        columns=["model_order"]
    )


def summarize_ablations(
    fold_metrics,
):
    pivot = fold_metrics.pivot(
        index="fold",
        columns="model",
        values="macro_f1",
    )

    rows = []

    for (
        component,
        base_model,
        extended_model,
    ) in ABLATION_COMPARISONS:
        base = pivot[base_model]

        extended = pivot[
            extended_model
        ]

        valid = (
            base.notna()
            & extended.notna()
        )

        base = base[valid]

        extended = extended[
            valid
        ]

        deltas = (
            extended
            - base
        )

        rows.append(
            {
                "component": component,
                "base_model": (
                    base_model
                ),
                "extended_model": (
                    extended_model
                ),
                "fold_count": int(
                    len(deltas)
                ),
                "base_mean_macro_f1": float(
                    base.mean()
                ),
                "extended_mean_macro_f1": float(
                    extended.mean()
                ),
                "mean_delta_macro_f1": float(
                    deltas.mean()
                ),
                "sd_delta_macro_f1": float(
                    deltas.std(ddof=1)
                ) if len(deltas) > 1 else 0.0,
                "min_delta_macro_f1": float(
                    deltas.min()
                ),
                "max_delta_macro_f1": float(
                    deltas.max()
                ),
                "improved_folds": int(
                    (deltas > 0).sum()
                ),
                "degraded_folds": int(
                    (deltas < 0).sum()
                ),
                "tied_folds": int(
                    (deltas == 0).sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def weighted_mean(
    values,
    weights,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    return float(
        np.average(
            values,
            weights=weights,
        )
    )


def summarize_company_generalization():
    frame = pd.read_csv(
        COMPANY_GENERALIZATION_PATH
    )

    rows = []

    for (
        model_name,
        company_segment,
    ), group in frame.groupby(
        [
            "model",
            "company_segment",
        ],
        sort=False,
    ):
        counts = group[
            "complaint_count"
        ].to_numpy(
            dtype=np.float64
        )

        macro_values = group[
            "macro_f1"
        ].to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "model": model_name,
                "company_segment": (
                    company_segment
                ),
                "fold_count": int(
                    group["fold"].nunique()
                ),
                "complaint_count": int(
                    counts.sum()
                ),
                "resolved_rate": weighted_mean(
                    group[
                        "resolved_rate"
                    ],
                    counts,
                ),
                "mean_macro_f1": float(
                    macro_values.mean()
                ),
                "sd_macro_f1": float(
                    macro_values.std(
                        ddof=1
                    )
                ) if len(macro_values) > 1 else 0.0,
                "mean_roc_auc": float(
                    group[
                        "roc_auc"
                    ].mean()
                ),
                "mean_pr_auc": float(
                    group[
                        "pr_auc"
                    ].mean()
                ),
                "mean_brier_score": float(
                    group[
                        "brier_score"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_temporal_stability():
    frame = pd.read_csv(
        MONTHLY_MODEL_METRICS_PATH
    )

    rows = []

    for model_name, group in frame.groupby(
        "model",
        sort=False,
    ):
        group = group.sort_values(
            "month"
        )

        values = group[
            "macro_f1"
        ].to_numpy(
            dtype=np.float64
        )

        x = np.arange(
            len(values),
            dtype=np.float64,
        )

        if len(values) > 1:
            slope = float(
                np.polyfit(
                    x,
                    values,
                    1,
                )[0]
            )

            sd_value = float(
                values.std(ddof=1)
            )
        else:
            slope = 0.0
            sd_value = 0.0

        rows.append(
            {
                "model": model_name,
                "month_count": int(
                    len(values)
                ),
                "mean_monthly_macro_f1": float(
                    values.mean()
                ),
                "sd_monthly_macro_f1": (
                    sd_value
                ),
                "min_monthly_macro_f1": float(
                    values.min()
                ),
                "max_monthly_macro_f1": float(
                    values.max()
                ),
                "macro_f1_slope_per_month": (
                    slope
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_risk_ranking():
    frame = pd.read_csv(
        RISK_RANKING_PATH
    )

    result = (
        frame.groupby(
            [
                "model",
                "fraction",
            ],
            sort=False,
        )
        .agg(
            fold_count=(
                "fold",
                "nunique",
            ),
            mean_precision_at_k=(
                "precision_at_k",
                "mean",
            ),
            sd_precision_at_k=(
                "precision_at_k",
                "std",
            ),
            mean_recall_at_k=(
                "recall_at_k",
                "mean",
            ),
            sd_recall_at_k=(
                "recall_at_k",
                "std",
            ),
            mean_lift_at_k=(
                "lift_at_k",
                "mean",
            ),
            sd_lift_at_k=(
                "lift_at_k",
                "std",
            ),
        )
        .reset_index()
    )

    return result


def summarize_calibration():
    frame = pd.read_csv(
        CALIBRATION_SUMMARY_PATH
    )

    result = (
        frame.groupby(
            "model",
            sort=False,
        )
        .agg(
            fold_count=(
                "fold",
                "nunique",
            ),
            mean_brier_score=(
                "brier_score",
                "mean",
            ),
            sd_brier_score=(
                "brier_score",
                "std",
            ),
            mean_expected_calibration_error=(
                "expected_calibration_error",
                "mean",
            ),
            sd_expected_calibration_error=(
                "expected_calibration_error",
                "std",
            ),
        )
        .reset_index()
    )

    return result


def get_significant_pairwise():
    frame = pd.read_csv(
        PAIRWISE_BOOTSTRAP_PATH
    )

    frame = frame[
        frame[
            "significant_holm_0_05"
        ] == 1
    ].copy()

    frame["absolute_delta"] = (
        frame[
            "observed_delta"
        ].abs()
    )

    frame = frame.sort_values(
        [
            "p_value_holm",
            "absolute_delta",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return frame.drop(
        columns=[
            "absolute_delta"
        ]
    )


def consolidate_final_results():
    create_project_directories()

    output_paths = [
        FINAL_FOLD_METRICS_PATH,
        FINAL_MODEL_SUMMARY_PATH,
        FINAL_EFFICIENCY_SUMMARY_PATH,
        FINAL_ABLATION_SUMMARY_PATH,
        FINAL_COMPANY_GENERALIZATION_PATH,
        FINAL_TEMPORAL_STABILITY_PATH,
        FINAL_RISK_RANKING_PATH,
        FINAL_CALIBRATION_PATH,
        FINAL_SIGNIFICANT_PAIRWISE_PATH,
    ]

    if all(
        path.exists()
        for path in output_paths
    ):
        print(
            "Final result consolidation already exists."
        )
        return

    for path in output_paths:
        if path.exists():
            path.unlink()

    print(
        "Consolidating final experimental results"
    )

    fold_metrics = load_fold_metrics()

    model_summary = summarize_models(
        fold_metrics=fold_metrics
    )

    efficiency_summary = (
        summarize_efficiency(
            fold_metrics=fold_metrics
        )
    )

    ablation_summary = (
        summarize_ablations(
            fold_metrics=fold_metrics
        )
    )

    company_generalization = (
        summarize_company_generalization()
    )

    temporal_stability = (
        summarize_temporal_stability()
    )

    risk_ranking = (
        summarize_risk_ranking()
    )

    calibration = (
        summarize_calibration()
    )

    significant_pairwise = (
        get_significant_pairwise()
    )

    fold_metrics.to_csv(
        FINAL_FOLD_METRICS_PATH,
        index=False,
    )

    model_summary.to_csv(
        FINAL_MODEL_SUMMARY_PATH,
        index=False,
    )

    efficiency_summary.to_csv(
        FINAL_EFFICIENCY_SUMMARY_PATH,
        index=False,
    )

    ablation_summary.to_csv(
        FINAL_ABLATION_SUMMARY_PATH,
        index=False,
    )

    company_generalization.to_csv(
        FINAL_COMPANY_GENERALIZATION_PATH,
        index=False,
    )

    temporal_stability.to_csv(
        FINAL_TEMPORAL_STABILITY_PATH,
        index=False,
    )

    risk_ranking.to_csv(
        FINAL_RISK_RANKING_PATH,
        index=False,
    )

    calibration.to_csv(
        FINAL_CALIBRATION_PATH,
        index=False,
    )

    significant_pairwise.to_csv(
        FINAL_SIGNIFICANT_PAIRWISE_PATH,
        index=False,
    )

    print()
    print(
        "Final result consolidation completed."
    )

    print(
        f"Saved to: "
        f"{FINAL_RESULTS_DIR}"
    )