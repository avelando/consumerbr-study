import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_macro_f1_threshold(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)

    order = np.argsort(-y_score, kind="mergesort")

    sorted_true = y_true[order]
    sorted_score = y_score[order]

    positive_total = sorted_true.sum()
    negative_total = len(sorted_true) - positive_total

    true_positive = np.cumsum(sorted_true)
    false_positive = np.cumsum(1 - sorted_true)

    group_end = np.concatenate(
        (
            sorted_score[:-1] != sorted_score[1:],
            np.array([True]),
        )
    )

    indices = np.flatnonzero(group_end)

    true_positive = true_positive[indices]
    false_positive = false_positive[indices]

    false_negative = positive_total - true_positive
    true_negative = negative_total - false_positive

    positive_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    negative_denominator = (
        2 * true_negative
        + false_positive
        + false_negative
    )

    positive_f1 = np.divide(
        2 * true_positive,
        positive_denominator,
        out=np.zeros_like(
            true_positive,
            dtype=np.float64,
        ),
        where=positive_denominator != 0,
    )

    negative_f1 = np.divide(
        2 * true_negative,
        negative_denominator,
        out=np.zeros_like(
            true_negative,
            dtype=np.float64,
        ),
        where=negative_denominator != 0,
    )

    macro_f1 = (
        positive_f1
        + negative_f1
    ) / 2.0

    thresholds = sorted_score[indices]

    all_negative_threshold = np.nextafter(
        sorted_score[0],
        np.inf,
    )

    all_negative_positive_f1 = 0.0
    all_negative_negative_f1 = (
        2 * negative_total
        / (
            2 * negative_total
            + positive_total
        )
    )

    thresholds = np.concatenate(
        (
            np.array([all_negative_threshold]),
            thresholds,
        )
    )

    macro_f1 = np.concatenate(
        (
            np.array(
                [
                    (
                        all_negative_positive_f1
                        + all_negative_negative_f1
                    )
                    / 2.0
                ]
            ),
            macro_f1,
        )
    )

    best_macro_f1 = macro_f1.max()

    best_indices = np.flatnonzero(
        np.isclose(
            macro_f1,
            best_macro_f1,
            rtol=0.0,
            atol=1e-12,
        )
    )

    distances = np.abs(
        thresholds[best_indices] - 0.5
    )

    best_index = best_indices[
        np.argmin(distances)
    ]

    return (
        float(thresholds[best_index]),
        float(macro_f1[best_index]),
    )


def calculate_binary_metrics(
    y_true,
    y_score,
    threshold,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float64,
    )

    y_pred = (
        y_score >= threshold
    ).astype(np.int8)

    unique_classes = np.unique(y_true)

    if len(unique_classes) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_score,
        )
    else:
        roc_auc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision_resolved": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_resolved": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_resolved": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "precision_unresolved": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "recall_unresolved": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_unresolved": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                y_score,
            )
        ),
    }