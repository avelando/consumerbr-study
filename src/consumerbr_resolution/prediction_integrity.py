import csv

import duckdb

from consumerbr_resolution.analysis_registry import (
    MODEL_PREDICTION_SPECS,
    get_prediction_path,
)
from consumerbr_resolution.config import (
    FEATURE_BASE_PATH,
    TABLES_DIR,
    TEMPORAL_FOLDS,
    create_project_directories,
)


PREDICTION_INTEGRITY_PATH = (
    TABLES_DIR
    / "prediction_integrity_audit.csv"
)


AUDIT_FIELDS = [
    "model",
    "fold",
    "expected_count",
    "prediction_count",
    "unique_prediction_record_count",
    "missing_record_count",
    "unexpected_record_count",
    "target_mismatch_count",
    "null_score_count",
    "invalid_score_count",
    "null_prediction_count",
    "invalid_prediction_count",
    "passed",
]


def audit_model_fold(
    connection,
    specification,
    fold,
):
    fold_number = fold["fold"]

    prediction_path = get_prediction_path(
        specification=specification,
        fold_number=fold_number,
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            "Prediction file not found: "
            f"{prediction_path}"
        )

    feature_source = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    prediction_source = str(
        prediction_path
    ).replace("'", "''")

    score_column = specification[
        "score_column"
    ]

    prediction_column = specification[
        "prediction_column"
    ]

    expected_count = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet(
                '{feature_source}'
            )
            WHERE opening_date BETWEEN
                DATE '{fold["test_start"]}'
                AND DATE '{fold["test_end"]}'
            """
        ).fetchone()[0]
    )

    stats = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT record_id),
            SUM(
                CASE
                    WHEN {score_column} IS NULL
                        THEN 1
                    ELSE 0
                END
            ),
            SUM(
                CASE
                    WHEN {score_column} IS NOT NULL
                        AND (
                            {score_column} < 0.0
                            OR {score_column} > 1.0
                        )
                        THEN 1
                    ELSE 0
                END
            ),
            SUM(
                CASE
                    WHEN {prediction_column} IS NULL
                        THEN 1
                    ELSE 0
                END
            ),
            SUM(
                CASE
                    WHEN {prediction_column} IS NOT NULL
                        AND {prediction_column}
                            NOT IN (0, 1)
                        THEN 1
                    ELSE 0
                END
            )
        FROM read_parquet(
            '{prediction_source}'
        )
        """
    ).fetchone()

    alignment = connection.execute(
        f"""
        WITH expected AS (
            SELECT
                record_id,
                CAST(
                    target_resolved
                    AS TINYINT
                ) AS target_resolved
            FROM read_parquet(
                '{feature_source}'
            )
            WHERE opening_date BETWEEN
                DATE '{fold["test_start"]}'
                AND DATE '{fold["test_end"]}'
        ),
        prediction AS (
            SELECT
                record_id,
                CAST(
                    target_resolved
                    AS TINYINT
                ) AS target_resolved
            FROM read_parquet(
                '{prediction_source}'
            )
        )
        SELECT
            SUM(
                CASE
                    WHEN prediction.record_id
                        IS NULL
                        THEN 1
                    ELSE 0
                END
            ),
            SUM(
                CASE
                    WHEN expected.record_id
                        IS NULL
                        THEN 1
                    ELSE 0
                END
            ),
            SUM(
                CASE
                    WHEN expected.record_id
                        IS NOT NULL
                        AND prediction.record_id
                        IS NOT NULL
                        AND expected.target_resolved
                            != prediction.target_resolved
                        THEN 1
                    ELSE 0
                END
            )
        FROM expected
        FULL OUTER JOIN prediction
            ON expected.record_id
                = prediction.record_id
        """
    ).fetchone()

    prediction_count = int(stats[0])

    unique_prediction_record_count = int(
        stats[1]
    )

    missing_record_count = int(
        alignment[0] or 0
    )

    unexpected_record_count = int(
        alignment[1] or 0
    )

    target_mismatch_count = int(
        alignment[2] or 0
    )

    null_score_count = int(
        stats[2] or 0
    )

    invalid_score_count = int(
        stats[3] or 0
    )

    null_prediction_count = int(
        stats[4] or 0
    )

    invalid_prediction_count = int(
        stats[5] or 0
    )

    passed = (
        expected_count > 0
        and prediction_count
        == expected_count
        and unique_prediction_record_count
        == prediction_count
        and missing_record_count == 0
        and unexpected_record_count == 0
        and target_mismatch_count == 0
        and null_score_count == 0
        and invalid_score_count == 0
        and null_prediction_count == 0
        and invalid_prediction_count == 0
    )

    return {
        "model": specification["model"],
        "fold": fold_number,
        "expected_count": expected_count,
        "prediction_count": prediction_count,
        "unique_prediction_record_count": (
            unique_prediction_record_count
        ),
        "missing_record_count": (
            missing_record_count
        ),
        "unexpected_record_count": (
            unexpected_record_count
        ),
        "target_mismatch_count": (
            target_mismatch_count
        ),
        "null_score_count": (
            null_score_count
        ),
        "invalid_score_count": (
            invalid_score_count
        ),
        "null_prediction_count": (
            null_prediction_count
        ),
        "invalid_prediction_count": (
            invalid_prediction_count
        ),
        "passed": passed,
    }


def validate_prediction_integrity():
    create_project_directories()

    if not FEATURE_BASE_PATH.exists():
        raise FileNotFoundError(
            "Feature base was not found: "
            f"{FEATURE_BASE_PATH}"
        )

    print(
        "Validating test-prediction integrity"
    )

    rows = []

    connection = duckdb.connect()

    try:
        for specification in (
            MODEL_PREDICTION_SPECS
        ):
            for fold in TEMPORAL_FOLDS:
                row = audit_model_fold(
                    connection=connection,
                    specification=(
                        specification
                    ),
                    fold=fold,
                )

                rows.append(row)

                if not row["passed"]:
                    raise RuntimeError(
                        "Prediction integrity "
                        "validation failed for "
                        f"model={row['model']}, "
                        f"fold={row['fold']}."
                    )

    finally:
        connection.close()

    expected_row_count = (
        len(MODEL_PREDICTION_SPECS)
        * len(TEMPORAL_FOLDS)
    )

    if len(rows) != expected_row_count:
        raise RuntimeError(
            "Prediction integrity audit did "
            "not cover every model-fold pair."
        )

    temporary_path = (
        PREDICTION_INTEGRITY_PATH
        .with_suffix(".csv.part")
    )

    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=AUDIT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(
        PREDICTION_INTEGRITY_PATH
    )

    print(
        "Prediction integrity validation "
        "completed."
    )

    print(
        f"Validated model-fold pairs: "
        f"{len(rows)}"
    )

    print(
        f"Saved to: "
        f"{PREDICTION_INTEGRITY_PATH}"
    )