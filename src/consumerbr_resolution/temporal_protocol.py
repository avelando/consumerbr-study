import csv
from datetime import date, timedelta

import duckdb

from consumerbr_resolution.config import (
    EXPECTED_CORPUS_OBSERVATION_END,
    FEATURE_BASE_PATH,
    TABLES_DIR,
    TEMPORAL_FIRST_VALIDATION_START,
    TEMPORAL_FOLDS,
    TEMPORAL_STEP_MONTHS,
    TEMPORAL_TEST_MONTHS,
    TEMPORAL_TRAIN_START,
    TEMPORAL_VALIDATION_MONTHS,
    TUNING_TRAIN_END,
    TUNING_VALIDATION_END,
    TUNING_VALIDATION_START,
    create_project_directories,
)
from consumerbr_resolution.temporal_design import (
    generate_temporal_folds,
    generate_test_window_candidates,
    validate_temporal_folds,
)


TEMPORAL_PROTOCOL_PATH = (
    TABLES_DIR / "temporal_protocol.csv"
)

TEMPORAL_FOLD_SUMMARY_PATH = (
    TABLES_DIR / "temporal_fold_summary.csv"
)

TEMPORAL_PROTOCOL_AUDIT_PATH = (
    TABLES_DIR / "temporal_protocol_audit.csv"
)

TEMPORAL_TEST_WINDOW_ELIGIBILITY_PATH = (
    TABLES_DIR
    / "temporal_test_window_eligibility.csv"
)


def write_csv(
    path,
    fieldnames,
    rows,
):
    temporary_path = path.with_suffix(
        path.suffix + ".part"
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
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


def get_dataset_bounds(
    connection,
    source_path,
):
    row = connection.execute(
        f"""
        SELECT
            COUNT(*),
            MIN(opening_date),
            MAX(opening_date)
        FROM read_parquet('{source_path}')
        """
    ).fetchone()

    return {
        "complaint_count": int(row[0]),
        "first_opening_date": row[1],
        "last_opening_date": row[2],
    }


def get_split_summary(
    connection,
    source_path,
    fold_number,
    split,
    start_date,
    end_date,
):
    row = connection.execute(
        f"""
        SELECT
            COUNT(*),
            SUM(
                CASE
                    WHEN target_resolved = 1
                        THEN 1
                    ELSE 0
                END
            ),
            SUM(
                CASE
                    WHEN target_resolved = 0
                        THEN 1
                    ELSE 0
                END
            ),
            AVG(target_resolved),
            MIN(opening_date),
            MAX(opening_date)
        FROM read_parquet('{source_path}')
        WHERE opening_date BETWEEN
            DATE '{start_date}'
            AND DATE '{end_date}'
        """
    ).fetchone()

    return {
        "fold": fold_number,
        "split": split,
        "start_date": start_date,
        "end_date": end_date,
        "complaint_count": int(row[0]),
        "resolved_count": int(
            row[1] or 0
        ),
        "unresolved_count": int(
            row[2] or 0
        ),
        "resolution_rate": (
            float(row[3])
            if row[3] is not None
            else None
        ),
        "first_opening_date": row[4],
        "last_opening_date": row[5],
    }


def build_temporal_protocol():
    create_project_directories()

    if not FEATURE_BASE_PATH.exists():
        raise FileNotFoundError(
            "Feature base was not found: "
            f"{FEATURE_BASE_PATH}"
        )

    print(
        "Building and auditing temporal "
        "evaluation protocol"
    )

    print(
        f"Source: {FEATURE_BASE_PATH}"
    )

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    connection = duckdb.connect()

    try:
        bounds = get_dataset_bounds(
            connection=connection,
            source_path=source_path,
        )

        dataset_start = (
            bounds[
                "first_opening_date"
            ]
        )

        dataset_end = (
            bounds[
                "last_opening_date"
            ]
        )

        if (
            dataset_start is None
            or dataset_end is None
        ):
            raise RuntimeError(
                "Feature base is empty."
            )

        observed_start = (
            dataset_start.isoformat()
        )

        observed_end = (
            dataset_end.isoformat()
        )

        if (
            observed_start
            != TEMPORAL_TRAIN_START
        ):
            raise RuntimeError(
                "Observed dataset start does "
                "not match configuration: "
                f"observed={observed_start}, "
                f"configured="
                f"{TEMPORAL_TRAIN_START}."
            )

        if (
            observed_end
            != EXPECTED_CORPUS_OBSERVATION_END
        ):
            raise RuntimeError(
                "Observed dataset end does "
                "not match the expected "
                "corpus end: "
                f"observed={observed_end}, "
                f"expected="
                f"{EXPECTED_CORPUS_OBSERVATION_END}."
            )

        actual_folds = (
            generate_temporal_folds(
                first_validation_start=(
                    TEMPORAL_FIRST_VALIDATION_START
                ),
                observation_end=(
                    observed_end
                ),
                validation_months=(
                    TEMPORAL_VALIDATION_MONTHS
                ),
                test_months=(
                    TEMPORAL_TEST_MONTHS
                ),
                step_months=(
                    TEMPORAL_STEP_MONTHS
                ),
            )
        )

        validate_temporal_folds(
            actual_folds
        )

        if actual_folds != TEMPORAL_FOLDS:
            raise RuntimeError(
                "Configured temporal folds "
                "do not match the folds "
                "supported by the dataset."
            )

        tuning_train_end = date.fromisoformat(
            TUNING_TRAIN_END
        )

        tuning_validation_start = (
            date.fromisoformat(
                TUNING_VALIDATION_START
            )
        )

        tuning_end = date.fromisoformat(
            TUNING_VALIDATION_END
        )

        first_evaluation_date = (
            date.fromisoformat(
                TEMPORAL_FIRST_VALIDATION_START
            )
        )

        if (
            tuning_train_end
            + timedelta(days=1)
            != tuning_validation_start
        ):
            raise RuntimeError(
                "Tuning train and validation "
                "windows are not contiguous."
            )

        if (
            tuning_end
            + timedelta(days=1)
            != first_evaluation_date
        ):
            raise RuntimeError(
                "Tuning validation and temporal "
                "evaluation are not contiguous."
            )

        protocol_rows = []
        fold_summary_rows = []

        for fold in actual_folds:
            fold_number = fold["fold"]

            protocol_rows.append(
                {
                    "fold": fold_number,
                    "train_start": (
                        TEMPORAL_TRAIN_START
                    ),
                    "train_end": (
                        fold["train_end"]
                    ),
                    "validation_start": (
                        fold[
                            "validation_start"
                        ]
                    ),
                    "validation_end": (
                        fold[
                            "validation_end"
                        ]
                    ),
                    "test_start": (
                        fold["test_start"]
                    ),
                    "test_end": (
                        fold["test_end"]
                    ),
                }
            )

            split_specs = (
                (
                    "train",
                    TEMPORAL_TRAIN_START,
                    fold["train_end"],
                ),
                (
                    "validation",
                    fold[
                        "validation_start"
                    ],
                    fold[
                        "validation_end"
                    ],
                ),
                (
                    "test",
                    fold["test_start"],
                    fold["test_end"],
                ),
            )

            for (
                split,
                start_date,
                end_date,
            ) in split_specs:
                summary = (
                    get_split_summary(
                        connection=connection,
                        source_path=source_path,
                        fold_number=(
                            fold_number
                        ),
                        split=split,
                        start_date=(
                            start_date
                        ),
                        end_date=end_date,
                    )
                )

                if (
                    summary[
                        "complaint_count"
                    ]
                    == 0
                ):
                    raise RuntimeError(
                        f"Fold {fold_number} "
                        f"{split} split is empty."
                    )

                if (
                    summary[
                        "resolved_count"
                    ]
                    == 0
                ):
                    raise RuntimeError(
                        f"Fold {fold_number} "
                        f"{split} has no "
                        "resolved cases."
                    )

                if (
                    summary[
                        "unresolved_count"
                    ]
                    == 0
                ):
                    raise RuntimeError(
                        f"Fold {fold_number} "
                        f"{split} has no "
                        "unresolved cases."
                    )

                fold_summary_rows.append(
                    summary
                )

        candidates = (
            generate_test_window_candidates(
                first_test_start=(
                    actual_folds[0][
                        "test_start"
                    ]
                ),
                observation_end=(
                    observed_end
                ),
                test_months=(
                    TEMPORAL_TEST_MONTHS
                ),
                step_months=(
                    TEMPORAL_STEP_MONTHS
                ),
            )
        )

        eligibility_rows = []

        for candidate in candidates:
            row = connection.execute(
                f"""
                SELECT
                    COUNT(*),
                    MIN(opening_date),
                    MAX(opening_date)
                FROM read_parquet(
                    '{source_path}'
                )
                WHERE opening_date BETWEEN
                    DATE '{
                        candidate[
                            "test_start"
                        ]
                    }'
                    AND DATE '{
                        candidate[
                            "test_end"
                        ]
                    }'
                """
            ).fetchone()

            eligibility_rows.append(
                {
                    "candidate": (
                        candidate[
                            "candidate"
                        ]
                    ),
                    "test_start": (
                        candidate[
                            "test_start"
                        ]
                    ),
                    "test_end": (
                        candidate[
                            "test_end"
                        ]
                    ),
                    "complete": (
                        candidate[
                            "complete"
                        ]
                    ),
                    "included": (
                        candidate[
                            "included"
                        ]
                    ),
                    "complaint_count": (
                        int(row[0])
                    ),
                    "first_opening_date": (
                        row[1]
                    ),
                    "last_opening_date": (
                        row[2]
                    ),
                }
            )

        complete_candidates = [
            row
            for row
            in eligibility_rows
            if row["complete"]
        ]

        incomplete_candidates = [
            row
            for row
            in eligibility_rows
            if not row["complete"]
        ]

        if (
            len(complete_candidates)
            != len(actual_folds)
        ):
            raise RuntimeError(
                "Complete test-window count "
                "does not match generated "
                "fold count."
            )

        if (
            len(incomplete_candidates)
            != 1
        ):
            raise RuntimeError(
                "Expected exactly one next "
                "incomplete test window."
            )

        expected_numbers = list(
            range(
                1,
                len(actual_folds) + 1,
            )
        )

        observed_numbers = [
            fold["fold"]
            for fold
            in actual_folds
        ]

        next_candidate = (
            incomplete_candidates[0]
        )

        audit_rows = [
            {
                "criterion": (
                    "dataset_start_matches_configuration"
                ),
                "value": observed_start,
                "passed": (
                    observed_start
                    == TEMPORAL_TRAIN_START
                ),
            },
            {
                "criterion": (
                    "dataset_end_matches_expected_corpus_end"
                ),
                "value": observed_end,
                "passed": (
                    observed_end
                    == EXPECTED_CORPUS_OBSERVATION_END
                ),
            },
            {
                "criterion": (
                    "tuning_precedes_evaluation"
                ),
                "value": (
                    f"{TUNING_VALIDATION_END} "
                    f"< "
                    f"{TEMPORAL_FIRST_VALIDATION_START}"
                ),
                "passed": (
                    tuning_end
                    < first_evaluation_date
                ),
            },
            {
                "criterion": (
                    "validation_window_months"
                ),
                "value": (
                    TEMPORAL_VALIDATION_MONTHS
                ),
                "passed": (
                    TEMPORAL_VALIDATION_MONTHS
                    == 3
                ),
            },
            {
                "criterion": (
                    "test_window_months"
                ),
                "value": (
                    TEMPORAL_TEST_MONTHS
                ),
                "passed": (
                    TEMPORAL_TEST_MONTHS
                    == 3
                ),
            },
            {
                "criterion": (
                    "step_months"
                ),
                "value": (
                    TEMPORAL_STEP_MONTHS
                ),
                "passed": (
                    TEMPORAL_STEP_MONTHS
                    == 3
                ),
            },
            {
                "criterion": (
                    "fold_numbering_is_contiguous"
                ),
                "value": ",".join(
                    map(
                        str,
                        observed_numbers,
                    )
                ),
                "passed": (
                    observed_numbers
                    == expected_numbers
                ),
            },
            {
                "criterion": (
                    "complete_test_window_count"
                ),
                "value": (
                    len(
                        complete_candidates
                    )
                ),
                "passed": (
                    len(
                        complete_candidates
                    )
                    == len(actual_folds)
                ),
            },
            {
                "criterion": (
                    "next_test_window_is_incomplete"
                ),
                "value": (
                    f"{next_candidate['test_start']}"
                    f".."
                    f"{next_candidate['test_end']}"
                ),
                "passed": (
                    not next_candidate[
                        "complete"
                    ]
                ),
            },
        ]

        if not all(
            bool(row["passed"])
            for row
            in audit_rows
        ):
            raise RuntimeError(
                "Temporal protocol audit failed."
            )

    finally:
        connection.close()

    write_csv(
        TEMPORAL_PROTOCOL_PATH,
        [
            "fold",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        ],
        protocol_rows,
    )

    write_csv(
        TEMPORAL_FOLD_SUMMARY_PATH,
        [
            "fold",
            "split",
            "start_date",
            "end_date",
            "complaint_count",
            "resolved_count",
            "unresolved_count",
            "resolution_rate",
            "first_opening_date",
            "last_opening_date",
        ],
        fold_summary_rows,
    )

    write_csv(
        TEMPORAL_TEST_WINDOW_ELIGIBILITY_PATH,
        [
            "candidate",
            "test_start",
            "test_end",
            "complete",
            "included",
            "complaint_count",
            "first_opening_date",
            "last_opening_date",
        ],
        eligibility_rows,
    )

    write_csv(
        TEMPORAL_PROTOCOL_AUDIT_PATH,
        [
            "criterion",
            "value",
            "passed",
        ],
        audit_rows,
    )

    print(
        "Temporal protocol completed "
        "and audited."
    )

    print(
        f"Saved to: "
        f"{TEMPORAL_PROTOCOL_PATH}"
    )

    print(
        f"Saved to: "
        f"{TEMPORAL_FOLD_SUMMARY_PATH}"
    )

    print(
        f"Saved to: "
        f"{TEMPORAL_TEST_WINDOW_ELIGIBILITY_PATH}"
    )

    print(
        f"Saved to: "
        f"{TEMPORAL_PROTOCOL_AUDIT_PATH}"
    )