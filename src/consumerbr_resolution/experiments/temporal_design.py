from datetime import date, timedelta


def add_months(value, months):
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1

    return date(year, month, 1)


def generate_temporal_folds(
    first_validation_start,
    observation_end,
    validation_months,
    test_months,
    step_months,
):
    validation_start = date.fromisoformat(
        first_validation_start
    )
    dataset_end = date.fromisoformat(observation_end)

    folds = []
    fold_number = 1

    while True:
        train_end = (
            validation_start
            - timedelta(days=1)
        )

        validation_end = (
            add_months(
                validation_start,
                validation_months,
            )
            - timedelta(days=1)
        )

        test_start = (
            validation_end
            + timedelta(days=1)
        )

        test_end = (
            add_months(
                test_start,
                test_months,
            )
            - timedelta(days=1)
        )

        if test_end > dataset_end:
            break

        folds.append(
            {
                "fold": fold_number,
                "train_end": train_end.isoformat(),
                "validation_start": (
                    validation_start.isoformat()
                ),
                "validation_end": (
                    validation_end.isoformat()
                ),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
            }
        )

        validation_start = add_months(
            validation_start,
            step_months,
        )

        fold_number += 1

    return tuple(folds)


def generate_test_window_candidates(
    first_test_start,
    observation_end,
    test_months,
    step_months,
):
    test_start = date.fromisoformat(
        first_test_start
    )
    dataset_end = date.fromisoformat(
        observation_end
    )

    candidates = []
    candidate_number = 1

    while True:
        test_end = (
            add_months(
                test_start,
                test_months,
            )
            - timedelta(days=1)
        )

        complete = test_end <= dataset_end

        candidates.append(
            {
                "candidate": candidate_number,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "complete": complete,
                "included": complete,
            }
        )

        if not complete:
            break

        test_start = add_months(
            test_start,
            step_months,
        )

        candidate_number += 1

    return tuple(candidates)


def validate_temporal_folds(folds):
    if not folds:
        raise ValueError(
            "Temporal protocol produced no folds."
        )

    previous_test_end = None

    for expected_fold, fold in enumerate(
        folds,
        start=1,
    ):
        fold_number = fold["fold"]

        train_end = date.fromisoformat(
            fold["train_end"]
        )

        validation_start = date.fromisoformat(
            fold["validation_start"]
        )

        validation_end = date.fromisoformat(
            fold["validation_end"]
        )

        test_start = date.fromisoformat(
            fold["test_start"]
        )

        test_end = date.fromisoformat(
            fold["test_end"]
        )

        if fold_number != expected_fold:
            raise ValueError(
                "Temporal fold numbering is not contiguous."
            )

        if validation_start > validation_end:
            raise ValueError(
                "Temporal validation window is invalid."
            )

        if test_start > test_end:
            raise ValueError(
                "Temporal test window is invalid."
            )

        if (
            train_end
            + timedelta(days=1)
            != validation_start
        ):
            raise ValueError(
                "Training and validation are not "
                "temporally contiguous."
            )

        if (
            validation_end
            + timedelta(days=1)
            != test_start
        ):
            raise ValueError(
                "Validation and test are not "
                "temporally contiguous."
            )

        if (
            previous_test_end is not None
            and test_start <= previous_test_end
        ):
            raise ValueError(
                "Test windows overlap."
            )

        previous_test_end = test_end