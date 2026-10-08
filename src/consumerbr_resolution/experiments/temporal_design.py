from datetime import date, timedelta


def add_months(value, months):
    month_index = value.month - 1 + months
    return date(value.year + month_index // 12, month_index % 12 + 1, 1)


def build_single_temporal_fold(
    train_start, train_end, validation_start, validation_end, test_start, test_end
):
    folds = ({
        "fold": 1,
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test_end,
    },)
    validate_temporal_folds(folds)
    return folds


def validate_temporal_folds(folds):
    if len(folds) != 1 or folds[0]["fold"] != 1:
        raise ValueError("The reduced protocol requires exactly one fold numbered 1.")
    fold = folds[0]
    windows = []
    for split in ("train", "validation", "test"):
        start = date.fromisoformat(fold[f"{split}_start"])
        end = date.fromisoformat(fold[f"{split}_end"])
        if start > end:
            raise ValueError(f"Invalid {split} window.")
        if start.day != 1 or (end + timedelta(days=1)).day != 1:
            raise ValueError(f"The {split} window must cover complete calendar months.")
        windows.append((start, end))
    for previous, following in zip(windows, windows[1:]):
        expected_start = add_months(previous[1] + timedelta(days=1), 1)
        if following[0] != expected_start:
            raise ValueError("Adjacent partitions require one excluded calendar month.")


def get_temporal_windows(fold, observation_end):
    validate_temporal_folds((fold,))
    observation_end = date.fromisoformat(observation_end)
    test_end = date.fromisoformat(fold["test_end"])
    if observation_end < test_end:
        raise ValueError("The observation horizon does not cover the test window.")
    windows = []
    for split in ("train", "validation", "test"):
        windows.append({
            "split": split,
            "start_date": fold[f"{split}_start"],
            "end_date": fold[f"{split}_end"],
            "included": True,
        })
    for name, previous, following in (
        ("gap_train_validation", "train", "validation"),
        ("gap_validation_test", "validation", "test"),
    ):
        windows.append({
            "split": name,
            "start_date": (
                date.fromisoformat(fold[f"{previous}_end"]) + timedelta(days=1)
            ).isoformat(),
            "end_date": (
                date.fromisoformat(fold[f"{following}_start"]) - timedelta(days=1)
            ).isoformat(),
            "included": False,
        })
    if observation_end > test_end:
        windows.append({
            "split": "after_test",
            "start_date": (test_end + timedelta(days=1)).isoformat(),
            "end_date": observation_end.isoformat(),
            "included": False,
        })
    return tuple(sorted(windows, key=lambda item: item["start_date"]))
