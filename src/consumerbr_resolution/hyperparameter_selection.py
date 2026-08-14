import json

from consumerbr_resolution.config import (
    SELECTED_HYPERPARAMETERS_PATH,
)


def load_selected_hyperparameters():
    if not SELECTED_HYPERPARAMETERS_PATH.exists():
        raise FileNotFoundError(
            "Selected hyperparameters were not found. "
            "Run the classical tuning stage first."
        )

    with SELECTED_HYPERPARAMETERS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_selected_sgd_alpha():
    parameters = (
        load_selected_hyperparameters()
    )

    return float(
        parameters["sgd"]["alpha"]
    )


def get_selected_complement_nb_alpha():
    parameters = (
        load_selected_hyperparameters()
    )

    return float(
        parameters[
            "complement_nb"
        ]["alpha"]
    )