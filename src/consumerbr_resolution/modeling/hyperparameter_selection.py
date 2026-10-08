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


def get_selected_catboost_hyperparameters():
    parameters = (
        load_selected_hyperparameters()
    )

    if "catboost" not in parameters:
        raise KeyError(
            "Selected CatBoost hyperparameters were not found. "
            "Run the CatBoost tuning stage first."
        )

    selected = parameters["catboost"]

    return {
        "iterations": int(
            selected["iterations"]
        ),
        "learning_rate": float(
            selected["learning_rate"]
        ),
        "depth": int(
            selected["depth"]
        ),
        "l2_leaf_reg": float(
            selected["l2_leaf_reg"]
        ),
    }


def get_selected_bertimbau_hyperparameters():
    parameters = (
        load_selected_hyperparameters()
    )

    if "bertimbau" not in parameters:
        raise KeyError(
            "Selected BERTimbau hyperparameters were not found. "
            "Run the BERTimbau tuning stage first."
        )

    selected = parameters["bertimbau"]

    return {
        "epochs": int(
            selected["epochs"]
        ),
        "learning_rate": float(
            selected["learning_rate"]
        ),
        "weight_decay": float(
            selected["weight_decay"]
        ),
    }


def get_selected_albertina_hyperparameters():
    parameters = (
        load_selected_hyperparameters()
    )

    if "albertina" not in parameters:
        raise KeyError(
            "Selected Albertina hyperparameters were not found. "
            "Run the Albertina tuning stage first."
        )

    selected = parameters["albertina"]

    return {
        "epochs": int(
            selected["epochs"]
        ),
        "learning_rate": float(
            selected["learning_rate"]
        ),
        "weight_decay": float(
            selected["weight_decay"]
        ),
    }