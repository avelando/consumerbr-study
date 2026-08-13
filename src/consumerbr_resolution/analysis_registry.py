from consumerbr_resolution.config import (
    PREDICTIONS_DIR,
)


MODEL_PREDICTION_SPECS = (
    {
        "model": "global_prior",
        "directory": (
            PREDICTIONS_DIR
            / "historical_baselines"
        ),
        "filename": "fold_{fold:02d}.parquet",
        "score_column": "global_score",
        "prediction_column": "global_prediction",
    },
    {
        "model": "company_historical_rate",
        "directory": (
            PREDICTIONS_DIR
            / "historical_baselines"
        ),
        "filename": "fold_{fold:02d}.parquet",
        "score_column": "company_score",
        "prediction_column": "company_prediction",
    },
    {
        "model": "tfidf_sgd",
        "directory": (
            PREDICTIONS_DIR
            / "tfidf_sgd"
        ),
        "filename": "fold_{fold:02d}.parquet",
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "metadata_without_company",
        "directory": (
            PREDICTIONS_DIR
            / "metadata_sgd"
            / "metadata_without_company"
        ),
        "filename": "fold_{fold:02d}.parquet",
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "metadata_with_company",
        "directory": (
            PREDICTIONS_DIR
            / "metadata_sgd"
            / "metadata_with_company"
        ),
        "filename": "fold_{fold:02d}.parquet",
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "tfidf_metadata_sgd",
        "directory": (
            PREDICTIONS_DIR
            / "tfidf_metadata_sgd"
        ),
        "filename": "fold_{fold:02d}.parquet",
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "tfidf_metadata_history_sgd",
        "directory": (
            PREDICTIONS_DIR
            / "tfidf_metadata_history_sgd"
        ),
        "filename": "fold_{fold:02d}.parquet",
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "tfidf_complement_nb",
        "directory": (
            PREDICTIONS_DIR
            / "tfidf_complement_nb"
        ),
        "filename": "fold_{fold:02d}.parquet",
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "catboost",
        "directory": (
            PREDICTIONS_DIR
            / "catboost"
        ),
        "filename": (
            "fold_{fold:02d}_test.parquet"
        ),
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "bertimbau",
        "directory": (
            PREDICTIONS_DIR
            / "bertimbau"
        ),
        "filename": (
            "fold_{fold:02d}_test.parquet"
        ),
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "albertina",
        "directory": (
            PREDICTIONS_DIR
            / "albertina"
        ),
        "filename": (
            "fold_{fold:02d}_test.parquet"
        ),
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "bertimbau_catboost_fusion",
        "directory": (
            PREDICTIONS_DIR
            / "bertimbau_catboost_fusion"
        ),
        "filename": (
            "fold_{fold:02d}_test.parquet"
        ),
        "score_column": "score",
        "prediction_column": "prediction",
    },
    {
        "model": "albertina_catboost_fusion",
        "directory": (
            PREDICTIONS_DIR
            / "albertina_catboost_fusion"
        ),
        "filename": (
            "fold_{fold:02d}_test.parquet"
        ),
        "score_column": "score",
        "prediction_column": "prediction",
    },
)


def get_prediction_path(
    specification,
    fold_number,
):
    return (
        specification["directory"]
        / specification[
            "filename"
        ].format(
            fold=fold_number
        )
    )