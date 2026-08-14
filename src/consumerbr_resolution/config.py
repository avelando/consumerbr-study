from pathlib import Path

from consumerbr_resolution.temporal_design import (
    generate_temporal_folds,
    validate_temporal_folds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
TFIDF_MODELS_DIR = MODELS_DIR / "tfidf"
TFIDF_CHAR_MODELS_DIR = MODELS_DIR / "tfidf_char"

METADATA_MODELS_DIR = MODELS_DIR / "metadata"
CLASSICAL_MODELS_DIR = MODELS_DIR / "classical"

TUNING_MODELS_DIR = MODELS_DIR / "tuning"

TUNING_WORD_TFIDF_PATH = (
    TUNING_MODELS_DIR
    / "word_tfidf.joblib"
)

TUNING_CHAR_TFIDF_PATH = (
    TUNING_MODELS_DIR
    / "char_tfidf.joblib"
)

TFIDF_SGD_MODELS_DIR = CLASSICAL_MODELS_DIR / "tfidf_sgd"

TFIDF_CHAR_SGD_MODELS_DIR = (
    CLASSICAL_MODELS_DIR / "tfidf_char_sgd"
)

TFIDF_WORD_CHAR_SGD_MODELS_DIR = (
    CLASSICAL_MODELS_DIR / "tfidf_word_char_sgd"
)
METADATA_SGD_MODELS_DIR = CLASSICAL_MODELS_DIR / "metadata_sgd"
TFIDF_METADATA_SGD_MODELS_DIR = (
    CLASSICAL_MODELS_DIR / "tfidf_metadata_sgd"
)
TFIDF_METADATA_HISTORY_SGD_MODELS_DIR = (
    CLASSICAL_MODELS_DIR
    / "tfidf_metadata_history_sgd"
)
TFIDF_METADATA_RECENT_HISTORY_SGD_MODELS_DIR = (
    CLASSICAL_MODELS_DIR
    / "tfidf_metadata_recent_history_sgd"
)
TFIDF_COMPLEMENT_NB_MODELS_DIR = (
    CLASSICAL_MODELS_DIR
    / "tfidf_complement_nb"
)

TABULAR_MODELS_DIR = MODELS_DIR / "tabular"

CATBOOST_MODELS_DIR = (
    TABULAR_MODELS_DIR
    / "catboost"
)

TRANSFORMER_MODELS_DIR = (
    MODELS_DIR / "transformers"
)

BERTIMBAU_MODELS_DIR = (
    TRANSFORMER_MODELS_DIR
    / "bertimbau_base"
)

BERTIMBAU_PRETRAINED_DIR = (
    BERTIMBAU_MODELS_DIR
    / "pretrained"
)

BERTIMBAU_FINETUNED_DIR = (
    BERTIMBAU_MODELS_DIR
    / "finetuned"
)

BERTIMBAU_HEAD_TAIL_FINETUNED_DIR = (
    BERTIMBAU_MODELS_DIR
    / "finetuned_head_tail_256"
)

BERTIMBAU_LONG_FINETUNED_DIR = (
    BERTIMBAU_MODELS_DIR
    / "finetuned_head_512"
)

ALBERTINA_MODELS_DIR = (
    TRANSFORMER_MODELS_DIR
    / "albertina_100m_ptbr"
)

ALBERTINA_PRETRAINED_DIR = (
    ALBERTINA_MODELS_DIR
    / "pretrained"
)

ALBERTINA_FINETUNED_DIR = (
    ALBERTINA_MODELS_DIR
    / "finetuned"
)

ALBERTINA_HEAD_TAIL_FINETUNED_DIR = (
    ALBERTINA_MODELS_DIR
    / "finetuned_head_tail_256"
)

ALBERTINA_LONG_FINETUNED_DIR = (
    ALBERTINA_MODELS_DIR
    / "finetuned_head_512"
)

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
SEED_STABILITY_DIR = (
    ANALYSIS_DIR / "seed_stability"
)

EFFECTIVE_CALIBRATION_DIR = (
    ANALYSIS_DIR / "effective_calibration"
)

FINAL_RESULTS_DIR = RESULTS_DIR / "final"

CLASSICAL_TUNING_RESULTS_PATH = (
    TABLES_DIR
    / "classical_tuning_results.csv"
)

CATBOOST_TUNING_RESULTS_PATH = (
    TABLES_DIR
    / "catboost_tuning_results.csv"
)

BERTIMBAU_TUNING_RESULTS_PATH = (
    TABLES_DIR
    / "bertimbau_tuning_results.csv"
)

ALBERTINA_TUNING_RESULTS_PATH = (
    TABLES_DIR
    / "albertina_tuning_results.csv"
)

SELECTED_HYPERPARAMETERS_PATH = (
    TABLES_DIR
    / "selected_hyperparameters.json"
)
SEED_STABILITY_METRICS_PATH = (
    SEED_STABILITY_DIR
    / "seed_stability_metrics.csv"
)

SEED_STABILITY_BY_SEED_PATH = (
    SEED_STABILITY_DIR
    / "seed_stability_by_seed.csv"
)

SEED_STABILITY_SUMMARY_PATH = (
    SEED_STABILITY_DIR
    / "seed_stability_summary.csv"
)

EFFECTIVE_CALIBRATION_FOLD_SUMMARY_PATH = (
    EFFECTIVE_CALIBRATION_DIR
    / "effective_calibration_fold_summary.csv"
)

EFFECTIVE_CALIBRATION_MODEL_SUMMARY_PATH = (
    EFFECTIVE_CALIBRATION_DIR
    / "effective_calibration_model_summary.csv"
)

EFFECTIVE_CALIBRATION_BINS_PATH = (
    EFFECTIVE_CALIBRATION_DIR
    / "effective_calibration_bins.csv"
)

LOGS_DIR = PROJECT_ROOT / "logs"

ZENODO_RECORD_ID = "18022568"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

CORPUS_ARCHIVE_NAME = "ConsumerBR.zip"
CORPUS_CSV_NAME = "ConsumerBR.csv"
CORPUS_PARQUET_NAME = "ConsumerBR.parquet"
MODELING_BASE_NAME = "consumerbr_modeling_base.parquet"
CLEAN_BASE_NAME = "consumerbr_clean_base.parquet"
FEATURE_BASE_NAME = "consumerbr_feature_base.parquet"

CORPUS_ARCHIVE_PATH = RAW_DATA_DIR / CORPUS_ARCHIVE_NAME
CORPUS_CSV_PATH = RAW_DATA_DIR / CORPUS_CSV_NAME
CORPUS_PARQUET_PATH = INTERIM_DATA_DIR / CORPUS_PARQUET_NAME
MODELING_BASE_PATH = PROCESSED_DATA_DIR / MODELING_BASE_NAME
CLEAN_BASE_PATH = PROCESSED_DATA_DIR / CLEAN_BASE_NAME
FEATURE_BASE_PATH = PROCESSED_DATA_DIR / FEATURE_BASE_NAME

COMPANY_HISTORY_DIR = PROCESSED_DATA_DIR / "company_history"

CATBOOST_TUNING_TRAIN_HISTORY_PATH = (
    COMPANY_HISTORY_DIR
    / "catboost_tuning_train.parquet"
)

CATBOOST_TUNING_VALIDATION_HISTORY_PATH = (
    COMPANY_HISTORY_DIR
    / "catboost_tuning_validation.parquet"
)

TOKENIZED_DATA_DIR = (
    INTERIM_DATA_DIR / "tokenized"
)

BERTIMBAU_TOKEN_CACHE_PATH = (
    TOKENIZED_DATA_DIR
    / "bertimbau_base_tokens.parquet"
)

BERTIMBAU_HEAD_TAIL_TOKEN_CACHE_PATH = (
    TOKENIZED_DATA_DIR
    / "bertimbau_base_head_tail_256_tokens.parquet"
)

BERTIMBAU_LONG_TOKEN_CACHE_PATH = (
    TOKENIZED_DATA_DIR
    / "bertimbau_base_head_512_tokens.parquet"
)

ALBERTINA_TOKEN_CACHE_PATH = (
    TOKENIZED_DATA_DIR
    / "albertina_100m_ptbr_tokens.parquet"
)

ALBERTINA_HEAD_TAIL_TOKEN_CACHE_PATH = (
    TOKENIZED_DATA_DIR
    / "albertina_100m_ptbr_head_tail_256_tokens.parquet"
)

ALBERTINA_LONG_TOKEN_CACHE_PATH = (
    TOKENIZED_DATA_DIR
    / "albertina_100m_ptbr_head_512_tokens.parquet"
)

RESOLVED_STATUS = "Resolvido"
UNRESOLVED_STATUS = "Não Resolvido"

MIN_TEXT_CHARS = 10
MIN_TEXT_WORDS = 2

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT = 120

VALID_UFS = (
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
)

# TEMPORAL_FOLDS = (
#     {
#         "fold": 1,
#         "train_end": "2023-06-30",
#         "validation_start": "2023-07-01",
#         "validation_end": "2023-09-30",
#         "test_start": "2023-10-01",
#         "test_end": "2023-12-31",
#     },
#     {
#         "fold": 2,
#         "train_end": "2023-09-30",
#         "validation_start": "2023-10-01",
#         "validation_end": "2023-12-31",
#         "test_start": "2024-01-01",
#         "test_end": "2024-03-31",
#     },
#     {
#         "fold": 3,
#         "train_end": "2023-12-31",
#         "validation_start": "2024-01-01",
#         "validation_end": "2024-03-31",
#         "test_start": "2024-04-01",
#         "test_end": "2024-06-30",
#     },
#     {
#         "fold": 4,
#         "train_end": "2024-03-31",
#         "validation_start": "2024-04-01",
#         "validation_end": "2024-06-30",
#         "test_start": "2024-07-01",
#         "test_end": "2024-09-30",
#     },
#     {
#         "fold": 5,
#         "train_end": "2024-06-30",
#         "validation_start": "2024-07-01",
#         "validation_end": "2024-09-30",
#         "test_start": "2024-10-01",
#         "test_end": "2024-12-31",
#     },
#     {
#         "fold": 6,
#         "train_end": "2024-09-30",
#         "validation_start": "2024-10-01",
#         "validation_end": "2024-12-31",
#         "test_start": "2025-01-01",
#         "test_end": "2025-03-31",
#     },
# )

TEMPORAL_TRAIN_START = "2021-05-01"
EXPECTED_CORPUS_OBSERVATION_END = "2025-04-03"

TUNING_TRAIN_END = "2023-03-31"
TUNING_VALIDATION_START = "2023-04-01"
TUNING_VALIDATION_END = "2023-06-30"

TEMPORAL_FIRST_VALIDATION_START = "2023-07-01"

TEMPORAL_VALIDATION_MONTHS = 3
TEMPORAL_TEST_MONTHS = 3
TEMPORAL_STEP_MONTHS = 3

TEMPORAL_FOLDS = generate_temporal_folds(
    first_validation_start=(
        TEMPORAL_FIRST_VALIDATION_START
    ),
    observation_end=(
        EXPECTED_CORPUS_OBSERVATION_END
    ),
    validation_months=(
        TEMPORAL_VALIDATION_MONTHS
    ),
    test_months=TEMPORAL_TEST_MONTHS,
    step_months=TEMPORAL_STEP_MONTHS,
)

validate_temporal_folds(TEMPORAL_FOLDS)

TUNING_TRAIN_END = "2023-03-31"
TUNING_VALIDATION_START = "2023-04-01"
TUNING_VALIDATION_END = "2023-06-30"

SGD_ALPHA_CANDIDATES = (
    1e-6,
    1e-5,
    1e-4,
)

COMPLEMENT_NB_ALPHA_CANDIDATES = (
    0.1,
    0.5,
    1.0,
)

CATBOOST_DEPTH_CANDIDATES = (
    6,
    8,
)

CATBOOST_LEARNING_RATE_CANDIDATES = (
    0.03,
    0.05,
)

CATBOOST_L2_LEAF_REG_CANDIDATES = (
    3.0,
    10.0,
)

TRANSFORMER_EPOCH_CANDIDATES = (
    1,
    2,
)

BERTIMBAU_LEARNING_RATE_CANDIDATES = (
    1e-5,
    2e-5,
)

ALBERTINA_LEARNING_RATE_CANDIDATES = (
    1e-5,
    2e-5,
)

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.90
TFIDF_MAX_FEATURES = 100_000
TFIDF_SUBLINEAR_TF = True
TFIDF_STRIP_ACCENTS = "unicode"
TFIDF_LOWERCASE = True
TFIDF_BATCH_SIZE = 10_000

TFIDF_CHAR_ANALYZER = "char_wb"
TFIDF_CHAR_NGRAM_RANGE = (3, 5)
TFIDF_CHAR_MIN_DF = 5
TFIDF_CHAR_MAX_DF = 0.95
TFIDF_CHAR_MAX_FEATURES = 100_000

RANDOM_SEED = 42

EXPERIMENT_SEEDS = (
    13,
    21,
    42,
)

PRIMARY_EXPERIMENT_SEED = (
    RANDOM_SEED
)

SGD_LOSS = "log_loss"
SGD_PENALTY = "l2"
SGD_EPOCHS = 5
SGD_BATCH_SIZE = 10_000

CATBOOST_ITERATIONS = 1000
CATBOOST_LEARNING_RATE = 0.05
CATBOOST_DEPTH = 8
CATBOOST_L2_LEAF_REG = 3.0
CATBOOST_EARLY_STOPPING_ROUNDS = 50
CATBOOST_TASK_TYPE = "GPU"
CATBOOST_DEVICES = "0"
CATBOOST_GPU_RAM_PART = 0.80

BERTIMBAU_MODEL_NAME = (
    "neuralmind/bert-base-portuguese-cased"
)

BERTIMBAU_REVISION = (
    "74364c8dbc30e651fee36aa714a772dcaae83815"
)

BERTIMBAU_MAX_LENGTH = 256
BERTIMBAU_LONG_MAX_LENGTH = 512

BERTIMBAU_TOKENIZATION_BATCH_SIZE = 2_048

BERTIMBAU_EPOCHS = 1
BERTIMBAU_TRAIN_BATCH_SIZE = 8
BERTIMBAU_EVAL_BATCH_SIZE = 32
BERTIMBAU_GRADIENT_ACCUMULATION_STEPS = 4
BERTIMBAU_LONG_TRAIN_BATCH_SIZE = 4
BERTIMBAU_LONG_EVAL_BATCH_SIZE = 16
BERTIMBAU_LONG_GRADIENT_ACCUMULATION_STEPS = 8
BERTIMBAU_LEARNING_RATE = 2e-5
BERTIMBAU_WEIGHT_DECAY = 0.01
BERTIMBAU_WARMUP_RATIO = 0.10
BERTIMBAU_MAX_GRAD_NORM = 1.0
BERTIMBAU_USE_AMP = True
BERTIMBAU_GRADIENT_CHECKPOINTING = True
BERTIMBAU_ARROW_BATCH_SIZE = 4_096

ALBERTINA_MODEL_NAME = (
    "PORTULAN/albertina-100m-portuguese-ptbr-encoder"
)

ALBERTINA_REVISION = (
    "d2a39d3d82408cc2a2618beb57a4fe908ee44b27"
)

ALBERTINA_MAX_LENGTH = 256

ALBERTINA_LONG_MAX_LENGTH = 512

ALBERTINA_TOKENIZATION_BATCH_SIZE = 2_048

ALBERTINA_EPOCHS = 1
ALBERTINA_TRAIN_BATCH_SIZE = 8
ALBERTINA_EVAL_BATCH_SIZE = 32
ALBERTINA_GRADIENT_ACCUMULATION_STEPS = 4
ALBERTINA_LONG_TRAIN_BATCH_SIZE = 4
ALBERTINA_LONG_EVAL_BATCH_SIZE = 16
ALBERTINA_LONG_GRADIENT_ACCUMULATION_STEPS = 8
ALBERTINA_LEARNING_RATE = 2e-5
ALBERTINA_WEIGHT_DECAY = 0.01
ALBERTINA_WARMUP_RATIO = 0.10
ALBERTINA_MAX_GRAD_NORM = 1.0
ALBERTINA_USE_AMP = True
ALBERTINA_GRADIENT_CHECKPOINTING = True
ALBERTINA_ARROW_BATCH_SIZE = 4_096

LATE_FUSION_WEIGHTS = tuple(
    index / 20
    for index in range(21)
)

RISK_RANKING_FRACTIONS = (
    0.01,
    0.05,
    0.10,
    0.20,
)

COMPANY_HISTORY_WINDOWS_DAYS = (
    30,
    90,
    180,
    365,
)

CALIBRATION_BIN_COUNT = 10
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
PERMUTATION_REPLICATES = 10_000

COMPANY_MIN_FREQUENCY = 100
RARE_COMPANY_LABEL = "RARE_OR_UNSEEN"

METADATA_NUMERIC_FEATURES = (
    "text_char_count",
    "text_word_count",
    "log_text_char_count",
    "log_text_word_count",
    "exclamation_count",
    "question_count",
    "anonymization_marker_count",
    "has_exclamation",
    "has_question",
    "has_anonymization_marker",
    "opening_month_sin",
    "opening_month_cos",
    "opening_weekday_sin",
    "opening_weekday_cos",
)

def create_project_directories():
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        COMPANY_HISTORY_DIR,
        TOKENIZED_DATA_DIR,
        MODELS_DIR,
        TFIDF_MODELS_DIR,
        TFIDF_CHAR_MODELS_DIR,
        METADATA_MODELS_DIR,
        CLASSICAL_MODELS_DIR,
        TUNING_MODELS_DIR,
        TFIDF_SGD_MODELS_DIR,
        TFIDF_CHAR_SGD_MODELS_DIR,
        TFIDF_WORD_CHAR_SGD_MODELS_DIR,
        METADATA_SGD_MODELS_DIR,
        TFIDF_METADATA_SGD_MODELS_DIR,
        TFIDF_METADATA_HISTORY_SGD_MODELS_DIR,
        TFIDF_METADATA_RECENT_HISTORY_SGD_MODELS_DIR,
        TFIDF_COMPLEMENT_NB_MODELS_DIR,
        TABULAR_MODELS_DIR,
        CATBOOST_MODELS_DIR,
        TRANSFORMER_MODELS_DIR,
        BERTIMBAU_MODELS_DIR,
        BERTIMBAU_PRETRAINED_DIR,
        BERTIMBAU_FINETUNED_DIR,
        BERTIMBAU_HEAD_TAIL_FINETUNED_DIR,
        BERTIMBAU_LONG_FINETUNED_DIR,
        ALBERTINA_MODELS_DIR,
        ALBERTINA_PRETRAINED_DIR,
        ALBERTINA_FINETUNED_DIR,
        ALBERTINA_HEAD_TAIL_FINETUNED_DIR,
        ALBERTINA_LONG_FINETUNED_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        ANALYSIS_DIR,
        SEED_STABILITY_DIR,
        EFFECTIVE_CALIBRATION_DIR,
        FINAL_RESULTS_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)