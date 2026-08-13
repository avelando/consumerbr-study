from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
TFIDF_MODELS_DIR = MODELS_DIR / "tfidf"
CLASSICAL_MODELS_DIR = MODELS_DIR / "classical"
TFIDF_SGD_MODELS_DIR = CLASSICAL_MODELS_DIR / "tfidf_sgd"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

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

TEMPORAL_FOLDS = (
    {
        "fold": 1,
        "train_end": "2023-06-30",
        "validation_start": "2023-07-01",
        "validation_end": "2023-09-30",
        "test_start": "2023-10-01",
        "test_end": "2023-12-31",
    },
    {
        "fold": 2,
        "train_end": "2023-09-30",
        "validation_start": "2023-10-01",
        "validation_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-03-31",
    },
    {
        "fold": 3,
        "train_end": "2023-12-31",
        "validation_start": "2024-01-01",
        "validation_end": "2024-03-31",
        "test_start": "2024-04-01",
        "test_end": "2024-06-30",
    },
    {
        "fold": 4,
        "train_end": "2024-03-31",
        "validation_start": "2024-04-01",
        "validation_end": "2024-06-30",
        "test_start": "2024-07-01",
        "test_end": "2024-09-30",
    },
    {
        "fold": 5,
        "train_end": "2024-06-30",
        "validation_start": "2024-07-01",
        "validation_end": "2024-09-30",
        "test_start": "2024-10-01",
        "test_end": "2024-12-31",
    },
    {
        "fold": 6,
        "train_end": "2024-09-30",
        "validation_start": "2024-10-01",
        "validation_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-03-31",
    },
)

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.90
TFIDF_MAX_FEATURES = 100_000
TFIDF_SUBLINEAR_TF = True
TFIDF_STRIP_ACCENTS = "unicode"
TFIDF_LOWERCASE = True
TFIDF_BATCH_SIZE = 10_000

RANDOM_SEED = 42

SGD_LOSS = "log_loss"
SGD_PENALTY = "l2"
SGD_ALPHA = 1e-5
SGD_EPOCHS = 5
SGD_BATCH_SIZE = 10_000

def create_project_directories():
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        TFIDF_MODELS_DIR,
        CLASSICAL_MODELS_DIR,
        TFIDF_SGD_MODELS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)