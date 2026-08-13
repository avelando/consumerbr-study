from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

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

CORPUS_ARCHIVE_PATH = RAW_DATA_DIR / CORPUS_ARCHIVE_NAME
CORPUS_CSV_PATH = RAW_DATA_DIR / CORPUS_CSV_NAME
CORPUS_PARQUET_PATH = INTERIM_DATA_DIR / CORPUS_PARQUET_NAME
MODELING_BASE_PATH = PROCESSED_DATA_DIR / MODELING_BASE_NAME

RESOLVED_STATUS = "Resolvido"
UNRESOLVED_STATUS = "Não Resolvido"

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT = 120


def create_project_directories():
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)