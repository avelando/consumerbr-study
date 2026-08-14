from consumerbr_resolution.config import (
    ALBERTINA_HEAD_TAIL_TOKEN_CACHE_PATH,
    ALBERTINA_LONG_MAX_LENGTH,
    ALBERTINA_LONG_TOKEN_CACHE_PATH,
    ALBERTINA_MAX_LENGTH,
    ALBERTINA_PRETRAINED_DIR,
    ALBERTINA_TOKENIZATION_BATCH_SIZE,
    ALBERTINA_TOKEN_CACHE_PATH,
    FEATURE_BASE_PATH,
    TABLES_DIR,
    create_project_directories,
)
from consumerbr_resolution.transformer_tokenization import (
    TokenCacheSpec,
    build_transformer_token_caches,
)


ALBERTINA_TOKEN_SUMMARY_PATH = (
    TABLES_DIR
    / "albertina_token_cache_summary.csv"
)


def build_albertina_token_cache():
    create_project_directories()

    cache_specs = (
        TokenCacheSpec(
            name="head_256",
            path=ALBERTINA_TOKEN_CACHE_PATH,
            max_length=ALBERTINA_MAX_LENGTH,
            strategy="head",
        ),
        TokenCacheSpec(
            name="head_tail_256",
            path=(
                ALBERTINA_HEAD_TAIL_TOKEN_CACHE_PATH
            ),
            max_length=ALBERTINA_MAX_LENGTH,
            strategy="head_tail",
        ),
        TokenCacheSpec(
            name="head_512",
            path=(
                ALBERTINA_LONG_TOKEN_CACHE_PATH
            ),
            max_length=ALBERTINA_LONG_MAX_LENGTH,
            strategy="head",
        ),
    )

    build_transformer_token_caches(
        model_label="albertina",
        pretrained_dir=(
            ALBERTINA_PRETRAINED_DIR
        ),
        feature_base_path=(
            FEATURE_BASE_PATH
        ),
        summary_path=(
            ALBERTINA_TOKEN_SUMMARY_PATH
        ),
        tokenization_batch_size=(
            ALBERTINA_TOKENIZATION_BATCH_SIZE
        ),
        cache_specs=cache_specs,
    )