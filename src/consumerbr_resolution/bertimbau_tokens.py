from consumerbr_resolution.config import (
    BERTIMBAU_HEAD_TAIL_TOKEN_CACHE_PATH,
    BERTIMBAU_LONG_MAX_LENGTH,
    BERTIMBAU_LONG_TOKEN_CACHE_PATH,
    BERTIMBAU_MAX_LENGTH,
    BERTIMBAU_PRETRAINED_DIR,
    BERTIMBAU_TOKENIZATION_BATCH_SIZE,
    BERTIMBAU_TOKEN_CACHE_PATH,
    FEATURE_BASE_PATH,
    TABLES_DIR,
    create_project_directories,
)
from consumerbr_resolution.transformer_tokenization import (
    TokenCacheSpec,
    build_transformer_token_caches,
)


BERTIMBAU_TOKEN_SUMMARY_PATH = (
    TABLES_DIR
    / "bertimbau_token_cache_summary.csv"
)


def build_bertimbau_token_cache():
    create_project_directories()

    cache_specs = (
        TokenCacheSpec(
            name="head_256",
            path=BERTIMBAU_TOKEN_CACHE_PATH,
            max_length=BERTIMBAU_MAX_LENGTH,
            strategy="head",
        ),
        TokenCacheSpec(
            name="head_tail_256",
            path=(
                BERTIMBAU_HEAD_TAIL_TOKEN_CACHE_PATH
            ),
            max_length=BERTIMBAU_MAX_LENGTH,
            strategy="head_tail",
        ),
        TokenCacheSpec(
            name="head_512",
            path=(
                BERTIMBAU_LONG_TOKEN_CACHE_PATH
            ),
            max_length=BERTIMBAU_LONG_MAX_LENGTH,
            strategy="head",
        ),
    )

    build_transformer_token_caches(
        model_label="bertimbau",
        pretrained_dir=(
            BERTIMBAU_PRETRAINED_DIR
        ),
        feature_base_path=(
            FEATURE_BASE_PATH
        ),
        summary_path=(
            BERTIMBAU_TOKEN_SUMMARY_PATH
        ),
        tokenization_batch_size=(
            BERTIMBAU_TOKENIZATION_BATCH_SIZE
        ),
        cache_specs=cache_specs,
        tokenizer_kwargs={
            "do_lower_case": False,
        },
    )