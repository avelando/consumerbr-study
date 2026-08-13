import csv
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from consumerbr_resolution.config import (
    ALBERTINA_MAX_LENGTH,
    ALBERTINA_PRETRAINED_DIR,
    ALBERTINA_TOKENIZATION_BATCH_SIZE,
    ALBERTINA_TOKEN_CACHE_PATH,
    FEATURE_BASE_PATH,
    TABLES_DIR,
    create_project_directories,
)


ALBERTINA_TOKEN_SUMMARY_PATH = (
    TABLES_DIR
    / "albertina_token_cache_summary.csv"
)


SUMMARY_FIELDS = [
    "document_count",
    "max_length",
    "mean_token_count",
    "max_token_count",
    "max_length_rate",
    "tokenization_seconds",
]


def build_arrow_table(
    rows,
    input_ids,
):
    record_ids = [
        row[0]
        for row in rows
    ]

    complaint_ids = [
        row[1]
        for row in rows
    ]

    opening_dates = [
        row[2]
        for row in rows
    ]

    targets = [
        row[3]
        for row in rows
    ]

    return pa.table(
        {
            "record_id": pa.array(
                record_ids,
                type=pa.string(),
            ),
            "complaint_id": pa.array(
                complaint_ids,
                type=pa.string(),
            ),
            "opening_date": pa.array(
                opening_dates,
                type=pa.date32(),
            ),
            "target_resolved": pa.array(
                targets,
                type=pa.int8(),
            ),
            "input_ids": pa.array(
                input_ids,
                type=pa.list_(
                    pa.int32()
                ),
            ),
        }
    )


def build_albertina_token_cache():
    create_project_directories()

    if (
        ALBERTINA_TOKEN_CACHE_PATH.exists()
        and ALBERTINA_TOKEN_SUMMARY_PATH.exists()
    ):
        print(
            "Albertina token cache already exists."
        )
        return

    temporary_path = (
        ALBERTINA_TOKEN_CACHE_PATH
        .with_suffix(
            ".parquet.part"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    source_path = str(
        FEATURE_BASE_PATH
    ).replace("'", "''")

    print(
        "Building Albertina token cache"
    )

    print(
        f"Source: {FEATURE_BASE_PATH}"
    )

    print(
        f"Destination: "
        f"{ALBERTINA_TOKEN_CACHE_PATH}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            ALBERTINA_PRETRAINED_DIR,
            local_files_only=True,
        )
    )

    connection = duckdb.connect()

    writer = None

    document_count = 0
    token_count_sum = 0
    max_token_count = 0
    max_length_count = 0

    start_time = time.perf_counter()

    try:
        cursor = connection.execute(
            f"""
            SELECT
                record_id,
                complaint_id,
                opening_date,
                target_resolved,
                complaint_text
            FROM read_parquet(
                '{source_path}'
            )
            ORDER BY HASH(record_id)
            """
        )

        while True:
            rows = cursor.fetchmany(
                ALBERTINA_TOKENIZATION_BATCH_SIZE
            )

            if not rows:
                break

            texts = [
                row[4]
                for row in rows
            ]

            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                max_length=(
                    ALBERTINA_MAX_LENGTH
                ),
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            input_ids = encoded[
                "input_ids"
            ]

            for token_ids in input_ids:
                length = len(token_ids)

                document_count += 1

                token_count_sum += length

                if length > max_token_count:
                    max_token_count = length

                if (
                    length
                    == ALBERTINA_MAX_LENGTH
                ):
                    max_length_count += 1

            table = build_arrow_table(
                rows=rows,
                input_ids=input_ids,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(
                table
            )

            print(
                f"\rTokenized: "
                f"{document_count}",
                end="",
                flush=True,
            )

    finally:
        if writer is not None:
            writer.close()

        connection.close()

    tokenization_seconds = (
        time.perf_counter()
        - start_time
    )

    print()

    temporary_path.replace(
        ALBERTINA_TOKEN_CACHE_PATH
    )

    mean_token_count = (
        token_count_sum
        / document_count
    )

    max_length_rate = (
        max_length_count
        / document_count
    )

    with ALBERTINA_TOKEN_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()

        writer.writerow(
            {
                "document_count": (
                    document_count
                ),
                "max_length": (
                    ALBERTINA_MAX_LENGTH
                ),
                "mean_token_count": (
                    mean_token_count
                ),
                "max_token_count": (
                    max_token_count
                ),
                "max_length_rate": (
                    max_length_rate
                ),
                "tokenization_seconds": (
                    tokenization_seconds
                ),
            }
        )

    print(
        "Albertina token cache completed."
    )

    print(
        f"Saved to: "
        f"{ALBERTINA_TOKEN_CACHE_PATH}"
    )

    print(
        f"Summary: "
        f"{ALBERTINA_TOKEN_SUMMARY_PATH}"
    )