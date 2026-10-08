import csv
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


@dataclass(frozen=True)
class TokenCacheSpec:
    name: str
    path: Path
    max_length: int
    strategy: str


SUMMARY_FIELDS = [
    "model",
    "cache_variant",
    "strategy",
    "max_length",
    "document_count",
    "mean_original_token_count",
    "p50_original_token_count",
    "p75_original_token_count",
    "p90_original_token_count",
    "p95_original_token_count",
    "p99_original_token_count",
    "max_original_token_count",
    "rate_over_128",
    "rate_over_256",
    "rate_over_384",
    "rate_over_512",
    "truncated_document_count",
    "truncation_rate",
    "mean_retained_token_count",
    "mean_discarded_token_count",
    "tokenization_seconds",
]


def build_arrow_table(
    rows,
    input_ids,
):
    return pa.table(
        {
            "record_id": pa.array(
                [
                    row[0]
                    for row in rows
                ],
                type=pa.string(),
            ),
            "complaint_id": pa.array(
                [
                    row[1]
                    for row in rows
                ],
                type=pa.string(),
            ),
            "opening_date": pa.array(
                [
                    row[2]
                    for row in rows
                ],
                type=pa.date32(),
            ),
            "target_resolved": pa.array(
                [
                    row[3]
                    for row in rows
                ],
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


def add_single_sequence_special_tokens(
    tokenizer,
    content_ids,
):
    cls_token_id = (
        tokenizer.cls_token_id
    )

    sep_token_id = (
        tokenizer.sep_token_id
    )

    if (
        cls_token_id is None
        or sep_token_id is None
    ):
        raise ValueError(
            "Tokenizer must define "
            "CLS and SEP token IDs."
        )

    special_token_count = (
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    if special_token_count != 2:
        raise ValueError(
            "Expected exactly two special "
            "tokens for a single sequence, "
            f"but found {special_token_count}."
        )

    return [
        int(cls_token_id),
        *content_ids,
        int(sep_token_id),
    ]


def validate_special_token_construction(
    tokenizer,
):
    probe_text = (
        "ConsumerBR tokenizer "
        "compatibility validation."
    )

    common_arguments = {
        "truncation": False,
        "padding": False,
        "return_attention_mask": False,
        "return_token_type_ids": False,
    }

    content_ids = tokenizer(
        probe_text,
        add_special_tokens=False,
        **common_arguments,
    )["input_ids"]

    reference_ids = tokenizer(
        probe_text,
        add_special_tokens=True,
        **common_arguments,
    )["input_ids"]

    constructed_ids = (
        add_single_sequence_special_tokens(
            tokenizer=tokenizer,
            content_ids=content_ids,
        )
    )

    if constructed_ids != reference_ids:
        raise ValueError(
            "Manual special-token construction "
            "does not match the tokenizer's "
            "native single-sequence encoding."
        )
    

def build_input_ids(
    tokenizer,
    content_ids,
    max_length,
    strategy,
):
    special_token_count = (
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    content_budget = (
        max_length
        - special_token_count
    )

    if content_budget <= 0:
        raise ValueError(
            "Maximum length is smaller "
            "than the required special tokens."
        )

    if len(content_ids) <= content_budget:
        selected_ids = content_ids
    elif strategy == "head":
        selected_ids = (
            content_ids[:content_budget]
        )
    elif strategy == "head_tail":
        head_size = (
            content_budget // 2
        )

        tail_size = (
            content_budget - head_size
        )

        selected_ids = (
            content_ids[:head_size]
            + content_ids[-tail_size:]
        )
    else:
        raise ValueError(
            f"Unknown truncation strategy: "
            f"{strategy}"
        )

    input_ids = (
        add_single_sequence_special_tokens(
            tokenizer=tokenizer,
            content_ids=selected_ids,
        )
    )

    if len(input_ids) > max_length:
        raise ValueError(
            "Tokenized sequence exceeds "
            "the configured maximum length."
        )

    return input_ids


def calculate_rate_over(
    lengths,
    threshold,
):
    return float(
        np.mean(
            lengths > threshold
        )
    )


def build_transformer_token_caches(
    model_label,
    pretrained_dir,
    feature_base_path,
    summary_path,
    tokenization_batch_size,
    cache_specs,
    tokenizer_kwargs=None,
):
    tokenizer_kwargs = (
        tokenizer_kwargs or {}
    )

    output_paths = [
        summary_path,
        *[
            specification.path
            for specification in cache_specs
        ],
    ]

    if all(
        path.exists()
        for path in output_paths
    ):
        print(
            f"{model_label} token caches "
            "already exist."
        )
        return

    for path in output_paths:
        if path.exists():
            path.unlink()

    temporary_paths = {
        specification.name: (
            specification.path.with_suffix(
                ".parquet.part"
            )
        )
        for specification in cache_specs
    }

    for path in temporary_paths.values():
        if path.exists():
            path.unlink()

    tokenizer = (
        AutoTokenizer.from_pretrained(
            pretrained_dir,
            local_files_only=True,
            **tokenizer_kwargs,
        )
    )

    validate_special_token_construction(
        tokenizer
    )

    source_path = str(
        feature_base_path
    ).replace("'", "''")

    connection = duckdb.connect()

    writers = {
        specification.name: None
        for specification in cache_specs
    }

    token_lengths = array(
        "I"
    )

    retained_token_sum = {
        specification.name: 0
        for specification in cache_specs
    }

    discarded_token_sum = {
        specification.name: 0
        for specification in cache_specs
    }

    truncated_document_count = {
        specification.name: 0
        for specification in cache_specs
    }

    document_count = 0

    special_token_count = (
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

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
                tokenization_batch_size
            )

            if not rows:
                break

            texts = [
                row[4]
                for row in rows
            ]

            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            content_batches = encoded[
                "input_ids"
            ]

            variant_batches = {
                specification.name: []
                for specification in cache_specs
            }

            for content_ids in content_batches:
                original_length = (
                    len(content_ids)
                    + special_token_count
                )

                token_lengths.append(
                    original_length
                )

                document_count += 1

                for specification in cache_specs:
                    input_ids = build_input_ids(
                        tokenizer=tokenizer,
                        content_ids=content_ids,
                        max_length=(
                            specification.max_length
                        ),
                        strategy=(
                            specification.strategy
                        ),
                    )

                    retained_count = len(
                        input_ids
                    )

                    discarded_count = max(
                        original_length
                        - retained_count,
                        0,
                    )

                    retained_token_sum[
                        specification.name
                    ] += retained_count

                    discarded_token_sum[
                        specification.name
                    ] += discarded_count

                    if discarded_count > 0:
                        truncated_document_count[
                            specification.name
                        ] += 1

                    variant_batches[
                        specification.name
                    ].append(
                        input_ids
                    )

            for specification in cache_specs:
                table = build_arrow_table(
                    rows=rows,
                    input_ids=(
                        variant_batches[
                            specification.name
                        ]
                    ),
                )

                writer = writers[
                    specification.name
                ]

                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_paths[
                            specification.name
                        ],
                        table.schema,
                        compression="zstd",
                    )

                    writers[
                        specification.name
                    ] = writer

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
        for writer in writers.values():
            if writer is not None:
                writer.close()

        connection.close()

    tokenization_seconds = (
        time.perf_counter()
        - start_time
    )

    print()

    for specification in cache_specs:
        temporary_paths[
            specification.name
        ].replace(
            specification.path
        )

    lengths = np.asarray(
        token_lengths,
        dtype=np.int32,
    )

    quantiles = np.quantile(
        lengths,
        [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ],
    )

    rows = []

    for specification in cache_specs:
        truncated_count = (
            truncated_document_count[
                specification.name
            ]
        )

        rows.append(
            {
                "model": model_label,
                "cache_variant": (
                    specification.name
                ),
                "strategy": (
                    specification.strategy
                ),
                "max_length": (
                    specification.max_length
                ),
                "document_count": (
                    document_count
                ),
                "mean_original_token_count": (
                    float(
                        np.mean(lengths)
                    )
                ),
                "p50_original_token_count": (
                    float(quantiles[0])
                ),
                "p75_original_token_count": (
                    float(quantiles[1])
                ),
                "p90_original_token_count": (
                    float(quantiles[2])
                ),
                "p95_original_token_count": (
                    float(quantiles[3])
                ),
                "p99_original_token_count": (
                    float(quantiles[4])
                ),
                "max_original_token_count": (
                    int(
                        np.max(lengths)
                    )
                ),
                "rate_over_128": (
                    calculate_rate_over(
                        lengths,
                        128,
                    )
                ),
                "rate_over_256": (
                    calculate_rate_over(
                        lengths,
                        256,
                    )
                ),
                "rate_over_384": (
                    calculate_rate_over(
                        lengths,
                        384,
                    )
                ),
                "rate_over_512": (
                    calculate_rate_over(
                        lengths,
                        512,
                    )
                ),
                "truncated_document_count": (
                    truncated_count
                ),
                "truncation_rate": (
                    truncated_count
                    / document_count
                ),
                "mean_retained_token_count": (
                    retained_token_sum[
                        specification.name
                    ]
                    / document_count
                ),
                "mean_discarded_token_count": (
                    discarded_token_sum[
                        specification.name
                    ]
                    / document_count
                ),
                "tokenization_seconds": (
                    tokenization_seconds
                ),
            }
        )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{model_label} token caches completed."
    )

    print(
        f"Saved summary to: "
        f"{summary_path}"
    )