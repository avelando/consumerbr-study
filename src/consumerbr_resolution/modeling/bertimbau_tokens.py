import hashlib
import json
import os
import shutil
from array import array
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from consumerbr_resolution.config import (
    BERTIMBAU_MAX_LENGTH, BERTIMBAU_TOKENIZATION_BATCH_SIZE, BERTIMBAU_TOKEN_CACHE_PATH,
    FEATURE_BASE_PATH, PROJECT_ROOT, TABLES_DIR,
)
from consumerbr_resolution.experiments.reproducibility import sha256_file, validate_execution, write_json
from consumerbr_resolution.experiments.stage_identity import compatible_source_hash
from consumerbr_resolution.experiments.temporal_protocol import write_csv
from consumerbr_resolution.modeling.bertimbau_assets import read_assets
from consumerbr_resolution.modeling import transformer_tokenization as encoding


REPORTS = ("bertimbau_token_summary.csv", "bertimbau_tokens_run.json")
SPLITS = ("train", "validation", "test")


def build_bertimbau_token_cache(root=None, source=None, tables=None, tokenizer=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    source = Path(source) if source is not None else FEATURE_BASE_PATH
    tables = Path(tables) if tables is not None else TABLES_DIR
    manifest = validate_execution(root, source, tables, stage="bertimbau_tokens")
    directory, assets = read_assets(root)
    cache = root / BERTIMBAU_TOKEN_CACHE_PATH.relative_to(PROJECT_ROOT)
    marker = cache.with_suffix(".run.json")
    summary_path = cache.with_suffix(".summary.csv")
    tokenizer_packages = {}
    for name in ("transformers", "tokenizers"):
        try:
            tokenizer_packages[name] = version(name)
        except PackageNotFoundError:
            tokenizer_packages[name] = None
    identity = {
        "source": manifest["identity"]["source_sha256"],
        "membership": manifest["identity"]["membership_sha256"],
        "assets": assets["artifacts"], "max_length": BERTIMBAU_MAX_LENGTH, "strategy": "head",
        "tokenizer_packages": tokenizer_packages,
        "processor": compatible_source_hash(Path(__file__)), "encoding": sha256_file(Path(encoding.__file__)),
    }
    token_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    if previous is not None and previous["token_fingerprint"] == token_fingerprint:
        required = {path.relative_to(root).as_posix() for path in (cache, summary_path)}
        if previous["identity"] != identity or set(previous["artifacts"]) != required:
            raise RuntimeError("Token cache manifest is inconsistent.")
        for name, digest in previous["artifacts"].items():
            path = root / name
            if not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"Token cache artifacts changed: {name}")
        print("Single BERTimbau token cache reused; artifact hashes verified.")
    else:
        if tokenizer is None:
            os.environ.setdefault("RAYON_NUM_THREADS", "4")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                                      use_fast=True, do_lower_case=False)
        encoding.validate_special_token_construction(tokenizer)
        cache.parent.mkdir(parents=True, exist_ok=True)
        lengths = {split: array("I") for split in SPLITS}
        fields = ("record_id", "complaint_id", "company", "opening_date", "target_resolved", "split")
        schema = pa.schema([
            ("record_id", pa.string()), ("complaint_id", pa.string()), ("company", pa.string()),
            ("opening_date", pa.date32()), ("target_resolved", pa.int8()), ("split", pa.string()),
            ("input_ids", pa.list_(pa.int32())), ("original_token_count", pa.int32()),
        ])
        temporary = cache.with_suffix(".parquet.part")
        membership = tables / "split_membership.parquet"
        with duckdb.connect(config={"memory_limit": "8GB", "threads": "4"}) as connection:
            counts = dict(connection.execute(
                "SELECT split, COUNT(*) FROM read_parquet(?) WHERE split IN "
                "('train', 'validation', 'test') GROUP BY split", [str(membership)],
            ).fetchall())
            cursor = connection.execute(
                "SELECT CAST(f.record_id AS VARCHAR), CAST(f.complaint_id AS VARCHAR), "
                "f.company, f.opening_date, f.target_resolved, m.split, f.complaint_text "
                "FROM read_parquet(?) f JOIN read_parquet(?) m USING(record_id) "
                "WHERE m.split IN ('train', 'validation', 'test') ORDER BY m.split, f.record_id",
                [str(source), str(membership)],
            )
            with pq.ParquetWriter(temporary, schema, compression="zstd") as writer, \
                 tqdm(total=sum(counts.values()), desc="BERTimbau tokens", unit="docs") as progress:
                while rows := cursor.fetchmany(BERTIMBAU_TOKENIZATION_BATCH_SIZE):
                    batches = tokenizer([row[-1] for row in rows], add_special_tokens=False,
                                        truncation=False, padding=False, return_attention_mask=False,
                                        return_token_type_ids=False)["input_ids"]
                    if len(batches) != len(rows):
                        raise RuntimeError("Tokenizer changed the batch row count.")
                    values = {name: [row[index] for row in rows] for index, name in enumerate(fields)}
                    original = [len(ids) + 2 for ids in batches]
                    values["input_ids"] = [encoding.build_input_ids(tokenizer, ids, BERTIMBAU_MAX_LENGTH)
                                           for ids in batches]
                    values["original_token_count"] = original
                    for row, length in zip(rows, original):
                        lengths[row[5]].append(length)
                    writer.write_table(pa.Table.from_pydict(values, schema=schema))
                    progress.update(len(rows))
        if any(len(lengths[split]) != counts[split] for split in SPLITS):
            raise RuntimeError("Token cache row counts differ from audited membership.")
        temporary.replace(cache)
        summary = []
        for split in SPLITS:
            values = np.asarray(lengths[split], dtype=np.int32)
            summary.append({
                "split": split, "strategy": "head", "max_length": BERTIMBAU_MAX_LENGTH,
                "document_count": len(values), "mean_original_tokens": float(values.mean()),
                "p50_original_tokens": float(np.quantile(values, .50)),
                "p95_original_tokens": float(np.quantile(values, .95)),
                "p99_original_tokens": float(np.quantile(values, .99)),
                "truncated_count": int((values > BERTIMBAU_MAX_LENGTH).sum()),
                "truncation_rate": float((values > BERTIMBAU_MAX_LENGTH).mean()),
                "rate_over_512": float((values > 512).mean()),
            })
        write_csv(summary_path, summary)
        previous = {
            "token_fingerprint": token_fingerprint, "identity": identity,
            "artifacts": {path.relative_to(root).as_posix(): sha256_file(path)
                          for path in (cache, summary_path)},
        }
        write_json(marker, previous)
    shutil.copyfile(summary_path, tables / REPORTS[0])
    report = {"fingerprint": manifest["fingerprint"], **previous,
              "artifacts": {**previous["artifacts"], marker.relative_to(root).as_posix(): sha256_file(marker)}}
    write_json(tables / REPORTS[1], report)
    print(f"Single head-{BERTIMBAU_MAX_LENGTH} cache verified: {cache}")
    return report
