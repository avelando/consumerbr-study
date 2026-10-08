import csv
import json
from pathlib import Path

import duckdb
import numpy as np

from consumerbr_resolution.config import (
    COMPANY_HISTORY_SMOOTHING, FEATURE_BASE_PATH, PREDICTIONS_DIR, TABLES_DIR,
)
from consumerbr_resolution.evaluation.metrics import (
    calculate_binary_metrics, find_best_macro_f1_threshold,
)
from consumerbr_resolution.experiments.reproducibility import (
    sha256_file, validate_execution, write_json,
)


def score_company_split(connection, source, membership, split, smoothing):
    if smoothing <= 0 or not np.isfinite(smoothing):
        raise ValueError("Smoothing must be finite and positive.")
    source_path = str(source).replace("'", "''")
    membership_path = str(membership).replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW assigned AS SELECT f.*, m.split "
        f"FROM read_parquet('{source_path}') f "
        f"JOIN read_parquet('{membership_path}') m USING (record_id)"
    )
    return connection.execute(
        "WITH prior AS (SELECT AVG(target_resolved) AS p FROM assigned "
        "WHERE split = 'train'), history AS (SELECT company, COUNT(*) AS n, "
        "SUM(target_resolved) AS resolved FROM assigned WHERE split = 'train' "
        "GROUP BY company) SELECT a.record_id, a.complaint_id, a.company, "
        "a.opening_date, a.target_resolved, "
        "COALESCE((h.resolved + ? * p.p) / (h.n + ?), p.p) AS score, "
        "h.n IS NOT NULL AS company_seen FROM assigned a CROSS JOIN prior p "
        "LEFT JOIN history h ON a.company IS NOT DISTINCT FROM h.company "
        "WHERE a.split = ? ORDER BY a.record_id",
        [float(smoothing), float(smoothing), split],
    ).df()


def write_predictions(connection, frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.part")
    escaped = str(temporary).replace("'", "''")
    connection.register("scored_predictions", frame)
    try:
        connection.execute(
            f"COPY scored_predictions TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.unregister("scored_predictions")
    temporary.replace(path)


def evaluate_historical_baselines(root=None, source=None, tables=None, predictions=None):
    source = Path(source) if source is not None else FEATURE_BASE_PATH
    tables = Path(tables) if tables is not None else TABLES_DIR
    predictions = Path(predictions) if predictions is not None else PREDICTIONS_DIR
    manifest = validate_execution(root, source, tables, stage="company_baseline")
    membership = tables / "split_membership.parquet"
    fingerprint = manifest["fingerprint"]
    completion = tables / "company_baseline_run.json"
    if completion.exists():
        previous = json.loads(completion.read_text(encoding="utf-8"))
        valid = previous["fingerprint"] == fingerprint and all(
            (tables.parent / name).is_file()
            and sha256_file(tables.parent / name) == digest
            for name, digest in previous["artifacts"].items()
        )
        if valid:
            print("Company baseline already completed; artifact hashes verified.")
            return previous
        raise RuntimeError("Baseline artifacts changed. Preserve or remove its completion file before rerunning.")
    smoothing = manifest["identity"]["protocol"]["company_history_smoothing"]
    if smoothing != COMPANY_HISTORY_SMOOTHING:
        raise RuntimeError("Protocol smoothing does not match configuration.")
    rows, artifacts = [], []
    with duckdb.connect(config={"memory_limit": "8GB", "threads": "4"}) as connection:
        validation = score_company_split(connection, source, membership, "validation", smoothing)
        threshold, _ = find_best_macro_f1_threshold(
            validation["target_resolved"], validation["score"],
        )
        for split in ("validation", "test"):
            frame = validation if split == "validation" else score_company_split(
                connection, source, membership, split, smoothing,
            )
            frame["prediction"] = (frame["score"] >= threshold).astype("int8")
            frame["threshold"] = threshold
            path = predictions / "company_historical_rate" / f"{split}.parquet"
            write_predictions(connection, frame, path)
            artifacts.append(path)
            metrics = calculate_binary_metrics(frame["target_resolved"], frame["score"], threshold)
            rows.append({
                "model": "company_historical_rate", "split": split,
                "fingerprint": fingerprint, "threshold_source": "validation_macro_f1",
                "smoothing": smoothing, "complaint_count": len(frame),
                "company_seen_rate": float(frame["company_seen"].mean()), **metrics,
            })
            print(f"{split}: Macro-F1={metrics['macro_f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
    path = tables / "company_baseline_metrics.csv"
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    artifacts.append(path)
    result = {"fingerprint": fingerprint, "threshold": threshold, "artifacts": {
        str(path.relative_to(tables.parent)): sha256_file(path) for path in artifacts
    }}
    write_json(completion, result)
    return result
