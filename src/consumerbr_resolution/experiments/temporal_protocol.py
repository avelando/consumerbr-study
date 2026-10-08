import csv
import json
from pathlib import Path

import duckdb

from consumerbr_resolution.config import (
    BERTIMBAU_EPOCHS,
    BERTIMBAU_LEARNING_RATE_CANDIDATES,
    BOOTSTRAP_REPLICATES,
    COMPANY_HISTORY_SMOOTHING,
    EXPECTED_CORPUS_OBSERVATION_END,
    EXPERIMENT_ID,
    EXPERIMENT_MODELS,
    EXPERIMENT_SEEDS,
    FEATURE_BASE_PATH,
    PRIMARY_EXPERIMENT_SEED,
    PRIMARY_METRIC,
    SGD_ALPHA_CANDIDATES,
    SGD_EPOCHS,
    TABLES_DIR,
    TEMPORAL_FOLDS,
    THRESHOLD_FIT_SPLIT,
)
from consumerbr_resolution.experiments.temporal_design import get_temporal_windows


REQUIRED_COLUMNS = {
    "record_id", "complaint_id", "opening_date", "target_resolved", "complaint_text"
}


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_temporal_protocol(source_path=None, output_dir=None):
    source = Path(source_path) if source_path is not None else FEATURE_BASE_PATH
    destination = Path(output_dir) if output_dir is not None else TABLES_DIR
    if not source.is_file():
        raise FileNotFoundError(f"Feature base was not found: {source}")
    windows = get_temporal_windows(
        TEMPORAL_FOLDS[0], EXPECTED_CORPUS_OBSERVATION_END
    )
    summary_rows = []
    audit_rows = []
    overlap_rows = []
    destination.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(config={"memory_limit": "8GB", "threads": "4"}) as connection:
        escaped_source = str(source).replace("'", "''")
        connection.execute(
            f"CREATE VIEW feature_base AS SELECT * FROM read_parquet('{escaped_source}')"
        )
        columns = {row[0] for row in connection.execute("DESCRIBE feature_base").fetchall()}
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        clauses = " ".join(
            f"WHEN opening_date BETWEEN DATE '{window['start_date']}' "
            f"AND DATE '{window['end_date']}' THEN '{window['split']}'"
            for window in windows
        )
        connection.execute(
            f"CREATE VIEW assigned AS SELECT *, CASE {clauses} "
            "ELSE 'outside_period' END AS protocol_split FROM feature_base"
        )
        total, unique_ids, invalid = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT record_id), "
            "COUNT(*) FILTER (WHERE record_id IS NULL OR complaint_id IS NULL "
            "OR opening_date IS NULL OR target_resolved IS NULL "
            "OR target_resolved NOT IN (0, 1) OR complaint_text IS NULL) FROM assigned"
        ).fetchone()
        outside = connection.execute(
            "SELECT COUNT(*) FROM assigned WHERE protocol_split = 'outside_period'"
        ).fetchone()[0]
        crossing = connection.execute(
            "SELECT COUNT(*) FROM (SELECT complaint_id FROM assigned "
            "WHERE protocol_split IN ('train', 'validation', 'test') "
            "GROUP BY complaint_id HAVING COUNT(DISTINCT protocol_split) > 1)"
        ).fetchone()[0]
        checks = (
            ("nonempty_source", total, total > 0),
            ("unique_record_ids", total - unique_ids, total == unique_ids),
            ("valid_required_values", invalid, invalid == 0),
            ("dates_inside_observation_horizon", outside, outside == 0),
            ("complaint_ids_do_not_cross_partitions", crossing, crossing == 0),
        )
        audit_rows.extend(
            {"criterion": name, "value": value, "passed": passed}
            for name, value, passed in checks
        )
        for window in windows:
            count, resolved, unresolved = connection.execute(
                "SELECT COUNT(*), COUNT(*) FILTER (WHERE target_resolved = 1), "
                "COUNT(*) FILTER (WHERE target_resolved = 0) "
                "FROM assigned WHERE protocol_split = ?",
                [window["split"]],
            ).fetchone()
            summary_rows.append({
                **window,
                "complaint_count": count,
                "resolved_count": resolved,
                "unresolved_count": unresolved,
                "resolution_rate": resolved / count if count else None,
            })
            if window["included"]:
                audit_rows.append({
                    "criterion": f"both_classes_in_{window['split']}",
                    "value": f"{resolved}/{unresolved}",
                    "passed": resolved > 0 and unresolved > 0,
                })
        audit_rows.append({
            "criterion": "all_rows_accounted_for",
            "value": sum(row["complaint_count"] for row in summary_rows),
            "passed": sum(row["complaint_count"] for row in summary_rows) == total,
        })
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            shared = connection.execute(
                "SELECT COUNT(*) FROM (SELECT complaint_text FROM assigned "
                "WHERE protocol_split IN (?, ?) GROUP BY complaint_text "
                "HAVING COUNT(DISTINCT protocol_split) = 2)",
                [left, right],
            ).fetchone()[0]
            overlap_rows.append({
                "left_split": left,
                "right_split": right,
                "shared_exact_texts": shared,
            })
        write_csv(destination / "temporal_protocol_audit.csv", audit_rows)
        failed = [row["criterion"] for row in audit_rows if not row["passed"]]
        if failed:
            raise RuntimeError(f"Temporal audit failed: {', '.join(failed)}")
        membership = destination / "split_membership.parquet"
        temporary = membership.with_suffix(".parquet.part")
        escaped = str(temporary).replace("'", "''")
        connection.execute(
            "COPY (SELECT record_id, complaint_id, opening_date, target_resolved, "
            "protocol_split AS split FROM assigned ORDER BY record_id) "
            f"TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        temporary.replace(membership)

    write_csv(destination / "temporal_split_summary.csv", summary_rows)
    write_csv(destination / "temporal_text_overlap.csv", overlap_rows)
    specification = {
        "experiment_id": EXPERIMENT_ID,
        "source_path": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "observation_end": EXPECTED_CORPUS_OBSERVATION_END,
        "windows": windows,
        "models": EXPERIMENT_MODELS,
        "seeds": EXPERIMENT_SEEDS,
        "selection_seed": PRIMARY_EXPERIMENT_SEED,
        "primary_metric": PRIMARY_METRIC,
        "threshold_fit_split": THRESHOLD_FIT_SPLIT,
        "sgd_alpha_candidates": SGD_ALPHA_CANDIDATES,
        "sgd_max_epochs": SGD_EPOCHS,
        "bertimbau_learning_rate_candidates": BERTIMBAU_LEARNING_RATE_CANDIDATES,
        "bertimbau_max_epochs": BERTIMBAU_EPOCHS,
        "company_history_smoothing": COMPANY_HISTORY_SMOOTHING,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "prior_test_inspection": True,
        "gap_is_label_maturation_approximation": True,
    }
    output = destination / "experimental_protocol.json"
    temporary = output.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(specification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Temporal protocol audited: {destination}")
    for row in summary_rows:
        print(f"{row['split']}: {row['complaint_count']} records")
    return summary_rows
