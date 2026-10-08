import json
from pathlib import Path

from consumerbr_resolution.config import PROJECT_ROOT, TABLES_DIR
from consumerbr_resolution.experiments.reproducibility import (
    sha256_file, validate_execution, write_json,
)
from consumerbr_resolution.modeling.tfidf_sgd import REPORTS, verify_artifacts


BERTIMBAU_REPORTS = (
    "bertimbau_selection.csv", "bertimbau_training_history.csv",
    "bertimbau_metrics.csv", "bertimbau_summary.csv", "bertimbau_run.json",
)


PREPARATION_REPORTS = (
    "dataset_integrity_audit.csv", "temporal_protocol_audit.csv",
    "temporal_split_summary.csv", "temporal_text_overlap.csv",
    "dataset_overview.csv", "class_distribution.csv", "monthly_distribution.csv",
    "uf_distribution.csv", "feature_summary.csv", "outcome_status_distribution.csv",
    "outcome_observation_summary.csv", "outcome_observation_monthly.csv",
    "outcome_observation_text.csv", "experimental_protocol.json", "execution_manifest.json",
)


def export_reports(root=None, source=None, tables=None, destination=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    tables = Path(tables) if tables is not None else TABLES_DIR
    destination = Path(destination) if destination is not None else root / "reports" / tables.parent.name
    manifest = validate_execution(root, source, tables)
    names = list(PREPARATION_REPORTS)
    completion = tables / "company_baseline_run.json"
    metrics = tables / "company_baseline_metrics.csv"
    if completion.exists() or metrics.exists():
        if not completion.is_file() or not metrics.is_file():
            raise RuntimeError("Company baseline is incomplete.")
        baseline = json.loads(completion.read_text(encoding="utf-8"))
        if baseline["fingerprint"] != validate_execution(root, source, tables, stage="company_baseline")["fingerprint"] or not all(
            (tables.parent / name).is_file()
            and sha256_file(tables.parent / name) == digest
            for name, digest in baseline["artifacts"].items()
        ):
            raise RuntimeError("Company baseline artifacts do not match this execution.")
        names.extend((completion.name, metrics.name))
    sgd_completion = tables / "tfidf_sgd_run.json"
    if any((tables / name).exists() for name in REPORTS):
        if not sgd_completion.is_file():
            raise RuntimeError("TF-IDF + SGD is incomplete.")
        sgd = json.loads(sgd_completion.read_text(encoding="utf-8"))
        verify_artifacts(sgd, root, validate_execution(root, source, tables, stage="tfidf_sgd")["fingerprint"])
        names.extend(REPORTS)
    transformer_names = (
        "bertimbau_assets.json", "bertimbau_preflight.json",
        "bertimbau_token_summary.csv", "bertimbau_tokens_run.json",
    )
    transformer_started = any((tables / name).exists() for name in transformer_names)
    if transformer_started:
        if not all((tables / name).is_file() for name in transformer_names):
            raise RuntimeError("BERTimbau preparation is incomplete.")
        for name in ("bertimbau_assets.json", "bertimbau_tokens_run.json"):
            record = json.loads((tables / name).read_text(encoding="utf-8"))
            stage = "bertimbau_assets" if name == "bertimbau_assets.json" else "bertimbau_tokens"
            registered = validate_execution(root, source, tables, stage=stage)
            if record["fingerprint"] != registered["fingerprint"] or not record["artifacts"] or not all(
                (root / path).is_file() and sha256_file(root / path) == digest
                for path, digest in record["artifacts"].items()
            ):
                raise RuntimeError(f"BERTimbau preparation artifacts changed: {name}")
        preflight = json.loads((tables / "bertimbau_preflight.json").read_text(encoding="utf-8"))
        if preflight["fingerprint"] != validate_execution(root, source, tables, stage="bertimbau_preflight")["fingerprint"] or preflight.get("passed") is not True:
            raise RuntimeError("BERTimbau GPU preflight does not match this execution.")
        summary = tables / "bertimbau_token_summary.csv"
        cached_summary = root / next(name for name in record["artifacts"] if name.endswith(".summary.csv"))
        if sha256_file(summary) != sha256_file(cached_summary):
            raise RuntimeError("BERTimbau token summary changed.")
        names.extend(transformer_names)
    bertimbau_completion = tables / BERTIMBAU_REPORTS[-1]
    if any((tables / name).exists() for name in BERTIMBAU_REPORTS):
        if not all((tables / name).is_file() for name in BERTIMBAU_REPORTS):
            raise RuntimeError("BERTimbau fine-tuning is incomplete.")
        from consumerbr_resolution.modeling.bertimbau_finetuning import verify_run

        record = json.loads(bertimbau_completion.read_text(encoding="utf-8"))
        registered = validate_execution(root, source, tables, stage="bertimbau")
        verify_run(record, root, registered["fingerprint"])
        names.extend(BERTIMBAU_REPORTS)
    missing = [name for name in names if not (tables / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing reports: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = tables / name
        target = destination / name
        if name == "experimental_protocol.json":
            protocol = json.loads(path.read_text(encoding="utf-8"))
            protocol["source_path"] = "data/processed/consumerbr_feature_base.parquet"
            write_json(target, protocol)
        else:
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
            temporary.replace(target)
    write_json(destination / "report_manifest.json", {
        "stage": ("bertimbau" if bertimbau_completion.exists() else
                  "bertimbau_preparation" if transformer_started else
                  "tfidf_sgd" if sgd_completion.exists() else
                  "company_baseline" if completion.exists() else "data_preparation_and_temporal_audit"),
        "fingerprint": manifest["fingerprint"],
        "files": {name: sha256_file(destination / name) for name in names},
    })
    print(f"Aggregate reports exported: {destination}")
    return destination
