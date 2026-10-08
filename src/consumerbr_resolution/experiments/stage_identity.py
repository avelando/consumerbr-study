import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from consumerbr_resolution import config


PREFIX = "src/consumerbr_resolution/"
COMMON_PROTOCOL = ("experiment_id", "observation_end", "windows", "primary_metric",
                   "threshold_fit_split", "prior_test_inspection",
                   "gap_is_label_maturation_approximation")
UTILITY_FILES = ("experiments/reproducibility.py#sha256_file",
                 "experiments/reproducibility.py#write_json")
STAGES = {
    "company_baseline": {
        "files": ("baselines.py", "evaluation/metrics.py"),
        "config": ("COMPANY_HISTORY_SMOOTHING",),
        "protocol": ("company_history_smoothing",),
        "packages": ("duckdb", "numpy", "pandas", "scikit-learn", "scipy"),
        "reports": ("company_baseline_metrics.csv", "company_baseline_run.json"),
        "predictions": "company_historical_rate",
    },
    "tfidf_sgd": {
        "files": ("modeling/tfidf_sgd.py", "modeling/tfidf.py", "evaluation/metrics.py",
                  "baselines.py#write_predictions", "experiments/temporal_protocol.py#write_csv"),
        "config": ("EXPERIMENT_SEEDS", "PRIMARY_EXPERIMENT_SEED", "SGD_ALPHA_CANDIDATES",
                   "SGD_BATCH_SIZE", "SGD_EPOCHS", "SGD_LOSS", "SGD_PENALTY",
                   "TFIDF_LOWERCASE", "TFIDF_MAX_DF", "TFIDF_MAX_FEATURES", "TFIDF_MIN_DF",
                   "TFIDF_NGRAM_RANGE", "TFIDF_STRIP_ACCENTS", "TFIDF_SUBLINEAR_TF"),
        "protocol": ("seeds", "selection_seed", "sgd_alpha_candidates", "sgd_max_epochs"),
        "packages": ("duckdb", "joblib", "numpy", "pandas", "pyarrow", "scikit-learn",
                     "scipy", "threadpoolctl", "tqdm"),
        "reports": ("tfidf_sgd_selection.csv", "tfidf_sgd_training_history.csv",
                    "tfidf_sgd_metrics.csv", "tfidf_sgd_summary.csv", "tfidf_sgd_run.json"),
        "predictions": "tfidf_sgd", "models": "classical/tfidf_sgd",
    },
    "bertimbau_assets": {
        "files": ("modeling/bertimbau_assets.py",),
        "config": ("BERTIMBAU_MODEL_NAME", "BERTIMBAU_REVISION", "BERTIMBAU_PRETRAINED_DIR"),
        "protocol": (), "packages": ("huggingface-hub",),
        "reports": ("bertimbau_assets.json",),
    },
    "bertimbau_preflight": {
        "files": ("modeling/bertimbau_preflight.py", "modeling/bertimbau_assets.py"),
        "config": ("BERTIMBAU_MODEL_NAME", "BERTIMBAU_REVISION", "BERTIMBAU_PRETRAINED_DIR",
                   "BERTIMBAU_EVAL_BATCH_SIZE", "BERTIMBAU_GRADIENT_ACCUMULATION_STEPS",
                   "BERTIMBAU_GRADIENT_CHECKPOINTING", "BERTIMBAU_LEARNING_RATE",
                   "BERTIMBAU_MAX_GRAD_NORM", "BERTIMBAU_MAX_LENGTH", "BERTIMBAU_TRAIN_BATCH_SIZE",
                   "BERTIMBAU_USE_AMP", "BERTIMBAU_WEIGHT_DECAY", "PRIMARY_EXPERIMENT_SEED"),
        "protocol": (), "packages": ("torch", "transformers", "tokenizers", "safetensors"),
        "upstream": ("bertimbau_assets",), "reports": ("bertimbau_preflight.json",),
    },
    "bertimbau_tokens": {
        "files": ("modeling/bertimbau_tokens.py", "modeling/transformer_tokenization.py",
                  "modeling/bertimbau_assets.py", "experiments/temporal_protocol.py#write_csv"),
        "config": ("BERTIMBAU_MODEL_NAME", "BERTIMBAU_REVISION", "BERTIMBAU_PRETRAINED_DIR",
                   "BERTIMBAU_MAX_LENGTH", "BERTIMBAU_TOKENIZATION_BATCH_SIZE",
                   "BERTIMBAU_TOKEN_CACHE_PATH"),
        "protocol": (), "packages": ("duckdb", "numpy", "pyarrow", "transformers",
                                      "tokenizers", "tqdm"),
        "upstream": ("bertimbau_assets",),
        "reports": ("bertimbau_token_summary.csv", "bertimbau_tokens_run.json"),
    },
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def json_value(value):
    if isinstance(value, Path):
        return value.relative_to(config.PROJECT_ROOT).as_posix()
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value


def source_digest(path, symbol=None):
    path = Path(path)
    reference_path = Path(__file__).with_name("stage_migration.json")
    if reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        compatible = reference["source_compatibility"].get(checksum, {})
        if (symbol or "module") in compatible:
            return compatible[symbol or "module"]
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    if symbol:
        tree = next(node for node in tree.body if getattr(node, "name", None) == symbol)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def source_path(root, name):
    path = Path(root) / PREFIX / name
    return path if path.is_file() else config.PROJECT_ROOT / PREFIX / name


def migration_reference():
    path = Path(__file__).with_name("stage_migration.json")
    return json.loads(path.read_text(encoding="utf-8"))


def compatible_source_hash(path):
    path = Path(path)
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = migration_reference()["token_processor"]
    return reference["previous"] if current == reference["installed"] else current


def stage_identities(root, execution, legacy=False):
    reference = migration_reference() if legacy else None
    identities = {}
    for stage, spec in STAGES.items():
        if legacy:
            expected = reference["stages"].get(stage)
            if expected is None:
                continue
            if any(execution["files"].get(name) != checksum
                   for name, checksum in expected["raw_files"].items()):
                continue
            if any(name not in identities for name in spec.get("upstream", ())):
                continue
            files, settings = expected["files"], expected["config"]
        else:
            files = {}
            for item in (*spec["files"], *UTILITY_FILES):
                name, _, symbol = item.partition("#")
                files[PREFIX + item] = source_digest(source_path(root, name), symbol or None)
            settings = {name: json_value(getattr(config, name)) for name in spec["config"]}
        protocol = execution["protocol"]
        uses_data = stage not in ("bertimbau_assets", "bertimbau_preflight")
        keys = (*(COMMON_PROTOCOL if uses_data else ()), *spec["protocol"])
        packages = {name: execution["packages"].get(name) for name in spec["packages"]}
        if legacy:
            for name in packages:
                if name not in execution["packages"]:
                    packages[name] = reference["locked_packages"].get(name)
        identity = {
            "schema": 1, "stage": stage, "definition": json_value(spec),
            "identity_engine": {name: source_digest(Path(__file__), name)
                                for name in ("digest", "json_value", "source_digest", "stage_identities")},
            "source_sha256": execution["source_sha256"] if uses_data else None,
            "membership_sha256": execution["membership_sha256"] if uses_data else None,
            "protocol": {key: protocol[key] for key in keys},
            "files": files, "config": settings, "python": execution["python"],
            "packages": packages,
            "upstream": {name: identities[name]["dependency_fingerprint"]
                         for name in spec.get("upstream", ())},
        }
        identities[stage] = {"identity": identity, "dependency_fingerprint": digest(identity)}
    return identities


def artifact_path(root, base, name):
    path = (Path(base) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise RuntimeError(f"Artifact path escapes the project: {name}")
    return path


def verify_stage_artifacts(root, tables, models, stage, fingerprint):
    from consumerbr_resolution.experiments.reproducibility import sha256_file

    spec = STAGES[stage]
    records = [Path(tables) / name for name in spec["reports"] if name.endswith(".json")]
    if spec.get("models"):
        records.extend((Path(models) / spec["models"]).rglob("*.json"))
    for path in records:
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Stage artifact belongs to another execution: {path.name}")
        if stage == "bertimbau_preflight":
            if record.get("passed") is not True:
                raise RuntimeError("BERTimbau GPU preflight did not pass.")
            continue
        artifacts = record.get("artifacts")
        if not artifacts:
            raise RuntimeError(f"Stage artifact manifest is empty: {path.name}")
        base = Path(tables).parent if stage == "company_baseline" else Path(root)
        for name, checksum in artifacts.items():
            target = artifact_path(root, base, name)
            if not target.is_file() or sha256_file(target) != checksum:
                raise RuntimeError(f"Stage artifacts changed: {name}")


def stage_paths(tables, models, stage):
    spec = STAGES[stage]
    paths = [Path(tables) / name for name in spec["reports"]]
    if spec.get("predictions"):
        paths.append(Path(tables).parent / "predictions" / spec["predictions"])
    if spec.get("models"):
        paths.append(Path(models) / spec["models"])
    return [path for path in paths if path.exists()]


def reconcile_stages(root, tables, models, previous, current, execution_fingerprint):
    from consumerbr_resolution.experiments.reproducibility import write_json

    if previous["fingerprint"] != digest(previous["identity"]):
        raise RuntimeError("Execution manifest identity is inconsistent.")
    old = previous.get("stages")
    if old is None:
        try:
            old = stage_identities(root, previous["identity"], legacy=True)
        except ValueError:
            old = {}
        for record in old.values():
            record["artifact_fingerprint"] = previous["fingerprint"]
            record["origin_execution_fingerprint"] = previous["fingerprint"]
    for stage, record in old.items():
        if stage in STAGES:
            if record["dependency_fingerprint"] != digest(record["identity"]):
                raise RuntimeError(f"Stage manifest identity is inconsistent: {stage}")
            verify_stage_artifacts(root, tables, models, stage, record["artifact_fingerprint"])
    changed = []
    for stage, record in current.items():
        before = old.get(stage)
        same = before is not None and before["identity"] == record["identity"]
        record["artifact_fingerprint"] = (
            before["artifact_fingerprint"] if same else record["dependency_fingerprint"]
        )
        record["origin_execution_fingerprint"] = (
            before.get("origin_execution_fingerprint", previous["fingerprint"])
            if same else execution_fingerprint
        )
        if not same:
            changed.append(stage)
    paths = {stage: stage_paths(tables, models, stage) for stage in changed}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = Path(root) / "results/archive" / Path(tables).parent.name / timestamp
    archive.mkdir(parents=True, exist_ok=False)
    write_json(archive / "execution_manifest.json", previous)
    write_json(archive / "archive_reason.json", {
        "reason": "Preserve execution provenance and artifacts of changed stages.",
        "changed_stages": changed,
        "reused_stages": [name for name in current if name not in changed],
    })
    for stage, entries in paths.items():
        for path in entries:
            relative = path.relative_to(root)
            target = archive / stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
    return current, archive
