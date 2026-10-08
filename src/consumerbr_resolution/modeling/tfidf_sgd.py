import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from consumerbr_resolution.baselines import write_predictions
from consumerbr_resolution.config import (
    EXPERIMENT_SEEDS, FEATURE_BASE_PATH, PRIMARY_EXPERIMENT_SEED, PROJECT_ROOT,
    SGD_ALPHA_CANDIDATES, SGD_BATCH_SIZE, SGD_EPOCHS, SGD_LOSS, SGD_PENALTY, TABLES_DIR,
)
from consumerbr_resolution.evaluation.metrics import (
    calculate_binary_metrics, find_best_macro_f1_threshold,
)
from consumerbr_resolution.experiments.reproducibility import (
    sha256_file, validate_execution, write_json,
)
from consumerbr_resolution.experiments.temporal_protocol import write_csv
from consumerbr_resolution.modeling.tfidf import create_tfidf_vectorizer, load_split


REPORTS = (
    "tfidf_sgd_selection.csv", "tfidf_sgd_training_history.csv",
    "tfidf_sgd_metrics.csv", "tfidf_sgd_summary.csv", "tfidf_sgd_run.json",
)


def verify_artifacts(record, root, fingerprint):
    if record["fingerprint"] != fingerprint or not record["artifacts"]:
        raise RuntimeError("TF-IDF execution does not match the registered inputs.")
    for name, digest in record["artifacts"].items():
        path = Path(root) / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"TF-IDF artifacts changed: {name}")


def artifact_hashes(paths, root):
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)


def train_candidate(x_train, y_train, x_validation, y_validation,
                    alpha, seed, directory, root, fingerprint, vectorizer_path):
    completion = directory / "run.json"
    model_path = directory / "model.joblib"
    if completion.exists():
        result = json.loads(completion.read_text(encoding="utf-8"))
        verify_artifacts(result, root, fingerprint)
        if result["alpha"] != alpha or result["seed"] != seed:
            raise RuntimeError("Cached training parameters do not match.")
        print(f"Training reused: alpha={alpha:g}, seed={seed}", flush=True)
        return result
    model = SGDClassifier(
        loss=SGD_LOSS, penalty=SGD_PENALTY, alpha=alpha,
        random_state=seed, shuffle=False,
    )
    generator = np.random.default_rng(seed)
    history, best = [], None
    for epoch in range(1, SGD_EPOCHS + 1):
        order = generator.permutation(len(y_train))
        batches = range(0, len(order), SGD_BATCH_SIZE)
        for start in tqdm(batches, desc=f"alpha={alpha:g} seed={seed} epoch={epoch}"):
            indices = order[start:start + SGD_BATCH_SIZE]
            model.partial_fit(x_train[indices], y_train[indices], classes=np.array([0, 1]))
        score = model.predict_proba(x_validation)[:, 1]
        threshold, macro_f1 = find_best_macro_f1_threshold(y_validation, score)
        history.append({
            "alpha": alpha, "seed": seed, "epoch": epoch,
            "validation_macro_f1": macro_f1, "threshold": threshold,
        })
        if best is None or macro_f1 > best["validation_macro_f1"]:
            best = dict(history[-1])
            save_model(model, model_path)
        print(f"Validation Macro-F1={macro_f1:.4f}; best epoch={best['epoch']}", flush=True)
    result = {
        "fingerprint": fingerprint, **best, "history": history,
        "model_path": model_path.relative_to(root).as_posix(),
        "artifacts": artifact_hashes([model_path, vectorizer_path], root),
    }
    write_json(completion, result)
    return result


def evaluate_tfidf_sgd(root=None, source=None, tables=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    source = Path(source) if source is not None else FEATURE_BASE_PATH
    tables = Path(tables) if tables is not None else TABLES_DIR
    manifest = validate_execution(root, source, tables)
    fingerprint = manifest["fingerprint"]
    protocol = manifest["identity"]["protocol"]
    expected = {
        "seeds": list(EXPERIMENT_SEEDS), "selection_seed": PRIMARY_EXPERIMENT_SEED,
        "sgd_alpha_candidates": list(SGD_ALPHA_CANDIDATES), "sgd_max_epochs": SGD_EPOCHS,
    }
    if any(protocol[key] != value for key, value in expected.items()):
        raise RuntimeError("Registered SGD protocol does not match configuration.")
    completion = tables / "tfidf_sgd_run.json"
    if completion.exists():
        result = json.loads(completion.read_text(encoding="utf-8"))
        verify_artifacts(result, root, fingerprint)
        print("TF-IDF + SGD already completed; artifact hashes verified.")
        return result
    models = root / "models" / tables.parent.name / "classical/tfidf_sgd"
    predictions = tables.parent / "predictions/tfidf_sgd"
    membership = tables / "split_membership.parquet"
    vectorizer_path = models / "vectorizer.joblib"
    vectorizer_record = models / "vectorizer_run.json"
    with threadpool_limits(limits=4), duckdb.connect(
        config={"memory_limit": "8GB", "threads": "4"},
    ) as connection:
        print("Loading training and validation partitions.", flush=True)
        train = load_split(connection, source, membership, "train")
        validation = load_split(connection, source, membership, "validation")
        if vectorizer_record.exists():
            cached = json.loads(vectorizer_record.read_text(encoding="utf-8"))
            verify_artifacts(cached, root, fingerprint)
            vectorizer = joblib.load(vectorizer_path)
            x_train = vectorizer.transform(train["complaint_text"])
        else:
            print("Fitting word TF-IDF on training texts only.", flush=True)
            vectorizer = create_tfidf_vectorizer()
            x_train = vectorizer.fit_transform(train["complaint_text"])
            save_model(vectorizer, vectorizer_path)
            write_json(vectorizer_record, {
                "fingerprint": fingerprint, "training_count": len(train),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "artifacts": artifact_hashes([vectorizer_path], root),
            })
        x_validation = vectorizer.transform(validation["complaint_text"])
        y_train = train["target_resolved"].to_numpy(dtype=np.int8)
        y_validation = validation["target_resolved"].to_numpy(dtype=np.int8)
        del train
        validation = validation.drop(columns="complaint_text")
        candidates, runs, artifacts = [], [], [vectorizer_path, vectorizer_record]
        for index, alpha in enumerate(SGD_ALPHA_CANDIDATES):
            directory = models / f"candidate_{index:02d}_seed_{PRIMARY_EXPERIMENT_SEED}"
            result = train_candidate(
                x_train, y_train, x_validation, y_validation, alpha,
                PRIMARY_EXPERIMENT_SEED, directory, root, fingerprint, vectorizer_path,
            )
            candidates.append(result)
            runs.append(result)
            artifacts.extend([directory / "run.json", root / result["model_path"]])
        selected_index = max(range(len(candidates)), key=lambda i: candidates[i]["validation_macro_f1"])
        selected = candidates[selected_index]
        chosen = {PRIMARY_EXPERIMENT_SEED: selected}
        selection = [{
            "alpha": item["alpha"], "seed": item["seed"], "best_epoch": item["epoch"],
            "validation_macro_f1": item["validation_macro_f1"], "threshold": item["threshold"],
            "selected": index == selected_index,
        } for index, item in enumerate(candidates)]
        write_csv(tables / REPORTS[0], selection)
        print(f"Selected alpha={selected['alpha']:g} using validation only.", flush=True)
        for seed in EXPERIMENT_SEEDS:
            if seed == PRIMARY_EXPERIMENT_SEED:
                continue
            directory = models / f"selected_seed_{seed}"
            result = train_candidate(
                x_train, y_train, x_validation, y_validation, selected["alpha"],
                seed, directory, root, fingerprint, vectorizer_path,
            )
            chosen[seed] = result
            runs.append(result)
            artifacts.extend([directory / "run.json", root / result["model_path"]])
        del x_train, y_train
        write_csv(tables / REPORTS[1], [row for item in runs for row in item["history"]])
        test = load_split(connection, source, membership, "test")
        x_test = vectorizer.transform(test["complaint_text"])
        test = test.drop(columns="complaint_text")
        rows = []
        for seed in EXPERIMENT_SEEDS:
            item = chosen[seed]
            model = joblib.load(root / item["model_path"])
            for split, frame, matrix in (
                ("validation", validation, x_validation), ("test", test, x_test),
            ):
                score = model.predict_proba(matrix)[:, 1]
                metrics = calculate_binary_metrics(frame["target_resolved"], score, item["threshold"])
                rows.append({
                    "model": "tfidf_sgd", "seed": seed, "split": split,
                    "fingerprint": fingerprint, "alpha": selected["alpha"],
                    "best_epoch": item["epoch"], "threshold_source": "validation_macro_f1",
                    "complaint_count": len(frame), **metrics,
                })
                scored = frame.assign(
                    score=score, prediction=(score >= item["threshold"]).astype("int8"),
                    threshold=item["threshold"], seed=seed,
                )
                path = predictions / f"seed_{seed}" / f"{split}.parquet"
                write_predictions(connection, scored, path)
                artifacts.append(path)
                print(f"seed={seed} {split}: Macro-F1={metrics['macro_f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
    write_csv(tables / REPORTS[2], rows)
    summary = []
    frame = pd.DataFrame(rows)
    for split, group in frame.groupby("split", sort=False):
        for metric in ("macro_f1", "roc_auc", "pr_auc", "accuracy", "brier_score"):
            summary.append({
                "model": "tfidf_sgd", "split": split, "metric": metric,
                "seed_count": len(group), "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=1)),
            })
    write_csv(tables / REPORTS[3], summary)
    artifacts.extend(tables / name for name in REPORTS[:-1])
    result = {
        "fingerprint": fingerprint, "selected_alpha": selected["alpha"],
        "selection_seed": PRIMARY_EXPERIMENT_SEED, "seeds": list(EXPERIMENT_SEEDS),
        "training_runs": len(runs), "tie_break": "first configured alpha; earliest epoch",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "artifacts": artifact_hashes(artifacts, root),
    }
    write_json(completion, result)
    return result
