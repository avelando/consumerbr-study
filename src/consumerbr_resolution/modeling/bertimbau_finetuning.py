import gc
import json
import math
import os
import random
import shutil
import time
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from consumerbr_resolution import config as cfg
from consumerbr_resolution.baselines import write_predictions
from consumerbr_resolution.evaluation.metrics import (
    calculate_binary_metrics, find_best_macro_f1_threshold,
)
from consumerbr_resolution.experiments.reproducibility import (
    sha256_file, validate_execution, write_json,
)
from consumerbr_resolution.experiments.temporal_protocol import write_csv
from consumerbr_resolution.modeling.bertimbau_assets import read_assets


REPORTS = (
    "bertimbau_selection.csv", "bertimbau_training_history.csv",
    "bertimbau_metrics.csv", "bertimbau_summary.csv", "bertimbau_run.json",
)


def verify_run(record, root, fingerprint):
    if record["fingerprint"] != fingerprint or not record.get("artifacts"):
        raise RuntimeError("BERTimbau artifacts do not match the registered execution.")
    root = Path(root).resolve()
    for name, digest in record["artifacts"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Artifact path escapes the project: {name}")
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"BERTimbau artifacts changed: {name}")


def hashes(paths, root):
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class TokenDataset(Dataset):
    def __init__(self, table):
        if table.num_rows == 0:
            raise ValueError("Token partition is empty.")
        self.table = table.combine_chunks()
        self.tokens = self.table["input_ids"].chunk(0)
        self.targets = self.table["target_resolved"].to_numpy()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "input_ids": self.tokens[index].as_py(),
            "labels": int(self.targets[index]),
        }

    def frame(self):
        columns = ("record_id", "complaint_id", "company", "opening_date", "target_resolved")
        return self.table.select(columns).to_pandas()


def collate(rows, pad_token_id):
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        mask[index, :length] = 1
    return {
        "input_ids": ids, "attention_mask": mask,
        "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.long),
    }


def make_loader(dataset, batch_size, pad_token_id, device, seed=None):
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=seed is not None,
        generator=generator, num_workers=0, pin_memory=device.type == "cuda",
        collate_fn=partial(collate, pad_token_id=pad_token_id),
    )


def amp_context(device, precision):
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device.type, dtype=dtype, enabled=precision != "fp32")


def create_scheduler(optimizer, total_steps):
    warmup = int(total_steps * cfg.BERTIMBAU_WARMUP_RATIO)

    def factor(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_epoch(model, dataset, tokenizer, device, precision, optimizer,
                scheduler, scaler, seed, description,
                batch_size=None, accumulation_steps=None):
    batch_size = batch_size or cfg.BERTIMBAU_TRAIN_BATCH_SIZE
    accumulation_steps = accumulation_steps or cfg.BERTIMBAU_GRADIENT_ACCUMULATION_STEPS
    loader = make_loader(dataset, batch_size, tokenizer.pad_token_id, device, seed)
    effective_batch = batch_size * accumulation_steps
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    start = time.perf_counter()
    with tqdm(loader, desc=description, unit="batch") as progress:
        for index, batch in enumerate(progress):
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            size = len(batch["labels"])
            group_start = (index // accumulation_steps) * effective_batch
            group_size = min(effective_batch, len(dataset) - group_start)
            with amp_context(device, precision):
                output = model(**batch)
                loss = output.loss * size / group_size
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite training loss.")
            scaler.scale(loss).backward()
            loss_sum += float(output.loss.detach()) * size
            if (index + 1) % accumulation_steps == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.BERTIMBAU_MAX_GRAD_NORM,
                    error_if_nonfinite=precision != "fp16",
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if index % 100 == 0:
                progress.set_postfix(loss=f"{float(output.loss.detach()):.4f}")
    return {
        "training_loss": loss_sum / len(dataset),
        "training_seconds": time.perf_counter() - start,
    }


def score_dataset(model, dataset, tokenizer, device, precision):
    loader = make_loader(dataset, cfg.BERTIMBAU_EVAL_BATCH_SIZE,
                         tokenizer.pad_token_id, device)
    scores = np.empty(len(dataset), dtype=np.float64)
    offset = 0
    model.eval()
    with torch.inference_mode(), tqdm(loader, desc="Scoring", unit="batch") as progress:
        for batch in progress:
            inputs = {name: batch[name].to(device, non_blocking=True)
                      for name in ("input_ids", "attention_mask")}
            with amp_context(device, precision):
                logits = model(**inputs).logits
            probabilities = logits.float().softmax(dim=1)[:, 1].cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Nonfinite prediction scores.")
            scores[offset:offset + len(probabilities)] = probabilities
            offset += len(probabilities)
    if offset != len(dataset):
        raise RuntimeError("Scoring changed the partition row count.")
    return scores


def save_checkpoint(model, path):
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    if path.exists():
        shutil.rmtree(path)
    temporary.rename(path)


def train_candidate(train, validation, pretrained, directory, root, fingerprint,
                    learning_rate, seed, device, precision):
    completion = directory / "run.json"
    if completion.exists():
        result = json.loads(completion.read_text(encoding="utf-8"))
        verify_run(result, root, fingerprint)
        if (result["learning_rate"] != learning_rate or result["seed"] != seed
                or result["precision"] != precision):
            raise RuntimeError("Cached BERTimbau training parameters do not match.")
        print(f"Training reused: lr={learning_rate:g}, seed={seed}", flush=True)
        return result
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True,
                                              do_lower_case=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained, num_labels=2, local_files_only=True,
    ).to(device)
    try:
        if cfg.BERTIMBAU_GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=cfg.BERTIMBAU_WEIGHT_DECAY,
        )
        steps = math.ceil(len(train) / (
            cfg.BERTIMBAU_TRAIN_BATCH_SIZE * cfg.BERTIMBAU_GRADIENT_ACCUMULATION_STEPS
        )) * cfg.BERTIMBAU_EPOCHS
        scheduler = create_scheduler(optimizer, steps)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024.)
        checkpoint = directory / "checkpoint"
        history, best = [], None
        for epoch in range(1, cfg.BERTIMBAU_EPOCHS + 1):
            training = train_epoch(
                model, train, tokenizer, device, precision, optimizer, scheduler,
                scaler, seed + epoch, f"lr={learning_rate:g} seed={seed} epoch={epoch}",
            )
            scores = score_dataset(model, validation, tokenizer, device, precision)
            threshold, macro_f1 = find_best_macro_f1_threshold(validation.targets, scores)
            row = {
                "learning_rate": learning_rate, "seed": seed, "epoch": epoch,
                "validation_macro_f1": macro_f1, "threshold": threshold, **training,
            }
            history.append(row)
            if best is None or macro_f1 > best["validation_macro_f1"]:
                best = dict(row)
                save_checkpoint(model, checkpoint)
            print(f"Validation Macro-F1={macro_f1:.6f}; best epoch={best['epoch']}",
                  flush=True)
        result = {
            "fingerprint": fingerprint, **best, "precision": precision, "history": history,
            "model_path": checkpoint.relative_to(root).as_posix(),
            "artifacts": hashes(sorted(path for path in checkpoint.rglob("*")
                                       if path.is_file()), root),
        }
        write_json(completion, result)
        return result
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def select_runs(worker):
    candidates = [
        worker(rate, cfg.PRIMARY_EXPERIMENT_SEED,
               f"candidate_{index:02d}_seed_{cfg.PRIMARY_EXPERIMENT_SEED}")
        for index, rate in enumerate(cfg.BERTIMBAU_LEARNING_RATE_CANDIDATES)
    ]
    selected_index = max(range(len(candidates)),
                         key=lambda index: candidates[index]["validation_macro_f1"])
    selected = candidates[selected_index]
    chosen = {cfg.PRIMARY_EXPERIMENT_SEED: selected}
    runs = list(candidates)
    for seed in cfg.EXPERIMENT_SEEDS:
        if seed != cfg.PRIMARY_EXPERIMENT_SEED:
            result = worker(selected["learning_rate"], seed, f"selected_seed_{seed}")
            chosen[seed] = result
            runs.append(result)
    selection = [{
        "learning_rate": result["learning_rate"], "seed": result["seed"],
        "best_epoch": result["epoch"], "validation_macro_f1": result["validation_macro_f1"],
        "threshold": result["threshold"], "selected": index == selected_index,
    } for index, result in enumerate(candidates)]
    return chosen, runs, selection


def verify_preparation(root, source, tables):
    tokens = json.loads((tables / "bertimbau_tokens_run.json").read_text())
    registered = validate_execution(root, source, tables, stage="bertimbau_tokens")
    verify_run(tokens, root, registered["fingerprint"])
    if tokens["identity"]["max_length"] != cfg.BERTIMBAU_MAX_LENGTH:
        raise RuntimeError("Token cache length does not match the training configuration.")
    preflight = json.loads((tables / "bertimbau_preflight.json").read_text())
    registered = validate_execution(root, source, tables, stage="bertimbau_preflight")
    precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if cfg.BERTIMBAU_USE_AMP else "fp32"
    expected = {
        "fingerprint": registered["fingerprint"], "passed": True,
        "max_length": cfg.BERTIMBAU_MAX_LENGTH,
        "train_batch_size": cfg.BERTIMBAU_TRAIN_BATCH_SIZE,
        "eval_batch_size": cfg.BERTIMBAU_EVAL_BATCH_SIZE,
        "accumulation_steps": cfg.BERTIMBAU_GRADIENT_ACCUMULATION_STEPS,
        "precision": precision, "gpu": torch.cuda.get_device_name(0),
    }
    if any(preflight.get(name) != value for name, value in expected.items()):
        raise RuntimeError("Run the GPU preflight again for this training configuration.")
    return precision


def evaluate_bertimbau(root=None, source=None, tables=None):
    root = Path(root) if root is not None else cfg.PROJECT_ROOT
    source = Path(source) if source is not None else cfg.FEATURE_BASE_PATH
    tables = Path(tables) if tables is not None else cfg.TABLES_DIR
    manifest = validate_execution(root, source, tables, stage="bertimbau")
    fingerprint = manifest["fingerprint"]
    protocol = manifest["identity"]["protocol"]
    expected = {
        "seeds": list(cfg.EXPERIMENT_SEEDS), "selection_seed": cfg.PRIMARY_EXPERIMENT_SEED,
        "bertimbau_learning_rate_candidates": list(cfg.BERTIMBAU_LEARNING_RATE_CANDIDATES),
        "bertimbau_max_epochs": cfg.BERTIMBAU_EPOCHS,
        "bertimbau_max_length": cfg.BERTIMBAU_MAX_LENGTH,
    }
    if any(protocol.get(name) != value for name, value in expected.items()):
        raise RuntimeError("Registered BERTimbau protocol does not match configuration.")
    completion = tables / REPORTS[-1]
    if completion.exists():
        result = json.loads(completion.read_text())
        verify_run(result, root, fingerprint)
        print("BERTimbau already completed; artifact hashes verified.")
        return result
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run BERTimbau training on the GPU environment.")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    precision = verify_preparation(root, source, tables)
    pretrained, _ = read_assets(root)
    cache = root / cfg.BERTIMBAU_TOKEN_CACHE_PATH.relative_to(cfg.PROJECT_ROOT)
    arrow = ds.dataset(cache, format="parquet")
    load = lambda split: TokenDataset(arrow.to_table(filter=ds.field("split") == split))
    train, validation = load("train"), load("validation")
    models = root / cfg.BERTIMBAU_FINETUNED_DIR.relative_to(cfg.PROJECT_ROOT)

    def worker(rate, seed, name):
        return train_candidate(train, validation, pretrained, models / name,
                               root, fingerprint, rate, seed, device, precision)

    chosen, runs, selection = select_runs(worker)
    write_csv(tables / REPORTS[0], selection)
    write_csv(tables / REPORTS[1], [row for run in runs for row in run["history"]])
    artifacts = [models / f"candidate_{index:02d}_seed_{cfg.PRIMARY_EXPERIMENT_SEED}" / "run.json"
                 for index in range(len(cfg.BERTIMBAU_LEARNING_RATE_CANDIDATES))]
    artifacts.extend(models / f"selected_seed_{seed}" / "run.json"
                     for seed in cfg.EXPERIMENT_SEEDS if seed != cfg.PRIMARY_EXPERIMENT_SEED)
    for run in runs:
        artifacts.extend(root / name for name in run["artifacts"])
    del train
    gc.collect()
    test = load("test")
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True,
                                              do_lower_case=False)
    rows = []
    for seed in cfg.EXPERIMENT_SEEDS:
        run = chosen[seed]
        model = AutoModelForSequenceClassification.from_pretrained(
            root / run["model_path"], local_files_only=True,
        ).to(device)
        try:
            for split, dataset in (("validation", validation), ("test", test)):
                scores = score_dataset(model, dataset, tokenizer, device, precision)
                metrics = calculate_binary_metrics(dataset.targets, scores, run["threshold"])
                rows.append({
                    "model": "bertimbau", "seed": seed, "split": split,
                    "fingerprint": fingerprint, "learning_rate": run["learning_rate"],
                    "best_epoch": run["epoch"], "precision": precision,
                    "threshold_source": "validation_macro_f1", "complaint_count": len(dataset),
                    **metrics,
                })
                frame = dataset.frame().assign(
                    score=scores, prediction=(scores >= run["threshold"]).astype("int8"),
                    threshold=run["threshold"], seed=seed,
                )
                path = tables.parent / "predictions/bertimbau" / f"seed_{seed}" / f"{split}.parquet"
                with duckdb.connect(config={"memory_limit": "8GB", "threads": "4"}) as connection:
                    write_predictions(connection, frame, path)
                artifacts.append(path)
                print(f"seed={seed} {split}: Macro-F1={metrics['macro_f1']:.6f}, "
                      f"ROC-AUC={metrics['roc_auc']:.6f}", flush=True)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    write_csv(tables / REPORTS[2], rows)
    frame = pd.DataFrame(rows)
    summary = [{
        "model": "bertimbau", "split": split, "metric": metric,
        "seed_count": len(group), "mean": float(group[metric].mean()),
        "std": float(group[metric].std(ddof=1)),
    } for split, group in frame.groupby("split", sort=False)
      for metric in ("macro_f1", "roc_auc", "pr_auc", "accuracy", "brier_score")]
    write_csv(tables / REPORTS[3], summary)
    artifacts.extend(tables / name for name in REPORTS[:-1])
    result = {
        "fingerprint": fingerprint, "selection_seed": cfg.PRIMARY_EXPERIMENT_SEED,
        "selected_learning_rate": chosen[cfg.PRIMARY_EXPERIMENT_SEED]["learning_rate"],
        "seeds": list(cfg.EXPERIMENT_SEEDS), "training_runs": len(runs),
        "max_length": cfg.BERTIMBAU_MAX_LENGTH, "max_epochs": cfg.BERTIMBAU_EPOCHS,
        "precision": precision, "gpu": torch.cuda.get_device_name(0),
        "tie_break": "first configured learning rate; earliest epoch",
        "artifacts": hashes(sorted(set(artifacts)), root),
    }
    write_json(completion, result)
    return result