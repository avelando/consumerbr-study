import gc
import os
import time
from pathlib import Path

from consumerbr_resolution.config import (
    BERTIMBAU_EVAL_BATCH_SIZE, BERTIMBAU_GRADIENT_ACCUMULATION_STEPS,
    BERTIMBAU_GRADIENT_CHECKPOINTING, BERTIMBAU_LEARNING_RATE, BERTIMBAU_MAX_GRAD_NORM,
    BERTIMBAU_MAX_LENGTH, BERTIMBAU_TRAIN_BATCH_SIZE, BERTIMBAU_USE_AMP,
    BERTIMBAU_WEIGHT_DECAY, PRIMARY_EXPERIMENT_SEED, PROJECT_ROOT, TABLES_DIR,
)
from consumerbr_resolution.experiments.reproducibility import validate_execution, write_json
from consumerbr_resolution.modeling.bertimbau_assets import read_assets


REPORT_NAME = "bertimbau_preflight.json"


def check_bertimbau_gpu(root=None, source=None, tables=None):
    import torch
    from transformers import AutoModelForSequenceClassification

    root = Path(root) if root is not None else PROJECT_ROOT
    tables = Path(tables) if tables is not None else TABLES_DIR
    manifest = validate_execution(root, source, tables, stage="bertimbau_preflight")
    (tables / REPORT_NAME).unlink(missing_ok=True)
    directory, assets = read_assets(root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Verify PyTorch and the selected GPU before continuing.")
    if min(BERTIMBAU_TRAIN_BATCH_SIZE, BERTIMBAU_EVAL_BATCH_SIZE,
           BERTIMBAU_GRADIENT_ACCUMULATION_STEPS) < 1:
        raise ValueError("Batch sizes and accumulation steps must be positive.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(PRIMARY_EXPERIMENT_SEED)
    torch.cuda.manual_seed_all(PRIMARY_EXPERIMENT_SEED)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    precision = ("bf16" if bf16 else "fp16") if BERTIMBAU_USE_AMP else "fp32"
    free, total = torch.cuda.mem_get_info(device)
    print(f"Checking {torch.cuda.get_device_name(device)} with {precision}; "
          f"effective batch={BERTIMBAU_TRAIN_BATCH_SIZE * BERTIMBAU_GRADIENT_ACCUMULATION_STEPS}.",
          flush=True)
    model = optimizer = inputs = output = loss = labels = scaler = None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            directory, num_labels=2, local_files_only=True,
        ).to(device)
        if BERTIMBAU_MAX_LENGTH > model.config.max_position_embeddings:
            raise ValueError("Maximum length exceeds the pretrained model's positional limit.")
        if BERTIMBAU_GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=BERTIMBAU_LEARNING_RATE,
                                      weight_decay=BERTIMBAU_WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024.)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model.train()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(BERTIMBAU_GRADIENT_ACCUMULATION_STEPS):
                inputs = torch.randint(model.config.vocab_size,
                                       (BERTIMBAU_TRAIN_BATCH_SIZE, BERTIMBAU_MAX_LENGTH), device=device)
                labels = torch.arange(len(inputs), device=device) % 2
                with torch.autocast("cuda", dtype=dtype, enabled=BERTIMBAU_USE_AMP):
                    output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels)
                    loss = output.loss / BERTIMBAU_GRADIENT_ACCUMULATION_STEPS
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite loss during the GPU preflight.")
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), BERTIMBAU_MAX_GRAD_NORM,
                                          error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            print(f"Synthetic optimizer step {step + 1}/2 passed.", flush=True)
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=BERTIMBAU_USE_AMP):
            inputs = torch.randint(model.config.vocab_size,
                                   (BERTIMBAU_EVAL_BATCH_SIZE, BERTIMBAU_MAX_LENGTH), device=device)
            output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
            if not torch.isfinite(output.logits).all().item():
                raise RuntimeError("Nonfinite evaluation logits during the GPU preflight.")
        torch.cuda.synchronize(device)
        result = {
            "fingerprint": manifest["fingerprint"], "passed": True, "synthetic_inputs": True,
            "model": assets["model"], "revision": assets["revision"],
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "precision": precision,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "max_length": BERTIMBAU_MAX_LENGTH, "train_batch_size": BERTIMBAU_TRAIN_BATCH_SIZE,
            "eval_batch_size": BERTIMBAU_EVAL_BATCH_SIZE,
            "accumulation_steps": BERTIMBAU_GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BERTIMBAU_TRAIN_BATCH_SIZE * BERTIMBAU_GRADIENT_ACCUMULATION_STEPS,
            "gradient_checkpointing": BERTIMBAU_GRADIENT_CHECKPOINTING,
            "initial_free_gib": free / 2**30, "total_gib": total / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "synthetic_seconds": time.perf_counter() - start,
        }
        write_json(tables / REPORT_NAME, result)
        print(f"BERTimbau CUDA forward/backward, optimizer and evaluation passed: {tables / REPORT_NAME}")
        return result
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError("GPU preflight ran out of memory. Adjust batch sizes and accumulation "
                           "before running the full experiment.") from error
    finally:
        del model, optimizer, inputs, output, loss, labels, scaler
        gc.collect()
        torch.cuda.empty_cache()
