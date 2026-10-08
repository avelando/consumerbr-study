import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import torch

from consumerbr_resolution import config as cfg
from consumerbr_resolution.experiments.stage_identity import verify_stage_artifacts
from consumerbr_resolution.modeling import bertimbau_finetuning as training


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        self.classifier = torch.nn.Linear(4, 2)
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids, attention_mask, labels=None):
        values = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        pooled = values.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        logits = self.classifier(pooled)
        loss = None if labels is None else torch.nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)

    def save_pretrained(self, directory, safe_serialization=True):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "weights.pt")
        (directory / "config.json").write_text("{}\n")


def dataset():
    return training.TokenDataset(pa.Table.from_pydict({
        "record_id": [str(index) for index in range(5)],
        "complaint_id": [f"id-{index}" for index in range(5)],
        "company": ["A"] * 5,
        "opening_date": ["2024-01-01"] * 5,
        "target_resolved": [0, 1, 0, 1, 0],
        "input_ids": [[1, 2], [3], [4, 5, 6], [7, 8], [9]],
    }))


class BertimbauFinetuningTests(unittest.TestCase):
    def test_selection_uses_four_runs_and_reuses_seed_42_winner(self):
        calls = []

        def worker(rate, seed, name):
            calls.append((rate, seed, name))
            return {
                "learning_rate": rate, "seed": seed, "epoch": 2,
                "validation_macro_f1": 0.7 if rate == 2e-5 else 0.6,
                "threshold": 0.45,
            }

        chosen, runs, selection = training.select_runs(worker)
        self.assertEqual([(rate, seed) for rate, seed, _ in calls],
                         [(1e-5, 42), (2e-5, 42), (2e-5, 13), (2e-5, 101)])
        self.assertEqual(len(runs), 4)
        self.assertIs(chosen[42], runs[1])
        self.assertEqual(sum(row["selected"] for row in selection), 1)

    def test_selection_tie_uses_first_learning_rate(self):
        def worker(rate, seed, name):
            return {
                "learning_rate": rate, "seed": seed, "epoch": 1,
                "validation_macro_f1": 0.7, "threshold": 0.5,
            }

        chosen, _, _ = training.select_runs(worker)
        self.assertEqual(chosen[42]["learning_rate"], 1e-5)

    def test_accumulation_matches_full_batches_including_last_group(self):
        training.set_seed(42)
        accumulated = TinyModel()
        reference = copy.deepcopy(accumulated)
        data = dataset()
        device = torch.device("cpu")
        tokenizer = SimpleNamespace(pad_token_id=0)
        optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with patch.object(cfg, "BERTIMBAU_MAX_GRAD_NORM", 1e6):
            training.train_epoch(
                accumulated, data, tokenizer, device, "fp32", optimizer,
                scheduler, scaler, 17, "CPU accumulation test",
                batch_size=2, accumulation_steps=2,
            )
        for batch in training.make_loader(data, 4, 0, device, seed=17):
            reference_optimizer.zero_grad(set_to_none=True)
            reference(**batch).loss.backward()
            reference_optimizer.step()
        for actual, expected in zip(accumulated.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_scoring_preserves_row_order(self):
        training.set_seed(42)
        model = TinyModel().eval()
        data = dataset()
        batch = training.collate([data[index] for index in range(len(data))], 0)
        with torch.inference_mode():
            expected = model(**batch).logits.softmax(dim=1)[:, 1].numpy()
        with patch.object(cfg, "BERTIMBAU_EVAL_BATCH_SIZE", 2):
            actual = training.score_dataset(
                model, data, SimpleNamespace(pad_token_id=0), torch.device("cpu"), "fp32",
            )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_best_checkpoint_threshold_and_completed_candidate_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = root / "results/study/tables"
            models = root / "models/study"
            directory = models / "transformers/bertimbau_base/finetuned/candidate_00_seed_42"
            data = dataset()
            scores = [
                np.full(5, 0.5),
                np.array([0.1, 0.9, 0.1, 0.9, 0.1]),
                np.full(5, 0.5),
            ]
            with patch.object(training.AutoTokenizer, "from_pretrained",
                              return_value=SimpleNamespace(pad_token_id=0)), \
                 patch.object(training.AutoModelForSequenceClassification, "from_pretrained",
                              return_value=TinyModel()), \
                 patch.object(cfg, "BERTIMBAU_GRADIENT_CHECKPOINTING", False), \
                 patch.object(training, "train_epoch",
                              return_value={"training_loss": 0.5, "training_seconds": 1.0}), \
                 patch.object(training, "score_dataset", side_effect=scores):
                result = training.train_candidate(
                    data, data, root / "pretrained", directory, root,
                    "fixture", 1e-5, 42, torch.device("cpu"), "fp32",
                )
            self.assertEqual(result["epoch"], 2)
            self.assertEqual(result["validation_macro_f1"], 1.0)
            prediction = (scores[1] >= result["threshold"]).astype("int8")
            np.testing.assert_array_equal(prediction, data.targets)
            verify_stage_artifacts(root, tables, models, "bertimbau", "fixture")
            with patch.object(training.AutoModelForSequenceClassification,
                              "from_pretrained") as factory:
                repeated = training.train_candidate(
                    None, None, root / "pretrained", directory, root,
                    "fixture", 1e-5, 42, torch.device("cpu"), "fp32",
                )
                factory.assert_not_called()
            self.assertEqual(result, repeated)
            (directory / "checkpoint/weights.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
                training.train_candidate(
                    None, None, root / "pretrained", directory, root,
                    "fixture", 1e-5, 42, torch.device("cpu"), "fp32",
                )


if __name__ == "__main__":
    unittest.main()