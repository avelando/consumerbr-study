import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow.parquet as pq

import test_company_baseline as fixtures
from consumerbr_resolution.experiments.report_export import PREPARATION_REPORTS, export_reports
from consumerbr_resolution.experiments.reproducibility import register_execution, source_files, validate_execution, write_json
from consumerbr_resolution.modeling.bertimbau_assets import prepare_bertimbau_assets
from consumerbr_resolution.modeling.bertimbau_preflight import check_bertimbau_gpu
from consumerbr_resolution.modeling.bertimbau_tokens import build_bertimbau_token_cache


class FakeTokenizer:
    cls_token_id, sep_token_id = 101, 102

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, texts, add_special_tokens=True, **kwargs):
        def encode(text):
            ids = list(range(200, 200 + len(text.split())))
            return [101, *ids, 102] if add_special_tokens else ids
        return {"input_ids": encode(texts) if isinstance(texts, str) else [encode(text) for text in texts]}


class BertimbauPreparationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.CompanyBaselineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root, self.source, self.tables = fixture.root, fixture.source, fixture.tables
        fixture.rows[0] = (*fixture.rows[0][:4], "word " * 600, fixture.rows[0][5])
        fixture.save_source()
        (self.tables / "execution_manifest.json").unlink()
        self.manifest = register_execution(self.root, self.source, self.tables)
        self.downloads = []
        def download(**kwargs):
            self.downloads.append(kwargs)
            for name in kwargs["allow_patterns"]:
                (kwargs["local_dir"] / name).write_bytes(b"fixture")
        self.prepare = lambda: prepare_bertimbau_assets(
            self.root, self.source, self.tables, downloader=download,
            repo_files=["config.json", "vocab.txt", "model.safetensors", "pytorch_model.bin"],
        )
        self.prepare()

    def tokenize(self, tokenizer=None):
        return build_bertimbau_token_cache(self.root, self.source, self.tables, tokenizer or FakeTokenizer())

    def test_assets_download_only_one_weight_format_and_are_reused(self):
        self.prepare()
        self.assertEqual(len(self.downloads), 1)
        self.assertNotIn("pytorch_model.bin", self.downloads[0]["allow_patterns"])

    def test_single_cache_preserves_ids_and_excludes_gaps(self):
        report = self.tokenize()
        path = self.root / next(name for name in report["artifacts"] if name.endswith(".parquet"))
        rows = pq.read_table(path).to_pylist()
        self.assertEqual({row["record_id"] for row in rows}, {"0", "1", "4", "5", "8", "9"})
        self.assertEqual({row["split"] for row in rows}, {"train", "validation", "test"})
        long = next(row for row in rows if row["record_id"] == "0")
        self.assertEqual(long["original_token_count"], 602)
        self.assertEqual(len(long["input_ids"]), 512)
        self.assertEqual((long["input_ids"][0], long["input_ids"][-1]), (101, 102))
        self.assertEqual(len(list(path.parent.glob("*.parquet"))), 1)

    def test_training_code_changes_reuse_the_token_cache(self):
        previous = self.tokenize()
        (self.root / "main.py").write_text("new training entry point\n")
        write_json(self.root / "logs/git_state.json", {"files": source_files(self.root)})
        register_execution(self.root, self.source, self.tables)
        tokenizer = FakeTokenizer()
        with patch.object(FakeTokenizer, "__call__", side_effect=AssertionError("retokenized")):
            current = self.tokenize(tokenizer)
        self.assertEqual(previous["token_fingerprint"], current["token_fingerprint"])
        self.assertEqual(previous["fingerprint"], current["fingerprint"])

    def test_corrupted_cache_is_rejected(self):
        report = self.tokenize()
        path = self.root / next(name for name in report["artifacts"] if name.endswith(".parquet"))
        path.write_bytes(b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "Token cache artifacts changed"):
            self.tokenize()

    def test_native_special_token_mismatch_is_rejected(self):
        class WrongTokenizer(FakeTokenizer):
            sep_token_id = 999
        with self.assertRaisesRegex(ValueError, "native tokenization"):
            self.tokenize(WrongTokenizer())

    def test_cuda_unavailable_blocks_preflight(self):
        write_json(self.tables / "bertimbau_preflight.json", {"passed": True})
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        fake_transformers = SimpleNamespace(AutoModelForSequenceClassification=None)
        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                check_bertimbau_gpu(self.root, self.source, self.tables)
        self.assertFalse((self.tables / "bertimbau_preflight.json").exists())

    def test_export_contains_preparation_aggregates_only(self):
        self.tokenize()
        write_json(self.tables / "bertimbau_preflight.json", {
            "fingerprint": validate_execution(self.root, self.source, self.tables,
                                               stage="bertimbau_preflight")["fingerprint"], "passed": True,
        })
        for name in PREPARATION_REPORTS:
            path = self.tables / name
            if not path.exists():
                path.write_text("field,value\nfixture,1\n")
        destination = export_reports(self.root, self.source, self.tables)
        manifest = json.loads((destination / "report_manifest.json").read_text())
        self.assertEqual(manifest["stage"], "bertimbau_preparation")
        self.assertIn("bertimbau_token_summary.csv", manifest["files"])
        self.assertFalse(list(destination.rglob("*.parquet")))
        self.assertFalse(list(destination.rglob("*.safetensors")))


if __name__ == "__main__":
    unittest.main()
