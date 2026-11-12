import sys
import unittest
from copy import deepcopy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT))
from train import INSTRUCTION, load_data, make_batches, prepare_query, validate_partitions
from prepare_splits import normalize
from test_retrieval import FakeRunnerMixin, FakeTokenizer


class TrainingTests(FakeRunnerMixin, unittest.TestCase):
    def test_batches_cover_each_row_once_and_exclude_duplicate_products(self):
        products = {"a": "Same Product", "b": " same   PRODUCT ", "c": "other", "d": "third"}
        rows = [{"qid": str(i), "gold_pid": pid} for i, pid in enumerate("aacbaddc")]
        batches = make_batches(rows, products, 3, 42)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(rows))))
        for batch in batches:
            self.assertGreaterEqual(len(batch), 2)
            self.assertLessEqual(len(batch), 3)
            self.assertEqual(len({rows[i]["gold_pid"] for i in batch}), len(batch))
            self.assertEqual(len({normalize(products[rows[i]["gold_pid"]]) for i in batch}), len(batch))
        self.assertEqual(batches, make_batches(rows, products, 3, 42))

    def test_batches_repair_singleton_without_dropping_training_row(self):
        products = {str(i): f"product {i}" for i in range(7)}
        rows = [{"gold_pid": str(i)} for i in range(7)]
        batches = make_batches(rows, products, 3, 42)
        self.assertEqual(sorted(map(len, batches)), [2, 2, 3])
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(7)))

    def test_unusable_negative_batches_raise(self):
        rows = [{"gold_pid": "a"}] * 4
        with self.assertRaises(ValueError):
            make_batches(rows, {"a": "one product"}, 3, 42)
        with self.assertRaises(ValueError):
            make_batches(rows, {"a": "one product"}, 1, 42)

    def test_training_query_matches_formal_short_and_long_inputs(self):
        queries = [{"qid": "short", "query": "user: no perfume\nassistant: which size?\nuser: small"},
                   {"qid": "long", "query": "user: original need " + "history " * 180 + "user: not scented"}]
        result, model = self.run_fake(queries, {"a": "first", "b": "second"}, {}, top_k=2)
        texts = [prepare_query(row["query"], FakeTokenizer()) for row in queries]
        self.assertEqual(model.calls[1], texts)
        self.assertEqual(texts[0], INSTRUCTION + queries[0]["query"])
        self.assertTrue(texts[1].endswith("user: not scented"))
        self.assertLessEqual(len(FakeTokenizer().encode(texts[1])), 512)

    def test_partition_guard_rejects_pid_text_and_manifest_leakage(self):
        train = [{"qid": "t", "gold_pid": "a", "query": "user: train"}]
        dev = [{"qid": "d", "gold_pid": "b", "query": "user: dev"}]
        manifest = {"partitions": {"train": {"count": 1, "qids": ["t"]},
                                   "dev": {"count": 1, "qids": ["d"]}}}
        products = {"a": "A product", "b": "B product"}
        validate_partitions(train, dev, products, manifest)
        for field, value in [("gold_pid", "a"), ("query", "USER: TRAIN"), ("qid", "wrong")]:
            changed = deepcopy(dev)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                validate_partitions(train, changed, products, manifest)
        with self.assertRaises(ValueError):
            validate_partitions(train, dev, {"a": "Same", "b": " SAME "}, manifest)

    def test_real_split_and_batches_do_not_include_dev_supervision(self):
        rows, products, hashes = load_data()
        batches = make_batches(rows, products, 16, 42)
        self.assertEqual(len(rows), 3864)
        self.assertEqual(len(products), 50000)
        self.assertEqual(sum(map(len, batches)), 3864)
        self.assertIn("data/derived/train.jsonl", hashes)
        self.assertTrue(all(2 <= len(batch) <= 16 for batch in batches))


if __name__ == "__main__":
    unittest.main()
