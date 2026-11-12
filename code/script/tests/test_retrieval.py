import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from retrieval_utils import retrieve_and_rerank


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text) + (["<s>", "</s>"] if add_special_tokens else [])

    def decode(self, ids, **kwargs):
        return "".join(ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors, self.calls = vectors, []
        self.tokenizer = FakeTokenizer()

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.array([self.vectors.get(text.removeprefix(
            "Represent this sentence for searching relevant passages: "), [1, 0]) for text in texts], dtype=np.float32)


class FakeRunnerMixin:
    def run_fake(self, queries, products, vectors, top_k=2, fallback=False):
        model = FakeEncoder(vectors)
        modules = {"sentence_transformers": SimpleNamespace(SentenceTransformer=lambda *a, **k: model)}
        if fallback:
            modules["faiss"] = None
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"HW2_MODEL_PATH": str(ROOT), "HW2_RERANK": "0", "HW2_RETRIEVER": "dense"}):
            result = retrieve_and_rerank(queries, products, top_k=top_k)
        return result, model


class IndexTests(FakeRunnerMixin, unittest.TestCase):
    def test_exact_oracle_and_mapping(self):
        rng = np.random.default_rng(7)
        vectors = {str(i): rng.normal(size=7) for i in range(14)}
        products = {f"PID-{i:02}": str(i) for i in reversed(range(12))}
        queries = [{"qid": "q1", "query": "12"}, {"qid": "q2", "query": "13"}]
        pids = sorted(products)
        pv = np.array([vectors[products[pid]] / np.linalg.norm(vectors[products[pid]]) for pid in pids])
        expected = [{"qid": q["qid"], "ranked_pids": [pids[i] for i in np.lexsort(
            (np.arange(len(pids)), -(pv @ (vectors[q["query"]] / np.linalg.norm(vectors[q["query"]])))))[:10]]}
                    for q in queries]
        for fallback in [False, True]:
            result, model = self.run_fake(queries, products, vectors, 10, fallback)
            self.assertEqual(result, expected)
            self.assertEqual(model.calls[0], [products[p] for p in pids])
            self.assertEqual(len(model.calls), 2)  # products encoded once per call

    def test_boundary_ties_order_and_pool_mutation(self):
        vectors = {"same": [1, 0], "query": [1, 0], "opposite": [-1, 0]}
        queries = [{"qid": "q", "query": "query"}]
        products = {f"P{i:02}": "same" for i in reversed(range(15))}
        for fallback in [False, True]:
            first, _ = self.run_fake(queries, products, vectors, 10, fallback)
            self.assertEqual(first[0]["ranked_pids"], sorted(products)[:10])
            products["P00"] = "opposite"
            del products["P01"]
            products["A-new"] = "same"
            second, _ = self.run_fake(queries, dict(reversed(list(products.items()))), vectors, 10, fallback)
            self.assertEqual(second[0]["ranked_pids"], sorted(p for p in products if p != "P00")[:10])
            products = {f"P{i:02}": "same" for i in reversed(range(15))}

    def test_nonfinite_or_zero_vectors_fail(self):
        for bad in [[float("nan"), 1], [0, 0]]:
            with self.assertRaises(ValueError):
                self.run_fake([{"qid": "q", "query": "q"}], {"a": "a", "b": "b"},
                              {"a": bad, "b": [1, 0], "q": [1, 0]})


class DialogueTests(FakeRunnerMixin, unittest.TestCase):
    def test_short_role_transcripts_unchanged_and_isolated(self):
        fixture = json.loads((ROOT / "code/script/tests/dialogue_fixtures.json").read_text())
        queries = [{"qid": r["qid"], "query": r["query"]} for r in fixture["queries"]]
        _, model = self.run_fake(queries, fixture["products"], {}, 10)
        prefix = "Represent this sentence for searching relevant passages: "
        self.assertEqual(model.calls[1], [prefix + q["query"] for q in queries])
        self.assertEqual(model.calls[0], [fixture["products"][p] for p in sorted(fixture["products"])])

    def test_long_query_preserves_start_and_final_negation(self):
        raw = "user: I need lotion. " + "filler " * 300 + "user: not scented."
        _, model = self.run_fake([{"qid": "long", "query": raw}], {"a": "a", "b": "b"}, {}, 2)
        text = model.calls[1][0]
        self.assertTrue(text.startswith("Represent this sentence for searching relevant passages: user: I need lotion."))
        self.assertTrue(text.endswith("user: not scented."))
        self.assertLessEqual(len(model.tokenizer.encode(text)), 512)
        self.assertNotEqual(text, "Represent this sentence for searching relevant passages: " + raw)


class ContractTests(FakeRunnerMixin, unittest.TestCase):
    def test_bad_inputs_and_empty_queries(self):
        products = {"a": "a", "b": "b"}
        result, model = self.run_fake([], products, {}, 2)
        self.assertEqual(result, [])
        self.assertEqual(model.calls, [])
        for queries, pool, k in [
            ([{"qid": "q", "query": ""}], products, 2),
            ([{"qid": "q", "query": "x"}] * 2, products, 2),
            ([{"qid": "q", "query": "x"}], {"a": ""}, 1),
            ([{"qid": "q", "query": "x"}], products, 3),
            ([{"qid": "q", "query": "x"}], products, 0),
        ]:
            with self.assertRaises(ValueError):
                self.run_fake(queries, pool, {}, k)

    def test_labels_are_ignored_and_topk_respected(self):
        query = {"qid": "q", "query": "x", "target": object()}
        result, _ = self.run_fake([query], {"a": "a", "b": "b"}, {}, 1)
        self.assertEqual(result, [{"qid": "q", "ranked_pids": ["a"]}])


if __name__ == "__main__":
    unittest.main()
