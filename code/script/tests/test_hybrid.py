"""Independent BM25 formula oracle and rank-fusion contracts."""
import math
import os
import sys
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from test_retrieval import ROOT, FakeEncoder
from retrieval_utils import retrieve_and_rerank


def oracle(products, query):
    counts = {pid: Counter(text.casefold().split()) for pid, text in products.items()}
    n = len(counts)
    average = max(sum(sum(c.values()) for c in counts.values()) / n, 1)
    scores = {}
    for pid, words in counts.items():
        score = 0
        for term in set(query.casefold().split()):
            tf = words[term]
            df = sum(term in c for c in counts.values())
            if tf:
                score += math.log(1 + (n - df + .5) / (df + .5)) * tf * 2.5 / (
                    tf + 1.5 * (.25 + .75 * sum(words.values()) / average))
        scores[pid] = score
    return sorted(products, key=lambda pid: (-scores[pid], pid)), scores


class HybridTests(unittest.TestCase):
    def run_mode(self, mode, products, query, k=2, vectors=None, candidates=100, **env):
        encoder = FakeEncoder(vectors or {})
        modules = {"sentence_transformers": SimpleNamespace(SentenceTransformer=lambda *a, **kw: encoder)}
        settings = {"HW2_RETRIEVER": mode, "HW2_RERANK": "0", "HW2_MODEL_PATH": str(ROOT),
                    "HW2_RRF_DENSE_WEIGHT": "1", "HW2_RRF_BM25_WEIGHT": "1",
                    "HW2_RRF_CANDIDATES": str(candidates), "HW2_BM25_QUERY_MODE": "full",
                    "HW2_BM25_LAST_WEIGHT": "1", "HW2_BM25_TITLE_WEIGHT": "1", **env}
        with patch.dict(os.environ, settings), patch.dict(sys.modules, modules):
            results = retrieve_and_rerank([{"qid": "q", "query": query}], products, top_k=k)
        return results[0]["ranked_pids"], encoder

    def test_bm25_against_independent_formula_and_pool_changes(self):
        products = {"p3": "red shampoo shampoo", "p1": "blue soap", "p2": "red red soap lotion", "p0": "soap"}
        for query in ["red shampoo", "red red shampoo", "soap", "missing"]:
            expected, _ = oracle(products, query)
            result, encoder = self.run_mode("bm25", products, query, k=4)
            self.assertEqual(result, expected)
            self.assertEqual(encoder.calls, [])  # lexical mode does not load/use an encoder
        products.pop("p1")
        products["new"] = "shampoo shampoo red"
        expected, _ = oracle(products, "shampoo")
        result, _ = self.run_mode("bm25", dict(reversed(list(products.items()))), "shampoo", k=4)
        self.assertEqual(result, expected)

    def test_decimals_models_negation_and_description_tail(self):
        products = {"wrong": "perfume 1.7 oz Model-X200", "right": "perfume 1.6 oz Model-X100",
                    "tail": "noise " * 1000 + "not scented"}
        for query, expected in [("1.6", "right"), ("MODEL-X100", "right"), ("not scented", "tail")]:
            result, _ = self.run_mode("bm25", products, query, k=1)
            self.assertEqual(result, [expected])

    def test_rrf_formula_union_and_unique_pids(self):
        products = {"a": "apple apple", "b": "apple", "c": "pear", "d": "banana"}
        vectors = {"apple": [1, 0], "apple apple": [-1, 0], "pear": [.9, .1], "banana": [0, 1]}
        dense = ["b", "c"]
        lexical = ["a", "b"]
        expected_scores = {}
        for stream in [dense, lexical]:
            for rank, pid in enumerate(stream, 1):
                expected_scores[pid] = expected_scores.get(pid, 0) + 1 / (60 + rank)
        expected = sorted(expected_scores, key=lambda pid: (-expected_scores[pid], pid))[:2]
        result, encoder = self.run_mode("hybrid", products, "apple", vectors=vectors, candidates=2)
        self.assertEqual(result, expected)
        self.assertEqual(len(result), len(set(result)))
        self.assertEqual(len(encoder.calls), 2)

    def test_zero_lexical_matches_preserve_dense_order_and_boundary_ties(self):
        products = {f"p{i:03}": "same" for i in reversed(range(105))}
        for query in ["unseen", "!!!"]:
            result, _ = self.run_mode("hybrid", products, query, k=10)
            self.assertEqual(result, sorted(products)[:10])
        result, _ = self.run_mode("bm25", products, "same", k=10)
        self.assertEqual(result, sorted(products)[:10])
        result, _ = self.run_mode("bm25", {"z": "!!!", "a": "..."}, "???", k=2)
        self.assertEqual(result, ["a", "z"])

    def test_candidate_floor_small_pool_and_invalid_modes(self):
        products = {"a": "red", "b": "green", "c": "blue"}
        result, _ = self.run_mode("hybrid", products, "red", k=3, candidates=1)
        self.assertEqual(set(result), set(products))
        for mode, env in [("wrong", {}), ("hybrid", {"HW2_RRF_CANDIDATES": "0"}),
                          ("bm25", {"HW2_RERANK": "1"})]:
            with self.assertRaises(ValueError):
                self.run_mode(mode, products, "red", **env)


if __name__ == "__main__":
    unittest.main()
