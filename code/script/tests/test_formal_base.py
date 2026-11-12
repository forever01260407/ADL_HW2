"""Weighted formal fusion must agree with an independent rank oracle at 768d."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import test_hybrid
from test_retrieval import ROOT, FakeEncoder
from retrieval_utils import retrieve_and_rerank


class FormalBaseTests(unittest.TestCase):
    def test_weighted_fusion_and_equal_weight_regression(self):
        runner = test_hybrid.HybridTests()
        products = {"a": "apple apple", "b": "apple", "c": "pear", "d": "banana"}
        vectors = {"apple": [1, 0], "apple apple": [-1, 0], "pear": [.9, .1], "banana": [0, 1]}
        for dw, bw in [(1, 1), (5, 1), (2.5, .5)]:
            scores = {}
            for stream, weight in [(["b", "c"], dw), (["a", "b"], bw)]:
                for rank, pid in enumerate(stream, 1):
                    scores[pid] = scores.get(pid, 0) + weight / (60 + rank)
            expected = sorted(scores, key=lambda p: (-scores[p], p))[:2]
            actual, _ = runner.run_mode("hybrid", products, "apple", vectors=vectors, candidates=2,
                HW2_RRF_DENSE_WEIGHT=str(dw), HW2_RRF_BM25_WEIGHT=str(bw))
            self.assertEqual(actual, expected)

    def test_invalid_weights_fail(self):
        for value in ["nan", "inf", "0", "-1", "bad"]:
            with self.assertRaises(ValueError):
                test_hybrid.HybridTests().run_mode("hybrid", {"a": "red", "b": "blue"}, "red",
                    HW2_RRF_DENSE_WEIGHT=value, HW2_RRF_BM25_WEIGHT="1")

    def test_768d_pid_mapping_pool_mutations_and_boundary_ties(self):
        class Encoder768(FakeEncoder):
            def encode(self, texts, **kwargs):
                low = super().encode(texts, **kwargs)
                return np.pad(low, ((0, 0), (0, 768 - low.shape[1])))
        queries = [{"qid": "q", "query": "unseen"}]
        products = {f"P{i:03}": "same" for i in reversed(range(105))}
        vectors = {"unseen": [1, 0], "same": [1, 0], "opposite": [-1, 0]}
        env = {"HW2_MODEL_PATH": str(ROOT), "HW2_RETRIEVER": "hybrid", "HW2_RERANK": "0",
            "HW2_RRF_CANDIDATES": "100", "HW2_RRF_DENSE_WEIGHT": "5", "HW2_RRF_BM25_WEIGHT": "1",
            "HW2_BM25_QUERY_MODE": "full", "HW2_BM25_LAST_WEIGHT": "1", "HW2_BM25_TITLE_WEIGHT": "1"}
        for fallback in [False, True]:
            for mutation in [False, True]:
                pool = dict(products)
                if mutation:
                    del pool["P001"]
                    pool["P000"] = "opposite"
                    pool["A-new"] = "same"
                modules = {"sentence_transformers": SimpleNamespace(SentenceTransformer=lambda *a, **k: Encoder768(vectors))}
                if fallback:
                    modules["faiss"] = None
                expected = sorted(pid for pid, text in pool.items() if text == "same")[:10]
                for ordered in [pool, dict(reversed(list(pool.items())))]:
                    with patch.dict(os.environ, env), patch.dict(sys.modules, modules):
                        result = retrieve_and_rerank(queries, ordered, top_k=10)
                    self.assertEqual(result, [{"qid": "q", "ranked_pids": expected}])


if __name__ == "__main__":
    unittest.main()
