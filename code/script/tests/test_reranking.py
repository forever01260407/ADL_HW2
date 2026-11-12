"""Candidate isolation, exact pair budgets, batching, and reranker failure contracts."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_retrieval import ROOT, FakeEncoder
from retrieval_utils import retrieve_and_rerank


class PairTokenizer:
    def __init__(self):
        self.pairs = []

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(c) + 3 for c in text]
        return ids[:max_length] if truncation else ids

    def prepare_for_model(self, ids, pair_ids, **kwargs):
        self.pairs.append((list(ids), list(pair_ids)))
        combined = [1] + ids + [2] + pair_ids + [2]
        return {"input_ids": combined, "attention_mask": [1] * len(combined),
                "token_type_ids": [0] * (len(ids) + 2) + [1] * (len(pair_ids) + 1)}

    def pad(self, rows, **kwargs):
        length = max(len(row["input_ids"]) for row in rows)
        return {key: torch.tensor([row[key] + [0] * (length - len(row[key])) for row in rows])
                for key in rows[0]}


class PairModel:
    def __init__(self, scoring):
        self.scoring, self.batch_lengths = scoring, []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, token_type_ids, attention_mask):
        self.batch_lengths.append(len(input_ids))
        scores = []
        for ids, types, mask in zip(input_ids.tolist(), token_type_ids.tolist(), attention_mask.tolist()):
            query = "".join(chr(i - 3) for i, t, m in zip(ids, types, mask) if m and not t and i > 2)
            product = "".join(chr(i - 3) for i, t, m in zip(ids, types, mask) if m and t and i > 2)
            scores.append([self.scoring(query, product)])
        return SimpleNamespace(logits=torch.tensor(scores, dtype=torch.float32))


class RerankingTests(unittest.TestCase):
    def run_rerank(self, queries, products, scoring, top_k=2, candidates=3, batch=4, vectors=None, **env):
        encoder = FakeEncoder(vectors or {})
        encoder.device = torch.device("cpu")
        encoder.to = lambda device: encoder
        tokenizer, scorer = PairTokenizer(), PairModel(scoring)
        modules = {
            "sentence_transformers": SimpleNamespace(SentenceTransformer=lambda *a, **k: encoder),
            "transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
                AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=lambda *a, **k: scorer)),
        }
        settings = {"HW2_MODEL_PATH": str(ROOT), "HW2_RERANK_MODEL_PATH": str(ROOT), "HW2_RETRIEVER": "dense",
                    "HW2_RERANK": "1", "HW2_RERANK_CANDIDATES": str(candidates),
                    "HW2_RERANK_BATCH_SIZE": str(batch), **env}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, settings):
            result = retrieve_and_rerank(queries, products, top_k=top_k)
        return result, tokenizer, scorer

    def test_only_candidates_and_cross_query_batch_mapping(self):
        products = {"a": "A", "b": "B", "c": "C", "d": "D"}
        queries = [{"qid": "first", "query": "forward"}, {"qid": "second", "query": "reverse"}]
        result, tokenizer, scorer = self.run_rerank(
            queries, products, lambda q, p: ord(p) * (-1 if q == "reverse" else 1))
        self.assertEqual(result, [{"qid": "first", "ranked_pids": ["c", "b"]},
                                  {"qid": "second", "ranked_pids": ["a", "b"]}])
        self.assertEqual(scorer.batch_lengths, [4, 2])  # first batch crosses query boundary
        self.assertTrue(all(chr(pair[1][0] - 3) != "D" for pair in tokenizer.pairs))

    def test_ties_preserve_dense_order_not_pid_order(self):
        products = {"a": "A", "b": "B", "c": "C"}
        vectors = {"A": [0, 1], "B": [1, 0], "C": [.5, .5]}
        result, _, _ = self.run_rerank([{"qid": "q", "query": "q"}], products,
                                     lambda q, p: 1, vectors=vectors)
        self.assertEqual(result[0]["ranked_pids"], ["b", "c"])

    def test_token_budget_preserves_last_correction_and_raw_roles(self):
        raw = "user: lotion " + "old details " * 100 + "user: not scented."
        _, tokenizer, _ = self.run_rerank([{"qid": "q", "query": raw}],
            {"a": "A" * 1000, "b": "B" * 1000}, lambda q, p: 1)
        for qids, pids in tokenizer.pairs:
            query = "".join(chr(i - 3) for i in qids)
            self.assertEqual(len(qids), 192)
            self.assertEqual(len(qids) + len(pids) + 3, 512)
            self.assertTrue(query.startswith("user: lotion "))
            self.assertTrue(query.endswith("user: not scented."))
            self.assertNotIn("Represent this sentence", query)
        short = "user: A\nassistant: B\nuser: not C"
        _, tokenizer, _ = self.run_rerank([{"qid": "q", "query": short}],
                                         {"a": "A", "b": "B"}, lambda q, p: 1)
        self.assertEqual("".join(chr(i - 3) for i in tokenizer.pairs[0][0]), short)

    def test_candidate_floor_pool_cap_and_topk(self):
        for count, requested, top_k in [(3, 50, 2), (5, 2, 4)]:
            products = {str(i): str(i) for i in range(count)}
            result, tokenizer, _ = self.run_rerank([{"qid": "q", "query": "q"}], products,
                lambda q, p: int(p), top_k=top_k, candidates=requested)
            candidate_count = min(count, max(top_k, requested))
            self.assertEqual(len(tokenizer.pairs), candidate_count)
            self.assertEqual(result[0]["ranked_pids"],
                             [str(i) for i in reversed(range(candidate_count))][:top_k])

    def test_nonfinite_scores_and_bad_configuration_fail(self):
        queries, products = [{"qid": "q", "query": "q"}], {"a": "A", "b": "B"}
        for score in [float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "finite score"):
                self.run_rerank(queries, products, lambda q, p: score)
        for settings in [{"HW2_RERANK": "invalid"}, {"HW2_RERANK_BATCH_SIZE": "0"},
                         {"HW2_RERANK_CANDIDATES": "0"}]:
            with self.assertRaises(ValueError):
                self.run_rerank(queries, products, lambda q, p: 0, **settings)
        with self.assertRaises(FileNotFoundError):
            self.run_rerank(queries, products, lambda q, p: 0,
                           HW2_RERANK_MODEL_PATH=str(ROOT / "missing-reranker-test"))


if __name__ == "__main__":
    unittest.main()
