"""Train-only curation, false-negative safeguards and independent loss oracle."""
import json
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hard_negative_data import choose_candidates, strict_negative, select_verified, batch_negatives
from hard_negative_loss import own_negative_loss


class HardNegativeTests(unittest.TestCase):
    def fixture(self):
        products = {f"p{i}": f"product {i}" for i in range(60)}
        products["duplicate"] = " PRODUCT  0 "
        rows = [{"qid": "train-a", "gold_pid": "p0", "query": "need 0"},
                {"qid": "train-b", "gold_pid": "p1", "query": "need 1"}]
        ranked = {r["qid"]: ["duplicate"] + [f"p{i}" for i in range(60)] for r in rows}
        return rows, products, ranked

    def test_candidate_window_excludes_positive_and_duplicate_text(self):
        rows, products, ranked = self.fixture()
        selected = choose_candidates(rows, products, ranked, starts=(1, 15), maximum=50)
        self.assertEqual(len(selected), 4)
        self.assertNotIn("duplicate", [r["candidate_pid"] for r in selected if r["qid"] == "train-a"])
        for row in selected:
            self.assertNotEqual(row["gold_pid"], row["candidate_pid"])
        with self.assertRaises(ValueError):
            choose_candidates(rows, products, {"dev-x": ["p0"]})

    def test_exact_teacher_false_only_and_alignment(self):
        row = {"raw_output": '{"relevant": false}', "parse_ok": True, "relevant": False}
        self.assertTrue(strict_negative(row))
        for raw in ['false', '{"relevant": true}', '{"relevant": 0}', 'bad', '{"relevant": false, "other": 1}']:
            self.assertFalse(strict_negative({**row, "raw_output": raw}))
        self.assertFalse(strict_negative({**row, "parse_ok": False}))
        rows, products, ranked = self.fixture()
        candidates = choose_candidates(rows, products, ranked)
        judgments = [{**c, **row} for c in candidates]
        selected = select_verified(rows, products, candidates, judgments)
        self.assertEqual(set(selected), {"train-a", "train-b"})
        self.assertEqual(selected["train-a"]["candidate_pid"], candidates[0]["candidate_pid"])
        judgments[0]["qid"] = "dev-x"
        with self.assertRaises(ValueError):
            select_verified(rows, products, candidates, judgments)

    def test_missing_negatives_fall_back_and_positive_stream_is_not_double_counted(self):
        rows, products, _ = self.fixture()
        selected = {"train-a": {"candidate_pid": "p1"}, "train-b": {"candidate_pid": "p9"}}
        indices, texts, masked = batch_negatives(rows, products, selected)
        self.assertEqual((indices, texts, masked), ([1], ["product 9"], 1))
        self.assertEqual(batch_negatives(rows, products, {}), ([], [], 0))

    def test_loss_matches_independent_scalar_probability_and_gradients(self):
        q = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        p = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        n = torch.tensor([[.8, .6]], requires_grad=True)
        loss = own_negative_loss(q, p, n, [0], scale=2)
        expected = (math.log(math.exp(2) + 1 + math.exp(1.6)) - 2 + math.log(math.exp(2) + 1) - 2) / 2
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        for tensor in [q, p, n]:
            self.assertTrue(torch.isfinite(tensor.grad).all())
        plain = own_negative_loss(q.detach(), p.detach(), scale=2)
        self.assertAlmostEqual(plain.item(), math.log(math.exp(2) + 1) - 2, places=6)

    def test_bad_negative_alignment_and_nonfinite_embeddings_fail(self):
        q = torch.eye(2)
        for vectors, indices in [(q, [0]), (q, [0, 0]), (q[:1], [2])]:
            with self.assertRaises(ValueError):
                own_negative_loss(q, q, vectors, indices)
        with self.assertRaises(ValueError):
            own_negative_loss(q * float("nan"), q)


if __name__ == "__main__":
    unittest.main()
