import unittest
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_bm25_representation import gate, fuse


class ScreeningTests(unittest.TestCase):
    def test_gate_requires_both_groups_and_all_three_metrics(self):
        metrics = {"gold_hit@10": .5, "gold_mrr@10": .3, "gold_hit@50": .7}
        baseline = {"metrics": dict(metrics), "by_qid_prefix": {"beauty-train": dict(metrics)}}
        candidate = deepcopy(baseline)
        self.assertFalse(gate(candidate, baseline))
        candidate["metrics"]["gold_mrr@10"] += .1
        self.assertFalse(gate(candidate, baseline))
        candidate["by_qid_prefix"]["beauty-train"]["gold_mrr@10"] += .1
        self.assertTrue(gate(candidate, baseline))
        candidate["metrics"]["gold_hit@50"] -= .01
        self.assertFalse(gate(candidate, baseline))

    def test_shared_dense_fusion_does_not_reward_zero_lexical_candidates(self):
        result = fuse({"q": ["b", "a"]}, {"q": ["a", "b"]}, {"q": 0}, k=2)
        self.assertEqual(result, [{"qid": "q", "ranked_pids": ["b", "a"]}])
        result = fuse({"q": ["b", "a"]}, {"q": ["a", "b"]}, {"q": 1}, k=2)
        self.assertEqual(result[0]["ranked_pids"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
