"""H2B pre-Judge selection and independent rank-formula verification."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extend_rrf_ablation import choose_variant, graded_group, verify_fusion
from run_rrf_ablation import weighted_fuse


class RRFExtensionTests(unittest.TestCase):
    def candidate(self, mrr, hit10, hit50):
        return {"metrics": {"gold_mrr@10": mrr, "gold_hit@10": hit10, "gold_hit@50": hit50}}

    def test_selection_uses_frozen_mrr_then_hit10_then_hit50(self):
        candidates = {"old": self.candidate(.5, .8, .9), "new": self.candidate(.51, .7, .85)}
        self.assertEqual(choose_variant(candidates), "new")
        candidates["new"] = self.candidate(.5, .81, .85)
        self.assertEqual(choose_variant(candidates), "new")
        candidates["new"] = self.candidate(.5, .8, .91)
        self.assertEqual(choose_variant(candidates), "new")
        candidates["new"] = self.candidate(.5, .8, .9)
        self.assertEqual(choose_variant(candidates), "old")
        for invalid in [{}, {"bad": self.candidate(float("nan"), .8, .9)}]:
            with self.assertRaises(ValueError):
                choose_variant(invalid)

    def test_independent_formula_all_extended_weights_and_tamper_rejection(self):
        dense, lexical, positive = {"q": ["z", "b", "c"]}, {"q": ["a", "c", "b"]}, {"q": 2}
        for dw in [1, 4, 5, 6, 8, 10, 10000]:
            rows = weighted_fuse(dense, lexical, positive, dw, 1, k=3)
            ranked = {"q": rows[0]["ranked_pids"]}
            self.assertEqual(verify_fusion(dense, lexical, positive, ranked, dw, 1), 1)
            with self.assertRaises(ValueError):
                verify_fusion(dense, lexical, positive, {"q": list(reversed(ranked["q"]))}, dw, 1)

    def test_graded_pairing_preserves_recall_mrr_tradeoff(self):
        old = {"a": {"graded_recall@10": 0, "graded_mrr@10": 0},
               "b": {"graded_recall@10": 1, "graded_mrr@10": 1}}
        new = {"a": {"graded_recall@10": 1, "graded_mrr@10": .1},
               "b": {"graded_recall@10": 1, "graded_mrr@10": .5}}
        result = graded_group(new, old, ["a", "b"])
        self.assertEqual(result["paired"]["graded_recall@10"]["mean_difference"], .5)
        self.assertAlmostEqual(result["paired"]["graded_mrr@10"]["mean_difference"], -.2)
        self.assertEqual(result["paired"]["graded_mrr@10"]["improved_qids"], ["a"])
        self.assertEqual(result["paired"]["graded_mrr@10"]["worsened_qids"], ["b"])


if __name__ == "__main__":
    unittest.main()
