"""H2 rank-fusion semantics and the mandatory W08 equivalence gate."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_bm25_representation import fuse
from run_rrf_ablation import category, paired, require_baseline, weighted_fuse


class RRFAbllationTests(unittest.TestCase):
    def test_equal_weights_match_existing_fusion_including_zero_votes(self):
        dense = {"q1": ["d", "b", "c"], "q2": ["c", "b", "a"]}
        lexical = {"q1": ["a", "b", "d"], "q2": ["a", "b", "c"]}
        for positive in [{"q1": 3, "q2": 0}, {"q1": 1, "q2": 2}]:
            self.assertEqual(weighted_fuse(dense, lexical, positive, k=3),
                             fuse(dense, lexical, positive, k=3))

    def test_weighted_independent_hand_calculation_and_pid_tie(self):
        # Shared c gets 2/62 + 1/62, d gets 2/61, b gets 1/61.
        self.assertEqual(weighted_fuse({"q": ["d", "c"]}, {"q": ["b", "c"]},
                                      {"q": 2}, 2, 1, k=3)[0]["ranked_pids"], ["c", "d", "b"])
        self.assertEqual(weighted_fuse({"q": ["z"]}, {"q": ["a"]},
                                      {"q": 1}, k=2)[0]["ranked_pids"], ["a", "z"])

    def test_dense_only_ignores_lexical_and_excludes_zero_weight_pids(self):
        dense = {"q": ["z", "b", "c"]}
        result = weighted_fuse(dense, {"q": ["a", "c", "b"]}, {"q": 3}, 1, 0, k=3)
        self.assertEqual(result[0]["ranked_pids"], dense["q"])

    def test_weight_two_excludes_bm25_only_from_top50_but_small_weight_does_not(self):
        dense = {"q": [f"d{i:03}" for i in range(100)]}
        lexical = {"q": [f"b{i:03}" for i in range(100)]}
        high = weighted_fuse(dense, lexical, {"q": 100}, 2, 1)[0]["ranked_pids"]
        self.assertEqual(high, dense["q"][:50])
        low = weighted_fuse(dense, lexical, {"q": 100}, 1.1, 1)[0]["ranked_pids"]
        self.assertIn("b000", low[:10])

    def test_invalid_weights_coverage_duplicates_counts_and_short_output(self):
        for dw, bw in [(-1, 1), (float("nan"), 1), (1, float("inf")), (0, 0), (True, 1)]:
            with self.assertRaises(ValueError):
                weighted_fuse({"q": ["a"]}, {"q": ["b"]}, {"q": 1}, dw, bw, k=1)
        for dense, lexical, counts, k in [({"q": ["a"]}, {"x": ["a"]}, {"q": 1}, 1),
                ({"q": ["a", "a"]}, {"q": ["b"]}, {"q": 1}, 1),
                ({"q": ["a"]}, {"q": ["b"]}, {"q": 2}, 1),
                ({"q": ["a"]}, {"q": ["b"]}, {"q": 0}, 2)]:
            with self.assertRaises(ValueError):
                weighted_fuse(dense, lexical, counts, k=k)

    def test_equivalence_gate_rejects_any_changed_rank_or_coverage(self):
        self.assertEqual(require_baseline({"q": ["a", "b"]}, {"q": ["a", "b"]})["changed_queries"], 0)
        for rebuilt in [{"q": ["b", "a"]}, {"other": ["a", "b"]}]:
            with self.assertRaises(ValueError):
                require_baseline(rebuilt, {"q": ["a", "b"]})

    def test_category_merge_and_paired_rank_boundaries(self):
        self.assertEqual(category("beauty-train-0001"), "beauty")
        self.assertEqual(category("train-beauty-official-0001"), "beauty")
        self.assertEqual(category("train-pet-supplies-0001"), "pet-supplies")
        old = {"records": [{"qid": "a", "gold_rank": 11}, {"qid": "b", "gold_rank": 1},
                           {"qid": "c", "gold_rank": None}]}
        new = {"records": [{"qid": "a", "gold_rank": 10}, {"qid": "b", "gold_rank": 2},
                           {"qid": "c", "gold_rank": 50}]}
        result = paired(new, old, ["a", "b", "c"])
        self.assertEqual(result["gold_hit@10"]["improved"], 1)
        self.assertEqual(result["gold_mrr@10"]["worsened"], 1)
        self.assertEqual(result["gold_hit@50"]["improved"], 1)
        self.assertAlmostEqual(result["gold_mrr@10"]["mean_difference"], (.1 - .5) / 3)


if __name__ == "__main__":
    unittest.main()
