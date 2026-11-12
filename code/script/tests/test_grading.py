import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_w10 import estimate_points


class GradingTests(unittest.TestCase):
    def score(self, recall, mrr, split="public"):
        return estimate_points({"graded_recall@10": recall, "graded_mrr@10": mrr}, split)

    def test_independent_cells_and_additive_public_points(self):
        score = self.score(.9, .7)
        self.assertEqual(score["estimated_points_inclusive"], 6)
        self.assertEqual(score["available_points"], 8)
        self.assertEqual(self.score(.96, .81)["estimated_points_inclusive"], 8)

    def test_recall_does_not_imply_mrr_points(self):
        self.assertEqual(self.score(.96, .60)["estimated_points_inclusive"], 4)
        self.assertEqual(self.score(.80, .70)["estimated_points_inclusive"], 3)

    def test_exact_boundaries_are_reported_both_ways(self):
        score = self.score(.860, .650)
        self.assertEqual(score["estimated_points_inclusive"], 4)
        self.assertEqual(score["estimated_points_strict"], 0)
        self.assertEqual(sum(c["relation"] == "equal" for c in score["cells"]), 2)

    def test_private_thresholds_and_unknown_scores_are_not_inferred(self):
        self.assertEqual(self.score(.9, .7, "private")["estimated_points_inclusive"], 4)
        score = self.score(.96, .81)
        self.assertFalse(score["private_points_known"])
        self.assertFalse(score["report_points_known"])
        self.assertEqual(self.score(.80, .60)["estimated_points_inclusive"], 0)

    def test_invalid_metrics_raise(self):
        for value in [float("nan"), float("inf"), -1, 1.01, "0.9"]:
            with self.assertRaises(ValueError):
                self.score(value, .7)


if __name__ == "__main__":
    unittest.main()
