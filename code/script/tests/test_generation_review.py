import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_generation import summarize


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.outputs = [{"qid": "q", "prompt_id": p, "stop_reason": "eos", "diagnostics": {
            "mentioned_pids": [], "formatted_pids": [], "out_of_evidence_pids": [], "word_count": 20}}
            for p in ["G-P1", "G-P2"]]
        self.review = {"reviewer": "assistant, not independent", "dimensions": ["a", "b", "c", "d", "e"],
            "records": [{"qid": "q", "prompt_id": p, "scores": [2, 1, 0, 2, 1], "assessment": "observed"}
                        for p in ["G-P1", "G-P2"]]}

    def test_subjective_scores_and_measured_diagnostics_remain_separate(self):
        summary = summarize(self.review, self.outputs)
        self.assertEqual(summary["by_prompt"]["G-P1"]["subjective_mean_total_out_of_10"], 6)
        self.assertEqual(summary["by_prompt"]["G-P1"]["measured_diagnostics"]["outputs_mentioning_any_pid"], 0)
        self.assertIn("not_official", summary["kind"])

    def test_missing_duplicate_or_reordered_reviews_raise(self):
        for records in [self.review["records"][:1], self.review["records"] * 2,
                        list(reversed(self.review["records"]))]:
            changed = {**self.review, "records": records}
            with self.assertRaises(ValueError):
                summarize(changed, self.outputs)

    def test_invalid_score_range_or_vector_raises(self):
        for scores in [[3, 1, 0, 2, 1], [float("nan"), 1, 0, 2, 1], [2, 1], [True, 1, 0, 2, 1]]:
            changed = deepcopy(self.review)
            changed["records"][0]["scores"] = scores
            with self.assertRaises(ValueError):
                summarize(changed, self.outputs)


if __name__ == "__main__":
    unittest.main()
