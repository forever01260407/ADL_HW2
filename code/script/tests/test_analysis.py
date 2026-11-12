import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_results import compare_prefix, gold_metrics
from evaluate import graded_mrr_at_k, graded_recall_at_k
from smoke_judge import length_summary
from audit_w06 import audit_graded
from evaluate import score_results


class AnalysisTests(unittest.TestCase):
    def test_gold_miss_and_rank_cutoffs(self):
        queries = [{"qid": "first", "gold_pid": "a"}, {"qid": "second", "gold_pid": "b"},
                   {"qid": "miss", "gold_pid": "z"}]
        ranked = {"first": ["a", "b", "c"], "second": ["c", "b", "a"], "miss": ["a", "b", "c"]}
        result = gold_metrics(queries, ranked, (1, 3))
        self.assertAlmostEqual(result["metrics"]["gold_hit@1"], 1 / 3)
        self.assertAlmostEqual(result["metrics"]["gold_hit@3"], 2 / 3)
        self.assertAlmostEqual(result["metrics"]["gold_mrr@3"], .5)
        self.assertIsNone(result["records"][2]["gold_rank"])

    def test_alternatives_differ_from_gold_only(self):
        diagnostic = gold_metrics([{"qid": "q", "gold_pid": "gold"}], {"q": ["alternative", "x"]}, (1, 2))
        self.assertEqual(diagnostic["metrics"]["gold_hit@2"], 0)
        self.assertEqual(graded_recall_at_k([1, 0], 2), 1)
        self.assertEqual(graded_mrr_at_k([0, 1], 2), .5)

    def test_prefix_changes_and_invalid_coverage(self):
        self.assertEqual(compare_prefix({"q": ["a", "b"]}, {"q": ["a", "b", "c"]})["changed_top10_count"], 0)
        self.assertEqual(compare_prefix({"q": ["b", "a"]}, {"q": ["a", "b", "c"]})["changed_qids"], ["q"])
        for queries, ranked in [([], {}), ([{"qid": "q", "gold_pid": "a"}], {"other": ["a"]})]:
            with self.assertRaises(ValueError):
                gold_metrics(queries, ranked, (1,))
        with self.assertRaises(ValueError):
            compare_prefix({"q": ["a"]}, {"z": ["a"]})

    def test_token_budget_boundary(self):
        summary = length_summary([10, 20, 30], 20)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["over_limit"], 1)  # equal to limit is valid
        self.assertEqual(summary["p50"], 20)
        self.assertEqual(summary["max"], 30)

    def test_audit_uses_saved_raw_decisions_and_detects_tampering(self):
        queries = [{"qid": "q", "gold_pid": "gold"}]
        ranking = {"q": ["alternative", "gold"] + [f"wrong-{i}" for i in range(8)]}
        decisions = {("q", pid): {"relevant": pid == "alternative", "parse_ok": pid != "wrong-0",
                     "raw_output": '{"relevant": true}' if pid == "alternative" else
                         ('invalid' if pid == "wrong-0" else '{"relevant": false}')}
                     for pid in ranking["q"] if pid != "gold"}
        metrics, records = score_results({"q": queries[0]}, ranking, decisions, top_k=10)
        summary = {"metrics": metrics, "records": records,
                   "judge": {"parse_failures": 1, "comparisons": 9, "runtime_seconds": 1}}
        audit = audit_graded(summary, queries, ranking)
        self.assertEqual(audit["metrics"]["graded_mrr@10"], 1)
        altered = deepcopy(summary)
        altered["records"][0]["candidates"][0]["raw_output"] = '{"relevant": false}'
        with self.assertRaisesRegex(ValueError, "gain/source"):
            audit_graded(altered, queries, ranking)


if __name__ == "__main__":
    unittest.main()
