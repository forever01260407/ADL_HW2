"""H3 preserves actual W08 negatives and evaluates encoder substitution fairly."""
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base_selection import adoption_gate, gold_gate, paired_bootstrap
from train import ROOT, load_data
from train_base import reference_batches, verify_checkpoint_shape


class BaseTrainingTests(unittest.TestCase):
    def policy(self):
        return json.loads((ROOT / "code/script/configs/h3_experiment.json").read_text())

    def test_real_batch_qids_exactly_match_w08(self):
        rows, products, _ = load_data()
        config = json.loads((ROOT / "code/script/configs/w08_train.json").read_text())
        batches, qids = reference_batches(rows, products, config)
        self.assertEqual(len(rows), 3864)
        self.assertEqual(len(batches), 242)
        self.assertEqual(len(batches[-1]), 8)
        self.assertEqual(len([qid for batch in qids for qid in batch]), 3864)

    def test_shape_checks_allow_768_without_accepting_small_shape(self):
        import numpy as np
        base = np.zeros((2, 768), dtype=np.float32)
        verify_checkpoint_shape(base, base, 768, 512)
        for actual, length in [(np.zeros((2, 384)), 512), (base, 256), (base + float("nan"), 512)]:
            with self.assertRaises(ValueError):
                verify_checkpoint_shape(actual, base, 768, length)

    def test_gold_gate_requires_both_views_and_clear_hybrid_gain(self):
        old = {"queries": 100, "gold_hit@10": .7, "gold_mrr@10": .5, "gold_hit@50": .8}
        new = {**old, "gold_hit@10": .71, "gold_mrr@10": .51}
        p = self.policy()["gold_gate"]
        self.assertTrue(gold_gate(new, new, old, old, p)["passed"])
        self.assertFalse(gold_gate(new, {**new, "gold_mrr@10": .509}, old, old, p)["passed"])
        self.assertFalse(gold_gate({**old, "gold_hit@10": .69}, new, old, old, p)["passed"])
        self.assertFalse(gold_gate(new, {**new, "gold_hit@50": .79}, old, old, p)["passed"])

    def test_adoption_checks_strongest_small_and_category_regression(self):
        old = {"graded_recall@10": .85, "graded_mrr@10": .63}
        new = {"graded_recall@10": .86, "graded_mrr@10": .65}
        groups = {"pet": dict(old)}
        p = self.policy()["adoption_gate"]
        self.assertTrue(adoption_gate(new, old, old, {"pet": dict(new)}, groups, p)["passed"])
        self.assertFalse(adoption_gate(new, old, {**old, "graded_mrr@10": .645}, {"pet": new}, groups, p)["passed"])
        bad = {"pet": {**new, "graded_mrr@10": .59}}
        self.assertFalse(adoption_gate(new, old, old, bad, groups, p)["passed"])

    def test_bootstrap_is_paired_reproducible_and_marks_scope(self):
        result = paired_bootstrap([.02] * 10, samples=100)
        self.assertAlmostEqual(result["ci95"][0], .02)
        self.assertAlmostEqual(result["ci95"][1], .02)
        self.assertEqual(result, paired_bootstrap([.02] * 10, samples=100))
        with self.assertRaises(ValueError):
            paired_bootstrap([float("nan")])


if __name__ == "__main__":
    unittest.main()
