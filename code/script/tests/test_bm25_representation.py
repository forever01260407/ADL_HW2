"""Role filtering and lexical-only weighting must not alter Dense evidence."""
import unittest
import json
import tempfile
from pathlib import Path

import test_hybrid


class RepresentationTests(unittest.TestCase):
    run_mode = test_hybrid.HybridTests.run_mode

    def test_assistant_only_terms_removed_and_earlier_user_constraints_kept(self):
        raw = "user: lotion\nassistant: unicorn glitter sparkle\nuser: red"
        products = {"a": "unicorn glitter sparkle", "b": "red lotion"}
        full, _ = self.run_mode("bm25", products, raw, k=1)
        users, _ = self.run_mode("bm25", products, raw, k=1, HW2_BM25_QUERY_MODE="users")
        self.assertEqual(full, ["a"])
        self.assertEqual(users, ["b"])
        for final in ["red", "not scented"]:
            result, _ = self.run_mode("bm25", {"a": "not scented red lotion", "b": "unicorn glitter sparkle"},
                "user: lotion\nassistant: unicorn glitter sparkle\nuser: " + final,
                k=1, HW2_BM25_QUERY_MODE="users")
            self.assertEqual(result, ["a"])

    def test_short_reference_restores_only_adjacent_assistant_and_does_not_boost(self):
        products = {"a": "facial mask", "b": "unicorn glitter sparkle"}
        for ending in ["yes", "that one", "Yes, exactly!", "sounds good"]:
            raw = "user: skin\nassistant: unicorn glitter sparkle\nuser: care\nassistant: facial mask\nuser: " + ending
            result, _ = self.run_mode("bm25", products, raw, k=1,
                HW2_BM25_QUERY_MODE="users", HW2_BM25_LAST_WEIGHT="2")
            self.assertEqual(result, ["a"])

    def test_multiline_and_embedded_colons_are_content_and_bad_roles_fall_back(self):
        products = {"a": "not scented mango", "b": "unicorn glitter sparkle"}
        raw = "user: lotion\nIngredients: not scented\nI wrote assistant: mango\nassistant: unicorn glitter sparkle\nuser: red"
        result, _ = self.run_mode("bm25", products, raw, k=1, HW2_BM25_QUERY_MODE="users")
        self.assertEqual(result, ["a"])
        for malformed in ["unicorn glitter sparkle", "assistant: unicorn glitter sparkle",
                          "user: mango\nuser: unicorn glitter sparkle", "user: mango\nassistant: unicorn"]:
            full, _ = self.run_mode("bm25", products, malformed)
            users, _ = self.run_mode("bm25", products, malformed, HW2_BM25_QUERY_MODE="users")
            self.assertEqual(users, full)

    def test_latest_query_weight_changes_score_despite_unique_term_policy(self):
        raw = "user: old\nassistant: old\nuser: new"
        products = {"a": "old old", "b": "new"}
        base, _ = self.run_mode("bm25", products, raw, k=1, HW2_BM25_QUERY_MODE="users")
        boosted, _ = self.run_mode("bm25", products, raw, k=1,
            HW2_BM25_QUERY_MODE="users", HW2_BM25_LAST_WEIGHT="2")
        self.assertEqual(base, ["a"])
        self.assertEqual(boosted, ["b"])

    def test_title_tf_and_user_filter_do_not_mutate_dense_input_or_documents(self):
        products = {"a": "neutral\nlotion", "b": "lotion\nneutral"}
        raw = "user: lotion\nassistant: unicorn\nuser: not scented"
        base, _ = self.run_mode("bm25", products, raw, k=1, HW2_BM25_QUERY_MODE="users")
        boosted, _ = self.run_mode("bm25", products, raw, k=1,
            HW2_BM25_QUERY_MODE="users", HW2_BM25_TITLE_WEIGHT="2")
        self.assertEqual(base, ["a"])
        self.assertEqual(boosted, ["b"])
        for extra in [{}, {"HW2_BM25_LAST_WEIGHT": "2"}, {"HW2_BM25_TITLE_WEIGHT": "2"}]:
            _, encoder = self.run_mode("hybrid", products, raw, HW2_BM25_QUERY_MODE="users", **extra)
            self.assertEqual(encoder.calls[0], [products[p] for p in sorted(products)])
            self.assertEqual(encoder.calls[1], ["Represent this sentence for searching relevant passages: " + raw])
        self.assertEqual(products, {"a": "neutral\nlotion", "b": "lotion\nneutral"})

    def test_invalid_representation_configuration_fails(self):
        for env in [{"HW2_BM25_QUERY_MODE": "last"}, {"HW2_BM25_LAST_WEIGHT": "0"},
                    {"HW2_BM25_TITLE_WEIGHT": "3"}, {"HW2_BM25_LAST_WEIGHT": "2"}]:
            with self.assertRaises(ValueError):
                self.run_mode("bm25", {"a": "a", "b": "b"}, "query", **env)

    def test_development_trace_excludes_zero_matches_and_preserves_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            ranked, _ = self.run_mode("bm25", {"a": "red", "b": "blue"}, "unseen",
                                     HW2_BM25_TRACE_PATH=str(path))
            self.assertEqual(ranked, ["a", "b"])
            self.assertEqual(json.loads(path.read_text())["results"], [{"qid": "q", "positive_count": 0}])


if __name__ == "__main__":
    unittest.main()
