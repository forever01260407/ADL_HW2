import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_splits import balanced_sample, normalize, q2_samples, split_rows


def row(qid, target, text):
    return {"dialogue_id": qid, "target": target,
            "conversation": [{"role": "user", "content": text}]}


class SplitTests(unittest.TestCase):
    def test_transitive_union_and_input_order(self):
        products = {"a": "A cream", "b": "A  CREAM", "c": "different", "d": "last"}
        rows = [row("1", "a", "one"), row("2", "a", "two"),
                row("3", "b", " three "), row("4", "c", "THREE"), row("5", "d", "five")]
        result = split_rows(rows, products)
        self.assertEqual(result, split_rows(list(reversed(rows)), dict(reversed(list(products.items())))))
        train, dev, groups = result
        self.assertEqual(sorted(len(g["qids"]) for g in groups), [1, 4])
        self.assertEqual(len(train) + len(dev), 5)
        self.assertFalse({r["target"] for r in train} & {r["target"] for r in dev})
        self.assertEqual(len(products), 4)  # split never removes candidate products

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            split_rows([row("", "a", "x")], {"a": "a"})
        with self.assertRaisesRegex(ValueError, "duplicate dialogue"):
            split_rows([row("x", "a", "x")] * 2, {"a": "a"})
        with self.assertRaisesRegex(ValueError, "missing target"):
            split_rows([row("x", "a", "x")], {})
        with self.assertRaisesRegex(ValueError, "independent groups"):
            split_rows([row("x", "a", "x")], {"a": "a"})

    def test_balanced_sample_and_label_free_q2(self):
        rows = [row(str(i), str(i), "not sure what I need") for i in range(120)]
        for i, r in enumerate(rows):
            r["conversation"] *= 3 if i % 2 else 5
        sample = balanced_sample(rows, 100, 42)
        self.assertEqual(sum(len(r["conversation"]) == 3 for r in sample), 50)
        self.assertEqual(sample, balanced_sample(list(reversed(rows)), 100, 42))
        q2 = q2_samples(rows, 42)
        self.assertEqual(len({r["dialogue_id"] for r in q2}), 5)
        self.assertNotIn('"target"', json.dumps(q2))
        self.assertEqual(normalize("Ａ  CREAM\n"), "a cream")

    def test_roles_order_and_negation_are_distinct(self):
        rows = [row("1", "a", "scented"), row("2", "b", "not scented"),
                row("3", "c", "scented")]
        rows[2]["conversation"][0]["role"] = "assistant"
        _, _, groups = split_rows(rows, {"a": "A", "b": "B", "c": "C"})
        self.assertEqual(len(groups), 3)

    def test_real_split_invariants(self):
        root = Path(__file__).resolve().parents[3]
        manifest = json.loads((root / "code/script/configs/split_manifest.json").read_text())
        from prepare_splits import read_jsonl, digest
        source = read_jsonl(root / "data/train/dialogues.jsonl")
        products = {r["parent_asin"]: normalize(r["title"] + "\n" + r["description"])
                    for r in read_jsonl(root / "data/train/products.jsonl")}
        train = read_jsonl(root / "data/derived/train.jsonl")
        dev = read_jsonl(root / "data/derived/dev.jsonl")
        tq, dq = ({r["dialogue_id"] for r in part} for part in [train, dev])
        self.assertFalse(tq & dq)
        self.assertEqual(tq | dq, {r["dialogue_id"] for r in source})
        for key in [lambda r: r["target"], lambda r: products[r["target"]],
                    lambda r: json.dumps([(t["role"], normalize(t["content"])) for t in r["conversation"]])]:
            self.assertFalse({key(r) for r in train} & {key(r) for r in dev})
        self.assertEqual(manifest["candidate_pool"]["count"], len(products))
        for name, part in manifest["partitions"].items():
            self.assertEqual(part["sha256"], digest(root / f"data/derived/{name}.jsonl"))
        self.assertEqual(len(set(manifest["partitions"]["dev_fast"]["qids"])), 100)
        self.assertTrue(set(manifest["partitions"]["dev_fast"]["qids"]) <= dq)
        self.assertTrue(set(manifest["partitions"]["q2_queries"]["qids"]) <= dq)


if __name__ == "__main__":
    unittest.main()
