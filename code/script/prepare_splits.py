#!/usr/bin/env python3
"""Create grouped train/dev supervision; keep the complete product pool intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_rows(rows: list[dict], products: dict[str, str], seed: int = 42,
               dev_fraction: float = .2) -> tuple[list[dict], list[dict], list[dict]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    rows = sorted(rows, key=lambda row: row["dialogue_id"])
    if any(not isinstance(row["dialogue_id"], str) or not row["dialogue_id"].strip() for row in rows):
        raise ValueError("dialogue_id must be a non-empty string")
    if len({row["dialogue_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate dialogue_id")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        if row["target"] not in products:
            raise ValueError(f"missing target product: {row['target']}")
        conversation = json.dumps([(turn["role"], normalize(turn["content"]))
                                   for turn in row["conversation"]], ensure_ascii=False)
        keys = [("pid", row["target"]), ("conversation", conversation),
                ("product_text", normalize(products[row["target"]]))]
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    groups = sorted(components.values(), key=lambda group: group[0]["dialogue_id"])
    if len(groups) < 2:
        raise ValueError("need at least two independent groups for train/dev")
    random.Random(seed).shuffle(groups)
    target = len(rows) * dev_fraction
    dev, train, assignments = [], [], []
    for i, group in enumerate(groups):
        use_dev = len(dev) < target and i < len(groups) - 1
        (dev if use_dev else train).extend(group)
        assignments.append({"group_id": group[0]["dialogue_id"],
                            "partition": "dev" if use_dev else "train",
                            "qids": [row["dialogue_id"] for row in group]})
    return (sorted(train, key=lambda r: r["dialogue_id"]),
            sorted(dev, key=lambda r: r["dialogue_id"]), assignments)


def balanced_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["dialogue_id"]):
        buckets[len(row["conversation"])].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < min(count, len(rows)):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
    return selected


def q2_samples(dev: list[dict], seed: int) -> list[dict]:
    # Fixed before generation. Tags describe lexical selection, not verified semantics.
    pool = balanced_sample(dev, len(dev), seed)
    selected, used = [], set()
    rules = [("three_messages", lambda r: len(r["conversation"]) == 3),
             ("five_messages", lambda r: len(r["conversation"]) == 5),
             ("negation_cue", lambda r: any(w in normalize(str(r["conversation"]))
                                            for w in [" not ", "without", "don't"])),
             ("history_cue", lambda r: any(w in normalize(str(r["conversation"]))
                                           for w in ["last time", "used to", "previous", "before"])),
             ("uncertainty_cue", lambda r: any(w in normalize(str(r["conversation"]))
                                               for w in ["not sure", "maybe", "unsure"]))]
    for tag, predicate in rules:
        remaining = [r for r in pool if r["dialogue_id"] not in used]
        if not remaining:
            raise ValueError("need five dev dialogues for Q2")
        matches = [r for r in remaining if predicate(r)]
        row = (matches or remaining)[0]
        used.add(row["dialogue_id"])
        selected.append({"dialogue_id": row["dialogue_id"],
                         "conversation": row["conversation"],
                         "selection_tag": tag if matches else "fallback"})
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dialogue_path, product_path = ROOT / "data/train/dialogues.jsonl", ROOT / "data/train/products.jsonl"
    rows, product_rows = read_jsonl(dialogue_path), read_jsonl(product_path)
    products = {r["parent_asin"]: r["title"] + "\n" + r["description"] for r in product_rows}
    if len(products) != len(product_rows):
        raise ValueError("duplicate product PID")
    train, dev, groups = split_rows(rows, products, args.seed)
    output = ROOT / "data/derived"
    output.mkdir(parents=True, exist_ok=True)
    selections = {"train": train, "dev": dev,
                  "dev_fast": balanced_sample(dev, 100, args.seed),
                  "q2_queries": q2_samples(dev, args.seed)}
    for name, selection in selections.items():
        write_jsonl(output / f"{name}.jsonl", selection)
    manifest = {
        "version": 1, "seed": args.seed, "requested_dev_fraction": .2,
        "grouping": ["same_target_pid", "NFKC_casefold_whitespace_exact_conversation",
                     "NFKC_casefold_whitespace_exact_target_product_text", "transitive_union"],
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in [dialogue_path, product_path]},
        "candidate_pool": {"path": "data/train/products.jsonl", "count": len(products)},
        "partitions": {name: {"count": len(selection),
            "qids": [r["dialogue_id"] for r in selection],
            "turn_counts": dict(Counter(len(r["conversation"]) for r in selection)),
            "sha256": digest(output / f"{name}.jsonl")} for name, selection in selections.items()},
        "groups": groups,
    }
    path = ROOT / "code/script/configs/split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": {k: len(v) for k, v in selections.items()},
                      "groups": len(groups), "candidate_products": len(products)}))


if __name__ == "__main__":
    main()
