#!/usr/bin/env python3
"""Describe actual MiniLM pair budgets and gold rank changes; no causal claims."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import sha256
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission


def main():
    from transformers import AutoTokenizer

    query_path = ROOT / "data/derived/dev.jsonl"
    product_path = ROOT / "data/train/products.jsonl"
    dense_path = ROOT / "results/w07_b0_dev_top50.json"
    reranked_path = ROOT / "results/w07_a1_dev_top10.json"
    queries, products = load_eval_queries(query_path), load_products(product_path)
    dense = load_submission(dense_path, queries, set(products), top_k=50)
    reranked = load_submission(reranked_path, queries, set(products), top_k=10)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "artifacts/models/ms-marco-MiniLM-L6-v2"),
                                              local_files_only=True, trust_remote_code=False)
    ids = sorted({pid for row in dense.values() for pid in row})
    lengths = {}
    for start in range(0, len(ids), 256):
        batch = ids[start:start + 256]
        encoded = tokenizer([products[pid] for pid in batch], add_special_tokens=False,
                            truncation=False, return_length=True)
        lengths.update(zip(batch, encoded["length"]))
    pairs, truncated, already_long = 0, 0, 0
    records = []
    special = tokenizer.num_special_tokens_to_add(pair=True)
    for query in queries:
        qid, gold = query["qid"], query["gold_pid"]
        query_length = len(tokenizer.encode(query["query"], add_special_tokens=False))
        budget = 512 - special - min(192, query_length)
        for pid in dense[qid]:
            pairs += 1
            truncated += lengths[pid] > budget
            already_long += lengths[pid] > 512 - tokenizer.num_special_tokens_to_add(pair=False)
        gold_in_candidates = gold in dense[qid]
        records.append({"qid": qid, "raw_query_tokens": query_length, "product_budget": budget,
            "gold_in_candidates": gold_in_candidates,
            "gold_fits_pair": lengths[gold] <= budget if gold_in_candidates else None,
            "dense_gold_rank": dense[qid].index(gold) + 1 if gold_in_candidates else None,
            "reranked_gold_rank": reranked[qid].index(gold) + 1 if gold in reranked[qid] else None})
    groups = {}
    for fits in [True, False]:
        subset = [r for r in records if r["gold_in_candidates"] and r["gold_fits_pair"] == fits]
        before = sum(r["dense_gold_rank"] <= 10 for r in subset)
        after = sum(r["reranked_gold_rank"] is not None for r in subset)
        groups["gold_fits_pair" if fits else "gold_truncated"] = {
            "queries": len(subset), "b0_gold_hits10": before, "a1_gold_hits10": after,
            "promoted_into_top10": sum(r["dense_gold_rank"] > 10 and
                r["reranked_gold_rank"] is not None for r in subset),
            "lost_from_top10": sum(r["dense_gold_rank"] <= 10 and
                r["reranked_gold_rank"] is None for r in subset)}
    summary = {"candidate_pairs": pairs, "unique_candidate_products": len(ids),
        "truncated_product_pairs": truncated, "truncated_pair_fraction": truncated / pairs,
        "pairs_exceeding_single_text_512_budget": already_long,
        "query_truncations_at_192": sum(r["raw_query_tokens"] > 192 for r in records),
        "min_product_budget": min(r["product_budget"] for r in records),
        "max_product_budget": max(r["product_budget"] for r in records), "groups": groups,
        "note": "candidate-frequency-weighted truncation; lengths use MiniLM tokenizer; descriptive association, not a truncation ablation",
        "records": records, "input_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
            for p in [query_path, product_path, dense_path, reranked_path, Path(__file__).resolve()]}}
    (ROOT / "results/w07_pair_diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "input_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
