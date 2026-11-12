#!/usr/bin/env python3
"""Gold-only retrieval diagnostics. These are NOT the official graded metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gold_metrics(queries: list[dict], ranked: dict[str, list[str]], cutoffs=(1, 10)) -> dict:
    if not queries:
        raise ValueError("cannot evaluate an empty query set")
    qids = [q["qid"] for q in queries]
    if len(set(qids)) != len(qids) or set(qids) != set(ranked):
        raise ValueError("query IDs and ranking IDs must match exactly")
    records = []
    for query in queries:
        pids = ranked[query["qid"]]
        if len(pids) < max(cutoffs) or len(set(pids)) != len(pids):
            raise ValueError("ranking is too short or has duplicate PIDs")
        gold = query["gold_pid"]
        rank = pids.index(gold) + 1 if gold in pids else None
        records.append({"qid": query["qid"], "gold_pid": gold, "gold_rank": rank})
    metrics = {"queries": len(records)}
    for k in cutoffs:
        metrics[f"gold_hit@{k}"] = sum(r["gold_rank"] is not None and r["gold_rank"] <= k
                                       for r in records) / len(records)
        metrics[f"gold_mrr@{k}"] = sum(1 / r["gold_rank"] if r["gold_rank"] is not None
                                       and r["gold_rank"] <= k else 0 for r in records) / len(records)
    return {"metrics": metrics, "records": records}


def compare_prefix(top10: dict[str, list[str]], top50: dict[str, list[str]]) -> dict:
    if set(top10) != set(top50):
        raise ValueError("cannot compare different query sets")
    changed = [qid for qid in top10 if top10[qid] != top50[qid][:len(top10[qid])]]
    return {"queries": len(top10), "changed_top10_count": len(changed), "changed_qids": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--products", type=Path, default=ROOT / "data/train/products.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-top10", type=Path)
    args = parser.parse_args()
    queries = load_eval_queries(str(args.dialogues))
    passages = load_products(str(args.products))
    if any(q["gold_pid"] not in passages for q in queries):
        raise ValueError("gold is absent from candidate products")
    k = json.loads(args.retrieval_results.read_text())["top_k"]
    if k not in (10, 50):
        raise ValueError("W06 diagnostics expect top-k 10 or 50")
    ranked = load_submission(args.retrieval_results, queries, set(passages), top_k=k)
    cutoffs = (1, 10) if k == 10 else (1, 10, 50)
    report = {"kind": "gold_only_diagnostic_not_official_graded", "candidate_products": len(passages),
              "retrieval_top_k": k, **gold_metrics(queries, ranked, cutoffs)}
    raw_rows = [json.loads(line) for line in args.dialogues.read_text(encoding="utf-8").splitlines() if line.strip()]
    turns = {r["dialogue_id"]: len(r["conversation"]) for r in raw_rows}
    report["by_message_count"] = {}
    for length in sorted(set(turns.values())):
        subset = [q for q in queries if turns[q["qid"]] == length]
        report["by_message_count"][str(length)] = gold_metrics(
            subset, {q["qid"]: ranked[q["qid"]] for q in subset}, cutoffs)["metrics"]
    report["input_sha256"] = {str(p): sha256(p) for p in
                              [args.retrieval_results, args.dialogues, args.products]}
    if args.compare_top10:
        other = load_submission(args.compare_top10, queries, set(passages), top_k=10)
        report["prefix_comparison"] = compare_prefix(other, ranked)
        report["input_sha256"][str(args.compare_top10)] = sha256(args.compare_top10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("records", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
