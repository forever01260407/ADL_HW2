#!/usr/bin/env python3
"""Audit saved W06 results and publish a compact, reproducible summary."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import compare_prefix, sha256
from eval_utils import load_eval_queries, load_products
from evaluate import graded_mrr_at_k, graded_recall_at_k, load_submission, parse_judgment
from smoke_retrieval import PROTECTED


def audit_graded(summary: dict, queries: list[dict], rankings: dict[str, list[str]]) -> dict:
    by_qid = {q["qid"]: q for q in queries}
    records = summary["records"]
    if len(records) != len(by_qid) or {r["qid"] for r in records} != set(by_qid):
        raise ValueError("official records do not cover the query set exactly")
    recalls, mrrs, parse_failures, comparisons = [], [], 0, 0
    for record in records:
        qid = record["qid"]
        candidates = record["candidates"]
        if record["gold_pid"] != by_qid[qid]["gold_pid"]:
            raise ValueError("official record gold differs from evaluation labels")
        if [c["pid"] for c in candidates] != rankings[qid]:
            raise ValueError("judge candidates differ from saved retrieval")
        for rank, candidate in enumerate(candidates, 1):
            if candidate["rank"] != rank:
                raise ValueError("invalid one-based rank")
            is_gold = candidate["pid"] == record["gold_pid"]
            if is_gold:
                gain, source = 1, "gold"
            else:
                relevant, parse_ok = parse_judgment(candidate["raw_output"])
                gain, source = int(relevant), "llm"
                comparisons += 1
                parse_failures += not parse_ok
                if candidate["parse_ok"] != parse_ok:
                    raise ValueError("parse flag differs from raw judge output")
            if candidate["gain"] != gain or candidate["source"] != source:
                raise ValueError("candidate gain/source differs from original rules")
        gains = [c["gain"] for c in candidates]
        recall, mrr = graded_recall_at_k(gains, 10), graded_mrr_at_k(gains, 10)
        if record["graded_recall@10"] != recall or record["graded_mrr@10"] != mrr:
            raise ValueError("per-query metric mismatch")
        recalls.append(recall)
        mrrs.append(mrr)
    expected = {"queries": len(records), "graded_recall@10": sum(recalls) / len(records),
                "graded_mrr@10": sum(mrrs) / len(records)}
    if any(not math.isclose(summary["metrics"][key], value, abs_tol=1e-12) for key, value in expected.items()):
        raise ValueError("aggregate official metric mismatch")
    if summary["judge"]["parse_failures"] != parse_failures or summary["judge"]["comparisons"] != comparisons:
        raise ValueError("judge comparison/parse count mismatch")
    return {"metrics": expected, "comparisons": comparisons, "parse_failures": parse_failures,
            "judge_resources": {k: v for k, v in summary["judge"].items() if k.startswith("peak_")},
            "judge_runtime_seconds": summary["judge"]["runtime_seconds"]}


def main():
    output_dir = ROOT / "results"
    read = lambda name: json.loads((output_dir / name).read_text(encoding="utf-8"))
    for name, expected in PROTECTED.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"protected source changed: {name}")
    products = load_products(str(ROOT / "data/train/products.jsonl"))
    report = {"scope": "B0 fixed grouped dev only; no public results or training", "candidate_products": len(products),
              "models": json.loads((ROOT / "code/script/configs/models.lock.json").read_text()),
              "splits": {}, "official_hardware_validation": False}
    rankings = {}
    for split in ["dev_fast", "dev"]:
        queries = load_eval_queries(str(ROOT / f"data/derived/{split}.jsonl"))
        ranking_path = output_dir / f"w06_b0_{split}_top10.json"
        rankings[split] = load_submission(ranking_path, queries, set(products), top_k=10)
        gold = read(f"w06_b0_{split}_gold.json")
        official = read(f"w06_b0_{split}_graded.json")
        retrieval_time = read(f"w06_b0_{split}_timing.json")
        judge_time = read(f"w06_b0_{split}_judge_timing.json")
        for timing in [retrieval_time, judge_time]:
            if timing["exit_code"] != 0 or timing["resources"]["exit_code"] != 0:
                raise ValueError("cannot certify a failed run")
            for name, expected in timing["input_sha256"].items():
                if sha256(ROOT / name) != expected:
                    raise ValueError(f"experiment source/input changed since run: {name}")
        audit = audit_graded(official, queries, rankings[split])
        raw_queries = [json.loads(line) for line in (ROOT / f"data/derived/{split}.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        turns = {row["dialogue_id"]: len(row["conversation"]) for row in raw_queries}
        by_turns = {}
        for length in sorted(set(turns.values())):
            rows = [r for r in official["records"] if turns[r["qid"]] == length]
            by_turns[str(length)] = {"queries": len(rows), **{
                key: sum(r[key] for r in rows) / len(rows) for key in ["graded_recall@10", "graded_mrr@10"]}}
        report["splits"][split] = {"gold_only": gold["metrics"], "official_graded": audit,
                                  "graded_by_message_count": by_turns,
                                  "retrieval_cold_wall_seconds": retrieval_time["cold_process_wall_seconds"],
                                  "retrieval_resources": retrieval_time["resources"],
                                  "judge_cold_wall_seconds": judge_time["cold_process_wall_seconds"]}
    subset = {q: rankings["dev"][q] for q in rankings["dev_fast"]}
    report["dev_fast_vs_full_dev_rankings"] = compare_prefix(rankings["dev_fast"], subset)
    fast_records = read("w06_b0_dev_fast_graded.json")["records"]
    full_records = {r["qid"]: r for r in read("w06_b0_dev_graded.json")["records"]}
    differing = [r["qid"] for r in fast_records if any(r[key] != full_records[r["qid"]][key]
                 for key in ["graded_recall@10", "graded_mrr@10"])]
    report["dev_fast_vs_full_dev_judge_metrics"] = {"different_qid_count": len(differing),
                                                   "different_qids": differing,
                                                   "note": "same queries may have different padding/grouping in batch-8 runs"}
    candidates = read("w06_b0_dev_top50_gold.json")
    report["candidate_diagnostic"] = {"metrics": candidates["metrics"],
                                      "top10_prefix_check": candidates["prefix_comparison"]}
    report["token_statistics"] = read("w06_token_statistics.json")
    report["protected_sources_unchanged"] = True
    report["notes"] = ["dev-fast is a subset of dev, not an independent holdout",
                       "gold-only misses may still be acceptable alternatives",
                       "Overall GPU memory usage was high and could not be attributed to this process; timings are observed runs, not a clean hardware benchmark",
                       "fresh process and full rebuilt index; OS disk cache was not flushed",
                       "RTX 4090 / Python 3.12.12 differs from required RTX 3070 / Python 3.12.9"]
    (output_dir / "w06_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": "passed", "splits": {k: {"gold_only": v["gold_only"],
          "official_graded": v["official_graded"]["metrics"]} for k, v in report["splits"].items()},
          "candidate_diagnostic": report["candidate_diagnostic"]}, indent=2))


if __name__ == "__main__":
    main()
