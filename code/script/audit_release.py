#!/usr/bin/env python3
"""Seal H4 experiment artifacts; no submission packaging or model selection."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from analyze_results import gold_metrics, sha256
from audit_w06 import audit_graded
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission
from validate_release import ROOT, RESULTS, MODEL, frozen, hashes, read, write


def main():
    plan = frozen()
    summary = read("results/h4_summary.json")
    hashes(summary["input_sha256"])
    queries = load_eval_queries(ROOT / "data/public/dialogues.jsonl")
    products = load_products(ROOT / "data/public/products.jsonl")
    ranked = load_submission(RESULTS / "h4_base5_public_top50.json", queries, set(products), top_k=50)
    top10 = load_submission(RESULTS / "h4_base5_public_top10.json", queries, set(products), top_k=10)
    if any(top10[qid] != pids[:10] for qid, pids in ranked.items()):
        raise ValueError("Public top10 is not the top50 prefix")
    gold = gold_metrics(queries, ranked, (1, 10, 50))
    if gold != {k: read("results/h4_base5_public_gold.json")[k] for k in ["metrics", "records"]}:
        raise ValueError("independent Public Gold recomputation differs")
    old = read("results/w10_w08_public_graded.json")
    new = read("results/h4_base5_public_graded.json")
    audited = audit_graded(new, queries, top10)
    if audited != summary["public"]:
        raise ValueError("final raw Judge audit differs")
    nongold = lambda result: {(r["qid"], c["pid"]): c for r in result["records"]
        for c in r["candidates"] if c["source"] == "llm"}
    before, after = nongold(old), nongold(new)
    common = before.keys() & after.keys()
    flips = [{"qid": qid, "pid": pid, "small1": before[qid, pid], "base5": after[qid, pid]}
        for qid, pid in sorted(common) if before[qid, pid]["gain"] != after[qid, pid]["gain"]]
    write("h4_judge_diagnostics.json", {"shared_non_gold_pairs": len(common),
        "shared_pair_gain_changes": len(flips), "changes": flips,
        "note": "Both original one-shot outputs preserved; cause of shared-pair changes not isolated. No judgment cherry-picking or rerun.",
        "input_sha256": {p: sha256(ROOT / p) for p in ["results/w10_w08_public_graded.json",
            "results/h4_base5_public_graded.json"]}})
    pp = read("results/h4_public_plan.json")
    retrieval = read("results/h4_base5_public_timing.json")
    if datetime.fromisoformat(pp["created_utc"]) >= datetime.fromisoformat(retrieval["started_utc"]):
        raise ValueError("Public configuration was not frozen before retrieval")
    history = read("results/h3_completion.json")
    hashes(history["artifact_sha256"])
    hashes(history["checkpoint_sha256"])
    paths = sorted(RESULTS.glob("h4_*.json")) + sorted(RESULTS.glob("h4_*.log"))
    paths += [ROOT / n for n in ["retrieval_utils.py", "code/script/run_validated.py",
        "code/script/validate_release.py", "code/script/audit_release.py",
        "code/script/tests/test_formal_base.py", "code/script/tests/test_measurement.py",
        "code/script/tests/test_hybrid.py", "H4_RESULTS.md"]]
    report = {"status": "experiment_complete_base5_adopted_before_public",
        "finished_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": MODEL,
        "dense_bm25_weights": [5, 1], "formal_dev_top50_equal_h3": True,
        "dev_retrieval_attempts": 2, "first_dev_parent_measurement_failed_preserved": True,
        "complete_suite_tests": 77, "focused_measurement_tests": 2,
        "new_training_runs": 0, "new_dev_judge_runs": 0,
        "public_retrieval_runs": 1, "public_judge_runs": 1,
        "public": audited["metrics"], "previous_small1": old["metrics"],
        "public_gold": gold["metrics"], "dev_and_public_allocator_budget_gib": 8,
        "public_configuration_frozen_before_retrieval": True,
        "public_used_for_selection": False, "historical_h3_seal_verified": True,
        "shared_pair_judge_gain_changes": len(flips), "official_hardware_validation": False,
        "training_8gib_validated": False, "private_evaluated": False, "w11_started": False,
        "mutable_worklog_references": ["PLAN.md", "DECISIONS.md", "PROGRESS.md"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "artifact_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}}
    write("h4_completion.json", report)
    print(json.dumps({"audit": "passed", "artifacts_sealed": len(report["artifact_sha256"]),
        "public": audited["metrics"], "shared_pair_judge_changes": len(flips)}, indent=2))


if __name__ == "__main__":
    main()
