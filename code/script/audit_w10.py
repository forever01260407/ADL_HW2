#!/usr/bin/env python3
"""Frozen public evaluation: original Judge preflight, raw audit and grade estimate."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import sha256
from audit_w06 import audit_graded
from audit_w07 import audit_timing
from eval_utils import load_eval_queries, load_products
from evaluate import LLMJudge, build_judge_items, load_submission
from retrieval import save_submission
from smoke_retrieval import PROTECTED

RESULTS = ROOT / "results"
THRESHOLDS = {
    "public": [("simple", .860, .650, 2), ("strong", .885, .680, 1), ("boss", .950, .800, 1)],
    "private": [("simple", .845, .635, 1), ("strong", .880, .675, 1), ("boss", .930, .780, 1)]}


def estimate_points(metrics, split="public"):
    """Each cell is a separate additive estimate; preserve boundary ambiguity."""
    if split not in THRESHOLDS:
        raise ValueError("unknown grading split")
    cells = []
    for tier, recall, mrr, points in THRESHOLDS[split]:
        for key, threshold in [("graded_recall@10", recall), ("graded_mrr@10", mrr)]:
            value = metrics[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid unrounded graded metric")
            delta = value - threshold
            relation = "equal" if abs(delta) <= 1e-12 else ("above" if delta > 0 else "below")
            cells.append({"tier": tier, "metric": key, "threshold": threshold, "actual": value,
                "course_percentage_points": points, "relation": relation,
                "estimated_if_inclusive": points if relation != "below" else 0,
                "estimated_if_strict": points if relation == "above" else 0})
    return {"split": split, "interpretation": "estimated additive per-cell course percentage points; exact boundary/grading rules not explicitly confirmed",
        "cells": cells, "estimated_points_inclusive": sum(r["estimated_if_inclusive"] for r in cells),
        "estimated_points_strict": sum(r["estimated_if_strict"] for r in cells),
        "available_points": 8 if split == "public" else 6,
        "private_points_known": False, "report_points_known": False}


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def write_new(name, value):
    with (RESULTS / name).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_frozen():
    plan = read("w10_frozen_plan.json")
    for name, expected in {**plan["input_sha256"], **PROTECTED}.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"frozen W10 input/source changed: {name}")
    model = json.loads((ROOT / "artifacts/training/w08_epoch1/training_manifest.json").read_text())
    for name, expected in model["checkpoint_files"].items():
        if sha256(ROOT / model["checkpoint"] / name) != expected:
            raise ValueError(f"frozen W08 checkpoint changed: {name}")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare-judge", "final"], required=True)
    args = parser.parse_args()
    plan = verify_frozen()
    queries = load_eval_queries(ROOT / "data/public/dialogues.jsonl")
    products = load_products(ROOT / "data/public/products.jsonl")
    top50 = load_submission(RESULTS / "w10_w08_public_top50.json", queries, set(products), top_k=50)
    if len(queries) != 1000 or len(products) != 75000 or list(top50) != [q["qid"] for q in queries]:
        raise ValueError("public scale/order mismatch")
    retrieval_time = audit_timing("w10_w08_public_timing.json")
    for key in ["HW2_MODEL_PATH", "HW2_RETRIEVER", "HW2_RERANK", "HW2_RRF_CANDIDATES",
                "HW2_BM25_QUERY_MODE", "HW2_BM25_LAST_WEIGHT", "HW2_BM25_TITLE_WEIGHT", "HW2_BM25_TRACE_PATH"]:
        if retrieval_time["environment"][key] is not None:
            raise ValueError(f"frozen default evaluation has an override: {key}")
    if args.stage == "prepare-judge":
        if (RESULTS / "w10_judge_plan.json").exists() or (RESULTS / "w10_w08_public_judge_timing.json").exists():
            raise ValueError("public Judge already authorized/completed; preserve initial single-run policy")
        save_submission(RESULTS / "w10_w08_public_top10.json",
            [{"qid": qid, "ranked_pids": pids[:10]} for qid, pids in top50.items()], top_k=10)
        ranked = load_submission(RESULTS / "w10_w08_public_top10.json", queries, set(products), top_k=10)
        from transformers import AutoConfig, AutoTokenizer
        model = json.loads((ROOT / "code/script/configs/models.lock.json").read_text())["generator"]
        download = read("generator_download.json")
        for name, details in download["files"].items():
            if sha256(ROOT / model["path"] / name) != details["sha256"]:
                raise ValueError(f"Judge model changed: {name}")
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
        config = AutoConfig.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
        items = build_judge_items({q["qid"]: q for q in queries}, ranked, products)
        renderer = SimpleNamespace(tokenizer=tokenizer)
        lengths = [len(tokenizer(LLMJudge._render(renderer, item))["input_ids"]) for item in items]
        if not lengths or max(lengths) + 12 > config.max_position_embeddings:
            raise ValueError("untruncated original Judge context cannot fit")
        report = {"candidate": plan["candidate"], "maximum_initial_public_judge_runs": 1,
            "batch_size": 8, "max_new_tokens": 12, "truncation": False, "queries": 1000,
            "non_gold_comparisons": len(items), "max_judge_input_tokens": max(lengths),
            "context_limit": config.max_position_embeddings, "model": model,
            "ranking_sha256": sha256(RESULTS / "w10_w08_public_top10.json"),
            "frozen_plan_sha256": sha256(RESULTS / "w10_frozen_plan.json")}
        write_new("w10_judge_plan.json", report)
        print(json.dumps(report, indent=2))
        return
    judge_plan = read("w10_judge_plan.json")
    if sha256(RESULTS / "w10_w08_public_top10.json") != judge_plan["ranking_sha256"]:
        raise ValueError("public Judge ranking changed")
    ranked = load_submission(RESULTS / "w10_w08_public_top10.json", queries, set(products), top_k=10)
    if any(ranked[qid] != pids[:10] for qid, pids in top50.items()):
        raise ValueError("official top-ten is not the fixed retrieval prefix")
    graded = read("w10_w08_public_graded.json")
    audited = audit_graded(graded, queries, ranked)
    judge_time = audit_timing("w10_w08_public_judge_timing.json")
    command = read("w10_w08_public_judge_timing.json")["command"]
    for flag, expected in [("--batch-size", "8"), ("--max-new-tokens", "12"), ("--top-k", "10"),
                           ("--model", judge_plan["model"]["path"])]:
        if command[command.index(flag) + 1] != expected:
            raise ValueError(f"original Judge argument differs: {flag}")
    if [r["qid"] for r in graded["records"]] != [q["qid"] for q in queries]:
        raise ValueError("public evaluation order changed")
    groups = {}
    for prefix in sorted({q["qid"].rsplit("-", 1)[0] for q in queries}):
        rows = [r for r in graded["records"] if r["qid"].rsplit("-", 1)[0] == prefix]
        groups[prefix] = {"queries": len(rows), **{key: sum(r[key] for r in rows) / len(rows)
            for key in ["graded_recall@10", "graded_mrr@10"]}}
    resources = retrieval_time["resources"]
    report = {"candidate": plan["candidate"], "official_public_judge_runs": 1, "public": audited,
        "grade_estimate": estimate_points(graded["metrics"]), "by_qid_prefix": groups,
        "retrieval": retrieval_time, "judge": judge_time,
        "development_retrieval_under_one_hour": retrieval_time["cold_process_wall_seconds"] <= 3600,
        "development_retrieval_rss_under_32gib": resources["peak_process_rss_bytes"] <= 32 * 2**30,
        "official_hardware_validation": False, "private_evaluated": False, "report_grade_known": False,
        "public_feedback_seen": True, "default_changed": False,
        "next_experiment_rule": "Any improvement is selected on train/dev; no public gradients, negative mining or per-qid/PID rules; subsequent public is not an untouched holdout",
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
            [RESULTS / "w10_frozen_plan.json", RESULTS / "w10_judge_plan.json", RESULTS / "w10_w08_public_graded.json",
             RESULTS / "w10_w08_public_top10.json", Path(__file__).resolve()]}}
    write_new("w10_summary.json", report)
    print(json.dumps({"metrics": audited["metrics"], "comparisons": audited["comparisons"],
        "parse_failures": audited["parse_failures"], "grade_estimate": report["grade_estimate"],
        "official_hardware_validation": False}, indent=2))


if __name__ == "__main__":
    main()
