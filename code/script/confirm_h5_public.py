#!/usr/bin/env python3
"""H6: one frozen Public confirmation of the already adopted H5 default.

Evaluation only. No training, mining, configuration selection or default changes.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from analyze_results import gold_metrics, sha256
from audit_w06 import audit_graded
from audit_w10 import estimate_points
from eval_utils import load_eval_queries, load_products
from evaluate import LLMJudge, build_judge_items, load_submission
from extend_rrf_ablation import graded_group
from retrieval import save_submission
from smoke_retrieval import PROTECTED
from validate_release import check_budget

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "results/h6_h5_public"
MODEL = "artifacts/training/h5_hard_negative/checkpoint"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(path, value):
    with (ROOT / path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def hashes(mapping):
    for path, expected in mapping.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"frozen file changed: {path}")


def history():
    seal = read("results/h5_completion.json")
    if seal["status"] != "experiment_complete_h5_adopted_dev_only" or not seal["default_changed"]:
        raise ValueError("H5 adoption not completed")
    hashes(seal["artifact_sha256"])
    hashes(seal["checkpoint_sha256"])
    hashes(PROTECTED)
    adoption = read("results/h5_adoption.json")
    if not adoption["default_changed"]:
        raise ValueError("H5 not adopted")
    return seal


def freeze():
    seal = history()
    paths = ["results/h5_completion.json", "results/h5_adoption.json", "retrieval_utils.py",
        "data/public/dialogues.jsonl", "data/public/products.jsonl", "code/script/run_validated.py",
        "code/script/confirm_h5_public.py", "code/script/analyze_results.py", "code/script/audit_w06.py",
        "code/script/audit_w10.py", "code/script/validate_release.py", "code/script/extend_rrf_ablation.py",
        "results/h4_base5_public_graded.json", "results/h4_base5_public_top10.json",
        "results/w10_w08_public_graded.json", "results/generator_download.json",
        "code/script/configs/models.lock.json", *PROTECTED]
    write("results/h6_frozen_plan.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "User requests Public test only, explicitly excludes training",
        "candidate": "adopted H5 hard-negative continued BGE-base + full-transcript BM25, fixed RRF 5:1",
        "checkpoint": MODEL, "dense_weight": 5, "bm25_weight": 1, "rrf_constant": 60,
        "route_candidates": 100, "queries": 1000, "products": 75000,
        "maximum_public_retrieval_runs": 1, "maximum_public_judge_runs": 1,
        "training_runs": 0, "mining_runs": 0, "public_previously_seen": True,
        "public_selection_or_retuning": False, "public_based_rollback": False,
        "default_changes": False, "w11_started": False, "official_hardware_validation": False,
        "historical_h5_artifacts_verified": len(seal["artifact_sha256"]),
        "input_sha256": {p: sha256(ROOT / p) for p in paths},
        "checkpoint_sha256": seal["checkpoint_sha256"]})
    print("H6 frozen: H5 seal/model verified, one Public test, no training/selection")


def frozen():
    plan = read("results/h6_frozen_plan.json")
    hashes(plan["input_sha256"])
    hashes(plan["checkpoint_sha256"])
    return plan


def timing(path):
    report = read(path)
    if report["exit_code"] or report["resources"]["exit_code"]:
        raise ValueError("failed evaluation process")
    hashes(report["input_sha256"])
    return report


def rankings():
    queries = load_eval_queries(ROOT / "data/public/dialogues.jsonl")
    products = load_products(ROOT / "data/public/products.jsonl")
    ranked = load_submission(ROOT / f"{PREFIX}_top50.json", queries, set(products), top_k=50)
    if len(queries) != 1000 or len(products) != 75000 or list(ranked) != [q["qid"] for q in queries]:
        raise ValueError("full Public scale/order mismatch")
    return queries, products, ranked


def prepare_judge():
    frozen()
    retrieval = timing(f"{PREFIX}_timing.json")
    check_budget(retrieval)
    queries, products, ranked = rankings()
    if (ROOT / "results/h6_judge_plan.json").exists() or (ROOT / f"{PREFIX}_judge_timing.log").exists():
        raise ValueError("one Judge already prepared/started")
    save_submission(ROOT / f"{PREFIX}_top10.json",
        [{"qid": qid, "ranked_pids": pids[:10]} for qid, pids in ranked.items()], top_k=10)
    top10 = load_submission(ROOT / f"{PREFIX}_top10.json", queries, set(products), top_k=10)
    from transformers import AutoConfig, AutoTokenizer
    model = read("code/script/configs/models.lock.json")["generator"]
    assets = read("results/generator_download.json")["files"]
    model_hashes = {f'{model["path"]}/{p}': details["sha256"] for p, details in assets.items()}
    hashes(model_hashes)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    config = AutoConfig.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    items = build_judge_items({q["qid"]: q for q in queries}, top10, products)
    renderer = SimpleNamespace(tokenizer=tokenizer)
    lengths = [len(tokenizer(LLMJudge._render(renderer, item))["input_ids"]) for item in items]
    if not lengths or max(lengths) + 12 > config.max_position_embeddings:
        raise ValueError("original Judge inputs cannot fit without truncation")
    gold = gold_metrics(queries, ranked, (1, 10, 50))
    write(f"{PREFIX}_gold.json", gold)
    report = {"maximum_public_judge_runs": 1, "queries": 1000, "batch_size": 8,
        "max_new_tokens": 12, "dtype": "bf16", "greedy": True, "truncation": False,
        "non_gold_comparisons": len(items), "max_judge_input_tokens": max(lengths),
        "context_limit": config.max_position_embeddings, "model": model,
        "model_sha256": model_hashes,
        "ranking_sha256": sha256(ROOT / f"{PREFIX}_top10.json"),
        "top50_sha256": sha256(ROOT / f"{PREFIX}_top50.json"),
        "frozen_plan_sha256": sha256(ROOT / "results/h6_frozen_plan.json")}
    write("results/h6_judge_plan.json", report)
    print(json.dumps({"preflight": report, "gold_diagnostic_not_official_score": gold["metrics"]}, indent=2))


def final():
    frozen()
    history()
    plan = read("results/h6_judge_plan.json")
    hashes(plan["model_sha256"])
    hashes({"results/h6_frozen_plan.json": plan["frozen_plan_sha256"],
        f"{PREFIX}_top10.json": plan["ranking_sha256"], f"{PREFIX}_top50.json": plan["top50_sha256"]})
    retrieval, judge = timing(f"{PREFIX}_timing.json"), timing(f"{PREFIX}_judge_timing.json")
    checks = check_budget(retrieval)
    for flag, expected in [("--batch-size", "8"), ("--max-new-tokens", "12"), ("--top-k", "10"),
        ("--model", plan["model"]["path"]), ("--dialogues", "data/public/dialogues.jsonl"),
        ("--products", "data/public/products.jsonl"), ("--retrieval-results", f"{PREFIX}_top10.json"),
        ("--device", "cuda")]:
        if judge["command"][judge["command"].index(flag) + 1] != expected:
            raise ValueError(f"Judge flag changed: {flag}")
    queries, products, top50 = rankings()
    ranked = load_submission(ROOT / f"{PREFIX}_top10.json", queries, set(products), top_k=10)
    if any(ranked[qid] != pids[:10] for qid, pids in top50.items()):
        raise ValueError("top10 prefix changed")
    gold = gold_metrics(queries, top50, (1, 10, 50))
    if gold != read(f"{PREFIX}_gold.json"):
        raise ValueError("Gold diagnostic changed")
    summary = read(f"{PREFIX}_graded.json")
    if [r["qid"] for r in summary["records"]] != [q["qid"] for q in queries]:
        raise ValueError("Judge query order changed")
    audited = audit_graded(summary, queries, ranked)
    if audited["comparisons"] != plan["non_gold_comparisons"]:
        raise ValueError("Judge comparisons differ from preflight")
    old = read("results/h4_base5_public_graded.json")
    old_ranked = load_submission(ROOT / "results/h4_base5_public_top10.json", queries, set(products), top_k=10)
    audit_graded(old, queries, old_ranked)
    new_by, old_by = ({r["qid"]: r for r in s["records"]} for s in [summary, old])
    ids = [q["qid"] for q in queries]
    groups = {c: graded_group(new_by, old_by, [q for q in ids if q.rsplit("-", 1)[0] == c])
        for c in sorted({q.rsplit("-", 1)[0] for q in ids})}
    shared, changes = 0, []
    for qid in ids:
        previous = {c["pid"]: c for c in old_by[qid]["candidates"] if c["source"] == "llm"}
        for c in new_by[qid]["candidates"]:
            if c["source"] == "llm" and c["pid"] in previous:
                shared += 1
                if c["gain"] != previous[c["pid"]]["gain"]:
                    changes.append({"qid": qid, "pid": c["pid"], "old_gain": previous[c["pid"]]["gain"],
                        "new_gain": c["gain"]})
    report = {"status": "frozen_h5_public_confirmation_complete", "public": audited,
        "gold_diagnostic": gold["metrics"], "previous_h4": old["metrics"],
        "previous_w10": read("results/w10_w08_public_graded.json")["metrics"],
        "vs_h4": graded_group(new_by, old_by, ids), "by_category_vs_h4": groups,
        "grade_estimate": estimate_points(audited["metrics"]), "budget_checks": checks,
        "retrieval": retrieval, "judge": judge,
        "shared_non_gold_pairs": shared, "shared_pair_gain_changes": changes,
        "judge_variation_note": "Candidate batches differ; cause of shared-pair changes unisolated; retain raw single runs",
        "official_public_judge_runs": 1, "public_retrieval_runs": 1, "training_runs": 0,
        "mining_runs": 0, "default_changed": False, "public_used_for_tuning_or_selection": False,
        "public_previously_seen": True, "official_hardware_validation": False, "w11_started": False,
        "finished_utc": datetime.now(timezone.utc).isoformat()}
    write("results/h6_summary.json", report)
    print(json.dumps({"public": audited, "previous_h4": old["metrics"],
        "grade_estimate": report["grade_estimate"], "shared_non_gold_pairs": shared,
        "shared_pair_gain_changes": len(changes), "budget_checks": checks}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["freeze", "prepare-judge", "final"])
    args = parser.parse_args()
    {"freeze": freeze, "prepare-judge": prepare_judge, "final": final}[args.stage]()
