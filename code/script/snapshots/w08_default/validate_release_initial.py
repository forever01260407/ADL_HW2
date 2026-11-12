#!/usr/bin/env python3
"""H4: audit the fixed base5 release, then one authorized frozen Public run.

No candidate or parameter selection is performed using Public labels/results.
Historical H3 gates stay failed; the new user-authorized decision accepts Pet risk.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import gold_metrics, sha256
from audit_w06 import audit_graded
from audit_w10 import estimate_points
from eval_utils import load_eval_queries, load_products
from evaluate import LLMJudge, build_judge_items, load_submission
from extend_rrf_ablation import graded_group
from retrieval import save_submission
from smoke_retrieval import PROTECTED

RESULTS = ROOT / "results"
OLD_SOURCE = ROOT / "code/script/snapshots/w08_default/retrieval_utils.py"
MODEL = "artifacts/training/h3_base_epoch1/checkpoint"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(name, data):
    with (RESULTS / name).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def hashes(mapping):
    for path, expected in mapping.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"frozen file changed: {path}")


def source_boundaries():
    hashes(PROTECTED)
    old, new = OLD_SOURCE.read_text(), (ROOT / "retrieval_utils.py").read_text()
    start = "# ==================== STUDENT CODE START ===================="
    end = "# ===================== STUDENT CODE END ====================="
    if (old.split(start)[0], old.split(end)[1]) != (new.split(start)[0], new.split(end)[1]):
        raise ValueError("source changed outside permitted student block")
    if sha256(OLD_SOURCE) != "d5be9dcba87dd70a6e79478306a7f770ba7030948cfd2bcae679c499dcc2b39f":
        raise ValueError("W08 default source snapshot changed")


def prepare():
    source_boundaries()
    history = read("results/h3_completion.json")
    hashes(history["checkpoint_sha256"])
    hashes(history["artifact_sha256"])
    tests = (RESULTS / "h4_tests_release.log").read_text()
    if "Ran 77 tests" not in tests or not tests.rstrip().endswith("OK"):
        raise ValueError("release tests not completed")
    paths = ["retrieval_utils.py", "code/script/run_validated.py", "code/script/validate_release.py",
        "code/script/tests/test_formal_base.py", "code/script/tests/test_hybrid.py",
        "results/h4_tests_initial.log", "results/h4_tests_release.log", "results/h3_completion.json",
        "results/h3_base5_top50.json", "results/h3_base5_top10.json", "results/h3_base5_graded.json",
        "results/h3_quality_final.json", "results/w10_summary.json", "results/w10_w08_public_graded.json",
        "artifacts/training/h3_base_epoch1/training_manifest.json", "code/script/configs/models.lock.json",
        "results/generator_download.json", "data/derived/dev.jsonl", "data/train/products.jsonl",
        "data/public/dialogues.jsonl", "data/public/products.jsonl", *PROTECTED]
    write("h4_validation_plan.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "H3 trained BGE-base, full transcript BM25, RRF 5:1, 100 each, constant60",
        "checkpoint": MODEL, "dense_weight": 5, "bm25_weight": 1, "route_candidates": 100,
        "decision": "D28: user accepts known Pet regression for overall dev benefit; H3 gate unchanged",
        "required_dev_top50_exact": "results/h3_base5_top50.json", "dev_queries": 966,
        "allocator_budget_gib": 8, "rss_budget_gib": 32, "time_limit_seconds": 3600,
        "maximum_public_retrieval_runs": 1, "maximum_public_judge_runs": 1,
        "public_not_used_for_selection": True, "no_public_based_rollback_or_retuning": True,
        "public_previously_seen": True, "official_hardware_validation": False,
        "w11_authorized": False, "historical_h3_seal_verified": True,
        "input_sha256": {p: sha256(ROOT / p) for p in paths},
        "checkpoint_sha256": history["checkpoint_sha256"]})
    print("H4 frozen: 77 tests passed, historical artifacts/checkpoint and starter boundaries verified")


def frozen():
    plan = read("results/h4_validation_plan.json")
    hashes(plan["input_sha256"])
    hashes(plan["checkpoint_sha256"])
    source_boundaries()
    return plan


def timing(path):
    report = read(path)
    if report["exit_code"] or report["resources"]["exit_code"]:
        raise ValueError("failed process cannot pass validation")
    hashes(report["input_sha256"])
    return report


def check_budget(report):
    resources = report["resources"]
    checks = {"allocator_budget_configured": report["cuda_allocator_budget_gib"] == 8
            and resources["cuda_allocator_budget_gib"] == 8,
        "cuda_allocated_within_budget": 0 < resources["peak_cuda_allocated_bytes"] <= 8 * 2**30,
        "cuda_reserved_within_budget": 0 < resources["peak_cuda_reserved_bytes"] <= 8 * 2**30,
        "rss_under_32gib": 0 < resources["peak_process_rss_bytes"] < 32 * 2**30,
        "under_one_hour": report["cold_process_wall_seconds"] < 3600,
        "batch32": report["environment"]["HW2_BATCH_SIZE"] == "32",
        "no_configuration_overrides": all(report["environment"][key] is None for key in [
            "HW2_MODEL_PATH", "HW2_RETRIEVER", "HW2_RERANK", "HW2_RRF_CANDIDATES",
            "HW2_RRF_DENSE_WEIGHT", "HW2_RRF_BM25_WEIGHT", "HW2_BM25_QUERY_MODE",
            "HW2_BM25_LAST_WEIGHT", "HW2_BM25_TITLE_WEIGHT", "HW2_BM25_TRACE_PATH"])}
    if not all(checks.values()):
        raise ValueError(f"formal budget/default check failed: {checks}")
    return checks


def dev_check():
    frozen()
    report = timing("results/h4_base5_dev_timing.json")
    checks = check_budget(report)
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    if len(queries) != 966 or len(products) != 50000:
        raise ValueError("incorrect dev scale")
    actual = load_submission(RESULTS / "h4_base5_dev_top50.json", queries, set(products), top_k=50)
    expected = load_submission(RESULTS / "h3_base5_top50.json", queries, set(products), top_k=50)
    changed = [q for q in expected if actual[q] != expected[q]]
    if list(actual) != [q["qid"] for q in queries] or changed:
        raise ValueError(f"formal default does not reproduce H3: {len(changed)} changed")
    audited = audit_graded(read("results/h3_base5_graded.json"), queries,
        {qid: pids[:10] for qid, pids in actual.items()})
    write("h4_dev_validation.json", {"passed": True, "checks": checks,
        "queries": len(actual), "products": len(products), "changed_top50_count": 0,
        "reuses_original_dev_judge_no_rerun": True, "graded": audited,
        "retrieval": report, "official_hardware_validation": False,
        "input_sha256": {p: sha256(ROOT / p) for p in ["results/h4_base5_dev_top50.json",
            "results/h4_base5_dev_timing.json", "results/h4_validation_plan.json"]}})
    print(json.dumps({"passed": True, "changed_top50_count": 0, "resources": report["resources"]}, indent=2))


def freeze_public():
    plan = frozen()
    dev = read("results/h4_dev_validation.json")
    hashes(dev["input_sha256"])
    if not dev["passed"]:
        raise ValueError("dev validation must pass before Public")
    if any(RESULTS.glob("h4_base5_public*")):
        raise ValueError("Public already attempted")
    write("h4_public_plan.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": plan["candidate"], "checkpoint": MODEL, "default_promoted_before_public": True,
        "selection_basis": "train/dev only; accept D28 Pet tradeoff; no selection based on new Public",
        "queries": 1000, "products": 75000, "maximum_public_judge_runs": 1,
        "batch_size": 8, "max_new_tokens": 12, "dtype": "bfloat16", "do_sample": False,
        "input_sha256": {p: sha256(ROOT / p) for p in ["results/h4_validation_plan.json",
            "results/h4_dev_validation.json", "retrieval_utils.py"]}})
    print("Formal base5 selected and frozen before new Public feedback")


def prepare_judge():
    frozen()
    public_plan = read("results/h4_public_plan.json")
    hashes(public_plan["input_sha256"])
    report = timing("results/h4_base5_public_timing.json")
    check_budget(report)
    queries = load_eval_queries(ROOT / "data/public/dialogues.jsonl")
    products = load_products(ROOT / "data/public/products.jsonl")
    if len(queries) != 1000 or len(products) != 75000:
        raise ValueError("incorrect Public scale")
    top50 = load_submission(RESULTS / "h4_base5_public_top50.json", queries, set(products), top_k=50)
    if list(top50) != [q["qid"] for q in queries]:
        raise ValueError("Public query order changed")
    save_submission(RESULTS / "h4_base5_public_top10.json", [{"qid": qid, "ranked_pids": pids[:10]}
        for qid, pids in top50.items()], top_k=10)
    gold = gold_metrics(queries, top50, (1, 10, 50))
    write("h4_base5_public_gold.json", {"kind": "diagnostic only after configuration frozen",
        **gold, "candidate_products": len(products)})
    import os
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoTokenizer
    model = read("code/script/configs/models.lock.json")["generator"]
    hashes({f'{model["path"]}/{name}': row["sha256"]
        for name, row in read("results/generator_download.json")["files"].items()})
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    config = AutoConfig.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    ranked = {qid: pids[:10] for qid, pids in top50.items()}
    items = build_judge_items({q["qid"]: q for q in queries}, ranked, products)
    renderer = SimpleNamespace(tokenizer=tokenizer)
    lengths = [len(tokenizer(LLMJudge._render(renderer, item))["input_ids"]) for item in items]
    if not lengths or max(lengths) + 12 > config.max_position_embeddings:
        raise ValueError("untruncated Judge context cannot fit")
    write("h4_judge_plan.json", {"model": model, "batch_size": 8, "max_new_tokens": 12,
        "dtype": "bfloat16", "do_sample": False, "truncation": False,
        "maximum_public_judge_runs": 1, "non_gold_comparisons": len(items),
        "max_input_tokens": max(lengths), "context_limit": config.max_position_embeddings,
        "input_sha256": {p: sha256(ROOT / p) for p in ["results/h4_public_plan.json",
            "results/h4_base5_public_top10.json", "results/h4_base5_public_top50.json"]}})
    print(json.dumps({"judge_preflight": "passed", "comparisons": len(items), "max_input_tokens": max(lengths)}))


def final():
    frozen()
    jp = read("results/h4_judge_plan.json")
    hashes(jp["input_sha256"])
    queries = load_eval_queries(ROOT / "data/public/dialogues.jsonl")
    products = load_products(ROOT / "data/public/products.jsonl")
    ranked = load_submission(RESULTS / "h4_base5_public_top10.json", queries, set(products), top_k=10)
    new = read("results/h4_base5_public_graded.json")
    audited = audit_graded(new, queries, ranked)
    if [r["qid"] for r in new["records"]] != [q["qid"] for q in queries]:
        raise ValueError("Judge query order changed")
    jt = timing("results/h4_base5_public_judge_timing.json")
    for flag, expected in [("--model", jp["model"]["path"]), ("--batch-size", "8"),
            ("--max-new-tokens", "12"), ("--top-k", "10"), ("--device", "cuda")]:
        if jt["command"][jt["command"].index(flag) + 1] != expected:
            raise ValueError(f"Judge command mismatch: {flag}")
    old = read("results/w10_w08_public_graded.json")
    old_ranked = load_submission(RESULTS / "w10_w08_public_top10.json", queries, set(products), top_k=10)
    audit_graded(old, queries, old_ranked)
    newrows, oldrows = ({r["qid"]: r for r in summary["records"]} for summary in [new, old])
    paired = {}
    for metric in ["graded_recall@10", "graded_mrr@10"]:
        diffs = [newrows[qid][metric] - oldrows[qid][metric] for qid in newrows]
        paired[metric] = {"difference": sum(diffs) / len(diffs), "improved": sum(d > 0 for d in diffs),
            "worsened": sum(d < 0 for d in diffs), "unchanged": sum(d == 0 for d in diffs)}
    sources = sorted({q["qid"].rsplit("-", 1)[0] for q in queries})
    report = {"status": "completed_fixed_base5_public_confirmation", "default": MODEL,
        "default_changed_before_public": True, "public_retrieval_runs": 1, "public_judge_runs": 1,
        "dev_validation": read("results/h4_dev_validation.json"), "public": audited,
        "gold": read("results/h4_base5_public_gold.json")["metrics"],
        "previous_small1_public": old["metrics"], "paired_public_vs_small1": paired,
        "grade_estimate": estimate_points(new["metrics"]),
        "by_qid_prefix": {prefix: {"base5": graded_group(new, [q["qid"] for q in queries if q["qid"].rsplit("-", 1)[0] == prefix]),
            "small1": graded_group(old, [q["qid"] for q in queries if q["qid"].rsplit("-", 1)[0] == prefix])} for prefix in sources},
        "retrieval": timing("results/h4_base5_public_timing.json"), "judge": jt,
        "official_hardware_validation": False, "private_evaluated": False, "w11_started": False,
        "known_limits": ["Pet dev recall tradeoff accepted in D28", "FP32 training exceeded 8GiB",
            "4090 allocator budget is not real RTX3070 validation", "Public previously seen; no Public parameter selection",
            "Judge candidate batch changes may change some shared decisions"],
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(RESULTS.glob("h4_*.json"))}}
    write("h4_summary.json", report)
    print(json.dumps({"public": audited["metrics"], "previous": old["metrics"], "paired": paired,
        "points": report["grade_estimate"]["estimated_points_inclusive"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "dev-check", "freeze-public", "prepare-judge", "final"], required=True)
    {"prepare": prepare, "dev-check": dev_check, "freeze-public": freeze_public,
        "prepare-judge": prepare_judge, "final": final}[parser.parse_args().stage]()
