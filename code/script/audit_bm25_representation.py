#!/usr/bin/env python3
"""Verify H1U Gold gate and one Judge run; audit raw grades before adoption."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import sha256
from audit_w06 import audit_graded
from audit_w07 import audit_timing
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission
from retrieval import save_submission
from run_bm25_representation import gate, CONFIGS
from smoke_retrieval import PROTECTED


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def verify_file(path, expected):
    actual = ROOT / path
    if sha256(actual) == expected:
        return
    if path == "retrieval_utils.py":
        archived = ROOT / "code/script/snapshots/h1u/retrieval_utils.py"
        if archived.is_file() and sha256(archived) == expected:
            return
    raise ValueError(f"H1U input/source changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-judge", action="store_true")
    args = parser.parse_args()
    screening = read("h1u_screening.json")
    chosen = screening["selected_variant"]
    if not chosen or not screening["variants"][chosen]["eligible"]:
        raise ValueError("no Gold-eligible winner; do not run a Judge")
    candidate = screening["variants"][chosen]
    baseline = read("h1u_baseline_gold.json")
    gold = read(f"h1u_{chosen}_gold.json")
    if not gate(gold, baseline):
        raise ValueError("saved winner no longer passes the predeclared gate")
    for path, expected in screening["input_sha256"].items():
        verify_file(path, expected)
    for variant in screening["variants"].values():
        for path, expected in variant["artifacts_sha256"].items():
            verify_file(path, expected)
    for name, expected in PROTECTED.items():
        verify_file(name, expected)
    before = (ROOT / "code/script/snapshots/w06/retrieval_utils.py").read_text(encoding="utf-8")
    after = (ROOT / "retrieval_utils.py").read_text(encoding="utf-8")
    for marker, side in [("# ==================== STUDENT CODE START ====================", 0),
                         ("# ===================== STUDENT CODE END =====================", 1)]:
        if before.split(marker)[side] != after.split(marker)[side]:
            raise ValueError("source changed outside the student block")
    audit_timing("h1u_dense_timing.json")
    for variant_name in screening["variants"]:
        audit_timing(f"h1u_{variant_name}_lexical_timing.json")
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    load = lambda name, k: load_submission(ROOT / f"results/{name}.json", queries, set(products), top_k=k)
    expected = load(f"h1u_{chosen}_top50", 50)
    actual = load("h1u_winner_top50", 50)
    if actual != expected:
        raise ValueError("uncached official Hybrid differs from shared-Dense screening")
    timing = audit_timing("h1u_winner_timing.json")
    mode, last, title = CONFIGS[chosen]
    for key, value in {"HW2_RETRIEVER": "hybrid", "HW2_RERANK": "0", "HW2_RRF_CANDIDATES": "100",
        "HW2_BM25_QUERY_MODE": mode, "HW2_BM25_LAST_WEIGHT": str(last),
        "HW2_BM25_TITLE_WEIGHT": str(title)}.items():
        if timing["environment"][key] != value:
            raise ValueError("official Hybrid settings differ from winner")
    judge_plan = ROOT / "results/h1u_judge_plan.json"
    if args.prepare_judge:
        if judge_plan.exists() or (ROOT / "results/h1u_winner_judge_timing.json").exists():
            raise ValueError("Judge was already authorized or completed; preserve the single-run limit")
        save_submission(ROOT / "results/h1u_winner_top10.json",
            [{"qid": qid, "ranked_pids": pids[:10]} for qid, pids in actual.items()], top_k=10)
        judge_plan.write_text(json.dumps({"winner": chosen, "gold_gate_verified": True,
            "uncached_matches_screening": True, "maximum_official_judge_runs": 1,
            "retrieval_sha256": sha256(ROOT / "results/h1u_winner_top10.json")}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"gold_gate": "passed", "uncached_match": True, "one_judge_authorized": chosen}))
        return
    plan = json.loads(judge_plan.read_text())
    if plan["retrieval_sha256"] != sha256(ROOT / "results/h1u_winner_top10.json"):
        raise ValueError("Judge input changed since the gate was checked")
    ranked = load("h1u_winner_top10", 10)
    graded = read("h1u_winner_graded.json")
    audit = audit_graded(graded, queries, ranked)
    judge_timing = audit_timing("h1u_winner_judge_timing.json")
    old = read("h1_hybrid_dev_graded.json")
    old_ranked = load("h1_hybrid_dev_top10", 10)
    audit_graded(old, queries, old_ranked)
    keys = ["graded_recall@10", "graded_mrr@10"]
    differences = {key: graded["metrics"][key] - old["metrics"][key] for key in keys}
    adopted = all(d >= 0 for d in differences.values()) and any(d > 0 for d in differences.values())
    old_rows = {r["qid"]: r for r in old["records"]}
    shared_pairs, changed_shared = 0, []
    for row in graded["records"]:
        old_candidates = {c["pid"]: c for c in old_rows[row["qid"]]["candidates"]}
        for candidate_row in row["candidates"]:
            previous = old_candidates.get(candidate_row["pid"])
            if previous is not None and candidate_row["source"] == "llm":
                shared_pairs += 1
                if previous["gain"] != candidate_row["gain"]:
                    changed_shared.append({"qid": row["qid"], "pid": candidate_row["pid"],
                                           "before": previous["gain"], "after": candidate_row["gain"]})
    paired = {key: {"improved": sum(r[key] > old_rows[r["qid"]][key] for r in graded["records"]),
                    "worsened": sum(r[key] < old_rows[r["qid"]][key] for r in graded["records"])} for key in keys}
    beauty = [r for r in graded["records"] if r["qid"].rsplit("-", 1)[0] == "beauty-train"]
    beauty_baseline = {key: sum(old_rows[r["qid"]][key] for r in beauty) / len(beauty) for key in keys}
    beauty_current = {key: sum(r[key] for r in beauty) / len(beauty) for key in keys}
    report = {"winner": chosen, "gold_gate_verified": True, "uncached_matches_screening": True,
        "official_judge_runs": 1, "baseline_metrics": old["metrics"],
        "winner_graded": audit, "differences": differences, "paired_changes": paired,
        "beauty_train_baseline": beauty_baseline, "beauty_train_graded": beauty_current,
        "beauty_train_difference": {key: beauty_current[key] - beauty_baseline[key] for key in keys},
        "adopt": adopted, "official_hardware_validation": False, "winner_timing": timing,
        "judge_timing": judge_timing, "default_verified": False,
        "default_action": "promotion_pending" if adopted else "retain_h1_full_1_1",
        "shared_non_gold_pairs": shared_pairs, "shared_pair_grade_disagreements": changed_shared}
    default_path = ROOT / "results/h1u_default_top50.json"
    if default_path.exists():
        if load("h1u_default_top50", 50) != expected:
            raise ValueError("new default differs from evaluated winner")
        report["default_verified"] = True
        report["default_action"] = "promoted_and_verified"
        report["default_timing"] = audit_timing("h1u_default_timing.json")
    (ROOT / "results/h1u_final.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["winner", "official_judge_runs", "differences", "adopt", "default_verified"]}, indent=2))


if __name__ == "__main__":
    main()
