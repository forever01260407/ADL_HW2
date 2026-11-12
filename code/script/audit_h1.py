#!/usr/bin/env python3
"""Audit fixed H1 ablations and compare original graded metrics with paired queries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import compare_prefix, gold_metrics, sha256
from audit_w06 import audit_graded
from audit_w07 import audit_timing
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission
from smoke_retrieval import PROTECTED


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def main():
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    order = [q["qid"] for q in queries]
    raw_queries = [json.loads(line) for line in (ROOT / "data/derived/dev.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    turns = {q["dialogue_id"]: len(q["conversation"]) for q in raw_queries}
    for name, expected in PROTECTED.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"official file changed: {name}")
    before = (ROOT / "code/script/snapshots/w06/retrieval_utils.py").read_text(encoding="utf-8")
    after = (ROOT / "retrieval_utils.py").read_text(encoding="utf-8")
    for marker, side in [("# ==================== STUDENT CODE START ====================", 0),
                         ("# ===================== STUDENT CODE END =====================", 1)]:
        if before.split(marker)[side] != after.split(marker)[side]:
            raise ValueError("retrieval code changed outside the student block")
    load = lambda name, k: load_submission(ROOT / f"results/{name}.json", queries, set(products), top_k=k)
    b0_candidates = load("h1_b0_dev_top50", 50)
    if b0_candidates != load("w06_b0_dev_top50", 50):
        raise ValueError("B0 regression: dense candidate rankings changed")
    rankings, graded_rows = {}, {}
    report = {"scope": "H1 fixed dev 966 / 50000 products; no public or training",
              "protected_files_and_student_boundary_verified": True, "b0_top50_reproduced": True,
              "official_hardware_validation": False, "variants": {},
              "selection_policy": "eligible if both graded metrics improve over B0; among eligible maximize Recall then MRR"}
    for label, prefix in [("dense", "w06_b0"), ("bm25", "h1_bm25"), ("hybrid", "h1_hybrid")]:
        ranking = load(f"{prefix}_dev_top10", 10)
        if list(ranking) != order:
            raise ValueError(f"query order changed: {label}")
        summary = read(f"{prefix}_dev_graded.json")
        if [r["qid"] for r in summary["records"]] != order:
            raise ValueError(f"judge order changed: {label}")
        rankings[label] = ranking
        graded_rows[label] = {r["qid"]: r for r in summary["records"]}
        report["variants"][label] = {
            "gold": gold_metrics(queries, ranking)["metrics"],
            "graded": audit_graded(summary, queries, ranking),
            "retrieval": audit_timing(f"{prefix}_dev_timing.json"),
            "judge": audit_timing(f"{prefix}_dev_judge_timing.json"),
            "by_message_count": {str(length): {metric: sum(r[metric] for r in summary["records"]
                if turns[r["qid"]] == length) / sum(turns[qid] == length for qid in order)
                for metric in ["graded_recall@10", "graded_mrr@10"]} for length in sorted(set(turns.values()))}}
        slices = {}
        for group in sorted({qid.rsplit("-", 1)[0] for qid in order}):
            subset = [q for q in queries if q["qid"].rsplit("-", 1)[0] == group]
            slices[group] = {"queries": len(subset),
                "gold": gold_metrics(subset, {q["qid"]: ranking[q["qid"]] for q in subset}, (10,))["metrics"],
                "graded": {metric: sum(graded_rows[label][q["qid"]][metric] for q in subset) / len(subset)
                           for metric in ["graded_recall@10", "graded_mrr@10"]}}
        report["variants"][label]["by_qid_prefix"] = slices
        if label != "dense":
            env = report["variants"][label]["retrieval"]["environment"]
            if env["HW2_RETRIEVER"] != label or env["HW2_RERANK"] != "0":
                raise ValueError("unexpected H1 inference configuration")
            top50 = load(f"{prefix}_dev_top50", 50)
            prefix_check = compare_prefix(ranking, top50)
            if prefix_check["changed_top10_count"]:
                raise ValueError("top-k 50 prefix differs from top-k 10")
            report["variants"][label]["candidate_gold"] = gold_metrics(queries, top50, (10, 50))["metrics"]
            report["variants"][label]["candidate_timing"] = audit_timing(f"{prefix}_dev_top50_timing.json")
            fast = read(f"{prefix}_dev_fast_top10.json")["results"]
            changed = [r["qid"] for r in fast if r["ranked_pids"] != ranking[r["qid"]]]
            report["variants"][label]["dev_fast_vs_full_changed_qids"] = changed
    report["b0_regression_timing"] = audit_timing("h1_b0_dev_top50_timing.json")
    report["paired_changes"] = {}
    cases = []
    for label in ["bm25", "hybrid"]:
        report["paired_changes"][label] = {}
        for metric in ["graded_recall@10", "graded_mrr@10"]:
            delta = [graded_rows[label][qid][metric] - graded_rows["dense"][qid][metric] for qid in order]
            report["paired_changes"][label][metric] = {
                "mean_difference": sum(delta) / len(delta), "improved_queries": sum(d > 0 for d in delta),
                "worsened_queries": sum(d < 0 for d in delta), "unchanged_queries": sum(d == 0 for d in delta)}
        for direction, sign in [("improved", 1), ("worsened", -1)]:
            selected = [q for q in queries if sign * (graded_rows[label][q["qid"]]["graded_recall@10"] -
                        graded_rows["dense"][q["qid"]]["graded_recall@10"]) > 0]
            for q in selected[:3]:
                cases.append({"variant": label, "direction": direction, "query": q,
                    "gold_title": products[q["gold_pid"]].split("\n")[0],
                    "rankings": {name: [{**c, "title": products[c["pid"]].split("\n")[0]}
                        for c in graded_rows[name][q["qid"]]["candidates"]] for name in ["dense", label]}})
    eligible = [label for label in ["bm25", "hybrid"] if all(
        report["paired_changes"][label][metric]["mean_difference"] > 0
        for metric in ["graded_recall@10", "graded_mrr@10"])]
    report["selected_variant"] = max(eligible, key=lambda name: (
        report["variants"][name]["graded"]["metrics"]["graded_recall@10"],
        report["variants"][name]["graded"]["metrics"]["graded_mrr@10"])) if eligible else "dense"
    default_path = ROOT / "results/h1_selected_default_dev_top10.json"
    if default_path.exists():
        default = load_submission(default_path, queries, set(products), top_k=10)
        if default != rankings[report["selected_variant"]]:
            raise ValueError("promoted default differs from selected experiment")
        report["selected_default_verified"] = True
        report["selected_default_timing"] = audit_timing("h1_selected_default_dev_timing.json")
    else:
        report["selected_default_verified"] = False
    report["notes"] = ["same frozen grouped dev; selecting on dev is not an unseen-test result",
        "BM25 reads full products, dense truncates at 512; this is a system ablation, not an isolated formula experiment",
        "same judge model/order/batch 8; changed candidates alter batch composition",
        "RRF uses rankings and filters zero-score lexical candidates; no MiniLM reranker",
        "historical source hashes resolved only against preserved exact snapshots"]
    report["analysis_sha256"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in
        [Path(__file__).resolve(), ROOT / "code/script/audit_w07.py", ROOT / "code/script/audit_w06.py"]}
    for name, value in [("h1_summary.json", report), ("h1_cases.json", cases)]:
        (ROOT / "results" / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": "passed", "metrics": {label: data["graded"]["metrics"]
        for label, data in report["variants"].items()}, "paired_changes": report["paired_changes"],
        "selected_variant": report["selected_variant"], "selected_default_verified": report["selected_default_verified"]}, indent=2))


if __name__ == "__main__":
    main()
