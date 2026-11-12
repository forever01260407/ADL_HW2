#!/usr/bin/env python3
"""Audit the W07 ablation, candidate containment, historical sources and raw grades."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import gold_metrics, sha256
from audit_w06 import audit_graded
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission
from smoke_retrieval import PROTECTED


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def audit_timing(name):
    data = read(name)
    if data["exit_code"] != 0 or data["resources"]["exit_code"] != 0:
        raise ValueError(f"failed run: {name}")
    archived_names = {"retrieval_utils.py": "retrieval_utils.py",
                      "code/script/run_measured.py": "run_measured.py"}
    resolved = {}
    for path, expected in data["input_sha256"].items():
        actual_path = ROOT / path
        if sha256(actual_path) != expected and path in archived_names:
            matches = [ROOT / "code/script/snapshots" / stage / archived_names[path]
                       for stage in ["w06", "w07", "h1", "h1_default", "h1u"]]
            actual_path = next((p for p in matches if p.is_file() and sha256(p) == expected), actual_path)
        if sha256(actual_path) != expected:
            raise ValueError(f"source/input fingerprint mismatch: {name}: {path}")
        resolved[path] = actual_path.relative_to(ROOT).as_posix()
    return {"cold_process_wall_seconds": data["cold_process_wall_seconds"],
            "resources": data["resources"], "environment": data["environment"],
            "verified_sources": resolved}


def main():
    products = load_products(str(ROOT / "data/train/products.jsonl"))
    queries = load_eval_queries(str(ROOT / "data/derived/dev.jsonl"))
    query_order = [q["qid"] for q in queries]
    for name, expected in PROTECTED.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"official source changed: {name}")
    before = (ROOT / "code/script/snapshots/w06/retrieval_utils.py").read_text(encoding="utf-8")
    after = (ROOT / "retrieval_utils.py").read_text(encoding="utf-8")
    start_marker = "# ==================== STUDENT CODE START ===================="
    end_marker = "# ===================== STUDENT CODE END ====================="
    if (before.split(start_marker)[0], before.split(end_marker)[1]) != (
            after.split(start_marker)[0], after.split(end_marker)[1]):
        raise ValueError("retrieval_utils.py changed outside the student block")
    rankings = {}
    for tag, k in [("w06_b0_dev_top10", 10), ("w06_b0_dev_top50", 50),
                   ("w07_b0_dev_top50", 50), ("w07_a1_dev_top10", 10)]:
        rankings[tag] = load_submission(ROOT / f"results/{tag}.json", queries, set(products), top_k=k)
        if list(rankings[tag]) != query_order:
            raise ValueError(f"query order changed: {tag}")
    if rankings["w06_b0_dev_top50"] != rankings["w07_b0_dev_top50"]:
        raise ValueError("B0 regression: candidate rankings differ from W06")
    for qid, pids in rankings["w07_a1_dev_top10"].items():
        if not set(pids) <= set(rankings["w07_b0_dev_top50"][qid]):
            raise ValueError(f"reranker added a product outside the candidate set: {qid}")
    report = {"scope": "fixed dev 966 / full 50000 products; A1 only; no training/public",
              "protected_sources_unchanged": True, "student_block_boundary_verified": True,
              "b0_top50_exactly_reproduced": True,
              "a1_candidate_containment": True, "official_hardware_validation": False,
              "models": json.loads((ROOT / "code/script/configs/models.lock.json").read_text()),
              "variants": {}}
    download = read("reranker_download.json")
    for name, details in download["files"].items():
        if sha256(ROOT / download["path"] / name) != details["sha256"]:
            raise ValueError(f"reranker model file changed: {name}")
    report["reranker_files_verified"] = True
    pair_diagnostics = read("w07_pair_diagnostics.json")
    for name, expected in pair_diagnostics["input_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"pair diagnostic input changed: {name}")
    report["pair_diagnostics"] = {k: v for k, v in pair_diagnostics.items()
                                  if k not in {"records", "input_sha256"}}
    summaries = {"B0": read("w06_b0_dev_graded.json"), "A1": read("w07_a1_dev_graded.json")}
    rows = {}
    for label, prefix in [("B0", "w06_b0"), ("A1", "w07_a1")]:
        summary = summaries[label]
        ranking = rankings[f"{prefix}_dev_top10"]
        if [r["qid"] for r in summary["records"]] != query_order:
            raise ValueError("judge record order differs from frozen dev")
        rows[label] = {r["qid"]: r for r in summary["records"]}
        report["variants"][label] = {
            "graded": audit_graded(summary, queries, ranking),
            "gold": gold_metrics(queries, ranking)["metrics"],
            "retrieval": audit_timing(f"{prefix}_dev_timing.json"),
            "judge": audit_timing(f"{prefix}_dev_judge_timing.json"),
        }
    report["b0_regression_timing"] = audit_timing("w07_b0_dev_top50_timing.json")
    report["a1_dev_fast_timing"] = audit_timing("w07_a1_dev_fast_timing.json")
    fast = read("w07_a1_dev_fast_top10.json")["results"]
    report["a1_dev_fast_vs_full_changed_qids"] = [r["qid"] for r in fast
        if r["ranked_pids"] != rankings["w07_a1_dev_top10"][r["qid"]]]
    report["paired_changes"] = {}
    for metric in ["graded_recall@10", "graded_mrr@10"]:
        differences = [rows["A1"][qid][metric] - rows["B0"][qid][metric] for qid in query_order]
        report["paired_changes"][metric] = {
            "mean_difference": sum(differences) / len(differences),
            "improved_queries": sum(d > 0 for d in differences),
            "worsened_queries": sum(d < 0 for d in differences),
            "unchanged_queries": sum(d == 0 for d in differences),
        }
    raw_queries = [json.loads(line) for line in (ROOT / "data/derived/dev.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    report["by_message_count"] = {}
    for count in sorted({len(q["conversation"]) for q in raw_queries}):
        ids = [q["dialogue_id"] for q in raw_queries if len(q["conversation"]) == count]
        report["by_message_count"][str(count)] = {label: {metric:
            sum(rows[label][qid][metric] for qid in ids) / len(ids)
            for metric in ["graded_recall@10", "graded_mrr@10"]} for label in rows}
    deltas = report["paired_changes"]
    recall_delta, mrr_delta = (deltas[key]["mean_difference"] for key in
                             ["graded_recall@10", "graded_mrr@10"])
    report["selection"] = ("A1 eligible: both graded metrics improved; hardware validation pending"
        if recall_delta > 0 and mrr_delta > 0 else "Keep B0 default; A1 did not improve both graded metrics")
    report["notes"] = ["one dev comparison, not an unseen-test guarantee",
                       "same judge model/order/batch 8; candidate changes alter batch composition",
                       "gold-only and official graded metrics are reported separately",
                       "W06 code hashes checked against preserved exact source snapshots"]
    cases = []
    for direction in ["improved", "worsened"]:
        sign = 1 if direction == "improved" else -1
        selected = [q for q in queries if sign * (rows["A1"][q["qid"]]["graded_recall@10"] -
                    rows["B0"][q["qid"]]["graded_recall@10"]) > 0]
        for query in selected[:3]:
            qid = query["qid"]
            cases.append({"direction": direction, "query": query,
                "gold_title": products[query["gold_pid"]].split("\n")[0],
                "variants": {label: [{**candidate, "title": products[candidate["pid"]].split("\n")[0]}
                    for candidate in rows[label][qid]["candidates"]] for label in rows}})
    for filename, content in [("w07_summary.json", report), ("w07_cases.json", cases)]:
        (ROOT / "results" / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": "passed", "metrics": {label: data["graded"]["metrics"]
        for label, data in report["variants"].items()}, "paired_changes": report["paired_changes"],
        "selection": report["selection"]}, indent=2))


if __name__ == "__main__":
    main()
