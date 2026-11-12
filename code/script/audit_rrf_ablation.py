#!/usr/bin/env python3
"""Independently reconstruct saved H2 rankings and audit Gold counts, without models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import sha256


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    check(abs(actual - expected) < 1e-12, f"metric mismatch: {actual} != {expected}")


def main():
    plan = read("results/h2_rrf_plan.json")
    summary = read("results/h2_rrf_summary.json")
    baseline_check = read("results/h2_rrf_baseline_check.json")
    for name, expected in {**plan["input_sha256"], **summary["input_sha256"]}.items():
        check(sha256(ROOT / name) == expected, f"changed source/input: {name}")
    check(summary["official_judge_runs"] == summary["public_runs"] == 0, "scope changed")
    check(not summary["default_changed"], "default changed")
    check(baseline_check["exact_rankings_reproduced"] and baseline_check["changed_queries"] == 0,
          "W08 reproduction did not pass")
    check(not baseline_check["old_untrained_dense_cache_used"], "used untrained Dense cache")
    dialogues = [json.loads(line) for line in (ROOT / "data/derived/dev.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    gold = {row["dialogue_id"]: row["target"] for row in dialogues}
    dense = {row["qid"]: row["ranked_pids"] for row in read(plan["dense_output"])["results"]}
    lexical = {row["qid"]: row["ranked_pids"] for row in read(plan["lexical_cache"])["results"]}
    positive = {row["qid"]: row["positive_count"] for row in read(plan["lexical_trace"])["results"]}
    dense_ranks = {qid: {pid: rank for rank, pid in enumerate(pids, 1)} for qid, pids in dense.items()}
    lexical_ranks = {qid: {pid: rank for rank, pid in enumerate(pids[:positive[qid]], 1)}
                     for qid, pids in lexical.items()}
    sources = {label: [qid for qid in gold if qid.rsplit("-", 1)[0] == label]
               for label in summary["variants"]["d1_b1"]["by_qid_prefix"]}
    # Explicit analysis-only category mapping; never used by retrieval or fusion.
    category_sources = {"beauty": ["beauty-train", "train-beauty-official"],
        **{name: [f"train-{name}"] for name in
           ["home-and-kitchen", "pet-supplies", "sports-and-outdoors", "toys-and-games"]}}
    categories = {name: [qid for label in labels for qid in sources[label]]
                  for name, labels in category_sources.items()}
    check(sum(map(len, categories.values())) == len(gold) == 966, "wrong category coverage")
    baseline = {row["qid"]: row["ranked_pids"] for row in read(plan["baseline"])["results"]}
    baseline_ranks = {qid: (pids.index(gold[qid]) + 1 if gold[qid] in pids else None)
                      for qid, pids in baseline.items()}
    audited = {}
    for name, (dw, bw) in plan["configs"].items():
        row = summary["variants"][name]
        for artifact, expected in row["artifacts_sha256"].items():
            check(sha256(ROOT / artifact) == expected, f"changed variant artifact: {artifact}")
        ranked_rows = read(f"results/h2_{name}_top50.json")["results"]
        check([r["qid"] for r in ranked_rows] == list(gold), "ranking query order changed")
        ranked = {r["qid"]: r["ranked_pids"] for r in ranked_rows}
        for qid in gold:
            dr, br = dense_ranks[qid], lexical_ranks[qid]
            pool = (set(dr) if dw else set()) | (set(br) if bw else set())
            def score(pid):
                return ((dw / (60 + dr[pid])) if pid in dr and dw else 0) + (
                    (bw / (60 + br[pid])) if pid in br and bw else 0)
            expected = sorted(pool, key=lambda pid: (-score(pid), pid))[:50]
            check(ranked[qid] == expected, f"independent RRF mismatch: {name}: {qid}")
        ranks = {qid: (pids.index(gold[qid]) + 1 if gold[qid] in pids else None)
                 for qid, pids in ranked.items()}
        def verify_metrics(metrics, ids):
            check(metrics["queries"] == len(ids), "wrong group query count")
            for cutoff in [1, 10, 50]:
                close(metrics[f"gold_hit@{cutoff}"], sum(ranks[q] is not None and ranks[q] <= cutoff
                                                         for q in ids) / len(ids))
                close(metrics[f"gold_mrr@{cutoff}"], sum(1 / ranks[q] if ranks[q] is not None and
                         ranks[q] <= cutoff else 0 for q in ids) / len(ids))
        verify_metrics(row["metrics"], list(gold))
        for label, ids in categories.items():
            verify_metrics(row["by_category"][label], ids)
        for label, ids in sources.items():
            verify_metrics(row["by_qid_prefix"][label], ids)
        for stats, ids in [(row["paired_overall"], list(gold)),
                *[(row["paired_by_category"][label], ids) for label, ids in categories.items()],
                *[(row["paired_by_qid_prefix"][label], ids) for label, ids in sources.items()]]:
            for metric in ["gold_hit@10", "gold_mrr@10", "gold_hit@50"]:
                cutoff = 50 if metric.endswith("50") else 10
                def value(rank):
                    return (1 / rank if "mrr" in metric else 1) if rank is not None and rank <= cutoff else 0
                changes = {qid: value(ranks[qid]) - value(baseline_ranks[qid]) for qid in ids}
                better = {qid for qid, delta in changes.items() if delta > 0}
                worse = {qid for qid, delta in changes.items() if delta < 0}
                saved = stats[metric]
                check(better == set(saved["improved_qids"]) and worse == set(saved["worsened_qids"]),
                      f"paired query IDs differ: {name}: {metric}")
                check(saved["improved"] == len(better) and saved["worsened"] == len(worse) and
                      saved["unchanged"] == len(ids) - len(better) - len(worse), "paired counts differ")
                close(saved["mean_difference"], sum(changes.values()) / len(ids))
        audited[name] = {"independent_rankings_verified": 966,
            "gold_hits_at_10": sum(r is not None and r <= 10 for r in ranks.values()),
            "gold_hits_at_50": sum(r is not None and r <= 50 for r in ranks.values())}
    expected_baseline = read(plan["baseline"])["results"]
    check(read("results/h2_d1_b1_top50.json")["results"] == expected_baseline,
          "saved 1:1 rankings differ from W08")
    output = {"status": "passed", "independent_rankings_verified": 966 * len(audited),
        "variants": audited, "five_categories_and_six_sources_verified": True,
        "official_judge_runs": 0, "public_runs": 0, "default_unchanged": True,
        "input_sha256": {name: sha256(ROOT / name) for name in
        ["code/script/audit_rrf_ablation.py", "results/h2_rrf_plan.json", "results/h2_rrf_summary.json",
         "results/h2_rrf_baseline_check.json", "results/h2_rrf_tests.log"]}}
    target = ROOT / "results/h2_rrf_audit.json"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
