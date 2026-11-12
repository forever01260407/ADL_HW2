#!/usr/bin/env python3
"""H1U Gold screening with one frozen Dense ranking and lexical-only ablations.

This script never invokes evaluate.py. Only the single gated winner may be
evaluated afterwards, after an uncached official-entrypoint equivalence check.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import sha256
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission
from retrieval import save_submission

CONFIGS = {"u0_full": ("full", 1, 1), "u1_users": ("users", 1, 1),
           "u2_latest": ("users", 2, 1), "u3_title": ("users", 1, 2)}
METRICS = ("gold_hit@10", "gold_mrr@10", "gold_hit@50")


def fuse(dense, lexical, positive_counts, k=50):
    if set(dense) != set(lexical) or set(dense) != set(positive_counts):
        raise ValueError("fusion query coverage differs")
    result = []
    for qid, dense_pids in dense.items():
        count = positive_counts[qid]
        if not 0 <= count <= len(lexical[qid]):
            raise ValueError("invalid positive candidate count")
        scores = {}
        for stream in (dense_pids, lexical[qid][:count]):
            for rank, pid in enumerate(stream, 1):
                scores[pid] = scores.get(pid, 0) + 1 / (60 + rank)
        result.append({"qid": qid, "ranked_pids": sorted(scores, key=lambda p: (-scores[p], p))[:k]})
    return result


def gate(candidate, baseline):
    for c, b in [(candidate["metrics"], baseline["metrics"]),
                 (candidate["by_qid_prefix"]["beauty-train"], baseline["by_qid_prefix"]["beauty-train"])]:
        delta = [c[key] - b[key] for key in METRICS]
        if any(d < -1e-12 for d in delta) or not any(d > 1e-12 for d in delta):
            return False
    return True


def paired_changes(candidate, baseline, beauty_only=False):
    old = {r["qid"]: r["gold_rank"] for r in baseline["records"]}
    changes = {}
    for metric, cutoff in [("gold_hit@10", 10), ("gold_mrr@10", 10), ("gold_hit@50", 50)]:
        better, worse = [], []
        for row in candidate["records"]:
            qid, rank = row["qid"], row["gold_rank"]
            if beauty_only and qid.rsplit("-", 1)[0] != "beauty-train":
                continue
            value = lambda r: (1 / r if metric == "gold_mrr@10" else 1) if r is not None and r <= cutoff else 0
            delta = value(rank) - value(old[qid])
            if delta > 0:
                better.append(qid)
            elif delta < 0:
                worse.append(qid)
        changes[metric] = {"improved": len(better), "worsened": len(worse),
                           "improved_qids": better, "worsened_qids": worse}
    return changes


def run(command, env=None):
    subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL)


def main():
    output = ROOT / "results"
    summary_path = output / "h1u_screening.json"
    if summary_path.exists():
        raise ValueError("screening already exists; preserve this fixed experiment")
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    load = lambda path, k: load_submission(path, queries, set(products), top_k=k)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    dense_path = output / "h1u_dense_top100.json"
    dense = load(dense_path, 100)
    old_dense = load(output / "h1_b0_dev_top50.json", 50)
    if any(dense[qid][:50] != old_dense[qid] for qid in dense):
        raise ValueError("Dense changed from H1 full-transcript baseline")
    baseline_path = output / "h1u_baseline_gold.json"
    run(["code/script/analyze_results.py", "--retrieval-results", "results/h1_hybrid_dev_top50.json",
         "--dialogues", "data/derived/dev.jsonl", "--output", baseline_path])
    baseline = read(baseline_path)
    report = {"scope": "BM25 representation only; Dense full transcript frozen; no training/reranker/public",
        "dense_top50_reproduced": True, "shared_dense": str(dense_path),
        "timing_note": "shared Dense cache is for development ablation only; lexical timings are not hybrid cold starts",
        "gate": "overall and beauty-train: no decrease in Hit10/MRR10/Hit50, at least one increase in each",
        "baseline": {"metrics": baseline["metrics"], "beauty_train": baseline["by_qid_prefix"]["beauty-train"]},
        "variants": {}}
    for name, (mode, last, title) in CONFIGS.items():
        prefix = output / f"h1u_{name}"
        timing = Path(str(prefix) + "_lexical_timing.json")
        lexical_path = Path(str(prefix) + "_lexical_top100.json")
        trace_path = Path(str(prefix) + "_trace.json")
        settings = {"HW2_RETRIEVER": "bm25", "HW2_RERANK": "0", "HW2_BM25_QUERY_MODE": mode,
            "HW2_BM25_LAST_WEIGHT": str(last), "HW2_BM25_TITLE_WEIGHT": str(title),
            "HW2_BM25_TRACE_PATH": str(trace_path), "HW2_RRF_CANDIDATES": "100"}
        if not timing.exists():
            run(["code/script/run_measured.py", "--output", timing, "--", "retrieval.py",
                 "--dialogues", "data/derived/dev.jsonl", "--products", "data/train/products.jsonl",
                 "--top-k", "100", "--output", lexical_path], {**os.environ, **settings})
        measured = read(timing)
        if measured["exit_code"] or any(measured["environment"].get(key) != value for key, value in settings.items()):
            raise ValueError("failed or mismatched lexical experiment")
        for path, expected in measured["input_sha256"].items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f"changed lexical source/input: {path}")
        lexical = load(lexical_path, 100)
        trace = read(trace_path)
        positive = {r["qid"]: r["positive_count"] for r in trace["results"]}
        fused = fuse(dense, lexical, positive)
        top50_path, top10_path = Path(str(prefix) + "_top50.json"), Path(str(prefix) + "_top10.json")
        save_submission(top50_path, fused, top_k=50)
        save_submission(top10_path, [{"qid": r["qid"], "ranked_pids": r["ranked_pids"][:10]} for r in fused], top_k=10)
        if name == "u0_full" and load(top50_path, 50) != load(output / "h1_hybrid_dev_top50.json", 50):
            raise ValueError("shared-Dense fusion did not reproduce original H1 exactly")
        gold_path = Path(str(prefix) + "_gold.json")
        run(["code/script/analyze_results.py", "--retrieval-results", top50_path,
             "--dialogues", "data/derived/dev.jsonl", "--compare-top10", top10_path, "--output", gold_path])
        gold = read(gold_path)
        report["variants"][name] = {"settings": settings, "metrics": gold["metrics"],
            "beauty_train": gold["by_qid_prefix"]["beauty-train"], "eligible": gate(gold, baseline),
            "paired_overall": paired_changes(gold, baseline), "paired_beauty": paired_changes(gold, baseline, True),
            "trace_diagnostics": {k: v for k, v in trace.items() if k != "results"},
            "artifacts_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in
                [lexical_path, trace_path, top50_path, top10_path, gold_path, timing]}}
    eligible = [name for name, data in report["variants"].items() if name != "u0_full" and data["eligible"]]
    report["selected_variant"] = max(eligible, key=lambda name: (
        report["variants"][name]["beauty_train"]["gold_mrr@10"],
        report["variants"][name]["metrics"]["gold_mrr@10"],
        report["variants"][name]["metrics"]["gold_hit@10"],
        report["variants"][name]["metrics"]["gold_hit@50"])) if eligible else None
    report["official_judge_runs"] = 0
    report["input_sha256"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in
        [dense_path, baseline_path, ROOT / "data/derived/dev.jsonl", ROOT / "data/train/products.jsonl",
         ROOT / "retrieval_utils.py", Path(__file__).resolve(), ROOT / "code/script/analyze_results.py"]}
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variants": {name: {k: data[k] for k in ["metrics", "beauty_train", "eligible"]}
        for name, data in report["variants"].items()}, "selected_variant": report["selected_variant"],
        "official_judge_runs": 0}, indent=2))


if __name__ == "__main__":
    main()
