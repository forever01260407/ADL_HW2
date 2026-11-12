#!/usr/bin/env python3
"""H2: frozen W08 Dense/BM25 rank-weight ablation on dev only; no Judge."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import gold_metrics, sha256
from audit_w07 import audit_timing
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission
from retrieval import save_submission
from smoke_retrieval import PROTECTED

RESULTS = ROOT / "results"
PLAN = RESULTS / "h2_rrf_plan.json"
SUMMARY = RESULTS / "h2_rrf_summary.json"
CHECKPOINT = "artifacts/training/w08_epoch1/checkpoint"
DENSE = "results/h2_w08_dense_top100.json"
LEXICAL = "results/h1u_u0_full_lexical_top100.json"
TRACE = "results/h1u_u0_full_trace.json"
TIMING = "results/h2_w08_dense_timing.json"
BASELINE = "results/w08_default_dev_top50.json"
CONFIGS = {"dense_only": (1, 0), "d075_b1": (.75, 1), "d1_b1": (1, 1),
           "d11_b1": (1.1, 1), "d15_b1": (1.5, 1), "d2_b1": (2, 1),
           "d3_b1": (3, 1), "d4_b1": (4, 1)}
METRICS = ("gold_hit@10", "gold_mrr@10", "gold_hit@50")


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"changed frozen source/input: {name}")


def weighted_fuse(dense, lexical, positive_counts, dense_weight=1, bm25_weight=1, k=50):
    """Gold-free, deterministic fusion; absence from a list earns no vote."""
    weights = (dense_weight, bm25_weight)
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w)
           or w < 0 for w in weights) or not any(weights):
        raise ValueError("weights must be finite, nonnegative, and not both zero")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if set(dense) != set(lexical) or set(dense) != set(positive_counts):
        raise ValueError("fusion query coverage differs")
    result = []
    for qid, dense_pids in dense.items():
        count = positive_counts[qid]
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= len(lexical[qid]):
            raise ValueError("invalid positive candidate count")
        if any(len(set(stream)) != len(stream) for stream in (dense_pids, lexical[qid])):
            raise ValueError("duplicate candidate PID")
        scores = {}
        for weight, stream in [(dense_weight, dense_pids), (bm25_weight, lexical[qid][:count])]:
            if weight:
                for rank, pid in enumerate(stream, 1):
                    scores[pid] = scores.get(pid, 0.0) + weight / (60 + rank)
        if len(scores) < k:
            raise ValueError("not enough candidates with a positive route weight")
        result.append({"qid": qid, "ranked_pids": sorted(scores, key=lambda p: (-scores[p], p))[:k]})
    return result


def require_baseline(rebuilt, baseline):
    if set(rebuilt) != set(baseline):
        raise ValueError("W08 reproduction query coverage differs")
    changed = [qid for qid in baseline if rebuilt[qid] != baseline[qid]]
    if changed:
        raise ValueError(f"1:1 did not reproduce W08: {len(changed)} changed queries; first={changed[0]}")
    return {"queries": len(baseline), "top_k": 50, "changed_queries": 0,
            "exact_rankings_reproduced": True}


def category(qid):
    source = qid.rsplit("-", 1)[0]
    return "beauty" if source in {"beauty-train", "train-beauty-official"} else source.removeprefix("train-")


def paired(candidate, baseline, ids):
    old = {r["qid"]: r["gold_rank"] for r in baseline["records"]}
    new = {r["qid"]: r["gold_rank"] for r in candidate["records"]}
    output = {"queries": len(ids)}
    for metric in METRICS:
        cutoff = 50 if metric.endswith("50") else 10
        def value(rank):
            return (1 / rank if "mrr" in metric else 1) if rank is not None and rank <= cutoff else 0
        deltas = {qid: value(new[qid]) - value(old[qid]) for qid in ids}
        improved = [qid for qid, delta in deltas.items() if delta > 0]
        worsened = [qid for qid, delta in deltas.items() if delta < 0]
        output[metric] = {"mean_difference": sum(deltas.values()) / len(ids),
            "improved": len(improved), "worsened": len(worsened),
            "unchanged": len(ids) - len(improved) - len(worsened),
            "improved_qids": improved, "worsened_qids": worsened}
    return output


def prepare():
    if PLAN.exists():
        raise ValueError("H2 plan already frozen; preserve experiment history")
    final = read("results/w08_final.json")
    if not final["default_verified"] or not final["adopt"]:
        raise ValueError("W08 default was not verified/adopted")
    verify_hashes({**final["finalization_sha256"], **PROTECTED})
    training = read("artifacts/training/w08_epoch1/training_manifest.json")
    if training["checkpoint"] != CHECKPOINT or training["dev_used_for_gradient"] or training["public_used"]:
        raise ValueError("wrong checkpoint or training data scope")
    checkpoint_hashes = {f"{CHECKPOINT}/{name}": expected
                         for name, expected in training["checkpoint_files"].items()}
    verify_hashes(checkpoint_hashes)
    lexical_history = read("results/h1u_screening.json")["variants"]["u0_full"]["artifacts_sha256"]
    verify_hashes({p: lexical_history[p] for p in [LEXICAL, TRACE,
                  "results/h1u_u0_full_lexical_timing.json"]})
    lexical_audit = audit_timing("h1u_u0_full_lexical_timing.json")
    expected = {"HW2_RETRIEVER": "bm25", "HW2_RERANK": "0", "HW2_RRF_CANDIDATES": "100",
                "HW2_BM25_QUERY_MODE": "full", "HW2_BM25_LAST_WEIGHT": "1", "HW2_BM25_TITLE_WEIGHT": "1"}
    if any(lexical_audit["environment"][key] != value for key, value in expected.items()):
        raise ValueError("lexical cache settings differ from W08")
    # The complete lexical code precedes the encoder import and is unchanged.
    old = (ROOT / "code/script/snapshots/h1u/retrieval_utils.py").read_text(encoding="utf-8")
    current = (ROOT / "retrieval_utils.py").read_text(encoding="utf-8")
    if old.split("    from sentence_transformers import SentenceTransformer")[0] != current.split(
            "    from sentence_transformers import SentenceTransformer")[0]:
        raise ValueError("BM25 implementation changed since cached run")
    inputs = ["data/derived/dev.jsonl", "data/train/products.jsonl", "retrieval_utils.py",
        "code/script/run_rrf_ablation.py", "code/script/run_measured.py", "code/script/analyze_results.py",
        "code/script/audit_w07.py", "code/script/snapshots/h1u/retrieval_utils.py", "results/w08_final.json",
        "results/w08_epoch1_dev_gold.json", "results/h1u_screening.json", BASELINE, LEXICAL, TRACE,
        "results/h1u_u0_full_lexical_timing.json", "artifacts/training/w08_epoch1/training_manifest.json"]
    plan = {"experiment": "h2_w08_rrf_weights", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "dev 966 / 50000 train products; rank weights only; no training/reranker/public/Judge/W11",
        "configs": CONFIGS, "rrf_constant": 60, "route_top_k": 100, "output_top_k": 50,
        "dense_checkpoint": CHECKPOINT, "dense_output": DENSE,
        "lexical_cache": LEXICAL, "lexical_trace": TRACE, "lexical_cache_audit": lexical_audit,
        "lexical_source_identical": True, "forbidden_dense_cache": "results/h1u_dense_top100.json",
        "baseline": BASELINE, "baseline_gate": "all 966 ordered top-50 rankings must match W08 exactly",
        "diagnostic_order": ["gold_mrr@10", "gold_hit@10", "gold_hit@50"],
        "selection_note": "global weights only; diagnostic best is not an adopted default or official score",
        "official_judge_runs": 0, "default_change_authorized": False,
        "input_sha256": {**{p: sha256(ROOT / p) for p in inputs}, **PROTECTED, **checkpoint_hashes}}
    write_new(PLAN, plan)
    print(json.dumps({"plan": str(PLAN), "checkpoint_files_verified": len(checkpoint_hashes),
                      "lexical_cache_verified": True, "configs": CONFIGS}))


def screen():
    if SUMMARY.exists():
        raise ValueError("H2 screening already exists; preserve results")
    plan = read(PLAN)
    verify_hashes(plan["input_sha256"])
    if plan["configs"] != {name: list(weights) for name, weights in CONFIGS.items()}:
        raise ValueError("ablation configurations changed after freeze")
    dense_timing = read(TIMING)
    timing_audit = audit_timing(Path(TIMING).name)
    expected = {"HW2_RETRIEVER": "dense", "HW2_RERANK": "0", "HW2_MODEL_PATH": CHECKPOINT}
    if any(dense_timing["environment"][key] != value for key, value in expected.items()):
        raise ValueError("Dense cache was not produced with the W08 checkpoint")
    command = dense_timing["command"]
    for flag, value in [("--dialogues", "data/derived/dev.jsonl"), ("--products", "data/train/products.jsonl"),
                        ("--output", DENSE), ("--top-k", "100")]:
        if command[command.index(flag) + 1] != value:
            raise ValueError(f"Dense retrieval command mismatch: {flag}")
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    if len(queries) != 966 or len(products) != 50000:
        raise ValueError("not the complete frozen dev/candidate pool")
    load = lambda path, k: load_submission(ROOT / path, queries, set(products), top_k=k)
    dense, lexical, baseline = load(DENSE, 100), load(LEXICAL, 100), load(BASELINE, 50)
    order = [q["qid"] for q in queries]
    trace_rows = read(TRACE)["results"]
    if [r["qid"] for r in trace_rows] != order or list(dense) != order or list(lexical) != order:
        raise ValueError("cached query order differs from frozen dev")
    positive = {r["qid"]: r["positive_count"] for r in trace_rows}
    rebuilt = {r["qid"]: r["ranked_pids"] for r in weighted_fuse(dense, lexical, positive)}
    reproduction = require_baseline(rebuilt, baseline)  # must pass BEFORE any variant is evaluated
    check = {**reproduction, "plan_sha256": sha256(PLAN), "old_untrained_dense_cache_used": False,
        "trained_dense_timing": timing_audit, "input_sha256": {p: sha256(ROOT / p)
        for p in [DENSE, TIMING, LEXICAL, TRACE, BASELINE]}}
    write_new(RESULTS / "h2_rrf_baseline_check.json", check)
    baseline_gold = read("results/w08_epoch1_dev_gold.json")
    if gold_metrics(queries, baseline, (1, 10, 50))["metrics"] != baseline_gold["metrics"]:
        raise ValueError("reproduced rankings disagree with W08 stored Gold metrics")
    categories = {name: [q for q in queries if category(q["qid"]) == name]
                  for name in sorted({category(q["qid"]) for q in queries})}
    source_groups = {name: [q["qid"] for q in queries if q["qid"].rsplit("-", 1)[0] == name]
                     for name in baseline_gold["by_qid_prefix"]}
    report = {"experiment": plan["experiment"], "kind": "gold_only_not_official_graded",
        "queries": len(queries), "candidate_products": len(products), "baseline_reproduction": reproduction,
        "official_judge_runs": 0, "public_runs": 0, "default_changed": False, "variants": {},
        "input_sha256": {**check["input_sha256"], "results/h2_rrf_plan.json": sha256(PLAN),
                          "results/h2_rrf_baseline_check.json": sha256(RESULTS / "h2_rrf_baseline_check.json")}}
    for name, (dw, bw) in CONFIGS.items():
        rows = weighted_fuse(dense, lexical, positive, dw, bw)
        path50, path10 = RESULTS / f"h2_{name}_top50.json", RESULTS / f"h2_{name}_top10.json"
        gold_path = RESULTS / f"h2_{name}_gold.json"
        if any(p.exists() for p in [path50, path10, gold_path]):
            raise ValueError(f"refusing to overwrite variant {name}")
        save_submission(path50, rows, top_k=50)
        save_submission(path10, [{"qid": r["qid"], "ranked_pids": r["ranked_pids"][:10]} for r in rows], top_k=10)
        subprocess.run([sys.executable, "code/script/analyze_results.py", "--retrieval-results", str(path50),
            "--dialogues", "data/derived/dev.jsonl", "--products", "data/train/products.jsonl",
            "--compare-top10", str(path10), "--output", str(gold_path)], cwd=ROOT, check=True,
            stdout=subprocess.DEVNULL)
        gold = read(gold_path)
        ranked = {r["qid"]: r["ranked_pids"] for r in rows}
        changes = paired(gold, baseline_gold, order)
        report["variants"][name] = {"dense_weight": dw, "bm25_weight": bw, "metrics": gold["metrics"],
            "by_category": {label: gold_metrics(subset, {q["qid"]: ranked[q["qid"]] for q in subset},
                            (1, 10, 50))["metrics"] for label, subset in categories.items()},
            "by_qid_prefix": gold["by_qid_prefix"], "paired_overall": changes,
            "paired_by_category": {label: paired(gold, baseline_gold, [q["qid"] for q in subset])
                                   for label, subset in categories.items()},
            "paired_by_qid_prefix": {label: paired(gold, baseline_gold, ids) for label, ids in source_groups.items()},
            "dominates_baseline_on_three_overall_metrics": all(changes[m]["mean_difference"] >= -1e-12
                for m in METRICS) and any(changes[m]["mean_difference"] > 1e-12 for m in METRICS),
            "bm25_exclusive_top10_count": sum(len(set(ranked[q][:10]) - set(dense[q])) for q in order),
            "bm25_exclusive_top50_count": sum(len(set(ranked[q]) - set(dense[q])) for q in order),
            "artifacts_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in [path50, path10, gold_path]}}
    report["diagnostic_best_by_gold_mrr"] = max(CONFIGS, key=lambda name: tuple(
        report["variants"][name]["metrics"][key] for key in plan["diagnostic_order"]))
    verify_hashes(plan["input_sha256"])
    write_new(SUMMARY, report)
    print(json.dumps({"baseline_reproduction": reproduction, "variants": {name: row["metrics"]
        for name, row in report["variants"].items()}, "diagnostic_best_by_gold_mrr":
        report["diagnostic_best_by_gold_mrr"], "official_judge_runs": 0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "screen"], required=True)
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else screen()
