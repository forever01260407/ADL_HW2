#!/usr/bin/env python3
"""H2B: four authorized RRF weights, then one frozen full-dev official Judge."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import gold_metrics, sha256
from audit_w06 import audit_graded
from audit_w07 import audit_timing
from eval_utils import load_eval_queries, load_products
from evaluate import LLMJudge, build_judge_items, load_submission
from retrieval import save_submission
from run_rrf_ablation import (BASELINE, DENSE, LEXICAL, TRACE, METRICS, category, paired,
                              require_baseline, verify_hashes, weighted_fuse)
from smoke_retrieval import PROTECTED

RESULTS = ROOT / "results"
CONFIGS = {"d5_b1": (5, 1), "d6_b1": (6, 1), "d8_b1": (8, 1), "d10_b1": (10, 1)}
ORDER = ("gold_mrr@10", "gold_hit@10", "gold_hit@50")


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_new(name, value):
    with (RESULTS / name).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_history():
    verify_hashes(read("results/h2_rrf_final.json")["artifact_sha256"])
    verify_hashes(read("results/h2_rrf_plan.json")["input_sha256"])
    verify_hashes(PROTECTED)


def frozen():
    verify_history()
    plan = read("results/h2b_rrf_plan.json")
    verify_hashes(plan["input_sha256"])
    if plan["configs"] != {k: list(v) for k, v in CONFIGS.items()} or plan["selection_order"] != list(ORDER):
        raise ValueError("settings changed after plan freeze")
    return plan


def choose_variant(variants):
    if not variants:
        raise ValueError("no candidates")
    for row in variants.values():
        if any(not isinstance(row["metrics"][key], (int, float)) or
               not math.isfinite(row["metrics"][key]) or not 0 <= row["metrics"][key] <= 1 for key in ORDER):
            raise ValueError("invalid Gold selection metric")
    return max(variants, key=lambda name: tuple(variants[name]["metrics"][key] for key in ORDER))


def verify_fusion(dense, lexical, positive, ranked, dw, bw):
    """Independent rank-map formula, separate from the accumulation implementation."""
    if set(dense) != set(ranked):
        raise ValueError("independent audit query coverage differs")
    for qid in dense:
        dr = {pid: rank for rank, pid in enumerate(dense[qid], 1)}
        br = {pid: rank for rank, pid in enumerate(lexical[qid][:positive[qid]], 1)}
        def score(pid):
            return (dw / (60 + dr[pid]) if pid in dr and dw else 0) + (
                    bw / (60 + br[pid]) if pid in br and bw else 0)
        pool = (set(dr) if dw else set()) | (set(br) if bw else set())
        if ranked[qid] != sorted(pool, key=lambda pid: (-score(pid), pid))[:len(ranked[qid])]:
            raise ValueError(f"independent RRF audit failed: {qid}")
    return len(ranked)


def prepare():
    if (RESULTS / "h2b_rrf_plan.json").exists():
        raise ValueError("H2B plan already frozen")
    verify_history()
    inputs = ["results/h2_rrf_final.json", "results/h2_rrf_summary.json", "results/w08_epoch1_dev_graded.json",
        "results/w08_epoch1_dev_top10.json", "code/script/extend_rrf_ablation.py",
        "code/script/tests/test_rrf_extension.py", "code/script/run_measured.py",
        "code/script/configs/models.lock.json", "results/generator_download.json",
        "data/derived/dev.jsonl", "data/train/products.jsonl", "retrieval_utils.py", DENSE, LEXICAL, TRACE]
    plan = {"experiment": "h2b_extended_rrf", "created_utc": datetime.now(timezone.utc).isoformat(),
        "configs": CONFIGS, "selection_order": ORDER, "maximum_full_dev_judge_runs": 1,
        "selection_scope": "original eight H2 plus four H2B configurations, dev Gold only",
        "scope": "dev 966/full 50000 products; no training/public/default change/W11",
        "rrf_constant": 60, "route_top_k": 100, "formal_inference_change": False,
        "input_sha256": {**{p: sha256(ROOT / p) for p in inputs}, **PROTECTED}}
    write_new("h2b_rrf_plan.json", plan)
    print(json.dumps({"configs": CONFIGS, "selection_order": ORDER, "history_verified": True}))


def screen():
    frozen()
    if (RESULTS / "h2b_rrf_screening.json").exists():
        raise ValueError("screening already complete")
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    if len(queries) != 966 or len(products) != 50000:
        raise ValueError("wrong full-dev scope")
    load = lambda path, k: load_submission(ROOT / path, queries, set(products), top_k=k)
    dense, lexical, baseline = load(DENSE, 100), load(LEXICAL, 100), load(BASELINE, 50)
    order = [q["qid"] for q in queries]
    trace = read(TRACE)["results"]
    if list(dense) != order or list(lexical) != order or [r["qid"] for r in trace] != order:
        raise ValueError("cached query order differs")
    positive = {r["qid"]: r["positive_count"] for r in trace}
    reproduction = require_baseline({r["qid"]: r["ranked_pids"] for r in
                                    weighted_fuse(dense, lexical, positive)}, baseline)
    old = read("results/h2_rrf_summary.json")
    variants = {name: {key: row[key] for key in ["dense_weight", "bm25_weight", "metrics", "by_category",
                "by_qid_prefix", "artifacts_sha256"]} | {"origin": "h2", "prefix": f"results/h2_{name}"}
                for name, row in old["variants"].items()}
    baseline_gold = read("results/h2_d1_b1_gold.json")
    categories = {label: [q for q in queries if category(q["qid"]) == label]
                  for label in variants["d1_b1"]["by_category"]}
    audited = 0
    for name, (dw, bw) in CONFIGS.items():
        prefix = f"results/h2b_{name}"
        paths = [f"{prefix}_{suffix}.json" for suffix in ["top50", "top10", "gold"]]
        if any((ROOT / p).exists() for p in paths):
            raise ValueError(f"refusing to overwrite {name}")
        rows = weighted_fuse(dense, lexical, positive, dw, bw)
        ranked = {r["qid"]: r["ranked_pids"] for r in rows}
        audited += verify_fusion(dense, lexical, positive, ranked, dw, bw)
        save_submission(ROOT / paths[0], rows, top_k=50)
        save_submission(ROOT / paths[1], [{"qid": r["qid"], "ranked_pids": r["ranked_pids"][:10]}
                                        for r in rows], top_k=10)
        subprocess.run([sys.executable, "code/script/analyze_results.py", "--retrieval-results", paths[0],
            "--dialogues", "data/derived/dev.jsonl", "--products", "data/train/products.jsonl",
            "--compare-top10", paths[1], "--output", paths[2]], cwd=ROOT, check=True,
            stdout=subprocess.DEVNULL)
        gold = read(paths[2])
        variants[name] = {"origin": "h2b", "prefix": prefix, "dense_weight": dw, "bm25_weight": bw,
            "metrics": gold["metrics"], "by_qid_prefix": gold["by_qid_prefix"],
            "by_category": {label: gold_metrics(subset, {q["qid"]: ranked[q["qid"]] for q in subset},
                             (1, 10, 50))["metrics"] for label, subset in categories.items()},
            "paired_gold_vs_1to1": paired(gold, baseline_gold, order),
            "artifacts_sha256": {p: sha256(ROOT / p) for p in paths}}
    best = choose_variant(variants)
    report = {"kind": "dev_gold_screening_not_graded", "configs": 12, "new_configs": 4,
        "baseline_reproduction": reproduction, "new_independent_rankings_verified": audited,
        "variants": variants, "selected_variant": best, "selection_order": ORDER,
        "selected_top10": variants[best]["prefix"] + "_top10.json",
        "selected_top50": variants[best]["prefix"] + "_top50.json",
        "selected_ranking_sha256": sha256(ROOT / (variants[best]["prefix"] + "_top10.json")),
        "official_judge_runs_at_selection": 0, "public_runs": 0,
        "plan_sha256": sha256(RESULTS / "h2b_rrf_plan.json")}
    frozen()
    write_new("h2b_rrf_screening.json", report)
    print(json.dumps({"selected_variant": best, "variants": {name: row["metrics"]
                     for name, row in variants.items()}}, indent=2))


def selected():
    frozen()
    screen = read("results/h2b_rrf_screening.json")
    if screen["plan_sha256"] != sha256(RESULTS / "h2b_rrf_plan.json"):
        raise ValueError("screening plan changed")
    if choose_variant(screen["variants"]) != screen["selected_variant"]:
        raise ValueError("winner differs from frozen selection rule")
    for row in screen["variants"].values():
        verify_hashes(row["artifacts_sha256"])
    return screen


def prepare_judge():
    screen = selected()
    if any((RESULTS / name).exists() for name in ["h2b_judge_plan.json", "h2b_winner_graded.json",
                                               "h2b_winner_judge_timing.json"]):
        raise ValueError("Judge already authorized/attempted")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoTokenizer
    model = read("code/script/configs/models.lock.json")["generator"]
    download = read("results/generator_download.json")
    model_hashes = {f'{model["path"]}/{name}': row["sha256"] for name, row in download["files"].items()}
    verify_hashes(model_hashes)
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    ranked = load_submission(ROOT / screen["selected_top10"], queries, set(products), top_k=10)
    top50 = load_submission(ROOT / screen["selected_top50"], queries, set(products), top_k=50)
    if any(ranked[qid] != top50[qid][:10] for qid in ranked):
        raise ValueError("winner top-ten differs from top-50 prefix")
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    config = AutoConfig.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    items = build_judge_items({q["qid"]: q for q in queries}, ranked, products)
    renderer = SimpleNamespace(tokenizer=tokenizer)
    lengths = [len(tokenizer(LLMJudge._render(renderer, item))["input_ids"]) for item in items]
    if not lengths or max(lengths) + 12 > config.max_position_embeddings:
        raise ValueError("original untruncated prompt exceeds context")
    plan = {"selected_variant": screen["selected_variant"], "ranking_path": screen["selected_top10"],
        "ranking_sha256": sha256(ROOT / screen["selected_top10"]), "model": model,
        "maximum_full_dev_judge_runs": 1, "queries": len(queries), "non_gold_comparisons": len(items),
        "max_input_tokens": max(lengths), "context_limit": config.max_position_embeddings,
        "batch_size": 8, "max_new_tokens": 12, "dtype": "bfloat16", "do_sample": False,
        "truncation": False, "input_sha256": {**model_hashes,
        "results/h2b_rrf_screening.json": sha256(RESULTS / "h2b_rrf_screening.json")}}
    write_new("h2b_judge_plan.json", plan)
    print(json.dumps({k: v for k, v in plan.items() if k != "input_sha256"}, indent=2))


def judge():
    selected()
    plan = read("results/h2b_judge_plan.json")
    verify_hashes(plan["input_sha256"])
    if sha256(ROOT / plan["ranking_path"]) != plan["ranking_sha256"]:
        raise ValueError("frozen winner ranking changed")
    if any((RESULTS / name).exists() for name in ["h2b_winner_graded.json", "h2b_winner_judge_timing.json",
                                                "h2b_winner_judge_timing.log"]):
        raise ValueError("one-shot Judge already attempted; do not rerun")
    subprocess.run([sys.executable, "code/script/run_measured.py", "--output",
        "results/h2b_winner_judge_timing.json", "--", "evaluate.py", "--retrieval-results", plan["ranking_path"],
        "--dialogues", "data/derived/dev.jsonl", "--products", "data/train/products.jsonl", "--model",
        plan["model"]["path"], "--batch-size", "8", "--max-new-tokens", "12", "--top-k", "10",
        "--device", "cuda", "--output", "results/h2b_winner_graded.json"], cwd=ROOT, check=True)


def graded_group(records, baseline, ids):
    result = {"queries": len(ids), "candidate": {}, "baseline": {}, "paired": {}}
    for key in ["graded_recall@10", "graded_mrr@10"]:
        differences = {qid: records[qid][key] - baseline[qid][key] for qid in ids}
        result["candidate"][key] = sum(records[qid][key] for qid in ids) / len(ids)
        result["baseline"][key] = sum(baseline[qid][key] for qid in ids) / len(ids)
        result["paired"][key] = {"mean_difference": sum(differences.values()) / len(ids),
            "improved": sum(d > 0 for d in differences.values()),
            "worsened": sum(d < 0 for d in differences.values()),
            "improved_qids": [q for q, d in differences.items() if d > 0],
            "worsened_qids": [q for q, d in differences.items() if d < 0]}
    return result


def final():
    screen = selected()
    plan = read("results/h2b_judge_plan.json")
    verify_hashes(plan["input_sha256"])
    timing = audit_timing("h2b_winner_judge_timing.json")
    command = read("results/h2b_winner_judge_timing.json")["command"]
    for flag, value in [("--retrieval-results", plan["ranking_path"]), ("--dialogues", "data/derived/dev.jsonl"),
        ("--products", "data/train/products.jsonl"), ("--model", plan["model"]["path"]),
        ("--batch-size", "8"), ("--max-new-tokens", "12"), ("--top-k", "10"), ("--device", "cuda")]:
        if command[command.index(flag) + 1] != value:
            raise ValueError(f"Judge command changed: {flag}")
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    ranked = load_submission(ROOT / plan["ranking_path"], queries, set(products), top_k=10)
    graded = read("results/h2b_winner_graded.json")
    audited = audit_graded(graded, queries, ranked)
    if [r["qid"] for r in graded["records"]] != [q["qid"] for q in queries]:
        raise ValueError("Judge query order changed")
    baseline = read("results/w08_epoch1_dev_graded.json")
    baseline_ranked = load_submission(ROOT / "results/w08_epoch1_dev_top10.json", queries, set(products), top_k=10)
    baseline_audit = audit_graded(baseline, queries, baseline_ranked)
    records, old_records = ({r["qid"]: r for r in value["records"]} for value in [graded, baseline])
    order = [q["qid"] for q in queries]
    overall = graded_group(records, old_records, order)
    groups = {label: graded_group(records, old_records, [qid for qid in order if category(qid) == label])
              for label in screen["variants"]["d1_b1"]["by_category"]}
    sources = {label: graded_group(records, old_records, [qid for qid in order if qid.rsplit("-", 1)[0] == label])
               for label in screen["variants"]["d1_b1"]["by_qid_prefix"]}
    non_gold = lambda data: {(r["qid"], c["pid"]): c["gain"] for r in data["records"]
                            for c in r["candidates"] if c["source"] == "llm"}
    new_pairs, old_pairs = non_gold(graded), non_gold(baseline)
    common = set(new_pairs) & set(old_pairs)
    report = {"status": "single_dev_judge_complete_not_promoted", "selected_variant": screen["selected_variant"],
        "gold": screen["variants"][screen["selected_variant"]]["metrics"], "candidate_graded": audited,
        "baseline_graded": baseline_audit, "overall": overall, "by_category": groups, "by_qid_prefix": sources,
        "shared_non_gold_pairs": len(common),
        "shared_pairs_with_changed_gain": sum(new_pairs[key] != old_pairs[key] for key in common),
        "both_graded_metrics_improved": all(row["mean_difference"] > 0 for row in overall["paired"].values()),
        "judge_timing": timing, "official_judge_runs": 1, "public_runs": 0, "default_changed": False,
        "formal_pipeline_weight_support_added": False, "w11_started": False, "official_hardware_validation": False,
        "input_sha256": {p: sha256(ROOT / p) for p in ["results/h2b_rrf_plan.json",
        "results/h2b_rrf_screening.json", "results/h2b_judge_plan.json", "results/h2b_winner_graded.json",
        "results/h2b_winner_judge_timing.json", "code/script/extend_rrf_ablation.py"]}}
    write_new("h2b_rrf_final.json", report)
    print(json.dumps({"selected_variant": report["selected_variant"], "candidate": audited["metrics"],
        "baseline": baseline_audit["metrics"], "paired": overall["paired"], "comparisons": audited["comparisons"],
        "parse_failures": audited["parse_failures"], "shared_non_gold_pairs": len(common),
        "shared_pairs_with_changed_gain": report["shared_pairs_with_changed_gain"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "screen", "prepare-judge", "judge", "final"], required=True)
    args = parser.parse_args()
    {"prepare": prepare, "screen": screen, "prepare-judge": prepare_judge, "judge": judge, "final": final}[args.stage]()
