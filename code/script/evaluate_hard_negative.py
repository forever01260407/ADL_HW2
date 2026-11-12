#!/usr/bin/env python3
"""H5 dev-only fixed-candidate screening and at most one original Judge run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from hard_negative_data import ROOT, RESULTS, PLAN, frozen, read, verify_hashes, write_new
from analyze_results import gold_metrics, sha256
from audit_w06 import audit_graded
from eval_utils import load_eval_queries, load_products
from evaluate import LLMJudge, build_judge_items, load_submission
from extend_rrf_ablation import graded_group, verify_fusion
from retrieval import save_submission
from run_rrf_ablation import category, paired, weighted_fuse


def training_verified():
    plan = frozen()
    config = plan["configuration"]
    manifest = read(f'{config["output"]}/training_manifest.json')
    if (manifest["status"] != "completed" or manifest["mode"] != "train" or manifest["train_count"] != 3864
            or manifest["optimizer_steps"] != 242 or manifest["public_used"] or manifest["dev_used_for_gradient"]
            or manifest["dev_used_for_mining"] or manifest["trained_own_negative_exposures"] <= 0
            or manifest["offline_reload_max_abs_difference"] > 1e-5):
        raise ValueError("full H5 training/integrity requirements failed")
    verify_hashes(manifest["input_sha256"])
    verify_hashes({f'{manifest["checkpoint"]}/{p}': h for p, h in manifest["checkpoint_files"].items()})
    if manifest["configuration"] != config:
        raise ValueError("training configuration was changed")
    losses = [json.loads(line) for line in (ROOT / config["output"] / "loss.jsonl").read_text().splitlines()]
    if (len(losses) != 242 or sum(r["batch_size"] for r in losses) != 3864
            or sha256(ROOT / config["output"] / "loss.jsonl") != manifest["loss_sha256"]
            or sha256(ROOT / config["output"] / "batches.json") != manifest["batches_sha256"]):
        raise ValueError("training data coverage or logs changed")
    return plan, manifest


def prepare():
    plan, manifest = training_verified()
    paths = ["code/script/evaluate_hard_negative.py", "code/script/analyze_results.py",
        "code/script/run_rrf_ablation.py", "code/script/extend_rrf_ablation.py", PLAN,
        f'{plan["configuration"]["output"]}/training_manifest.json', "results/h3_base_dense_gold.json",
        "results/h3_base5_gold.json", "results/h3_base5_graded.json",
        "results/h1u_u0_full_lexical_top100.json", "results/h1u_u0_full_trace.json"]
    old = read("results/h3_completion.json")
    verify_hashes(old["artifact_sha256"])
    verify_hashes(old["checkpoint_sha256"])
    original = read("results/h3_evaluation_plan.json")
    lexical_paths = ["results/h1u_u0_full_lexical_top100.json", "results/h1u_u0_full_trace.json"]
    # H3 sealed the exact lexical lists used for the current base5 baseline.
    for p in lexical_paths:
        expected = original["input_sha256"][p]
        if sha256(ROOT / p) != expected:
            raise ValueError("historical BM25 cache changed")
    write_new("h5_evaluation_plan.json", {"candidate_checkpoint": manifest["checkpoint"],
        "configuration": plan["configuration"], "baseline": "H3 trained base + fixed 5:1",
        "dev_queries": 966, "candidate_products": 50000, "maximum_dev_judge_runs": 1,
        "public_runs": 0, "input_sha256": {p: sha256(ROOT / p) for p in paths}})
    print(json.dumps({"training_verified": True, "steps": 242, "evaluation_plan_frozen": True}))


def evaluation_frozen():
    plan, manifest = training_verified()
    evaluation = read("results/h5_evaluation_plan.json")
    verify_hashes(evaluation["input_sha256"])
    return plan, manifest, evaluation


def screen():
    plan, manifest, _ = evaluation_frozen()
    config = plan["configuration"]
    timing = read("results/h5_dev_dense_timing.json")
    verify_hashes(timing["input_sha256"])
    if (timing["exit_code"] or timing["environment"]["HW2_RETRIEVER"] != "dense"
            or timing["environment"]["HW2_MODEL_PATH"] != manifest["checkpoint"]):
        raise ValueError("new Dense retrieval used wrong route/model or failed")
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    if len(queries) != 966 or len(products) != 50000:
        raise ValueError("complete dev/full product pool required")
    dense = load_submission(RESULTS / "h5_dev_dense_top100.json", queries, set(products), top_k=100)
    lexical = load_submission(RESULTS / "h1u_u0_full_lexical_top100.json", queries, set(products), top_k=100)
    positive = {r["qid"]: r["positive_count"] for r in read("results/h1u_u0_full_trace.json")["results"]}
    ids = [q["qid"] for q in queries]
    golds = {}
    for name, dw, bw in [("dense", 1, 0), ("base5", 5, 1)]:
        rows = weighted_fuse(dense, lexical, positive, dw, bw)
        ranked = {r["qid"]: r["ranked_pids"] for r in rows}
        verify_fusion(dense, lexical, positive, ranked, dw, bw)
        prefix = f"results/h5_{name}"
        if any((ROOT / f"{prefix}_{suffix}.json").exists() for suffix in ["top50", "top10", "gold"]):
            raise ValueError("refusing to overwrite H5 screening artifacts")
        save_submission(ROOT / f"{prefix}_top50.json", rows, top_k=50)
        save_submission(ROOT / f"{prefix}_top10.json", [{"qid": qid, "ranked_pids": pids[:10]} for qid, pids in ranked.items()], top_k=10)
        subprocess.run([sys.executable, "code/script/analyze_results.py", "--retrieval-results", f"{prefix}_top50.json",
            "--dialogues", "data/derived/dev.jsonl", "--products", "data/train/products.jsonl",
            "--compare-top10", f"{prefix}_top10.json", "--output", f"{prefix}_gold.json"], cwd=ROOT,
            check=True, stdout=subprocess.DEVNULL)
        gold = read(f"{prefix}_gold.json")
        if gold_metrics(queries, ranked, (1, 10, 50)) != {k: gold[k] for k in ["metrics", "records"]}:
            raise ValueError("independent Gold rank recomputation differs")
        gold["by_category"] = {label: gold_metrics([q for q in queries if category(q["qid"]) == label],
            {q["qid"]: ranked[q["qid"]] for q in queries if category(q["qid"]) == label}, (1, 10, 50))["metrics"]
            for label in sorted({category(qid) for qid in ids})}
        golds[name] = gold
    old_dense, old_hybrid = read("results/h3_base_dense_gold.json"), read("results/h3_base5_gold.json")
    d, h, od, oh = golds["dense"]["metrics"], golds["base5"]["metrics"], old_dense["metrics"], old_hybrid["metrics"]
    gate = config["gold_gate"]
    checks = {"hit10_gain": (h["gold_hit@10"] - oh["gold_hit@10"]) * 966 >= gate["minimum_hit10_count_gain"] - 1e-9,
        "mrr10_gain": h["gold_mrr@10"] - oh["gold_mrr@10"] >= gate["minimum_mrr10_gain"] - 1e-12,
        "hit50_nondecreasing": h["gold_hit@50"] >= oh["gold_hit@50"] - 1e-12,
        "dense_hit10_nondecreasing": d["gold_hit@10"] >= od["gold_hit@10"] - 1e-12,
        "dense_mrr10_nondecreasing": d["gold_mrr@10"] >= od["gold_mrr@10"] - 1e-12}
    write_new("h5_gold_screening.json", {"kind": "dev Gold diagnostic, not official graded",
        "gold_gate": {"passed": all(checks.values()), "checks": checks}, "variants": golds,
        "baseline_dense": old_dense, "baseline_base5": old_hybrid,
        "paired_base5": paired(golds["base5"], old_hybrid, ids), "new_dev_judge_runs": 0, "public_runs": 0,
        "independent_rankings_verified": 1932,
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in [RESULTS / "h5_evaluation_plan.json",
            RESULTS / "h5_dev_dense_top100.json", RESULTS / "h5_dev_dense_timing.json",
            *[RESULTS / f"h5_{name}_{suffix}.json" for name in ["dense", "base5"] for suffix in ["top50", "top10", "gold"]]]}})
    print(json.dumps({"gold_gate": checks, "passed": all(checks.values()), "baseline_base5": oh,
        "new_base5": h, "baseline_dense": od, "new_dense": d}, indent=2))


def prepare_judge():
    evaluation_frozen()
    screening = read("results/h5_gold_screening.json")
    verify_hashes(screening["input_sha256"])
    if not screening["gold_gate"]["passed"]:
        raise ValueError("Gold gate failed: stop, do not run official Judge")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import AutoConfig, AutoTokenizer
    model = read("code/script/configs/models.lock.json")["generator"]
    verify_hashes({f'{model["path"]}/{p}': row["sha256"] for p, row in read("results/generator_download.json")["files"].items()})
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    ranked = load_submission(RESULTS / "h5_base5_top10.json", queries, set(products), top_k=10)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    model_config = AutoConfig.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    renderer = SimpleNamespace(tokenizer=tokenizer)
    items = build_judge_items({q["qid"]: q for q in queries}, ranked, products)
    lengths = [len(tokenizer(LLMJudge._render(renderer, i))["input_ids"]) for i in items]
    if not lengths or max(lengths) + 12 > model_config.max_position_embeddings:
        raise ValueError("untruncated Judge input cannot fit")
    write_new("h5_judge_plan.json", {"model": model, "batch_size": 8, "max_new_tokens": 12,
        "dtype": "bfloat16", "do_sample": False, "truncation": False, "non_gold_comparisons": len(items),
        "max_input_tokens": max(lengths), "maximum_dev_judge_runs": 1,
        "input_sha256": {p: sha256(ROOT / p) for p in ["results/h5_gold_screening.json", "results/h5_base5_top10.json"]}})
    print(json.dumps({"judge_preflight": "passed", "comparisons": len(items), "max_input_tokens": max(lengths)}))


def judge():
    evaluation_frozen()
    plan = read("results/h5_judge_plan.json")
    verify_hashes(plan["input_sha256"])
    if any((RESULTS / f"h5_base5_{s}").exists() for s in ["graded.json", "judge_timing.json", "judge_timing.log"]):
        raise ValueError("preserve the single Judge attempt")
    subprocess.run([sys.executable, "code/script/run_validated.py", "--output", "results/h5_base5_judge_timing.json",
        "--", "evaluate.py", "--retrieval-results", "results/h5_base5_top10.json", "--dialogues", "data/derived/dev.jsonl",
        "--products", "data/train/products.jsonl", "--model", plan["model"]["path"], "--batch-size", "8",
        "--max-new-tokens", "12", "--top-k", "10", "--device", "cuda", "--output", "results/h5_base5_graded.json"], cwd=ROOT, check=True)


def final():
    plan, manifest, _ = evaluation_frozen()
    screening = read("results/h5_gold_screening.json")
    verify_hashes(screening["input_sha256"])
    if not screening["gold_gate"]["passed"]:
        write_new("h5_quality_final.json", {"status": "completed_gold_gate_failed_keep_h4",
            "gold_gate": screening["gold_gate"], "quality_training_runs": 1, "new_dev_judge_runs": 0,
            "public_runs": 0, "default_changed": False})
        print("Gold gate failed: experiment completed; retain H4, no official Judge or Public")
        return
    jp = read("results/h5_judge_plan.json")
    verify_hashes(jp["input_sha256"])
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    ranked = load_submission(RESULTS / "h5_base5_top10.json", queries, set(products), top_k=10)
    new, old = read("results/h5_base5_graded.json"), read("results/h3_base5_graded.json")
    audit = audit_graded(new, queries, ranked)
    t = read("results/h5_base5_judge_timing.json")
    verify_hashes(t["input_sha256"])
    if t["exit_code"] or t["resources"]["exit_code"]:
        raise ValueError("official Judge did not finish")
    before, after = old["metrics"], new["metrics"]
    checks = {"recall_nondecreasing": after["graded_recall@10"] >= before["graded_recall@10"] - 1e-12,
        "mrr_gain": after["graded_mrr@10"] - before["graded_mrr@10"] >= plan["configuration"]["graded_gate"]["minimum_mrr10_gain"] - 1e-12}
    nr, br = ({r["qid"]: r for r in obj["records"]} for obj in [new, old])
    ids = [q["qid"] for q in queries]
    report = {"status": "quality_eligible_pending_formal_validation" if all(checks.values()) else "completed_graded_gate_failed_keep_h4",
        "quality_gate": {"passed": all(checks.values()), "checks": checks}, "candidate": audit,
        "baseline": before, "paired": graded_group(nr, br, ids),
        "by_category": {label: graded_group(nr, br, [qid for qid in ids if category(qid) == label])
            for label in sorted({category(qid) for qid in ids})},
        "by_qid_prefix": {prefix: graded_group(nr, br, [qid for qid in ids if qid.rsplit("-", 1)[0] == prefix])
            for prefix in sorted({qid.rsplit("-", 1)[0] for qid in ids})},
        "quality_training_runs": 1, "new_dev_judge_runs": 1, "public_runs": 0, "default_changed": False,
        "input_sha256": {p: sha256(ROOT / p) for p in ["results/h5_judge_plan.json", "results/h5_base5_graded.json", "results/h5_base5_judge_timing.json"]}}
    write_new("h5_quality_final.json", report)
    print(json.dumps({"quality_gate": checks, "candidate": after, "baseline": before, "status": report["status"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "screen", "prepare-judge", "judge", "final"], required=True)
    {"prepare": prepare, "screen": screen, "prepare-judge": prepare_judge, "judge": judge, "final": final}[parser.parse_args().stage]()
