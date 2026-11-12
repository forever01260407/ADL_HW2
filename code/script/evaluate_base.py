#!/usr/bin/env python3
"""H3 full-dev Gold gate and two conditional one-shot official comparisons."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from analyze_results import gold_metrics, sha256
from audit_w06 import audit_graded
from audit_w07 import audit_timing
from base_selection import adoption_gate, gold_gate, paired_bootstrap
from eval_utils import load_eval_queries, load_products
from evaluate import LLMJudge, build_judge_items, load_submission
from extend_rrf_ablation import graded_group, verify_fusion
from retrieval import save_submission
from run_rrf_ablation import category, paired, verify_hashes, weighted_fuse
from train import ROOT

RESULTS = ROOT / "results"
TRAINING = ROOT / "artifacts/training/h3_base_epoch1"
DENSE = "results/h3_base_dense_top100.json"
SMALL5 = "results/h2b_d5_b1_top10.json"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_new(name, value):
    with (RESULTS / name).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def training():
    plan = read("results/h3_experiment_plan.json")
    verify_hashes(plan["input_sha256"])
    manifest = read(TRAINING / "training_manifest.json")
    if (manifest["status"] != "completed" or manifest["mode"] != "train" or
            manifest["train_count"] != 3864 or manifest["optimizer_steps"] != 242 or
            manifest["dev_used_for_gradient"] or manifest["public_used"] or
            manifest["offline_reload_max_abs_difference"] > 1e-5):
        raise ValueError("base training completion/integrity check failed")
    if manifest["config"] != plan["training_hyperparameters"]:
        raise ValueError("training hyperparameters differ from W08")
    verify_hashes({f'{manifest["checkpoint"]}/{name}': expected
                   for name, expected in manifest["checkpoint_files"].items()})
    batches = read(TRAINING / "batches.json")
    if batches != read("artifacts/training/w08_epoch1/batches.json"):
        raise ValueError("base batch qids differ from W08")
    if sha256(TRAINING / "batches.json") != manifest["batches_sha256"] or sha256(TRAINING / "loss.jsonl") != manifest["loss_sha256"]:
        raise ValueError("training batch/loss logs changed")
    losses = [json.loads(line) for line in (TRAINING / "loss.jsonl").read_text().splitlines()]
    if len(losses) != 242 or sum(row["batch_size"] for row in losses) != 3864:
        raise ValueError("training step/data coverage mismatch")
    for i, (row, qids) in enumerate(zip(losses, batches), 1):
        if row["step"] != i or row["batch_size"] != len(qids) or row["grad_norm_before_clip"] <= 0:
            raise ValueError("invalid actual training step")
    return plan, manifest


def prepare():
    plan, manifest = training()
    files = ["code/script/evaluate_base.py", "code/script/base_selection.py", "code/script/analyze_results.py",
        "code/script/extend_rrf_ablation.py", "results/h3_experiment_plan.json",
        "artifacts/training/h3_base_epoch1/training_manifest.json", "results/h2_rrf_summary.json",
        "results/h2b_rrf_screening.json", "results/h2b_rrf_final.json", "results/h2b_completion.json",
        "results/h2_w08_dense_top100.json", "results/h1u_u0_full_lexical_top100.json",
        "results/h1u_u0_full_trace.json", "results/h2_dense_only_gold.json", "results/h2b_d5_b1_gold.json",
        "results/h2b_d5_b1_top50.json", SMALL5, "results/h2b_winner_graded.json",
        "results/w08_epoch1_dev_graded.json", "code/script/configs/models.lock.json", "results/generator_download.json"]
    write_new("h3_evaluation_plan.json", {"training_manifest_sha256": sha256(TRAINING / "training_manifest.json"),
        "policy": plan["experiment"], "quality_training_runs": 1, "maximum_full_dev_judge_runs": 2,
        "input_sha256": {p: sha256(ROOT / p) for p in files}})
    print(json.dumps({"training_verified": True, "steps": manifest["optimizer_steps"], "evaluation_plan_frozen": True}))


def frozen():
    plan, manifest = training()
    evaluation = read("results/h3_evaluation_plan.json")
    verify_hashes(evaluation["input_sha256"])
    return plan, manifest, evaluation


def screen():
    plan, manifest, _ = frozen()
    if (RESULTS / "h3_gold_screening.json").exists():
        raise ValueError("H3 Gold screening already saved")
    timing = audit_timing("h3_base_dense_timing.json")
    env = timing["environment"]
    if env["HW2_MODEL_PATH"] != manifest["checkpoint"] or env["HW2_RETRIEVER"] != "dense" or env["HW2_RERANK"] != "0":
        raise ValueError("Dense ranking did not use the frozen fine-tuned base")
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    if len(queries) != 966 or len(products) != 50000:
        raise ValueError("not complete dev/full candidate pool")
    dense = load_submission(ROOT / DENSE, queries, set(products), top_k=100)
    lexical = load_submission(ROOT / "results/h1u_u0_full_lexical_top100.json", queries, set(products), top_k=100)
    positive = {r["qid"]: r["positive_count"] for r in read("results/h1u_u0_full_trace.json")["results"]}
    qids = [q["qid"] for q in queries]
    if list(dense) != qids or list(lexical) != qids:
        raise ValueError("ranked query order changed")
    configurations = {"base_dense": (1, 0), "base5": (5, 1)}
    golds = {}
    for name, (dw, bw) in configurations.items():
        rows = weighted_fuse(dense, lexical, positive, dw, bw)
        ranked = {r["qid"]: r["ranked_pids"] for r in rows}
        verify_fusion(dense, lexical, positive, ranked, dw, bw)
        prefix = f"results/h3_{name}"
        paths = [f"{prefix}_{suffix}.json" for suffix in ["top50", "top10", "gold"]]
        if any((ROOT / path).exists() for path in paths):
            raise ValueError("refusing to overwrite H3 quality results")
        save_submission(ROOT / paths[0], rows, top_k=50)
        save_submission(ROOT / paths[1], [{"qid": row["qid"], "ranked_pids": row["ranked_pids"][:10]} for row in rows], top_k=10)
        subprocess.run([sys.executable, "code/script/analyze_results.py", "--retrieval-results", paths[0],
            "--dialogues", "data/derived/dev.jsonl", "--products", "data/train/products.jsonl", "--compare-top10",
            paths[1], "--output", paths[2]], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        gold = read(paths[2])
        gold["by_category"] = {}
        for label in sorted({category(qid) for qid in qids}):
            subset = [q for q in queries if category(q["qid"]) == label]
            gold["by_category"][label] = gold_metrics(subset, {q["qid"]: ranked[q["qid"]] for q in subset}, (1, 10, 50))["metrics"]
        golds[name] = gold
    small_dense, small5 = read("results/h2_dense_only_gold.json"), read("results/h2b_d5_b1_gold.json")
    decision = gold_gate(golds["base_dense"]["metrics"], golds["base5"]["metrics"],
                         small_dense["metrics"], small5["metrics"], plan["experiment"]["gold_gate"])
    report = {"kind": "Gold-only dev screening, not official graded", "gold_gate": decision,
        "variants": {name: {key: data[key] for key in ["metrics", "by_category", "by_qid_prefix", "records"]}
                     for name, data in golds.items()},
        "small_dense": small_dense["metrics"], "small5": small5["metrics"],
        "paired_base5_vs_small5": paired(golds["base5"], small5, qids),
        "official_judge_runs": 0, "public_runs": 0, "independent_rankings_verified": 1932,
        "dense_timing": timing, "input_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in
            [RESULTS / "h3_evaluation_plan.json", ROOT / DENSE, RESULTS / "h3_base_dense_timing.json",
             *sorted(RESULTS.glob("h3_base*_top*.json")), *sorted(RESULTS.glob("h3_base*_gold.json"))]}}
    write_new("h3_gold_screening.json", report)
    print(json.dumps({"gold_gate": decision, "small_dense": small_dense["metrics"], "small5": small5["metrics"],
        "base_dense": golds["base_dense"]["metrics"], "base5": golds["base5"]["metrics"]}, indent=2))


def prepare_judge():
    frozen()
    screen = read("results/h3_gold_screening.json")
    verify_hashes(screen["input_sha256"])
    if not screen["gold_gate"]["passed"]:
        raise ValueError("Gold did not pass: no official Judge is authorized")
    if (RESULTS / "h3_judge_plan.json").exists():
        raise ValueError("H3 Judge plan already exists")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoTokenizer
    model = read("code/script/configs/models.lock.json")["generator"]
    hashes = {f'{model["path"]}/{name}': row["sha256"] for name, row in read("results/generator_download.json")["files"].items()}
    verify_hashes(hashes)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    config = AutoConfig.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    variants = {}
    for label, path in [("small5", SMALL5), ("base5", "results/h3_base5_top10.json")]:
        ranked = load_submission(ROOT / path, queries, set(products), top_k=10)
        top50_path = "results/h2b_d5_b1_top50.json" if label == "small5" else "results/h3_base5_top50.json"
        top50 = load_submission(ROOT / top50_path, queries, set(products), top_k=50)
        if any(ranked[qid] != top50[qid][:10] for qid in ranked):
            raise ValueError("top-ten prefix differs")
        items = build_judge_items({q["qid"]: q for q in queries}, ranked, products)
        renderer = SimpleNamespace(tokenizer=tokenizer)
        lengths = [len(tokenizer(LLMJudge._render(renderer, item))["input_ids"]) for item in items]
        if max(lengths) + 12 > config.max_position_embeddings:
            raise ValueError("original untruncated Judge input exceeds context")
        variants[label] = {"ranking_path": path, "ranking_sha256": sha256(ROOT / path),
            "non_gold_comparisons": len(items), "max_input_tokens": max(lengths)}
    write_new("h3_judge_plan.json", {"variants": variants, "model": model, "batch_size": 8,
        "max_new_tokens": 12, "dtype": "bfloat16", "do_sample": False, "truncation": False,
        "maximum_full_dev_judge_runs": 2, "input_sha256": {**hashes,
            "results/h3_gold_screening.json": sha256(RESULTS / "h3_gold_screening.json")}})
    print(json.dumps({"variants": variants, "maximum_full_dev_judge_runs": 2}, indent=2))


def judge(label):
    frozen()
    plan = read("results/h3_judge_plan.json")
    verify_hashes(plan["input_sha256"])
    variant = plan["variants"][label]
    if sha256(ROOT / variant["ranking_path"]) != variant["ranking_sha256"]:
        raise ValueError("frozen ranking changed")
    if any((RESULTS / f"h3_{label}_{suffix}").exists() for suffix in ["graded.json", "judge_timing.json", "judge_timing.log"]):
        raise ValueError("this candidate Judge was already attempted")
    subprocess.run([sys.executable, "code/script/run_measured.py", "--output", f"results/h3_{label}_judge_timing.json",
        "--", "evaluate.py", "--retrieval-results", variant["ranking_path"], "--dialogues", "data/derived/dev.jsonl",
        "--products", "data/train/products.jsonl", "--model", plan["model"]["path"], "--batch-size", "8",
        "--max-new-tokens", "12", "--top-k", "10", "--device", "cuda", "--output",
        f"results/h3_{label}_graded.json"], cwd=ROOT, check=True)


def final():
    plan, manifest, _ = frozen()
    screen = read("results/h3_gold_screening.json")
    verify_hashes(screen["input_sha256"])
    if not screen["gold_gate"]["passed"]:
        write_new("h3_quality_final.json", {"status": "gold_gate_failed", "gold_gate": screen["gold_gate"],
            "official_judge_runs": 0, "default_changed": False, "public_runs": 0})
        return
    judge_plan = read("results/h3_judge_plan.json")
    verify_hashes(judge_plan["input_sha256"])
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    graded, audits, times = {}, {}, {}
    for name, details in judge_plan["variants"].items():
        ranked = load_submission(ROOT / details["ranking_path"], queries, set(products), top_k=10)
        if sha256(ROOT / details["ranking_path"]) != details["ranking_sha256"]:
            raise ValueError("Judge ranking changed")
        graded[name] = read(f"results/h3_{name}_graded.json")
        audits[name] = audit_graded(graded[name], queries, ranked)
        times[name] = audit_timing(f"h3_{name}_judge_timing.json")
        command = read(f"results/h3_{name}_judge_timing.json")["command"]
        for flag, value in [("--retrieval-results", details["ranking_path"]), ("--dialogues", "data/derived/dev.jsonl"),
            ("--products", "data/train/products.jsonl"), ("--batch-size", "8"), ("--max-new-tokens", "12"),
            ("--model", judge_plan["model"]["path"]), ("--top-k", "10"), ("--device", "cuda")]:
            if command[command.index(flag) + 1] != value:
                raise ValueError("Judge command differs from frozen conditions")
        if [r["qid"] for r in graded[name]["records"]] != [q["qid"] for q in queries]:
            raise ValueError("Judge qid order changed")
    graded["small6"] = read("results/h2b_winner_graded.json")
    graded["small1"] = read("results/w08_epoch1_dev_graded.json")
    records = {name: {r["qid"]: r for r in data["records"]} for name, data in graded.items()}
    qids = [q["qid"] for q in queries]
    comparisons = {}
    categories = sorted({category(qid) for qid in qids})
    for name in ["small5", "small6", "small1"]:
        comparisons[name] = {"overall": graded_group(records["base5"], records[name], qids),
            "by_category": {label: graded_group(records["base5"], records[name], [qid for qid in qids if category(qid) == label])
                            for label in categories}}
        comparisons[name]["bootstrap"] = {key: paired_bootstrap([records["base5"][qid][key] - records[name][qid][key]
            for qid in qids]) for key in ["graded_recall@10", "graded_mrr@10"]}
    base_categories = {label: comparisons["small5"]["by_category"][label]["candidate"] for label in categories}
    small_categories = {label: comparisons["small5"]["by_category"][label]["baseline"] for label in categories}
    decision = adoption_gate(audits["base5"]["metrics"], audits["small5"]["metrics"], graded["small6"]["metrics"],
        base_categories, small_categories, plan["experiment"]["adoption_gate"])
    report = {"status": "graded_quality_complete_resource_and_formal_checks_pending" if decision["passed"] else "graded_adoption_gate_failed",
        "gold_gate": screen["gold_gate"], "quality_adoption_gate": decision, "official_judge_runs": 2,
        "graded": audits, "comparisons": comparisons, "judge_timings": times,
        "default_changed": False, "public_runs": 0, "w11_started": False, "official_hardware_validation": False,
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in [RESULTS / "h3_judge_plan.json",
            RESULTS / "h3_gold_screening.json", RESULTS / "h3_base5_graded.json", RESULTS / "h3_small5_graded.json",
            Path(__file__).resolve()]}}
    write_new("h3_quality_final.json", report)
    print(json.dumps({"status": report["status"], "quality_adoption_gate": decision,
        "small5": audits["small5"]["metrics"], "base5": audits["base5"]["metrics"],
        "small6": graded["small6"]["metrics"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "screen", "prepare-judge", "judge", "final"], required=True)
    parser.add_argument("--candidate", choices=["small5", "base5"])
    args = parser.parse_args()
    if args.stage == "judge":
        if args.candidate is None:
            parser.error("judge requires --candidate")
        judge(args.candidate)
    else:
        {"prepare": prepare, "screen": screen, "prepare-judge": prepare_judge, "final": final}[args.stage]()
