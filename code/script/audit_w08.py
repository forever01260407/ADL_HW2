#!/usr/bin/env python3
"""Audit the single W08 training run, Gold gate and optional one-shot Judge."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
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
from run_bm25_representation import gate, paired_changes
from smoke_retrieval import PROTECTED
from train import load_data, make_batches

RESULTS = ROOT / "results"
TRAINING = ROOT / "artifacts/training/w08_epoch1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_file(path, expected):
    if sha256(ROOT / path) == expected:
        return
    if path == "retrieval_utils.py":
        old = ROOT / "code/script/snapshots/h1u/retrieval_utils.py"
        if old.is_file() and sha256(old) == expected:
            return
    raise ValueError(f"changed experiment artifact/source: {path}")


def verify_training():
    manifest = read(TRAINING / "training_manifest.json")
    plan = read(RESULTS / "w08_experiment_plan.json")
    if manifest["status"] != "completed" or manifest["mode"] != "train":
        raise ValueError("full quality training did not complete")
    for path, expected in {**manifest["input_sha256"], **plan["baseline_sha256"], **PROTECTED}.items():
        verify_file(path, expected)
    if sha256(ROOT / "code/script/configs/w08_train.json") != plan["config_sha256"]:
        raise ValueError("configuration changed after plan freeze")
    if manifest["config"] != read(ROOT / "code/script/configs/w08_train.json"):
        raise ValueError("training manifest configuration mismatch")
    checkpoint = ROOT / manifest["checkpoint"]
    for name, expected in manifest["checkpoint_files"].items():
        if sha256(checkpoint / name) != expected:
            raise ValueError(f"checkpoint changed: {name}")
    train, products, _ = load_data()
    batches = make_batches(train, products, manifest["config"]["batch_size"], manifest["config"]["seed"])
    expected_batches = [[train[i]["qid"] for i in b] for b in batches]
    if read(TRAINING / "batches.json") != expected_batches:
        raise ValueError("actual training batch plan differs from frozen sampler")
    if manifest["gradient_qids"] != sorted(row["qid"] for row in train):
        raise ValueError("training gradient qids do not cover exactly the training partition")
    if sha256(TRAINING / "batches.json") != manifest["batches_sha256"]:
        raise ValueError("batch metadata changed")
    if sha256(TRAINING / "loss.jsonl") != manifest["loss_sha256"]:
        raise ValueError("loss log changed")
    losses = [json.loads(line) for line in (TRAINING / "loss.jsonl").read_text().splitlines()]
    if len(losses) != len(batches) or len(losses) != manifest["optimizer_steps"]:
        raise ValueError("training step count mismatch")
    for step, (row, batch) in enumerate(zip(losses, batches), 1):
        if row["step"] != step or row["batch_size"] != len(batch) or row["in_batch_negatives_per_query"] != len(batch) - 1:
            raise ValueError("training batch/loss step metadata mismatch")
        if not all(math.isfinite(row[k]) for k in ["loss", "grad_norm_before_clip", "learning_rate"]):
            raise ValueError("non-finite training metadata")
        if row["grad_norm_before_clip"] <= 0:
            raise ValueError("zero training gradient")
    if manifest["tracked_parameter_max_abs_change"] <= 0 or manifest["offline_reload_max_abs_difference"] > 1e-5:
        raise ValueError("weight update/reload verification failed")
    if manifest["dev_used_for_gradient"] or manifest["public_used"]:
        raise ValueError("evaluation supervision was used for training")
    old = (ROOT / "code/script/snapshots/w06/retrieval_utils.py").read_text()
    current = (ROOT / "retrieval_utils.py").read_text()
    for marker, side in [("# ==================== STUDENT CODE START ====================", 0),
                         ("# ===================== STUDENT CODE END =====================", 1)]:
        if old.split(marker)[side] != current.split(marker)[side]:
            raise ValueError("inference changed outside student block")
    return manifest, plan, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screen", "prepare-judge", "final"], required=True)
    args = parser.parse_args()
    training, plan, losses = verify_training()
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    load = lambda name, k: load_submission(RESULTS / name, queries, set(products), top_k=k)
    ranking = load("w08_epoch1_dev_top50.json", 50)
    if list(ranking) != [q["qid"] for q in queries]:
        raise ValueError("dev order changed")
    timing = audit_timing("w08_epoch1_dev_timing.json")
    settings = training["config"]["retrieval"]
    for key, expected in settings.items():
        if timing["environment"][key] != expected:
            raise ValueError(f"retrieval configuration changed: {key}")
    if timing["environment"]["HW2_MODEL_PATH"] != training["checkpoint"]:
        raise ValueError("retrieval did not use the trained checkpoint")
    baseline = read(RESULTS / "h1u_baseline_gold.json")
    screen_path = RESULTS / "w08_screening.json"
    if args.stage == "screen":
        if screen_path.exists():
            raise ValueError("screening already saved; do not replace experiment history")
        save_submission(RESULTS / "w08_epoch1_dev_top10.json",
            [{"qid": qid, "ranked_pids": pids[:10]} for qid, pids in ranking.items()], top_k=10)
        subprocess.run([sys.executable, "code/script/analyze_results.py", "--retrieval-results",
            "results/w08_epoch1_dev_top50.json", "--dialogues", "data/derived/dev.jsonl", "--compare-top10",
            "results/w08_epoch1_dev_top10.json", "--output", "results/w08_epoch1_dev_gold.json"],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        gold = read(RESULTS / "w08_epoch1_dev_gold.json")
        report = {"experiment": plan["experiment"], "baseline": baseline["metrics"], "candidate": gold["metrics"],
            "beauty_baseline": baseline["by_qid_prefix"]["beauty-train"],
            "beauty_candidate": gold["by_qid_prefix"]["beauty-train"], "by_qid_prefix": gold["by_qid_prefix"],
            "eligible": gate(gold, baseline), "gate": plan["gold_gate"],
            "paired_overall": paired_changes(gold, baseline), "paired_beauty": paired_changes(gold, baseline, True),
            "training_steps": len(losses), "retrieval_timing": timing,
            "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                [TRAINING / "training_manifest.json", RESULTS / "w08_epoch1_dev_top50.json",
                 RESULTS / "w08_epoch1_dev_top10.json", RESULTS / "w08_epoch1_dev_gold.json", screen_source()]}}
        write_new(screen_path, report)
        print(json.dumps({k: report[k] for k in ["baseline", "candidate", "beauty_baseline", "beauty_candidate", "eligible"]}, indent=2))
        return
    screening = read(screen_path)
    for path, expected in screening["input_sha256"].items():
        verify_file(path, expected)
    gold = read(RESULTS / "w08_epoch1_dev_gold.json")
    eligible = gate(gold, baseline)
    if eligible != screening["eligible"]:
        raise ValueError("saved Gold gate result differs from raw diagnostic")
    judge_plan_path = RESULTS / "w08_judge_plan.json"
    if args.stage == "prepare-judge":
        if not eligible:
            raise ValueError("Gold gate failed; do not run official Judge")
        if judge_plan_path.exists() or (RESULTS / "w08_epoch1_dev_judge_timing.json").exists():
            raise ValueError("Judge already authorized/completed; single-run limit")
        write_new(judge_plan_path, {"maximum_official_judge_runs": 1, "gold_gate_verified": True,
            "top10_sha256": sha256(RESULTS / "w08_epoch1_dev_top10.json"),
            "checkpoint_manifest_sha256": sha256(TRAINING / "training_manifest.json"),
            "generator_lock": read(ROOT / "code/script/configs/models.lock.json")["generator"],
            "batch_size": 8, "query_count": 966})
        print("Gold gate passed; one original Judge run authorized")
        return
    report = {"experiment": plan["experiment"], "gold_gate_passed": eligible, "adopt": False,
        "official_judge_runs": 0, "default_action": "retain_h1", "screening": screening,
        "training": {k: training[k] for k in ["optimizer_steps", "training_wall_seconds", "wall_seconds",
            "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "offline_reload_max_abs_difference",
            "tracked_parameter_max_abs_change"]}, "official_hardware_validation": False}
    if eligible:
        judge_plan = read(judge_plan_path)
        if judge_plan["top10_sha256"] != sha256(RESULTS / "w08_epoch1_dev_top10.json"):
            raise ValueError("Judge ranking changed since authorization")
        graded = read(RESULTS / "w08_epoch1_dev_graded.json")
        ranked10 = load("w08_epoch1_dev_top10.json", 10)
        current = audit_graded(graded, queries, ranked10)
        old = read(RESULTS / "h1_hybrid_dev_graded.json")
        audit_graded(old, queries, load("h1_hybrid_dev_top10.json", 10))
        judge_timing = audit_timing("w08_epoch1_dev_judge_timing.json")
        for flag, expected in [("--batch-size", "8"), ("--top-k", "10")]:
            command = read(RESULTS / "w08_epoch1_dev_judge_timing.json")["command"]
            if command[command.index(flag) + 1] != expected:
                raise ValueError("official Judge flags changed")
        keys = ["graded_recall@10", "graded_mrr@10"]
        deltas = {key: graded["metrics"][key] - old["metrics"][key] for key in keys}
        old_rows = {r["qid"]: r for r in old["records"]}
        beauty = [r for r in graded["records"] if r["qid"].rsplit("-", 1)[0] == "beauty-train"]
        report.update({"official_judge_runs": 1, "baseline_graded": old["metrics"], "candidate_graded": current,
            "graded_differences": deltas, "judge_timing": judge_timing,
            "adopt": all(d >= 0 for d in deltas.values()) and any(d > 0 for d in deltas.values()),
            "paired_graded": {key: {"improved": sum(r[key] > old_rows[r["qid"]][key] for r in graded["records"]),
                "worsened": sum(r[key] < old_rows[r["qid"]][key] for r in graded["records"])} for key in keys},
            "beauty_graded": {key: sum(r[key] for r in beauty) / len(beauty) for key in keys},
            "beauty_baseline_graded": {key: sum(old_rows[r["qid"]][key] for r in beauty) / len(beauty) for key in keys}})
        report["default_action"] = "promotion_pending" if report["adopt"] else "retain_h1"
    elif judge_plan_path.exists() or (RESULTS / "w08_epoch1_dev_judge_timing.json").exists():
        raise ValueError("Judge ran despite failed Gold gate")
    write_new(RESULTS / "w08_summary.json", report)
    print(json.dumps({k: report[k] for k in ["gold_gate_passed", "official_judge_runs", "adopt", "default_action"]}, indent=2))


def screen_source():
    return Path(__file__).resolve()


if __name__ == "__main__":
    main()
