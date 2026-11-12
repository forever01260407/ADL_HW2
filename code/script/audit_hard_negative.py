#!/usr/bin/env python3
"""Final H5 provenance/data/metric audit; never trains or calls a Judge."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from hard_negative_data import ROOT, RESULTS, PLAN, read, write_new, select_verified
from analyze_results import gold_metrics, sha256
from audit_w06 import audit_graded
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission, parse_judgment
from smoke_retrieval import PROTECTED


def verify(mapping):
    for name, expected in mapping.items():
        path = ROOT / name
        if name == "retrieval_utils.py" and sha256(path) != expected:
            path = ROOT / "code/script/snapshots/h4_default/retrieval_utils.py"
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source/artifact changed: {name}")


def main():
    plan = read(PLAN)
    verify(plan["input_sha256"])
    verify(plan["checkpoint_sha256"])
    verify(PROTECTED)
    config = plan["configuration"]
    train = load_eval_queries(ROOT / "data/derived/train.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    selected_data = read("results/h5_verified_negatives.json")
    verify(selected_data["input_sha256"])
    teacher = [json.loads(line) for line in (RESULTS / "h5_negative_teacher.jsonl").read_text().splitlines()]
    for row in teacher:
        relevant, parsed = parse_judgment(row["raw_output"])
        if relevant != row["relevant"] or parsed != row["parse_ok"]:
            raise ValueError("teacher raw judgment differs from recorded decision")
    selected = select_verified(train, products, read("results/h5_negative_candidates.json")["records"], teacher)
    if selected != selected_data["negatives"] or len(teacher) != 7728:
        raise ValueError("teacher selection changed")
    encoder_audit = read("results/h5_encoder_input_audit.json")
    verify(encoder_audit["input_sha256"])
    if encoder_audit["identical_encoder_input_pairs"] != 0:
        raise ValueError("unlearnable duplicate encoder inputs")
    manifest = read(f'{config["output"]}/training_manifest.json')
    verify(manifest["input_sha256"])
    checkpoint = {f'{manifest["checkpoint"]}/{p}': h for p, h in manifest["checkpoint_files"].items()}
    verify(checkpoint)
    batches = read(f'{config["output"]}/batches.json')
    expected_ids = [r["qid"] for r in train]
    if (batches != read("artifacts/training/h3_base_epoch1/batches.json")
            or Counter(qid for batch in batches for qid in batch) != Counter(expected_ids)
            or manifest["gradient_qids"] != sorted(expected_ids)):
        raise ValueError("actual gradient batches differ from complete Train only")
    losses = [json.loads(line) for line in (ROOT / config["output"] / "loss.jsonl").read_text().splitlines()]
    if (len(losses) != 242 or sum(r["batch_size"] for r in losses) != 3864
            or sum(r["own_verified_negatives"] for r in losses) != manifest["trained_own_negative_exposures"]
            or sum(r["duplicate_stream_negatives_masked"] for r in losses) != manifest["duplicate_stream_negatives_masked"]
            or manifest["status"] != "completed" or manifest["mode"] != "train"
            or manifest["tracked_parameter_max_abs_change"] <= 0 or manifest["offline_reload_max_abs_difference"] > 1e-5):
        raise ValueError("actual training coverage/update/reload audit failed")
    verify({f'{config["output"]}/loss.jsonl': manifest["loss_sha256"],
        f'{config["output"]}/batches.json': manifest["batches_sha256"]})
    if max(manifest["peak_cuda_allocated_bytes"], manifest["peak_cuda_reserved_bytes"]) > 8 * 2**30:
        raise ValueError("training allocator budget exceeded")
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    screening = read("results/h5_gold_screening.json")
    verify(screening["input_sha256"])
    for tag in ["dense", "base5"]:
        ranked = load_submission(RESULTS / f"h5_{tag}_top50.json", queries, set(products), top_k=50)
        actual = gold_metrics(queries, ranked, (1, 10, 50))
        if actual != {k: screening["variants"][tag][k] for k in ["metrics", "records"]}:
            raise ValueError("independent dev Gold recomputation differs")
    quality = read("results/h5_quality_final.json")
    if "input_sha256" in quality:
        verify(quality["input_sha256"])
    if quality.get("new_dev_judge_runs"):
        ranked = load_submission(RESULTS / "h5_base5_top10.json", queries, set(products), top_k=10)
        audited = audit_graded(read("results/h5_base5_graded.json"), queries, ranked)
        if audited != quality["candidate"]:
            raise ValueError("final graded audit differs")
    adopted = read("results/h5_adoption.json") if (RESULTS / "h5_adoption.json").exists() else None
    if quality.get("quality_gate", {}).get("passed") and (not adopted or not adopted["passed"]):
        raise ValueError("quality eligible but formal adoption work remains unfinished")
    if adopted:
        verify(adopted["input_sha256"])
    history = read("results/h4_completion.json")
    verify(history["artifact_sha256"])
    verify(history["checkpoint_sha256"])
    paths = sorted(RESULTS.glob("h5_*.json")) + sorted(RESULTS.glob("h5_*.jsonl")) + sorted(RESULTS.glob("h5_*.log"))
    paths += [ROOT / p for p in ["H5_RESULTS.md", "retrieval_utils.py", "code/script/configs/h5_hard_negative.json",
        "code/script/hard_negative_data.py", "code/script/hard_negative_loss.py", "code/script/train_hard_negative.py",
        "code/script/tests/test_hard_negatives.py", "code/script/adopt_hard_negative.py", "code/script/evaluate_hard_negative.py",
        "code/script/audit_hard_negative.py", f'{config["output"]}/training_manifest.json',
        f'{config["output"]}/batches.json', f'{config["output"]}/loss.jsonl',
        f'{config["smoke_output"]}/training_manifest.json']]
    archive = ROOT / "code/script/snapshots/h4_default/retrieval_utils.py"
    if archive.exists():
        paths.append(archive)
    write_new("h5_completion.json", {"status": "experiment_complete_h5_adopted_dev_only" if adopted else quality["status"],
        "finished_utc": datetime.now(timezone.utc).isoformat(), "quality_training_runs": 1, "engineering_smoke_runs": 1,
        "train_qids": 3864, "steps": 242, "teacher_pairs": len(teacher), "verified_negative_queries": len(selected),
        "trained_negative_exposures": manifest["trained_own_negative_exposures"], "training_qid_boundary_verified": True,
        "new_dev_judge_runs": quality["new_dev_judge_runs"], "public_runs": 0, "default_changed": bool(adopted),
        "training_allocator_budget_gib": 8, "official_hardware_validation": False,
        "source_h3_fp32_training_under_8gib_verified": False, "w11_started": False,
        "gold_gate": screening["gold_gate"], "quality": quality,
        "checkpoint_sha256": checkpoint, "historical_h4_seal_verified": True,
        "artifact_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
        "mutable_worklog_references": ["PLAN.md", "DECISIONS.md", "PROGRESS.md"]})
    print(json.dumps({"audit": "passed", "status": "H5 adopted" if adopted else quality["status"], "artifact_count": len(paths),
        "all3864_train_gradient_qids_once": True, "teacher_raw_outputs_rechecked": len(teacher), "public_runs": 0}, indent=2))


if __name__ == "__main__":
    main()
