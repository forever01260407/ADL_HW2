#!/usr/bin/env python3
"""Promote the one H5 candidate only after quality and formal dev validation.

No Public run, packaging or model/parameter selection is performed here.
On a formal validation failure the previous inference source is restored.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

from analyze_results import sha256
from audit_w06 import audit_graded
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission
from evaluate_hard_negative import evaluation_frozen
from hard_negative_data import ROOT, RESULTS, read, verify_hashes, write_new
from smoke_retrieval import PROTECTED
from validate_release import check_budget


def main():
    plan, manifest, _ = evaluation_frozen()
    quality = read("results/h5_quality_final.json")
    if not quality.get("quality_gate", {}).get("passed"):
        raise ValueError("H5 did not pass the fixed official dev quality gate")
    verify_hashes(quality["input_sha256"])
    before = (ROOT / "retrieval_utils.py").read_bytes()
    old_path = plan["configuration"]["source_checkpoint"].encode()
    new_path = manifest["checkpoint"].encode()
    if before.count(old_path) != 1:
        raise ValueError("source checkpoint occurrence ambiguous")
    after = before.replace(old_path, new_path)
    for marker, index in [(b"# ==================== STUDENT CODE START ====================", 0),
            (b"# ===================== STUDENT CODE END =====================", 1)]:
        if before.split(marker)[index] != after.split(marker)[index]:
            raise ValueError("change outside student block")
    archive = ROOT / "code/script/snapshots/h4_default/retrieval_utils.py"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if archive.read_bytes() != before:
            raise ValueError("existing H4 snapshot differs")
    else:
        with archive.open("xb") as handle:
            handle.write(before)
    import hashlib
    expected_after = hashlib.sha256(after).hexdigest()
    write_new("h5_promotion_plan.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": manifest["checkpoint"], "quality_sha256": sha256(RESULTS / "h5_quality_final.json"),
        "source_before_sha256": sha256(archive), "source_after_sha256": expected_after,
        "public_runs": 0, "decision_basis": "train/dev only, fixed 5:1 and one trained candidate",
        "expected_top50_sha256": sha256(RESULTS / "h5_base5_top50.json"),
        "input_sha256": {p: sha256(ROOT / p) for p in ["code/script/adopt_hard_negative.py",
            "results/h5_quality_final.json", "results/h5_base5_top50.json", "results/h5_base5_top10.json"]}})
    try:
        (ROOT / "retrieval_utils.py").write_bytes(after)
        with (RESULTS / "h5_tests_selected.log").open("x", encoding="utf-8") as handle:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "code/script/tests", "-v"],
                cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
        log = (RESULTS / "h5_tests_selected.log").read_text()
        if "Ran 84 tests" not in log or not log.rstrip().endswith("OK"):
            raise ValueError("selected-default tests did not pass")
        subprocess.run([sys.executable, "code/script/run_validated.py", "--budget-gib", "8",
            "--output", "results/h5_default_dev_timing.json", "--", "retrieval.py", "--dialogues",
            "data/derived/dev.jsonl", "--products", "data/train/products.jsonl", "--top-k", "50",
            "--output", "results/h5_default_dev_top50.json"], cwd=ROOT, check=True)
        timing = read("results/h5_default_dev_timing.json")
        if timing["exit_code"] or timing["resources"]["exit_code"]:
            raise ValueError("formal process failed")
        verify_hashes(timing["input_sha256"])
        verify_hashes(PROTECTED)
        budget = check_budget(timing)
        queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
        products = load_products(ROOT / "data/train/products.jsonl")
        actual = load_submission(RESULTS / "h5_default_dev_top50.json", queries, set(products), top_k=50)
        expected = load_submission(RESULTS / "h5_base5_top50.json", queries, set(products), top_k=50)
        if list(actual) != [q["qid"] for q in queries] or any(actual[qid] != expected[qid] for qid in actual):
            raise ValueError("formal default does not reproduce the evaluated H5 ranking")
        audit_graded(read("results/h5_base5_graded.json"), queries, {qid: pids[:10] for qid, pids in actual.items()})
        if sha256(ROOT / "retrieval_utils.py") != expected_after:
            raise ValueError("formal source changed since promotion")
        paths = ["retrieval_utils.py", "code/script/snapshots/h4_default/retrieval_utils.py",
            "code/script/adopt_hard_negative.py", "results/h5_tests_selected.log", "results/h5_promotion_plan.json",
            "results/h5_default_dev_top50.json", "results/h5_default_dev_timing.json", "results/h5_quality_final.json"]
        write_new("h5_adoption.json", {"passed": True, "default_changed": True,
            "checkpoint": manifest["checkpoint"], "dense_bm25_weights": [5, 1], "public_runs": 0,
            "dev_queries": 966, "changed_top50_count": 0, "budget_checks": budget,
            "official_hardware_validation": False, "cold_seconds": timing["cold_process_wall_seconds"],
            "resources": timing["resources"], "input_sha256": {p: sha256(ROOT / p) for p in paths}})
        print(json.dumps({"adopted": True, "default_top50_matches_all966": True,
            "cold_seconds": timing["cold_process_wall_seconds"], "public_runs": 0}, indent=2))
    except BaseException:
        (ROOT / "retrieval_utils.py").write_bytes(before)
        raise


if __name__ == "__main__":
    main()
