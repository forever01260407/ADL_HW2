#!/usr/bin/env python3
"""Freeze H3 once, after tests and before any base quality experiment."""
import json
from datetime import datetime, timezone
from pathlib import Path

from analyze_results import sha256
from audit_w08 import verify_training
from run_rrf_ablation import verify_hashes
from train import ROOT


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    output = ROOT / "results/h3_experiment_plan.json"
    if output.exists():
        raise ValueError("H3 is already frozen")
    verify_training()
    verify_hashes(read("results/h2_rrf_final.json")["artifact_sha256"])
    verify_hashes(read("results/h2b_completion.json")["artifacts_sha256"])
    experiment = read("code/script/configs/h3_experiment.json")
    lock = read(experiment["model_lock"])
    download = read(lock["download_manifest"])
    base_hashes = {f'{lock["path"]}/{name}': row["sha256"] for name, row in download["files"].items()}
    verify_hashes(base_hashes)
    original = read("results/embedding_download.json")
    same_tokenizer = all(download["files"][name]["sha256"] == original["files"][name]["sha256"]
                         for name in ["tokenizer.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"])
    if not same_tokenizer:
        raise ValueError("tokenizers differ; encoder substitution would change tokenization")
    paths = ["code/script/train.py", "code/script/train_base.py", "code/script/base_selection.py",
        "code/script/prepare_base_experiment.py", "code/script/tests/test_base_training.py",
        "code/script/configs/h3_experiment.json", experiment["reference_training_config"], experiment["model_lock"],
        lock["download_manifest"], "results/h3_base_remote_metadata.json", "results/h2b_completion.json",
        "results/h2_rrf_summary.json", "results/h2b_rrf_screening.json", "results/h2b_rrf_final.json",
        "artifacts/training/w08_epoch1/training_manifest.json", "artifacts/training/w08_epoch1/batches.json",
        "data/derived/train.jsonl", "data/derived/dev.jsonl", "data/train/products.jsonl",
        "code/script/configs/split_manifest.json", "retrieval_utils.py", "retrieval.py", "evaluate.py", "eval_utils.py"]
    report = {"experiment": experiment, "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_hyperparameters": read(experiment["reference_training_config"]),
        "base_model": lock, "tokenizer_assets_identical_to_small": same_tokenizer,
        "train_count": 3864, "dev_count": 966, "candidate_products": 50000,
        "one_quality_training_run": True, "smoke_is_not_quality_candidate": True,
        "gold_gate_and_adoption_gate_frozen": True,
        "input_sha256": {**{name: sha256(ROOT / name) for name in paths}, **base_hashes}}
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"plan": output.name, "tokenizer_assets_identical_to_small": same_tokenizer,
                      "base_revision": lock["revision"], "one_quality_training_run": True}))


if __name__ == "__main__":
    main()
