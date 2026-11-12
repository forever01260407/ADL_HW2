#!/usr/bin/env python3
"""H5 train-only mining, conservative fixed-teacher filtering and provenance."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from analyze_results import sha256
from prepare_splits import normalize
from train import ROOT, load_data
from smoke_retrieval import PROTECTED

RESULTS = ROOT / "results"
CONFIG = "code/script/configs/h5_hard_negative.json"
PLAN = "results/h5_training_plan.json"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_new(name, data):
    with (RESULTS / name).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def verify_hashes(values):
    for name, expected in values.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"frozen H5 input changed: {name}")


def prepare():
    rows, products, data_hashes = load_data()
    config = read(CONFIG)
    manifest = read("artifacts/training/h3_base_epoch1/training_manifest.json")
    checkpoint_hashes = {f'{config["source_checkpoint"]}/{p}': h
        for p, h in manifest["checkpoint_files"].items()}
    verify_hashes(checkpoint_hashes)
    verify_hashes(PROTECTED)
    files = [CONFIG, "code/script/hard_negative_data.py", "code/script/hard_negative_loss.py",
        "code/script/train_hard_negative.py", "code/script/tests/test_hard_negatives.py",
        "code/script/train.py", "code/script/prepare_splits.py", "code/script/run_validated.py",
        "retrieval_utils.py", "code/script/configs/models.lock.json", "results/generator_download.json",
        "artifacts/training/h3_base_epoch1/training_manifest.json", "results/h3_base_dense_gold.json",
        "results/h3_base5_gold.json", "results/h3_base5_graded.json"]
    write_new("h5_training_plan.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "train_queries": len(rows), "candidate_products": len(products),
        "negative_source": "train3864 dialogues/targets and full train50000 product pool only",
        "dev_access": "original load_data verifies split integrity only; no dev-derived training examples, mining or gradients",
        "public_used": False, "maximum_quality_training_runs": 1, "maximum_dev_judge_runs": 1,
        "attribution_limit": "continued training plus curated negatives as one intervention; no matched no-negative continuation control",
        "input_sha256": {**data_hashes, **PROTECTED, **{p: sha256(ROOT / p) for p in files}},
        "checkpoint_sha256": checkpoint_hashes})
    print(json.dumps({"plan_frozen": True, "train_queries": len(rows), "products": len(products)}))


def frozen():
    plan = read(PLAN)
    verify_hashes(plan["input_sha256"])
    verify_hashes(plan["checkpoint_sha256"])
    return plan


def choose_candidates(rows, products, ranked, starts=(5, 15), maximum=50):
    if list(ranked) != [r["qid"] for r in rows]:
        raise ValueError("mining rankings must exactly cover the train query order")
    result = []
    for row in rows:
        gold = row["gold_pid"]
        used, texts = {gold}, {normalize(products[gold])}
        for start in starts:
            for rank, pid in enumerate(ranked[row["qid"]], 1):
                if start <= rank <= maximum and pid not in used and normalize(products[pid]) not in texts:
                    result.append({"qid": row["qid"], "gold_pid": gold, "candidate_pid": pid, "dense_rank": rank})
                    used.add(pid)
                    texts.add(normalize(products[pid]))
                    break
    return result


def strict_negative(judgment):
    try:
        parsed = json.loads(judgment["raw_output"])
    except (ValueError, TypeError):
        return False
    return (judgment["parse_ok"] is True and judgment["relevant"] is False
        and isinstance(parsed, dict) and set(parsed) == {"relevant"} and parsed["relevant"] is False)


def select_verified(rows, products, candidates, judgments):
    if len(candidates) != len(judgments):
        raise ValueError("teacher candidate coverage differs")
    by_qid = {r["qid"]: r for r in rows}
    selected = {}
    for candidate, judgment in zip(candidates, judgments):
        qid, pid = candidate["qid"], candidate["candidate_pid"]
        if qid not in by_qid or any(judgment[k] != candidate[k] for k in ["qid", "gold_pid", "candidate_pid"]):
            raise ValueError("teacher/source alignment differs")
        gold = by_qid[qid]["gold_pid"]
        if candidate["gold_pid"] != gold or pid == gold or normalize(products[pid]) == normalize(products[gold]):
            raise ValueError("positive or duplicate product cannot be a mined negative")
        if strict_negative(judgment) and qid not in selected:
            selected[qid] = {**candidate, "raw_output": judgment["raw_output"]}
    return selected


def batch_negatives(rows, products, selected):
    positive_pids = {r["gold_pid"] for r in rows}
    positive_texts = {normalize(products[p]) for p in positive_pids}
    indices, texts, masked = [], [], 0
    for i, row in enumerate(rows):
        negative = selected.get(row["qid"])
        if negative is None:
            continue
        pid = negative["candidate_pid"]
        if pid in positive_pids or normalize(products[pid]) in positive_texts:
            masked += 1  # already in the positive stream: do not double-count it
            continue
        indices.append(i)
        texts.append(products[pid])
    return indices, texts, masked


def candidates():
    plan = frozen()
    rows, products, _ = load_data()
    from evaluate import load_submission
    timing = read("results/h5_train_dense_timing.json")
    verify_hashes(timing["input_sha256"])
    if timing["exit_code"] or timing["environment"]["HW2_RETRIEVER"] != "dense":
        raise ValueError("train Dense mining failed or wrong route")
    if timing["environment"]["HW2_MODEL_PATH"] != plan["configuration"]["source_checkpoint"]:
        raise ValueError("mining used the wrong checkpoint")
    ranked = load_submission(RESULTS / "h5_train_dense_top100.json", rows, set(products), top_k=100)
    config = plan["configuration"]
    selected = choose_candidates(rows, products, ranked, config["candidate_start_ranks"], config["candidate_max_rank"])
    if len(selected) != 2 * len(rows):
        raise ValueError("expected exactly two eligible candidates per train query")
    write_new("h5_negative_candidates.json", {"train_queries": len(rows), "candidate_pairs": len(selected),
        "records": selected, "public_used": False, "dev_used_for_mining": False,
        "input_sha256": {p: sha256(ROOT / p) for p in [PLAN, "results/h5_train_dense_top100.json",
            "results/h5_train_dense_timing.json"]}})
    print(json.dumps({"train_candidate_pairs": len(selected), "dev_or_public_pairs": 0}))


def filter_candidates():
    plan = frozen()
    data = read("results/h5_negative_candidates.json")
    verify_hashes(data["input_sha256"])
    rows, products, _ = load_data()
    by_qid = {r["qid"]: r for r in rows}
    config = plan["configuration"]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4")
    import torch
    from evaluate import JudgeItem, LLMJudge
    model = read("code/script/configs/models.lock.json")["generator"]
    verify_hashes({f'{model["path"]}/{p}': row["sha256"] for p, row in read("results/generator_download.json")["files"].items()})
    for name in ["h5_negative_teacher.jsonl", "h5_verified_negatives.json"]:
        if (RESULTS / name).exists():
            raise ValueError("preserve the original teacher attempt; refusing to rerun")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(8 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    judge = LLMJudge(model["path"], "cuda", max_new_tokens=config["teacher_max_new_tokens"])
    decisions, max_tokens = [], 0
    records = data["records"]
    with (RESULTS / "h5_negative_teacher.jsonl").open("x", encoding="utf-8", buffering=1) as handle:
        for offset in range(0, len(records), config["teacher_batch_size"]):
            part = records[offset:offset + config["teacher_batch_size"]]
            items = [JudgeItem(c["qid"], c["dense_rank"], c["gold_pid"], c["candidate_pid"],
                by_qid[c["qid"]]["query"], products[c["gold_pid"]], products[c["candidate_pid"]]) for c in part]
            max_tokens = max(max_tokens, max(len(judge.tokenizer(judge._render(i))["input_ids"]) for i in items))
            batch = judge.judge_batch(items)
            for row in batch:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            decisions.extend(batch)
            if offset == 0 or len(decisions) % 400 == 0 or len(decisions) == len(records):
                print(json.dumps({"teacher_pairs_completed": len(decisions), "total": len(records),
                    "strict_negative_pairs": sum(strict_negative(r) for r in decisions)}), flush=True)
    selected = select_verified(rows, products, records, decisions)
    if len(selected) < config["minimum_verified_negative_fraction"] * len(rows):
        raise ValueError("insufficient conservatively verified negative coverage; no training authorized")
    report = {"train_queries": len(rows), "teacher_pairs": len(decisions), "verified_negative_queries": len(selected),
        "without_verified_negative_queries": len(rows) - len(selected), "negatives": selected,
        "teacher": model, "max_input_tokens": max_tokens, "truncation": False,
        "parse_failures": sum(not r["parse_ok"] for r in decisions),
        "teacher_strict_negative_pairs": sum(strict_negative(r) for r in decisions),
        "teacher_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "label_limit": "fixed Qwen pseudo-negative; not guaranteed human-verified irrelevant",
        "public_used": False, "dev_used_for_mining": False,
        "input_sha256": {p: sha256(ROOT / p) for p in [PLAN, "results/h5_negative_candidates.json", "results/h5_negative_teacher.jsonl"]}}
    write_new("h5_verified_negatives.json", report)
    print(json.dumps({k: report[k] for k in ["verified_negative_queries", "without_verified_negative_queries",
        "parse_failures", "teacher_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "candidates", "filter"], required=True)
    {"prepare": prepare, "candidates": candidates, "filter": filter_candidates}[parser.parse_args().stage]()
