#!/usr/bin/env python3
"""Check pinned Qwen and original judge at batch 8 without changing its rules."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate import LLMJudge, build_judge_items, load_submission
from eval_utils import load_eval_queries, load_products
from analyze_results import sha256
from smoke_retrieval import PROTECTED


def length_summary(lengths, limit):
    import numpy as np
    return {"count": len(lengths), "min": min(lengths), "max": max(lengths),
            "p50": float(np.quantile(lengths, .5)), "p95": float(np.quantile(lengths, .95)),
            "over_limit": sum(n > limit for n in lengths), "limit": limit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new experiment name")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name, expected in PROTECTED.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"protected source changed: {name}")
    from transformers import AutoConfig, AutoTokenizer
    lock = json.loads((ROOT / "code/script/configs/models.lock.json").read_text())["generator"]
    model_path = ROOT / lock["path"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    queries = load_eval_queries(str(args.dialogues))
    passages = load_products(str(ROOT / "data/train/products.jsonl"))
    ranked = load_submission(args.retrieval_results, queries, set(passages), top_k=10)
    items = build_judge_items({q["qid"]: q for q in queries}, ranked, passages)
    if len(items) < 8:
        raise ValueError("need at least eight non-gold candidate pairs for batch-8 smoke")
    # Reuse the original renderer, with only its tokenizer dependency, before
    # loading weights. No copy of its prompt and no truncation of any pair.
    renderer = SimpleNamespace(tokenizer=tokenizer)
    lengths = [len(tokenizer(LLMJudge._render(renderer, item))["input_ids"]) for item in items]
    limit = config.max_position_embeddings - 12
    summary = length_summary(lengths, limit)
    if summary["over_limit"]:
        raise ValueError(f"original judge prompt exceeds context: {summary}")
    report = {"kind": "official_judge_compatibility_smoke_not_quality_benchmark",
              "model": lock, "judge_prompt_tokens": summary,
              "settings": {"batch_size": 8, "max_new_tokens": 12, "dtype": "bfloat16",
                           "do_sample": False, "truncation": False},
              "input_sha256": {str(path): sha256(path) for path in [args.dialogues, args.retrieval_results]},
              "original_evaluate_sha256": sha256(ROOT / "evaluate.py")}
    report["status"] = "preflight_passed"
    if not args.preflight_only:
        started = time.perf_counter()
        try:
            judge = LLMJudge(str(model_path), "cuda", max_new_tokens=12)
            hardest = sorted(range(len(items)), key=lambda i: (-lengths[i], i))[:8]
            selections = {"first_eight": list(range(8)), "eight_longest_prompts": hardest}
            report["batches"] = {}
            for label, indices in selections.items():
                rows = judge.judge_batch([items[i] for i in indices])
                if len(rows) != 8:
                    raise ValueError("judge returned wrong number of decisions")
                report["batches"][label] = {"input_lengths": [lengths[i] for i in indices],
                                          "parse_failures": sum(not row["parse_ok"] for row in rows),
                                          "raw_decisions": rows}
            report["elapsed_seconds_including_model_load"] = time.perf_counter() - started
            report["peak_cuda_allocated_bytes"] = judge.torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = judge.torch.cuda.max_memory_reserved()
            report["gpu_name"] = judge.torch.cuda.get_device_name()
            report["status"] = "passed"
        except Exception:
            report["status"] = "failed"
            report["error"] = traceback.format_exc()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ["batches", "input_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
