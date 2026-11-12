#!/usr/bin/env python3
"""Verify an adopted W08 default against its evaluated rankings; never run a model."""
from __future__ import annotations

import json
from pathlib import Path

from analyze_results import sha256
from audit_w08 import RESULTS, ROOT, TRAINING, read, verify_file, verify_training, write_new
from audit_w07 import audit_timing
from eval_utils import load_eval_queries, load_products
from evaluate import load_submission


def main():
    training, _, _ = verify_training()
    selection = read(RESULTS / "w08_summary.json")
    if not selection["adopt"] or selection["official_judge_runs"] != 1:
        raise ValueError("official selection did not authorize promotion")
    for path, expected in selection["screening"]["input_sha256"].items():
        verify_file(path, expected)
    measured = read(RESULTS / "w08_default_dev_timing.json")
    timing = audit_timing("w08_default_dev_timing.json")
    for key in ["HW2_MODEL_PATH", "HW2_RETRIEVER", "HW2_RERANK", "HW2_RRF_CANDIDATES",
                "HW2_BM25_QUERY_MODE", "HW2_BM25_LAST_WEIGHT", "HW2_BM25_TITLE_WEIGHT", "HW2_BM25_TRACE_PATH"]:
        if measured["environment"][key] is not None:
            raise ValueError(f"default verification has an explicit override: {key}")
    queries = load_eval_queries(ROOT / "data/derived/dev.jsonl")
    products = load_products(ROOT / "data/train/products.jsonl")
    load = lambda name: load_submission(RESULTS / name, queries, set(products), top_k=50)
    if load("w08_default_dev_top50.json") != load("w08_epoch1_dev_top50.json"):
        raise ValueError("new default rankings differ from evaluated W08 checkpoint")
    log = Path(measured["log"]).read_text(encoding="utf-8")
    if f"model={ROOT / training['checkpoint']} " not in log:
        raise ValueError("formal default log did not load the selected checkpoint")
    final = {**selection, "default_verified": True, "default_action": "promoted_and_verified",
        "default_timing": timing, "default_query_count": len(queries),
        "finalization_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
            [RESULTS / "w08_summary.json", RESULTS / "w08_default_dev_top50.json",
             RESULTS / "w08_default_dev_timing.json", ROOT / "retrieval_utils.py",
             TRAINING / "training_manifest.json", Path(__file__).resolve()]}}
    write_new(RESULTS / "w08_final.json", final)
    print(json.dumps({k: final[k] for k in ["adopt", "official_judge_runs", "default_verified",
                                           "default_action", "default_query_count"]}, indent=2))


if __name__ == "__main__":
    main()
