#!/usr/bin/env python3
"""Exercise the unchanged official CLI on handwritten data, using real weights."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = {
    "retrieval.py": "7c0a0c0a6a5f466a3a3bcf78e7d410bacaf1876eeb44aa3cae10c6cebba8f774",
    "evaluate.py": "0f640c6b126594d75d39baa1c3b7447185fda9010a5e8486a88e29198b23acbf",
    "eval_utils.py": "47b1dd2dee9b6c9769d4952dc32712ac8e25f0cb57665ceb2247880d43e32e0b",
}


def main():
    for name, expected in PROTECTED.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"protected starter changed: {name}")
    fixture = json.loads((ROOT / "code/script/tests/dialogue_fixtures.json").read_text())
    output = ROOT / "results/fixtures"
    output.mkdir(parents=True, exist_ok=True)
    products = [{"parent_asin": pid, "title": text.split("\n", 1)[0],
                 "description": text.split("\n", 1)[1]} for pid, text in fixture["products"].items()]
    # One role wraps the fixture's complete transcript. Loader and ranker still use
    # their actual public contracts; semantic probes below are descriptive only.
    queries = [{"dialogue_id": q["qid"], "conversation": [{"role": "user", "content": q["query"]}]}
               for q in fixture["queries"]]
    for name, rows in [("products", products), ("dialogues", queries)]:
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HW2_RETRIEVER": "dense", "HW2_RERANK": "0"}
    elapsed, runs = [], []
    for i in range(2):
        path = output / f"retrieval_{i}.json"
        command = [sys.executable, str(ROOT / "retrieval.py"), "--dialogues", str(output / "dialogues.jsonl"),
                   "--products", str(output / "products.jsonl"), "--output", str(path)]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        elapsed.append(time.perf_counter() - started)
        (output / f"retrieval_{i}.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        payload = json.loads(path.read_text())
        assert payload["top_k"] == 10 and len(payload["results"]) == len(queries)
        assert [r["qid"] for r in payload["results"]] == [q["dialogue_id"] for q in queries]
        for row in payload["results"]:
            assert len(set(row["ranked_pids"])) == 10
            assert set(row["ranked_pids"]) <= fixture["products"].keys()
        runs.append(payload)
    assert runs[0] == runs[1], "repeat rankings differ"
    report = {"kind": "real_model_cli_fixture_smoke_not_official_quality",
              "runs_equal": True, "valid_top10": True, "protected_hashes_verified": PROTECTED,
              "process_wall_seconds": elapsed, "sample_count": len(queries),
              "semantic_observations": [{"qid": q["qid"], "expected_pid": q["semantic_expected_pid"],
                  "observed_rank": r["ranked_pids"].index(q["semantic_expected_pid"]) + 1
                    if q["semantic_expected_pid"] in r["ranked_pids"] else None}
                  for q, r in zip(fixture["queries"], runs[0]["results"])]}
    (ROOT / "results/w05_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
