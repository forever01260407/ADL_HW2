#!/usr/bin/env python3
"""Measure an unchanged official entrypoint in a fresh, offline Python process.

The outer timer includes child startup/imports/model/data/index/search/output.
The child uses runpy only to collect process RSS and PyTorch peak memory after
the entrypoint returns; it does not patch models, inputs, or evaluator methods.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import runpy
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from analyze_results import sha256
from smoke_retrieval import PROTECTED

ROOT = Path(__file__).resolve().parents[2]


def worker(command: list[str], output: Path) -> int:
    import resource
    sys.argv = command
    started = time.perf_counter()
    code, error = 0, None
    try:
        runpy.run_path(command[0], run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code:
            error = str(exc)
    except BaseException:
        code, error = 1, traceback.format_exc()
        print(error, file=sys.stderr)
    torch = sys.modules.get("torch")
    cuda = torch is not None and torch.cuda.is_initialized()
    report = {
        "entrypoint_wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "gpu_name": torch.cuda.get_device_name() if cuda else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "versions": {name: getattr(sys.modules[name], "__version__", None) for name in
                     ["torch", "transformers", "sentence_transformers", "numpy", "faiss"] if name in sys.modules},
        "python": sys.version, "platform": platform.platform(), "exit_code": code, "error": error,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or Path(command[0]).resolve() not in [ROOT / "retrieval.py", ROOT / "evaluate.py"]:
        parser.error("command must start with this project's retrieval.py or evaluate.py")
    if platform.system() != "Linux":
        parser.error("run W06 measurements in the Linux development environment")
    if args.worker:
        return worker(command, args.output)
    if args.output.exists():
        parser.error("measurement already exists; use a new output name to preserve experiment history")
    for name, expected in PROTECTED.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"official source changed: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    child_output = args.output.with_suffix(".worker.json")
    log_path = args.output.with_suffix(".log")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "HW2_BATCH_SIZE": "32"}
    sources = [ROOT / name for name in PROTECTED] + [ROOT / "retrieval_utils.py", Path(__file__).resolve()]
    for flag in ["--dialogues", "--products", "--retrieval-results"]:
        if flag in command:
            sources.append(Path(command[command.index(flag) + 1]).resolve())
    # Fingerprinting is intentionally outside the measured official workload.
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *command],
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
        "models": json.loads((ROOT / "code/script/configs/models.lock.json").read_text()),
        "environment": {key: env.get(key) for key in
                        ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM",
                         "OMP_NUM_THREADS", "HW2_BATCH_SIZE", "HW2_MODEL_PATH",
                         "HW2_RERANK", "HW2_RERANK_CANDIDATES", "HW2_RERANK_BATCH_SIZE",
                         "HW2_RERANK_MODEL_PATH"]},
        "timing_scope": "fresh Python child including wrapper startup, imports, data/model load, indexing, search, output; no index cache; OS disk caches not flushed",
        "official_hardware_validation": False,
    }
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--output",
                                 str(child_output), "--", *command], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    metadata["cold_process_wall_seconds"] = time.perf_counter() - started
    metadata["exit_code"] = result.returncode
    metadata["log"] = str(log_path)
    if child_output.exists():
        metadata["resources"] = json.loads(child_output.read_text())
    args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"measurement": str(args.output), "exit_code": result.returncode,
                      "cold_process_wall_seconds": metadata["cold_process_wall_seconds"]}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
