#!/usr/bin/env python3
"""Fresh official process measurement with an optional PyTorch CUDA budget.

The budget constrains the PyTorch allocator on the development GPU; it does
not emulate RTX 3070 speed or cap CUDA context/other processes' allocations.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
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
ENV_KEYS = ["HW2_MODEL_PATH", "HW2_RETRIEVER", "HW2_RERANK", "HW2_BATCH_SIZE",
    "HW2_RRF_CANDIDATES", "HW2_RRF_DENSE_WEIGHT", "HW2_RRF_BM25_WEIGHT",
    "HW2_BM25_QUERY_MODE", "HW2_BM25_LAST_WEIGHT", "HW2_BM25_TITLE_WEIGHT",
    "HW2_BM25_TRACE_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS"]


def worker(command, output, budget_gib):
    started = time.perf_counter()
    code, error, fraction = 0, None, None
    sys.argv = command
    try:
        if budget_gib:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for this budget validation")
            total = torch.cuda.get_device_properties(0).total_memory
            fraction = min(budget_gib * 2**30 / total, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            torch.cuda.reset_peak_memory_stats()
        runpy.run_path(command[0], run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        error = str(exc) if code else None
    except BaseException:
        code, error = 1, traceback.format_exc()
        print(error, file=sys.stderr)
    torch = sys.modules.get("torch")
    cuda = torch is not None and torch.cuda.is_initialized()
    report = {"exit_code": code, "error": error,
        "entrypoint_wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "gpu_name": torch.cuda.get_device_name() if cuda else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "cuda_allocator_budget_gib": budget_gib, "cuda_allocator_fraction": fraction,
        "versions": {n: getattr(sys.modules[n], "__version__", None) for n in
            ["torch", "transformers", "sentence_transformers", "numpy", "faiss"] if n in sys.modules},
        "python": sys.version, "platform": platform.platform()}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-gib", type=float, default=0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or Path(command[0]).resolve() not in [ROOT / "retrieval.py", ROOT / "evaluate.py"]:
        parser.error("only this project's unchanged official entrypoints are supported")
    if platform.system() != "Linux" or not 0 <= args.budget_gib <= 8:
        parser.error("Linux and CUDA allocator budget between 0 and 8 GiB required")
    if args.worker:
        return worker(command, args.output, args.budget_gib)
    child_output = args.output.with_suffix(".worker.json")
    log_path = args.output.with_suffix(".log")
    if any(p.exists() for p in [args.output, child_output, log_path]):
        parser.error("refusing to overwrite a measurement or previous attempt")
    for name, expected in PROTECTED.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"official source changed: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "HW2_BATCH_SIZE": "32"}
    sources = [ROOT / n for n in PROTECTED] + [ROOT / "retrieval_utils.py", Path(__file__).resolve()]
    for flag in ["--dialogues", "--products", "--retrieval-results"]:
        if flag in command:
            sources.append(Path(command[command.index(flag) + 1]).resolve())
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *command], "input_sha256": {
            p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
        "environment": {k: env.get(k) for k in ENV_KEYS},
        "cuda_allocator_budget_gib": args.budget_gib, "official_hardware_validation": False,
        "timing_scope": "fresh child includes imports, full data/model load, indexing and output; no persistent index; OS disk caches not flushed"}
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
            "--budget-gib", str(args.budget_gib), "--output", str(child_output), "--", *command],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    report.update(exit_code=result.returncode, cold_process_wall_seconds=time.perf_counter() - started,
        resources=json.loads(child_output.read_text()), log=log_path.relative_to(ROOT).as_posix())
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["exit_code", "cold_process_wall_seconds", "resources"]}, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
