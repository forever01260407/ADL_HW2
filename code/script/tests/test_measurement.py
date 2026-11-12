"""Regression coverage for relative measurement paths and resource gates."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_validated
from validate_release import check_budget


class MeasurementTests(unittest.TestCase):
    def test_relative_log_path_is_saved_and_previous_attempt_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root / "code/script/run_validated.py"
            source.parent.mkdir(parents=True)
            source.write_text("measurement source")
            for name in ["retrieval.py", "retrieval_utils.py"]:
                (root / name).write_text("fixture source")

            def fake_process(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"exit_code": 0}))
                return SimpleNamespace(returncode=0)

            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(run_validated, "ROOT", root), patch.object(run_validated, "PROTECTED", {}), \
                        patch.object(run_validated, "__file__", str(source)), \
                        patch.object(sys, "argv", ["run_validated.py", "--output", "results/timing.json", "--", "retrieval.py"]), \
                        patch.object(run_validated.subprocess, "run", side_effect=fake_process), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(run_validated.main(), 0)
                    report = json.loads((root / "results/timing.json").read_text())
                    self.assertEqual(report["log"], "results/timing.log")
                    self.assertEqual(report["exit_code"], 0)
                    with self.assertRaises(SystemExit):
                        run_validated.main()
            finally:
                os.chdir(previous)

    def test_resource_gate_requires_real_budget_and_no_candidate_overrides(self):
        environment = dict.fromkeys(["HW2_MODEL_PATH", "HW2_RETRIEVER", "HW2_RERANK", "HW2_RRF_CANDIDATES",
            "HW2_RRF_DENSE_WEIGHT", "HW2_RRF_BM25_WEIGHT", "HW2_BM25_QUERY_MODE",
            "HW2_BM25_LAST_WEIGHT", "HW2_BM25_TITLE_WEIGHT", "HW2_BM25_TRACE_PATH"])
        environment["HW2_BATCH_SIZE"] = "32"
        report = {"cuda_allocator_budget_gib": 8, "cold_process_wall_seconds": 200,
            "environment": environment, "resources": {"cuda_allocator_budget_gib": 8,
                "peak_cuda_allocated_bytes": 2**30, "peak_cuda_reserved_bytes": 2 * 2**30,
                "peak_process_rss_bytes": 2 * 2**30}}
        self.assertTrue(all(check_budget(report).values()))
        for field, value in [("peak_cuda_reserved_bytes", 9 * 2**30), ("peak_process_rss_bytes", 32 * 2**30)]:
            changed = {**report, "resources": {**report["resources"], field: value}}
            with self.assertRaises(ValueError):
                check_budget(changed)
        with self.assertRaises(ValueError):
            check_budget({**report, "environment": {**environment, "HW2_MODEL_PATH": "override"}})


if __name__ == "__main__":
    unittest.main()
