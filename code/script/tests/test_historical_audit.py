"""Source-history resolution must preserve strict data-file fingerprints."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import audit_w07
from analyze_results import sha256


class HistoricalAuditTests(unittest.TestCase):
    def test_exact_source_snapshot_is_accepted_but_data_tampering_is_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root / "retrieval_utils.py"
            archived = root / "code/script/snapshots/w07/retrieval_utils.py"
            archived.parent.mkdir(parents=True)
            archived.write_text("old source", encoding="utf-8")
            current.write_text("new source", encoding="utf-8")
            data = root / "data.json"
            data.write_text("original", encoding="utf-8")
            timing = {"exit_code": 0, "resources": {"exit_code": 0}, "cold_process_wall_seconds": 1,
                      "environment": {}, "input_sha256": {"retrieval_utils.py": sha256(archived),
                                                          "data.json": sha256(data)}}
            with patch.object(audit_w07, "ROOT", root), patch.object(audit_w07, "read", return_value=timing):
                result = audit_w07.audit_timing("h1_example.json")
                self.assertEqual(result["verified_sources"]["retrieval_utils.py"],
                                 "code/script/snapshots/w07/retrieval_utils.py")
                data.write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    audit_w07.audit_timing("h1_example.json")
                data.write_text("original", encoding="utf-8")
                archived.write_text("tampered source", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    audit_w07.audit_timing("h1_example.json")


if __name__ == "__main__":
    unittest.main()
