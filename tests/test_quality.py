import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_tool.quality import audit_baseline


class QualityTests(unittest.TestCase):
    def test_quality_gate_reports_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.yaml"
            policy = root / "policy.yaml"
            baseline.write_text(
                yaml.safe_dump(
                    {
                        "sources": {},
                        "regressions": {"intrinsics": {"values": {"test_count": 0}}},
                    }
                ),
                encoding="utf-8",
            )
            policy.write_text(
                yaml.safe_dump(
                    {
                        "metric_gates": [
                            {
                                "id": "independent_test",
                                "metric": "intrinsics.test_count",
                                "op": "ge",
                                "expected": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit_baseline(baseline, policy)
            self.assertEqual(report["overall"], "fail")
            self.assertEqual(report["counts"]["fail"], 1)


if __name__ == "__main__":
    unittest.main()
