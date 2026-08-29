import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml

from calibration_tool.acceptance import build_acceptance_report
from calibration_tool.cli import main
from calibration_tool.errors import CalibrationToolError
from calibration_tool.io_utils import sha256_file


class AcceptanceTests(unittest.TestCase):
    def _write_fixture(self, root: Path, *, independent: bool = True, release: bool = False) -> Path:
        (root / "workflow.yaml").write_text(yaml.safe_dump({
            "status": "completed",
            "stages": [{
                "stage": "intrinsics", "status": "completed",
                "quality_gates": [{"id": "rmse", "status": "pass", "actual": 0.1, "expected": "<= 0.3"}],
            }],
        }), encoding="utf-8")
        (root / "quality.yaml").write_text(yaml.safe_dump({
            "overall": "pass", "counts": {"pass": 1, "warn": 0, "fail": 0},
            "gates": [{"id": "independent", "status": "pass", "actual": 2, "expected": ">= 1"}],
        }), encoding="utf-8")
        evaluation = 2 if independent else 10
        (root / "compensation.json").write_text(json.dumps({
            "loaded_frame_count": 10,
            "build_frame_count": 8 if independent else 10,
            "evaluation_frame_count": evaluation,
            "repeatability_sigma_median_mm": 0.004,
            "metrics": {
                "profile_before_pv_mm": 2.0, "profile_after_pv_mm": 0.1,
                "profile_before_rms_mm": 0.5, "profile_after_rms_mm": 0.02,
            },
        }), encoding="utf-8")
        (root / "evidence.csv").write_text("x,value\n1,0.1\n", encoding="utf-8")
        (root / "policy.yaml").write_text(yaml.safe_dump({
            "schema_version": 1,
            "required": {
                "workflow_completed": True, "quality_report": True,
                "runtime_profile_clean": False, "golden_match": False,
                "compensation": True, "artifacts": True,
            },
            "compensation": {
                "max_profile_after_pv_mm": 0.25,
                "max_profile_after_rms_mm": 0.05,
                "min_pv_reduction_fraction": 0.8,
                "min_rms_reduction_fraction": 0.8,
                "max_repeatability_sigma_mm": 0.02,
            },
        }), encoding="utf-8")
        plan = {
            "schema_version": 1, "report_id": "test", "output_dir": "report", "policy": "policy.yaml",
            "inputs": {
                "workflow_report": "workflow.yaml", "quality_reports": ["quality.yaml"],
                "compensation_metrics": "compensation.json", "artifacts": ["evidence.csv"],
            },
            "release": {"enabled": release},
        }
        path = root / "plan.yaml"
        path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        return path

    def test_pass_report_contains_compensation_comparison_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = build_acceptance_report(self._write_fixture(root))
            self.assertEqual(report["decision"], "accepted")
            self.assertEqual(report["overall"], "pass")
            self.assertAlmostEqual(report["compensation"]["reduction"]["pv_fraction"], 0.95)
            self.assertTrue((root / "report" / "acceptance_report.html").is_file())
            self.assertTrue(all(len(item["sha256"]) == 64 for item in report["artifacts"]))
            with self.assertRaises(CalibrationToolError):
                build_acceptance_report(root / "plan.yaml")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["acceptance-report", str(root / "plan.yaml"), "--overwrite"]), 0)

    def test_self_evaluated_compensation_is_rejected_and_release_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = build_acceptance_report(
                self._write_fixture(root, independent=False, release=True)
            )
            self.assertEqual(report["decision"], "rejected")
            self.assertEqual(report["release"]["status"], "blocked")
            gate = next(item for item in report["gates"] if item["id"] == "compensation.independent_validation")
            self.assertEqual(gate["status"], "fail")

    def test_accepted_release_embeds_final_published_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = self._write_fixture(root)
            (root / "camera_intrinsics.yaml").write_text(yaml.safe_dump({
                "image_width": 64, "image_height": 48,
                "camera_matrix": [[100, 0, 32], [0, 100, 24], [0, 0, 1]],
                "dist_coeffs": [0, 0, 0, 0, 0],
            }), encoding="utf-8")
            (root / "laser_plane.yaml").write_text(
                "coefficients: [0, 1, 0, -1]\n", encoding="utf-8"
            )
            (root / "extrinsics.yaml").write_text(
                "R: [[1,0,0],[0,1,0],[0,0,1]]\nt: [0,0,0]\n", encoding="utf-8"
            )
            (root / "runtime.yaml").write_text(yaml.safe_dump({
                "schema_version": 1,
                "calibration": {
                    "intrinsics": "camera_intrinsics.yaml",
                    "laser_plane": "laser_plane.yaml",
                    "extrinsics": "extrinsics.yaml",
                },
                "extraction": {"method": "shared_steger", "shared_steger": {"sigma_px": 1.2}},
            }), encoding="utf-8")
            plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
            plan["release"] = {
                "enabled": True,
                "runtime_config": "runtime.yaml",
                "output_dir": "bundle",
                "package_id": "accepted-v1",
            }
            plan_path.write_text(yaml.safe_dump(plan), encoding="utf-8")

            report = build_acceptance_report(plan_path)
            bundled_report = yaml.safe_load(
                (root / "bundle" / "acceptance_report.yaml").read_text(encoding="utf-8")
            )
            manifest = yaml.safe_load(
                (root / "bundle" / "calibration_bundle.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(report["release"]["status"], "published")
            self.assertEqual(bundled_report["release"]["status"], "published")
            self.assertEqual(
                manifest["quality"]["reports"]["yaml"]["sha256"],
                sha256_file(root / "bundle" / "acceptance_report.yaml"),
            )


if __name__ == "__main__":
    unittest.main()
