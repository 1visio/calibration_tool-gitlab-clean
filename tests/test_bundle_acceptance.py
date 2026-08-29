import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_tool.bundle import build_calibration_bundle


class BundleAcceptanceTests(unittest.TestCase):
    def test_accepted_report_is_embedded_with_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            intrinsics = root / "camera_intrinsics.yaml"
            intrinsics.write_text(yaml.safe_dump({
                "image_width": 64, "image_height": 48,
                "camera_matrix": [[100, 0, 32], [0, 100, 24], [0, 0, 1]],
                "dist_coeffs": [0, 0, 0, 0, 0],
            }), encoding="utf-8")
            laser = root / "laser_plane.yaml"; laser.write_text("coefficients: [0, 1, 0, -1]\n", encoding="utf-8")
            extrinsics = root / "extrinsics.yaml"; extrinsics.write_text("R: [[1,0,0],[0,1,0],[0,0,1]]\nt: [0,0,0]\n", encoding="utf-8")
            config = root / "runtime.yaml"
            config.write_text(yaml.safe_dump({
                "schema_version": 1,
                "calibration": {
                    "intrinsics": intrinsics.name,
                    "laser_plane": laser.name,
                    "extrinsics": extrinsics.name,
                },
                "extraction": {"method": "shared_steger", "shared_steger": {"sigma_px": 1.2}},
            }), encoding="utf-8")
            quality = root / "acceptance_report.yaml"
            quality.write_text(yaml.safe_dump({
                "overall": "pass", "decision": "accepted",
                "counts": {"pass": 10, "warn": 0, "fail": 0},
            }), encoding="utf-8")
            quality.with_suffix(".html").write_text("<html>accepted</html>", encoding="utf-8")
            output = root / "bundle"
            manifest = build_calibration_bundle(
                config, output, "accepted-v1",
                expected_extractor="shared_steger", quality_report=quality,
            )
            self.assertTrue((output / "acceptance_report.yaml").is_file())
            self.assertTrue((output / "acceptance_report.html").is_file())
            self.assertEqual(manifest["quality"]["decision"], "accepted")
            self.assertEqual(len(manifest["quality"]["reports"]["yaml"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
