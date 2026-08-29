import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from calibration_tool.laser_models import (
    DEFAULT_LASER_MODEL,
    LaserModelConfigError,
    load_laser_model,
    normalize_laser_model_document,
    normalize_model_type,
    select_default_model,
)
from calibration_tool.profiles import load_runtime_profile


class LaserModelTests(unittest.TestCase):
    def test_default_and_supported_aliases(self):
        self.assertEqual(DEFAULT_LASER_MODEL, "circular_cone")
        self.assertEqual(normalize_model_type("plane_abcd"), "global_plane")
        self.assertEqual(
            select_default_model({"laser_model": {"default": "quadratic_graph"}}),
            "quadratic_graph",
        )

    def test_legacy_plane_abcd_is_normalized(self):
        model = normalize_laser_model_document({"plane_abcd": [0, 2, 0, -4]})
        self.assertEqual(model["model_type"], "global_plane")
        np.testing.assert_allclose(model["normal"], [0, 1, 0])
        self.assertEqual(model["d_mm"], -2.0)

    def test_model_file_is_loaded_and_old_runtime_key_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "laser_plane.yaml"
            model_path.write_text("coefficients: [0, 1, 0, -1]\n", encoding="utf-8")
            loaded = load_laser_model(model_path)
            self.assertEqual(loaded["model_type"], "global_plane")

            (root / "intrinsics.yaml").write_text(yaml.safe_dump({
                "camera_matrix": [[100, 0, 32], [0, 100, 24], [0, 0, 1]],
                "dist_coeffs": [0, 0, 0, 0, 0],
            }), encoding="utf-8")
            (root / "extrinsics.yaml").write_text(
                "R: [[1,0,0],[0,1,0],[0,0,1]]\nt: [0,0,0]\n",
                encoding="utf-8",
            )
            config = root / "runtime.yaml"
            config.write_text(yaml.safe_dump({
                "schema_version": 1,
                "calibration": {
                    "intrinsics": "intrinsics.yaml",
                    "laser_model": "laser_plane.yaml",
                    "extrinsics": "extrinsics.yaml",
                },
                "extraction": {"method": "steger", "steger": {}},
            }), encoding="utf-8")
            profile = load_runtime_profile(config)
            self.assertEqual(profile["laser_model"]["model_type"], "global_plane")
            self.assertEqual(profile["calibration_files"]["laser_plane"]["path"], str(model_path.resolve()))

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(LaserModelConfigError):
            normalize_model_type("cylinder")


if __name__ == "__main__":
    unittest.main()

