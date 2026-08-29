import tempfile
import unittest
from pathlib import Path

from calibration_tool.camera.config import load_camera_config, load_capture_plan
from calibration_tool.errors import ConfigError


TOOL_ROOT = Path(__file__).resolve().parents[1]


class CameraConfigTests(unittest.TestCase):
    def test_laser_orientation_defaults_to_horizontal_and_accepts_vertical(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "camera.yaml"
            path.write_text("backend: synthetic\n", encoding="utf-8")
            self.assertEqual(load_camera_config(path)["laser"].orientation, "horizontal")

            path.write_text(
                "backend: synthetic\nlaser:\n  orientation: vertical\n",
                encoding="utf-8",
            )
            self.assertEqual(load_camera_config(path)["laser"].orientation, "vertical")

    def test_daheng_defaults_to_vertical_but_explicit_orientation_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "camera.yaml"
            path.write_text("backend: daheng\n", encoding="utf-8")
            self.assertEqual(load_camera_config(path)["laser"].orientation, "vertical")

            path.write_text(
                "backend: daheng\nlaser:\n  orientation: horizontal\n",
                encoding="utf-8",
            )
            self.assertEqual(load_camera_config(path)["laser"].orientation, "horizontal")

    def test_invalid_laser_orientation_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "camera.yaml"
            path.write_text(
                "backend: synthetic\nlaser:\n  orientation: diagonal\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, r"laser\.orientation.*horizontal.*vertical"):
                load_camera_config(path)

    def test_example_camera_geometry_is_explicit(self):
        runtime = load_camera_config(TOOL_ROOT / "configs" / "camera.example.yaml")
        self.assertEqual((runtime["camera"].width, runtime["camera"].height), (2448, 2048))
        self.assertEqual(runtime["calibration_src"], TOOL_ROOT / "calibration" / "src")

    def test_missing_calibration_src_uses_bundled_algorithms(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "camera.yaml"
            path.write_text("backend: synthetic\n", encoding="utf-8")
            runtime = load_camera_config(path)
            self.assertEqual(runtime["calibration_src"], TOOL_ROOT / "calibration" / "src")

    def test_capture_plan_task_camera_inherits_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.yaml"
            path.write_text(
                """dataset_id: test
output_dir: output
backend: synthetic
camera:
  exposure_us: 1000
  gain_db: 2
  width: 64
  height: 48
tasks:
  - task_id: one
    frames: 1
    filename_template: image{suffix}
    camera:
      exposure_us: 2000
""",
                encoding="utf-8",
            )
            plan = load_capture_plan(path)
            self.assertEqual(plan.tasks[0].config.exposure_us, 2000)
            self.assertEqual(plan.tasks[0].config.gain_db, 2)
            self.assertEqual(plan.output_dir, Path(temporary) / "output")

    def test_daheng_capture_plan_defaults_to_vertical(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.yaml"
            path.write_text(
                """dataset_id: test
output_dir: output
backend: daheng
tasks:
  - task_id: one
    frames: 1
    filename_template: image{suffix}
""",
                encoding="utf-8",
            )
            self.assertEqual(load_capture_plan(path).laser.orientation, "vertical")


if __name__ == "__main__":
    unittest.main()
