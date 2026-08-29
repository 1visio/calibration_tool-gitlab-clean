import unittest
from pathlib import Path

from calibration_tool.camera.models import CameraConfig, CapturePlan, CaptureTask


class CameraModelTests(unittest.TestCase):
    def test_capture_task_rejects_path_escape(self):
        with self.assertRaises(ValueError):
            CaptureTask(
                task_id="bad",
                frames=1,
                filename_template="../outside{suffix}",
                config=CameraConfig(width=32, height=24),
            )

    def test_capture_task_rejects_duplicate_names(self):
        with self.assertRaises(ValueError):
            CaptureTask(
                task_id="duplicate",
                frames=2,
                filename_template="same{suffix}",
                config=CameraConfig(width=32, height=24),
            )

    def test_capture_plan_rejects_cross_task_collision(self):
        config = CameraConfig(width=32, height=24)
        first = CaptureTask("first", 1, "same{suffix}", config)
        second = CaptureTask("second", 1, "same{suffix}", config)
        with self.assertRaises(ValueError):
            CapturePlan("set", Path("out"), "synthetic", "", config, (first, second))


if __name__ == "__main__":
    unittest.main()
