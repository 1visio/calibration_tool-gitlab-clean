import tempfile
import threading
import time
import unittest
from pathlib import Path

from calibration_tool.camera.capture import run_capture_plan
from calibration_tool.camera.models import CameraConfig, CapturePlan, CaptureTask
from calibration_tool.camera.synthetic import SyntheticCameraProvider
from calibration_tool.errors import CaptureError


class CaptureBeforeTaskTests(unittest.TestCase):
    def _plan(self, output: Path) -> CapturePlan:
        config = CameraConfig(width=32, height=24, timeout_ms=100)
        task = CaptureTask(
            task_id="one",
            frames=1,
            filename_template="fit/{pose_id}{suffix}",
            config=config,
            pose_id="001",
            role="laser",
            settle_frames=0,
            image_format="tif",
            quality_mode="laser",
            tags={"laser_state": "on", "split": "fit"},
        )
        return CapturePlan("gate", output, "synthetic", "SIM-001", config, (task,))

    def test_before_task_blocks_until_approval_and_preserves_progress_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            requested = threading.Event()
            approve = threading.Event()
            events: list[dict] = []

            def before_task(_task):
                requested.set()
                return approve.wait(2.0)

            holder: list[object] = []
            thread = threading.Thread(
                target=lambda: holder.append(
                    run_capture_plan(
                        self._plan(output),
                        SyntheticCameraProvider(target_fps=1000),
                        before_task=before_task,
                        progress=events.append,
                    )
                )
            )
            thread.start()
            self.assertTrue(requested.wait(1.0))
            time.sleep(0.05)
            self.assertFalse(output.exists())
            approve.set(); thread.join(5.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(holder[0].frame_count, 1)
            self.assertEqual([event["event"] for event in events], ["task_started", "frame", "task_completed"])

    def test_before_task_false_cancels_and_closes_without_output_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            with self.assertRaisesRegex(CaptureError, "采集已取消"):
                run_capture_plan(
                    self._plan(output),
                    SyntheticCameraProvider(target_fps=1000),
                    before_task=lambda _task: False,
                )
            self.assertFalse(output.exists())
            self.assertTrue((output.parent / ".dataset.inprogress" / "dataset_manifest.yaml").is_file())
            changed = self._plan(output)
            changed_task = CaptureTask(
                task_id="one",
                frames=1,
                filename_template="fit/{pose_id}{suffix}",
                config=changed.base_config.updated({"exposure_us": 2400}),
                pose_id="001",
                role="laser",
                settle_frames=0,
                image_format="tif",
                quality_mode="laser",
                tags={"laser_state": "on", "split": "fit"},
            )
            changed = CapturePlan(
                "gate", output, "synthetic", "SIM-001", changed.base_config, (changed_task,)
            )
            with self.assertRaisesRegex(CaptureError, "不一致"):
                run_capture_plan(
                    changed,
                    SyntheticCameraProvider(target_fps=1000),
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
