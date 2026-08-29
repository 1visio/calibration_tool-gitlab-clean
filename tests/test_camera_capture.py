import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from calibration_tool.camera import capture as capture_module
from calibration_tool.camera.capture import run_capture_plan
from calibration_tool.camera.capture import _new_manifest
from calibration_tool.camera.models import CameraConfig, CapturePlan, CaptureTask
from calibration_tool.camera.synthetic import SyntheticCameraProvider
from calibration_tool.errors import CaptureError
from calibration_tool.laser import LaserConfig


class _FailOnConfigureSession:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def configure(self, config):
        raise RuntimeError("intentional configure failure")


class _FailOnConfigureProvider(SyntheticCameraProvider):
    def open(self, serial_number, config):
        return _FailOnConfigureSession(super().open(serial_number, config))


class _RecordingProvider(SyntheticCameraProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sessions = []

    def open(self, serial_number, config):
        session = _RecordingSession(super().open(serial_number, config))
        self.sessions.append(session)
        return session


class _RecordingSession:
    def __init__(self, inner):
        self.inner = inner
        self.configure_calls = []
        self.exposure_gain_calls = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def configure(self, config):
        self.configure_calls.append(config)
        return self.inner.configure(config)

    def update_exposure_gain(self, exposure_us, gain_db):
        self.exposure_gain_calls.append((exposure_us, gain_db))
        return self.inner.update_exposure_gain(exposure_us, gain_db)


class CameraCaptureTests(unittest.TestCase):
    def test_manifest_records_actual_laser_orientation(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary) / "dataset")
            plan = CapturePlan(
                dataset_id=plan.dataset_id,
                output_dir=plan.output_dir,
                backend=plan.backend,
                serial_number=plan.serial_number,
                base_config=plan.base_config,
                tasks=plan.tasks,
                quality_thresholds=plan.quality_thresholds,
                laser=LaserConfig("vertical"),
            )
            manifest = _new_manifest(plan)
            self.assertEqual(manifest["laser"], {"orientation": "vertical"})
            self.assertEqual(manifest["plan"]["laser"], {"orientation": "vertical"})

    def _plan(self, output: Path) -> CapturePlan:
        base = CameraConfig(exposure_us=800, width=64, height=48, timeout_ms=100)
        tasks = (
            CaptureTask(
                "exposure_01", 2, "800us/frame_{index02}{suffix}", base,
                settle_frames=0, image_format="png", quality_mode="laser",
            ),
            CaptureTask(
                "exposure_02", 1, "1200us/frame_{index02}{suffix}",
                base.updated({"exposure_us": 1200}),
                settle_frames=0, image_format="png", quality_mode="laser",
            ),
        )
        return CapturePlan("test_set", output, "synthetic", "SIM-001", base, tasks)

    def test_capture_writes_manifest_csv_images_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            result = run_capture_plan(
                self._plan(output),
                SyntheticCameraProvider(target_fps=1000),
            )
            self.assertEqual(result.frame_count, 3)
            manifest = yaml.safe_load((output / "dataset_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["laser"], {"orientation": "horizontal"})
            self.assertEqual(len(manifest["frames"]), 3)
            self.assertTrue((output / "frames.csv").is_file())
            self.assertFalse((output / ".task_staging").exists())
            for frame in manifest["frames"]:
                self.assertTrue((output / frame["filename"]).is_file())
                self.assertEqual(len(frame["sha256"]), 64)
                self.assertIn("applied_camera", frame)
                self.assertIn("quality", frame)

    def test_task_config_is_applied_before_before_task_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = _RecordingProvider(target_fps=1000)
            seen = []

            def before_task(task):
                seen.append((task.task_id, provider.sessions[0].config.exposure_us))
                return True

            run_capture_plan(
                self._plan(Path(temporary) / "dataset"),
                provider,
                before_task=before_task,
            )

            self.assertEqual(seen, [("exposure_01", 800.0), ("exposure_02", 1200.0)])
            self.assertEqual(provider.sessions[0].configure_calls, [])
            self.assertEqual(provider.sessions[0].exposure_gain_calls, [(1200.0, 0.0)])

    def test_structural_change_still_uses_full_configure(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = _RecordingProvider(target_fps=1000)
            plan = self._plan(Path(temporary) / "dataset")
            second = replace(
                plan.tasks[1],
                config=plan.tasks[1].config.updated({"width": 32}),
            )
            run_capture_plan(replace(plan, tasks=(plan.tasks[0], second)), provider)
            self.assertEqual(len(provider.sessions[0].configure_calls), 1)
            self.assertEqual(provider.sessions[0].exposure_gain_calls, [])

    def test_failed_plan_resumes_without_recapturing_completed_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            plan = self._plan(output)
            second = replace(
                plan.tasks[1],
                config=plan.tasks[1].config.updated({"width": 32}),
            )
            plan = replace(plan, tasks=(plan.tasks[0], second))
            with self.assertRaises(CaptureError):
                run_capture_plan(plan, _FailOnConfigureProvider(target_fps=1000))
            work = output.parent / f".{output.name}.inprogress"
            failed = yaml.safe_load((work / "dataset_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(failed["tasks"]["exposure_01"]["status"], "completed")
            self.assertEqual(failed["tasks"]["exposure_02"]["status"], "failed")

            result = run_capture_plan(
                plan,
                SyntheticCameraProvider(target_fps=1000),
                resume=True,
            )
            self.assertEqual(result.frame_count, 3)
            completed = yaml.safe_load((output / "dataset_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(len([x for x in completed["frames"] if x["task_id"] == "exposure_01"]), 2)

    def test_atomic_text_retries_transient_windows_permission_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "frames.csv"
            real_replace = capture_module.os.replace
            attempts = []

            def flaky_replace(source, destination):
                attempts.append((source, destination))
                if len(attempts) < 3:
                    raise PermissionError(13, "temporarily locked", str(destination))
                return real_replace(source, destination)

            with (
                patch.object(capture_module.os, "replace", side_effect=flaky_replace),
                patch.object(capture_module.time, "sleep"),
            ):
                capture_module._atomic_text(target, "header\nvalue\n")

            self.assertEqual(len(attempts), 3)
            self.assertEqual(target.read_text(encoding="utf-8"), "header\nvalue\n")
            self.assertEqual(list(target.parent.glob(".frames.csv.*.tmp")), [])

    def test_save_state_can_defer_derived_frames_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            manifest = capture_module._new_manifest(self._plan(work_dir / "dataset"))
            with (
                patch.object(capture_module, "_atomic_yaml") as write_manifest,
                patch.object(capture_module, "_write_frames_csv") as write_csv,
            ):
                capture_module._save_state(work_dir, manifest, write_frames_csv=False)
                write_manifest.assert_called_once()
                write_csv.assert_not_called()

                capture_module._save_state(work_dir, manifest)
                self.assertEqual(write_manifest.call_count, 2)
                write_csv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
