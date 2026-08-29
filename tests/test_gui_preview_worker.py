import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from calibration_tool.camera.models import CameraConfig, QualityThresholds
from calibration_tool.camera.quality import analyze_frame as real_analyze_frame
from calibration_tool.camera.synthetic import SyntheticCameraProvider
from calibration_tool.gui.workers import PreviewThread


class PreviewWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_thread_emits_frames_and_stops(self):
        thread = PreviewThread(
            SyntheticCameraProvider(target_fps=500), "SIM-001",
            CameraConfig(width=64, height=48, timeout_ms=100), "laser",
            QualityThresholds(), None,
        )
        loop = QEventLoop(); frames = []; errors = []; applied = []
        def on_frame(frame, quality):
            frames.append((frame, quality))
            if len(frames) == 2:
                thread.request_exposure_gain(2400, 1.5)
            if len(frames) >= 4 and applied:
                thread.requestInterruption()
        thread.frame_ready.connect(on_frame); thread.failed.connect(errors.append)
        thread.settings_applied.connect(applied.append); thread.finished.connect(loop.quit)
        QTimer.singleShot(3000, lambda: (thread.requestInterruption(), loop.quit()))
        thread.start(); loop.exec(); thread.wait(3000)
        self.assertFalse(thread.isRunning())
        self.assertFalse(errors)
        self.assertGreaterEqual(len(frames), 2)
        self.assertTrue(applied)
        self.assertEqual(applied[-1].exposure_us, 2400)
        self.assertEqual(applied[-1].gain_db, 1.5)
        self.assertIn("laser_coverage", frames[-1][1])

    def test_settle_frames_are_visible_and_marked_until_stable(self):
        thread = PreviewThread(
            SyntheticCameraProvider(target_fps=500), "SIM-001",
            CameraConfig(width=64, height=48, timeout_ms=100), "laser",
            QualityThresholds(), None, initial_discard_frames=3,
        )
        loop = QEventLoop(); states = []; errors = []

        def on_frame(_frame, quality):
            states.append((bool(quality.get("settling")), quality.get("settle_frames_remaining")))
            if len(states) >= 3:
                thread.requestInterruption()

        thread.frame_ready.connect(on_frame)
        thread.failed.connect(errors.append)
        thread.finished.connect(loop.quit)
        QTimer.singleShot(3000, lambda: (thread.requestInterruption(), loop.quit()))
        thread.start(); loop.exec(); thread.wait(3000)

        self.assertFalse(thread.isRunning())
        self.assertFalse(errors)
        self.assertEqual(states[:3], [(True, 2), (True, 1), (False, 0)])

    def test_preview_quality_is_rate_limited_and_reuses_latest_complete_result(self):
        calls = []

        def counted_analyze(*args, **kwargs):
            calls.append(1)
            return real_analyze_frame(*args, **kwargs)

        thread = PreviewThread(
            SyntheticCameraProvider(target_fps=200), "SIM-001",
            CameraConfig(width=64, height=48, timeout_ms=100), "generic",
            QualityThresholds(), None, initial_discard_frames=0,
            quality_refresh_interval_s=10.0,
        )
        loop = QEventLoop(); payloads = []; errors = []

        def on_frame(_frame, quality):
            payloads.append(quality)
            if len(payloads) >= 6:
                thread.requestInterruption()

        thread.frame_ready.connect(on_frame)
        thread.failed.connect(errors.append)
        thread.finished.connect(loop.quit)
        with patch("calibration_tool.gui.workers.analyze_frame", side_effect=counted_analyze):
            QTimer.singleShot(3000, lambda: (thread.requestInterruption(), loop.quit()))
            thread.start(); loop.exec(); thread.wait(3000)

        self.assertFalse(errors)
        self.assertEqual(len(calls), 1)
        self.assertTrue(payloads[0]["preview_quality_fresh"])
        self.assertTrue(all(item["preview_quality_reused"] for item in payloads[1:]))
        self.assertEqual(
            {item["preview_quality_source_frame_number"] for item in payloads},
            {payloads[0]["preview_quality_source_frame_number"]},
        )

    def test_forced_capture_quality_matches_current_frame(self):
        thread = PreviewThread(
            SyntheticCameraProvider(target_fps=100), "SIM-001",
            CameraConfig(width=64, height=48, timeout_ms=100), "generic",
            QualityThresholds(), None, initial_discard_frames=0,
            quality_refresh_interval_s=10.0,
        )
        loop = QEventLoop(); payloads = []; errors = []

        def on_frame(frame, quality):
            payloads.append((frame.camera_frame_number, quality))
            if len(payloads) == 1:
                thread.request_fresh_quality_after_frames(0)
            elif quality["preview_quality_fresh"]:
                thread.requestInterruption()

        thread.frame_ready.connect(on_frame)
        thread.failed.connect(errors.append)
        thread.finished.connect(loop.quit)
        QTimer.singleShot(3000, lambda: (thread.requestInterruption(), loop.quit()))
        thread.start(); loop.exec(); thread.wait(3000)

        self.assertFalse(errors)
        self.assertGreaterEqual(len(payloads), 2)
        forced_frame, forced_quality = next(
            item for item in payloads[1:] if item[1]["preview_quality_fresh"]
        )
        self.assertEqual(
            forced_quality["preview_quality_source_frame_number"],
            forced_frame,
        )


if __name__ == "__main__":
    unittest.main()
