import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

from calibration_tool.camera import load_capture_plan
from calibration_tool.camera.capture import _new_manifest, _save_state
from calibration_tool.camera.config import capture_plan_hash
from calibration_tool.camera.models import CapturedFrame
from calibration_tool.camera.quality import analyze_frame, quality_to_dict
from calibration_tool.gui.capture_controller import CaptureTaskGate
from calibration_tool.gui.main_window import CalibrationWizardWindow
from calibration_tool.gui.pages import _SEARCH_REGION_QUALITY_ENABLED
from calibration_tool.io_utils import load_document


ROOT = Path(__file__).resolve().parents[1]


class GuiCapturePageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _window_page(self, root: Path):
        window = CalibrationWizardWindow(default_camera_config=ROOT / "configs" / "camera.example.yaml")
        page = window.capture_page
        camera = window.camera_page
        camera.width.setValue(64); camera.height.setValue(48)
        page.output.setText(str(root / "dataset"))
        page.plan_path.setText(str(root / "plans" / "capture.yaml"))
        page.dataset_id.setText("gui_capture")
        for row in range(page.recipe_table.table.rowCount()):
            settle = page.recipe_table.table.cellWidget(row, 7)
            settle.setValue(0)
        return window, page

    def test_default_project_uses_project_data_laser_plane_output(self):
        window = CalibrationWizardWindow(default_camera_config=ROOT / "configs" / "camera.example.yaml")
        try:
            self.assertEqual(window.capture_page.dataset_id.text(), "laser_plane")
            self.assertEqual(window.capture_page.output.text(), str(ROOT / "projects" / "default" / "data" / "laser_plane"))
        finally:
            window.close(); self.app.processEvents()

    def test_search_region_quality_is_temporarily_disabled_in_gui(self):
        window = CalibrationWizardWindow(default_camera_config=ROOT / "configs" / "camera.example.yaml")
        try:
            self.assertFalse(_SEARCH_REGION_QUALITY_ENABLED)
            self.assertTrue(window.camera_page.search_region_quality.isHidden())
            self.assertTrue(window.capture_page.live_search_region_quality.isHidden())
        finally:
            window.close(); self.app.processEvents()

    def test_selecting_daheng_camera_config_updates_laser_orientation(self):
        window = CalibrationWizardWindow(default_camera_config=ROOT / "configs" / "camera.example.yaml")
        try:
            self.assertEqual(window.project_page.laser_orientation.currentText(), "horizontal")
            window.project_page.camera_config.setText(
                str(ROOT / "configs" / "camera.daheng.example.yaml")
            )
            self.assertEqual(window.project_page.laser_orientation.currentText(), "vertical")
        finally:
            window.close(); self.app.processEvents()

    def test_only_visible_page_renders_live_frame(self):
        window = CalibrationWizardWindow(default_camera_config=ROOT / "configs" / "camera.example.yaml")
        frame = CapturedFrame(
            image=np.zeros((48, 64), dtype=np.uint8),
            camera_frame_number=1,
            camera_timestamp_ticks=None,
            host_timestamp_ns=1,
            host_monotonic_ns=1,
        )
        quality = quality_to_dict(analyze_frame(frame.image, sensor_max_value=255))
        camera_render = Mock()
        capture_render = Mock()
        window.camera_page.preview.set_array = camera_render
        window.capture_page.live_preview.set_array = capture_render
        try:
            window.show()
            window.steps.setCurrentRow(1)
            self.app.processEvents()
            window.camera_page._on_frame(frame, quality)
            self.assertEqual(camera_render.call_count, 1)
            self.assertEqual(capture_render.call_count, 0)

            window.steps.setCurrentRow(2)
            self.app.processEvents()
            window.camera_page._on_frame(frame, quality)
            self.assertEqual(camera_render.call_count, 1)
            self.assertEqual(capture_render.call_count, 1)
        finally:
            window.close(); self.app.processEvents()

    def test_recipe_defaults_use_500us_for_nolaser_and_laser(self):
        window = CalibrationWizardWindow(default_camera_config=ROOT / "configs" / "camera.example.yaml")
        try:
            page = window.capture_page
            self.assertEqual(page.recipe_table.table.cellWidget(1, 3).value(), 500.0)
            self.assertEqual(page.recipe_table.table.cellWidget(2, 3).value(), 500.0)
            self.assertIs(page.ready_button, page.capture_task_button)
            self.assertNotIn(
                "已准备，采集当前任务",
                [button.text() for button in window.findChildren(QPushButton)],
            )
        finally:
            window.close(); self.app.processEvents()

    def test_default_plan_preview_has_72_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            window, page = self._window_page(Path(temporary))
            try:
                self.assertTrue(page.generate_plan())
                self.assertEqual(page.plan_table.rowCount(), 72)
                self.assertIn("任务 72", page.plan_summary.text())
                self.assertEqual(len(page.loaded_plan.tasks), 72)
            finally:
                window.close(); self.app.processEvents()

    def test_task_exposure_is_individually_editable_and_saved_to_yaml(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window, page = self._window_page(root)
            try:
                page.include_validation.setChecked(False)
                page.fit_groups.setValue(2)
                self.assertTrue(page.generate_plan())
                exposure_item = page.plan_table.item(3, 4)
                self.assertTrue(exposure_item.flags() & Qt.ItemFlag.ItemIsEditable)
                self.assertFalse(
                    page.plan_table.item(3, 3).flags() & Qt.ItemFlag.ItemIsEditable
                )

                exposure_item.setText("42000")
                self.app.processEvents()

                self.assertEqual(page.loaded_plan.tasks[0].config.exposure_us, 35000.0)
                self.assertEqual(page.loaded_plan.tasks[3].config.exposure_us, 42000.0)
                saved = load_capture_plan(root / "plans" / "capture.yaml")
                self.assertEqual(saved.tasks[3].config.exposure_us, 42000.0)
                self.assertIn("42000 μs", page.plan_tasks.item(3).text())
            finally:
                window.close(); self.app.processEvents()

    def test_pending_task_exposure_edit_updates_resume_plan_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window, page = self._window_page(root)
            try:
                page.include_validation.setChecked(False)
                page.fit_groups.setValue(1)
                self.assertTrue(page.generate_plan())
                work_dir = root / ".dataset.inprogress"
                work_dir.mkdir()
                _save_state(work_dir, _new_manifest(page.loaded_plan))

                page.plan_table.item(1, 4).setText("750")
                self.app.processEvents()

                manifest = load_document(work_dir / "dataset_manifest.yaml")
                self.assertEqual(manifest["plan_sha256"], capture_plan_hash(page.loaded_plan))
                self.assertEqual(
                    manifest["plan"]["tasks"][1]["camera"]["exposure_us"],
                    750.0,
                )
            finally:
                window.close(); self.app.processEvents()

    def test_stale_capture_does_not_block_new_task_exposure_edit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window, page = self._window_page(root)
            try:
                page.include_validation.setChecked(False)
                page.fit_groups.setValue(1)
                self.assertTrue(page.generate_plan())
                work_dir = root / ".dataset.inprogress"
                work_dir.mkdir()
                manifest = _new_manifest(page.loaded_plan)
                manifest["plan_sha256"] = "historical-plan-hash"
                manifest["status"] = "in_progress"
                manifest["tasks"][page.loaded_plan.tasks[1].task_id].update(
                    status="completed",
                    frames_captured=1,
                )
                _save_state(work_dir, manifest)

                # 引导采集中的实时预览仍允许按质量反馈调整曝光；此时没有
                # 正在写帧，不应被旧现场 hash 阻塞。
                page._guided_capture_active = True
                page.plan_table.item(1, 4).setText("750")
                self.app.processEvents()

                self.assertEqual(page.loaded_plan.tasks[1].config.exposure_us, 750.0)
                self.assertEqual(
                    load_capture_plan(root / "plans" / "capture.yaml").tasks[1].config.exposure_us,
                    750.0,
                )
                self.assertEqual(
                    load_document(work_dir / "dataset_manifest.yaml")["plan_sha256"],
                    "historical-plan-hash",
                )
            finally:
                page._guided_capture_active = False
                window.close(); self.app.processEvents()

    def test_capture_page_has_vertical_scrollbar_and_wheel_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            window, page = self._window_page(Path(temporary))
            try:
                window.resize(900, 500)
                window.show()
                window.stack.setCurrentIndex(2)
                self.app.processEvents()
                scroll_bar = page.scroll_area.verticalScrollBar()
                self.assertGreater(scroll_bar.maximum(), 0)
                scroll_bar.setValue(scroll_bar.maximum())
                self.assertEqual(scroll_bar.value(), scroll_bar.maximum())
            finally:
                window.close(); self.app.processEvents()

    def test_validation_and_nolaser_toggles_change_task_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            window, page = self._window_page(Path(temporary))
            try:
                page.include_validation.setChecked(False)
                self.assertTrue(page.generate_plan())
                self.assertEqual(len(page.loaded_plan.tasks), 54)
                enabled = page.recipe_table.table.cellWidget(1, 0)
                self.assertIsInstance(enabled, QCheckBox)
                enabled.setChecked(False)
                self.assertTrue(page.generate_plan())
                self.assertEqual(len(page.loaded_plan.tasks), 36)
            finally:
                window.close(); self.app.processEvents()

    def test_dirty_configuration_disables_start_until_regenerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            window, page = self._window_page(Path(temporary))
            try:
                self.assertTrue(page.generate_plan())
                self.assertTrue(page.start_button.isEnabled())
                page.fit_groups.setValue(2)
                self.assertFalse(page.start_button.isEnabled())
                self.assertTrue(page.generate_plan())
                self.assertTrue(page.start_button.isEnabled())
            finally:
                window.close(); self.app.processEvents()

    def test_batch_task_request_selects_next_task_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            window, page = self._window_page(Path(temporary))
            try:
                self.assertTrue(page.generate_plan())
                page._capture_worker = object()
                page._capture_cancel_event = threading.Event()
                next_task = page.loaded_plan.tasks[1]
                page._on_task_requested(next_task)
                self.assertEqual(page.plan_tasks.currentRow(), 1)
                self.assertEqual(page.plan_table.currentRow(), 1)
                self.assertEqual(page.camera_page.exposure.value(), 500.0)
                self.assertEqual(page.camera_page.quality_mode.currentData(), "laser")
                self.assertIn(next_task.task_id, page.live_camera.text())
            finally:
                page._capture_worker = None
                page._capture_cancel_event = None
                window.close(); self.app.processEvents()

    def test_yaml_round_trip_is_loaded_after_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window, page = self._window_page(root)
            try:
                page.fit_groups.setValue(2); page.validation_groups.setValue(1)
                self.assertTrue(page.generate_plan())
                saved = root / "plans" / "capture.yaml"
                self.assertTrue(saved.is_file())
                self.assertEqual(len(load_capture_plan(saved).tasks), 9)
                self.assertFalse((root / "dataset").exists())
            finally:
                window.close(); self.app.processEvents()

    def test_synthetic_capture_requires_confirmation_and_completes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window, page = self._window_page(root)
            try:
                page.fit_groups.setValue(2); page.validation_groups.setValue(1)
                self.assertTrue(page.generate_plan())
                page.start_capture(resume=False)
                self.assertIsNotNone(page._capture_worker)
                time.sleep(0.15)
                self.app.processEvents()
                self.assertTrue(page.ready_button.isEnabled())
                self.assertEqual(page.plan_tasks.currentRow(), 0)
                self.assertEqual(page.camera_page.exposure.value(), 35000.0)
                page.capture_task_button.click()
                self.assertFalse(page.ready_button.isEnabled())

                loop = QEventLoop()
                timer = QTimer(); timer.setInterval(10)
                ticks = [0]
                worker = page._capture_worker
                self.assertIsNotNone(worker)
                worker.signals.finished.connect(loop.quit)

                def drive_gate_and_finish():
                    ticks[0] += 1
                    if page.ready_button.isEnabled():
                        page.capture_task_button.click()
                    elif page._capture_worker is None and ticks[0] > 10:
                        loop.quit()

                timer.timeout.connect(drive_gate_and_finish)
                timer.start(); QTimer.singleShot(30_000, loop.quit)
                loop.exec(); timer.stop()
                self.assertIsNone(page._capture_worker)
                output = root / "dataset"
                self.assertTrue((output / "dataset_manifest.yaml").is_file())
                self.assertTrue((output / "frames.csv").is_file())
                manifest = load_document(output / "dataset_manifest.yaml")
                self.assertEqual(len(manifest["frames"]), 9)
                self.assertEqual(len(load_capture_plan(root / "plans" / "capture.yaml").tasks), 9)
            finally:
                page.cancel_capture()
                window.close(); self.app.processEvents()

    def test_start_button_guided_capture_keeps_live_preview_and_switches_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window, page = self._window_page(root)
            try:
                page.include_validation.setChecked(False)
                page.fit_groups.setValue(1)
                self.assertTrue(page.generate_plan())
                page.start_button.click()
                self.assertTrue(page._guided_capture_active)
                self.assertIsNone(page._capture_worker)

                expected_exposures = [35000.0, 500.0, 500.0]
                preview_identity = None
                for row, exposure in enumerate(expected_exposures):
                    deadline = time.monotonic() + 8.0
                    while time.monotonic() < deadline:
                        self.app.processEvents()
                        preview = page.camera_page.preview_thread
                        if (
                            page.plan_tasks.currentRow() == row
                            and page.capture_task_button.isEnabled()
                            and preview is not None
                            and preview.isRunning()
                        ):
                            break
                        time.sleep(0.02)
                    self.assertEqual(page.plan_tasks.currentRow(), row)
                    self.assertIsNotNone(page.camera_page.preview_thread)
                    if preview_identity is None:
                        preview_identity = page.camera_page.preview_thread
                    else:
                        self.assertIs(page.camera_page.preview_thread, preview_identity)
                    self.assertAlmostEqual(page.camera_page.preview_thread.config.exposure_us, exposure)
                    self.assertIn(page.loaded_plan.tasks[row].task_id, page.current_task.text())
                    page.capture_task_button.click()

                deadline = time.monotonic() + 8.0
                while page._guided_capture_active and time.monotonic() < deadline:
                    self.app.processEvents(); time.sleep(0.02)
                self.assertFalse(page._guided_capture_active)
                manifest = load_document(root / "dataset" / "dataset_manifest.yaml")
                self.assertEqual(len(manifest["frames"]), 3)
                self.assertEqual(
                    [frame["requested_camera"]["exposure_us"] for frame in manifest["frames"]],
                    [35000.0, 500.0, 500.0],
                )
            finally:
                page.cancel_capture()
                window.close(); self.app.processEvents()

    def test_guided_preview_does_not_discard_before_capture_click(self):
        with tempfile.TemporaryDirectory() as temporary:
            window, page = self._window_page(Path(temporary))
            try:
                page.include_validation.setChecked(False)
                page.fit_groups.setValue(1)
                settle = page.recipe_table.table.cellWidget(0, 7)
                settle.setValue(3)
                self.assertTrue(page.generate_plan())
                states = []
                page.camera_page.frame_ready.connect(
                    lambda _frame, quality: states.append(bool(quality.get("settling")))
                )

                page.start_button.click()
                deadline = time.monotonic() + 8.0
                while not page.capture_task_button.isEnabled() and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.02)

                self.assertFalse(any(states))
                self.assertTrue(page.capture_task_button.isEnabled())
            finally:
                page.cancel_capture()
                window.close(); self.app.processEvents()

    def test_one_guided_click_captures_all_task_frames_without_resettling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window, page = self._window_page(root)
            try:
                page.include_validation.setChecked(False)
                page.fit_groups.setValue(1)
                for row in (1, 2):
                    page.recipe_table.table.cellWidget(row, 0).setChecked(False)
                page.recipe_table.table.cellWidget(0, 6).setValue(4)
                page.recipe_table.table.cellWidget(0, 7).setValue(3)
                self.assertTrue(page.generate_plan())
                page.start_button.click()
                deadline = time.monotonic() + 8.0
                while not page.capture_task_button.isEnabled() and time.monotonic() < deadline:
                    self.app.processEvents(); time.sleep(0.02)
                self.assertTrue(page.capture_task_button.isEnabled())
                frame_before_click = page.camera_page.last_frame.camera_frame_number

                page.capture_task_button.click()
                deadline = time.monotonic() + 8.0
                while page._guided_capture_active and time.monotonic() < deadline:
                    self.app.processEvents(); time.sleep(0.02)

                self.assertFalse(page._guided_capture_active)
                manifest = load_document(root / "dataset" / "dataset_manifest.yaml")
                task_id = page.loaded_plan.tasks[0].task_id
                self.assertEqual(len(manifest["frames"]), 4)
                self.assertEqual(manifest["tasks"][task_id]["frames_captured"], 4)
                self.assertEqual(manifest["tasks"][task_id]["status"], "completed")
                frame_numbers = [int(frame["camera_frame_number"]) for frame in manifest["frames"]]
                self.assertGreaterEqual(frame_numbers[0] - int(frame_before_click), 4)
                self.assertEqual(
                    [right - left for left, right in zip(frame_numbers, frame_numbers[1:])],
                    [1, 1, 1],
                )
            finally:
                page.cancel_capture()
                window.close(); self.app.processEvents()


class CaptureTaskGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_gate_waits_for_approval_and_cancel_releases_waiter(self):
        gate = CaptureTaskGate()
        requested = threading.Event()
        gate.task_requested.connect(lambda _task: requested.set(), Qt.ConnectionType.DirectConnection)
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(gate.wait_for_task("task")))
        thread.start()
        self.assertTrue(requested.wait(1.0))
        self.assertTrue(thread.is_alive())
        gate.approve(); thread.join(1.0)
        self.assertEqual(result, [True])

        requested.clear(); result.clear()
        thread = threading.Thread(target=lambda: result.append(gate.wait_for_task("task-2")))
        thread.start(); self.assertTrue(requested.wait(1.0)); gate.cancel(); thread.join(1.0)
        self.assertEqual(result, [False])


if __name__ == "__main__":
    unittest.main()
