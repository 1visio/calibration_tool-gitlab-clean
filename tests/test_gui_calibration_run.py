import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from calibration_tool.calibration_run import CalibrationRun, CalibrationStage
from calibration_tool.calibration_run_io import save_calibration_run
from calibration_tool.gui.pages import ResultsPage
from calibration_tool.gui.project import WizardProject
from calibration_tool.io_utils import dump_yaml


class GuiCalibrationRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _save_run(root: Path) -> Path:
        run_root = root / "runs" / "20260828"
        intrinsics_dir = run_root / "intrinsics"
        intrinsics_dir.mkdir(parents=True)
        laser_dir = run_root / "laser_model"
        laser_dir.mkdir(parents=True)
        stages = [
            CalibrationStage(
                stage="intrinsics",
                status="completed",
                output_dir=intrinsics_dir,
                result_file=intrinsics_dir / "result.yaml",
                metrics={
                    "fit_rmse_px": 0.12,
                    "test_rmse_px": 0.14,
                    "fit_image_count": 16,
                    "test_image_count": 6,
                },
            ),
            CalibrationStage(
                stage="laser_surface_models",
                status="completed",
                output_dir=laser_dir,
                result_file=laser_dir / "result.yaml",
                metrics={
                    "model_type": "circular_cone",
                    "validation_rmse_mm": 0.13,
                    "validation_p95_mm": 0.25,
                    "validation_valid_rate": 1.0,
                },
            ),
        ]
        run = CalibrationRun(
            run_id="20260828",
            project_id="demo",
            workflow_path=root / "workflow.yaml",
            started_utc=datetime(2026, 8, 28, 3, 24, tzinfo=timezone.utc),
            completed_utc=datetime(2026, 8, 28, 3, 26, tzinfo=timezone.utc),
            status="completed",
            laser_orientation="horizontal",
            stages=stages,
            gates=[],
            counts={"pass": 0, "warn": 0, "fail": 0},
            overall="pass",
        )
        return save_calibration_run(run, run_root / "calibration_run.yaml")

    @staticmethod
    def _project(root: Path, run_path: Path | None = None) -> WizardProject:
        camera = root / "camera.yaml"
        camera.write_text("backend: synthetic\n", encoding="utf-8")
        return WizardProject(
            "demo",
            root / "project",
            camera,
            last_calibration_run=run_path,
        )

    def test_project_manifest_is_loaded_when_project_is_reopened(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._save_run(root)
            project_path = self._project(root, manifest).save(root / "project.yaml")
            page = ResultsPage(QThreadPool())
            try:
                page.set_project(WizardProject.load(project_path))

                self.assertIsNotNone(page.current_run)
                self.assertEqual(page.current_run.run_id, "20260828")
                self.assertEqual(page.current_run_path, manifest.resolve())
                self.assertIn("当前 Calibration Run：20260828", page.run_status.text())
                self.assertEqual(page.overview_run_id.text(), "20260828")
                self.assertIn("2026-08-28T03:24:00+00:00", page.overview_time.text())
                self.assertEqual(page.overview_overall.text(), "通过")
                self.assertEqual(page.intrinsics_fit_images.text(), "16")
                self.assertEqual(page.laser_model.text(), "circular_cone")
                self.assertEqual(page.laser_valid_rate.text(), "100.0%")
                self.assertEqual(page.ground_status.text(), "未执行")
            finally:
                page.close()

    def test_project_without_manifest_shows_empty_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = ResultsPage(QThreadPool())
            try:
                page.set_project(self._project(root))

                self.assertIsNone(page.current_run)
                self.assertIsNone(page.current_run_path)
                self.assertEqual(page.run_status.text(), "当前项目暂无标定结果")
                self.assertEqual(page.overview_run_id.text(), "暂无")
                self.assertEqual(page.intrinsics_status.text(), "未执行")
            finally:
                page.close()

    def test_missing_manifest_does_not_block_project_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "runs" / "missing" / "calibration_run.yaml"
            page = ResultsPage(QThreadPool())
            try:
                page.set_project(self._project(root, missing))

                self.assertIsNone(page.current_run)
                self.assertEqual(page.current_run_path, missing.resolve())
                self.assertIn("文件不存在", page.run_status.text())
            finally:
                page.close()

    def test_corrupt_manifest_does_not_block_project_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corrupt = root / "calibration_run.yaml"
            corrupt.write_text("stages: [", encoding="utf-8")
            page = ResultsPage(QThreadPool())
            try:
                page.set_project(self._project(root, corrupt))

                self.assertIsNone(page.current_run)
                self.assertIn("标定结果读取失败", page.run_status.text())
            finally:
                page.close()

    def test_manual_load_of_legacy_workflow_report_uses_calibration_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "projects" / "haikang" / "reports" / "workflow_20260828.yaml"
            dump_yaml(
                report,
                {
                    "workflow": "../plans/workflow_haikang.yaml",
                    "status": "completed",
                    "overall": "pass",
                    "stages": [
                        {
                            "stage": "intrinsics",
                            "status": "completed",
                            "metrics": {"fit_rmse_px": 0.12},
                        }
                    ],
                },
            )
            page = ResultsPage(QThreadPool())
            try:
                run = page.load_calibration_run_path(report, silent=True)

                self.assertIsNotNone(run)
                self.assertEqual(page.current_run.run_id, "workflow_20260828")
                self.assertEqual(page.current_run_path, report.resolve())
                self.assertEqual(page.current_run.project_id, "haikang")
            finally:
                page.close()


if __name__ == "__main__":
    unittest.main()
