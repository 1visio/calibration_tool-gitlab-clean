import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from calibration_tool.calibration_run_io import load_calibration_run
from calibration_tool.gui.main_window import CalibrationWizardWindow
from calibration_tool.gui.project import WizardProject


ROOT = Path(__file__).resolve().parents[1]


class GuiWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_window_contains_five_functional_steps(self):
        window = CalibrationWizardWindow(
            default_camera_config=ROOT / "configs" / "camera.example.yaml"
        )
        default_plan = ROOT / "projects" / "default" / "plans" / "acceptance_plan.yaml"
        had_default_plan = default_plan.is_file()
        try:
            self.assertEqual(window.stack.count(), 5)
            self.assertEqual(window.steps.count(), 5)
            self.assertEqual(window.steps.item(4).text(), "5  标定结果")
            self.assertIsNotNone(window.project)
            self.assertEqual(window.camera_page.runtime["backend"], "synthetic")
            window.calibration_page.workflow.setText(str(ROOT / "configs" / "workflow.example.yaml"))
            self.assertTrue(window.calibration_page.refresh_plan())
            self.assertEqual(window.calibration_page.stage_table.rowCount(), 5)
            window.steps.setCurrentRow(4)
            self.assertEqual(window.stack.currentIndex(), 4)
        finally:
            window.close()
            self.app.processEvents()
            if not had_default_plan:
                default_plan.unlink(missing_ok=True)

    def test_completed_workflow_creates_manifest_and_records_project_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            camera = root / "camera.yaml"
            camera.write_text("backend: synthetic\n", encoding="utf-8")
            project = WizardProject("demo", root / "project", camera)
            project_path = project.save(root / "project.yaml")
            run_root = root / "runs" / "20260828"
            workflow_path = root / "workflow.yaml"
            report = {
                "schema_version": 1,
                "workflow": str(workflow_path),
                "started_utc": "2026-08-28T03:24:38+00:00",
                "completed_utc": "2026-08-28T03:26:27+00:00",
                "status": "completed",
                "laser": {"orientation": "horizontal"},
                "stages": [
                    {
                        "stage": "intrinsics",
                        "status": "completed",
                        "output_dir": str(run_root / "intrinsics"),
                        "result_file": str(run_root / "intrinsics" / "calibration_result.yaml"),
                        "metrics": {"test_rmse_px": 0.1},
                        "quality_gates": [],
                    },
                    {
                        "stage": "laser_surface_models",
                        "status": "completed",
                        "output_dir": str(run_root / "laser_model"),
                        "result_file": str(run_root / "laser_model" / "laser_model.yaml"),
                        "metrics": {"validation_rmse_mm": 0.1},
                        "quality_gates": [],
                    },
                ],
                "gates": [],
                "counts": {"pass": 0, "warn": 0, "fail": 0},
                "overall": "pass",
            }
            window = CalibrationWizardWindow(
                project_path=project_path,
                default_camera_config=ROOT / "configs" / "camera.example.yaml",
            )
            try:
                window.capture_page.capture_finished.emit(
                    SimpleNamespace(
                        capture_artifacts={
                            "dataset_root": str(root / "dataset"),
                            "fit_dir": str(root / "dataset" / "fit"),
                            "validation_dir": str(root / "dataset" / "validation"),
                        }
                    )
                )
                self.app.processEvents()
                self.assertEqual(
                    window.calibration_page.capture_artifacts["dataset_root"],
                    str(root / "dataset"),
                )
                self.assertEqual(
                    window.results_page.capture_artifacts["validation_dir"],
                    str(root / "dataset" / "validation"),
                )

                with patch.object(window.results_page, "update_acceptance_from_workflow"), patch.object(
                    window.results_page, "show_result"
                ):
                    window._workflow_finished(report)

                manifest_path = run_root / "calibration_run.yaml"
                self.assertTrue(manifest_path.is_file())
                self.assertEqual(window.project.last_calibration_run, manifest_path.resolve())
                self.assertIsNotNone(window.results_page.current_run)
                self.assertEqual(window.results_page.current_run_path, manifest_path.resolve())
                self.assertEqual(
                    WizardProject.load(project_path).last_calibration_run,
                    manifest_path.resolve(),
                )
                self.assertEqual(load_calibration_run(manifest_path).run_id, "20260828")
                self.assertEqual(load_calibration_run(manifest_path).project_id, "demo")
                self.assertEqual(window.steps.currentRow(), 4)

                reopened = CalibrationWizardWindow(
                    project_path=project_path,
                    default_camera_config=ROOT / "configs" / "camera.example.yaml",
                )
                try:
                    self.assertIsNotNone(reopened.results_page.current_run)
                    self.assertEqual(reopened.results_page.current_run.run_id, "20260828")
                    self.assertEqual(reopened.results_page.current_run_path, manifest_path.resolve())
                finally:
                    reopened.close()
            finally:
                window.close()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
