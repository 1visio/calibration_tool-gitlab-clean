import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from calibration_tool.calibration_run import CalibrationRun, CalibrationStage
from calibration_tool.gui.pages import ResultsPage
from calibration_tool.io_utils import dump_yaml


class GuiIntrinsicsResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _run(root: Path, *, with_intrinsics: bool = True, result_exists: bool = True) -> CalibrationRun:
        stages = []
        if with_intrinsics:
            result_file = root / "intrinsics" / "calibration_result.yaml"
            if result_exists:
                result_file.parent.mkdir(parents=True)
                result_file.write_text(
                    "camera_matrix:\n"
                    "- [100.0, 0.0, 50.0]\n"
                    "- [0.0, 101.0, 60.0]\n"
                    "- [0.0, 0.0, 1.0]\n"
                    "dist_coeffs: [-0.1, 0.02, 0.003, -0.004, 0.0]\n",
                    encoding="utf-8",
                )
                result_file.parent.joinpath("fit_images.csv").write_text(
                    "image,status,per_image_rmse,mean_euclidean_reprojection_error,error_stage\n"
                    "fit.tif,used,0.12,0.08,final_fit\n",
                    encoding="utf-8",
                )
                result_file.parent.joinpath("test_images.csv").write_text(
                    "image,status,per_image_rmse,mean_euclidean_reprojection_error,error_stage\n"
                    "test.tif,evaluated,0.15,0.09,frozen_intrinsics\n",
                    encoding="utf-8",
                )
            stages.append(
                CalibrationStage(
                    stage="intrinsics",
                    status="completed",
                    result_file=result_file,
                    metrics={
                        "fit_rmse_px": 0.1,
                        "test_rmse_px": 0.2,
                        "fit_image_count": 1,
                        "test_image_count": 1,
                    },
                )
            )
        stages.append(
            CalibrationStage(
                stage="laser_surface_models",
                status="completed",
                metrics={"model_type": "circular_cone"},
            )
        )
        return CalibrationRun(
            run_id="run-001",
            project_id="demo",
            workflow_path=root / "workflow.yaml",
            started_utc=datetime(2026, 8, 28, 3, 24, tzinfo=timezone.utc),
            completed_utc=datetime(2026, 8, 28, 3, 26, tzinfo=timezone.utc),
            status="completed",
            stages=stages,
            overall="pass",
        )

    def test_complete_run_shows_intrinsics_details_and_hides_debug_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            page = ResultsPage(QThreadPool())
            try:
                page.show_calibration_run(self._run(Path(temporary)))
                page.show()
                self.app.processEvents()

                self.assertEqual(
                    [page.result_tabs.tabText(index) for index in range(page.result_tabs.count())],
                    ["结果总览", "相机内参", "激光表面", "地面外参"],
                )
                self.assertEqual(page.intrinsics_fx.text(), "100.000000")
                self.assertEqual(page.intrinsics_fy.text(), "101.000000")
                self.assertEqual(page.intrinsics_cx.text(), "50.000000")
                self.assertEqual(page.intrinsics_cy.text(), "60.000000")
                self.assertEqual(page.intrinsics_k1.text(), "-0.100000")
                self.assertEqual(page.intrinsics_p2.text(), "-0.004000")
                self.assertEqual(page.intrinsics_detail_fit_rmse.text(), "0.1000 px")
                self.assertEqual(page.intrinsics_detail_test_images.text(), "1")
                self.assertEqual(page.intrinsics_reprojection_table.rowCount(), 2)
                self.assertEqual(
                    page.intrinsics_reprojection_table.item(1, 0).text(),
                    "test",
                )
                self.assertIn("已读取 2 条", page.intrinsics_reprojection_status.text())
                self.assertFalse(page.acceptance_button.isVisible())
                self.assertFalse(page.artifact_tree.isVisible())
                self.assertFalse(page.plot.isVisible())
            finally:
                page.close()

    def test_run_without_intrinsics_shows_not_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            page = ResultsPage(QThreadPool())
            try:
                page.show_calibration_run(self._run(Path(temporary), with_intrinsics=False))

                self.assertEqual(page.intrinsics_detail_status.text(), "未执行")
                self.assertEqual(page.intrinsics_fx.text(), "未执行")
                self.assertEqual(page.intrinsics_reprojection_status.text(), "未执行")
                self.assertEqual(page.intrinsics_reprojection_table.rowCount(), 0)
                self.assertEqual(page.laser_detail_status.text(), "已完成")
            finally:
                page.close()

    def test_missing_intrinsics_result_file_shows_state_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            page = ResultsPage(QThreadPool())
            try:
                page.show_calibration_run(
                    self._run(Path(temporary), result_exists=False)
                )

                self.assertIn("不存在", page.intrinsics_detail_status.text())
                self.assertEqual(page.intrinsics_fx.text(), "暂无")
                self.assertEqual(page.intrinsics_detail_fit_rmse.text(), "0.1000 px")
                self.assertEqual(page.intrinsics_reprojection_table.rowCount(), 0)
            finally:
                page.close()

    def test_legacy_workflow_history_loads_intrinsics_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_run = self._run(root)
            stage = fixture_run.stages[0]
            report = root / "projects" / "haikang" / "reports" / "workflow_20260828.yaml"
            dump_yaml(
                report,
                {
                    "workflow": "../plans/workflow_haikang.yaml",
                    "status": "completed",
                    "overall": "pass",
                    "stages": [stage.to_dict()],
                },
            )
            page = ResultsPage(QThreadPool())
            try:
                run = page.load_calibration_run_path(report, silent=True)

                self.assertIsNotNone(run)
                self.assertEqual(page.intrinsics_fx.text(), "100.000000")
                self.assertEqual(page.intrinsics_k3.text(), "0.000000")
                self.assertEqual(page.intrinsics_reprojection_table.rowCount(), 2)
                self.assertIn("已读取 2 条", page.intrinsics_reprojection_status.text())
            finally:
                page.close()


if __name__ == "__main__":
    unittest.main()
