import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QScrollArea, QVBoxLayout

from calibration_tool.calibration_run import CalibrationRun, CalibrationStage
from calibration_tool.gui.pages import ResultsPage


class GuiLaserSurfaceResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_result(path: Path, *, corrupt: bool = False) -> None:
        if corrupt:
            path.write_text("model_type: [", encoding="utf-8")
            return
        path.write_text(
            "model_type: circular_cone\n"
            "model_selection:\n"
            "  default_model: circular_cone\n"
            "metrics:\n"
            "  train: {board_rmse_mm: 0.12, board_p95_abs_mm: 0.22, board_max_abs_mm: 0.50, valid_rate: 0.99}\n"
            "  validation: {board_rmse_mm: 0.14, board_p95_abs_mm: 0.25, board_max_abs_mm: 0.60, valid_rate: 0.98}\n"
            "axis_unit_camera: [0.0, 1.0, 0.0]\n"
            "apex_camera_mm: [1.0, 2.0, 3.0]\n"
            "half_apex_angle_deg: 20.0\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_comparison(root: Path, models: tuple[str, ...]) -> None:
        values = {
            "global_plane": ("0.30", "0.40", "0.80", "0.90"),
            "quadratic_graph": ("0.20", "0.30", "0.70", "0.95"),
            "circular_cone": ("0.12", "0.25", "0.60", "0.98"),
        }
        lines = [
            "model,split,board_rmse_mm,board_p95_abs_mm,board_max_abs_mm,valid_rate"
        ]
        for model in models:
            rmse, p95, max_value, rate = values[model]
            lines.extend(
                (
                    f"{model},train,{rmse},{p95},{max_value},{rate}",
                    f"{model},validation,{rmse},{p95},{max_value},{rate}",
                )
            )
        (root / "model_comparison.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _write_models(root: Path, models: tuple[str, ...]) -> None:
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        documents = {
            "global_plane": "model_type: global_plane\nplane_abcd: [1.0, 2.0, 3.0, 4.0]\n",
            "quadratic_graph": "model_type: quadratic_graph\ncoefficients: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]\n",
            "circular_cone": (
                "model_type: circular_cone\n"
                "axis_unit_camera: [0.0, 1.0, 0.0]\n"
                "apex_camera_mm: [1.0, 2.0, 3.0]\n"
                "half_apex_angle_deg: 20.0\n"
            ),
        }
        for model in models:
            (model_dir / f"{model}.yaml").write_text(documents[model], encoding="utf-8")

    @classmethod
    def _run(
        cls,
        root: Path,
        *,
        comparison_models: tuple[str, ...] | None = None,
        model_files: tuple[str, ...] | None = None,
        corrupt: bool = False,
        plots: bool = False,
    ) -> CalibrationRun:
        laser_dir = root / "laser_model"
        laser_dir.mkdir(parents=True, exist_ok=True)
        result_file = laser_dir / "laser_model.yaml"
        cls._write_result(result_file, corrupt=corrupt)
        if comparison_models is not None:
            cls._write_comparison(laser_dir, comparison_models)
        if model_files is not None:
            cls._write_models(laser_dir, model_files)
        if plots:
            for name in (
                "validation_error_vs_u.png",
                "validation_error_vs_v.png",
                "validation_error_vs_depth.png",
            ):
                image = QImage(900, 540, QImage.Format.Format_RGB32)
                image.fill(0x00FFFFFF)
                self_path = laser_dir / name
                image.save(str(self_path))
        return CalibrationRun(
            run_id="run-001",
            project_id="demo",
            workflow_path=root / "workflow.yaml",
            started_utc=datetime(2026, 8, 28, 3, 24, tzinfo=timezone.utc),
            completed_utc=datetime(2026, 8, 28, 3, 26, tzinfo=timezone.utc),
            status="completed",
            laser_orientation="horizontal",
            stages=[
                CalibrationStage(
                    stage="laser_surface_models",
                    status="completed",
                    output_dir=laser_dir,
                    result_file=result_file,
                    metrics={
                        "model_type": "circular_cone",
                        "validation_rmse_mm": 0.14,
                        "validation_p95_mm": 0.25,
                        "validation_valid_rate": 0.98,
                    },
                )
            ],
            overall="pass",
        )

    def _show(self, run: CalibrationRun) -> ResultsPage:
        page = ResultsPage(QThreadPool())
        page.show_calibration_run(run)
        self.app.processEvents()
        return page

    def test_complete_run_shows_comparison_selected_model_parameters_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._run(
                Path(temporary),
                comparison_models=("global_plane", "quadratic_graph", "circular_cone"),
                model_files=("global_plane", "quadratic_graph", "circular_cone"),
                plots=True,
            )
            page = self._show(run)
            try:
                self.assertEqual(page.laser_detail_model.text(), "circular_cone")
                self.assertEqual(page.laser_detail_validation_rmse.text(), "0.1400 mm")
                self.assertEqual(page.laser_detail_validation_p95.text(), "0.2500 mm")
                self.assertEqual(page.laser_detail_valid_rate.text(), "98.0%")
                self.assertEqual(page.laser_model_comparison_table.rowCount(), 3)
                self.assertEqual(
                    page.laser_model_comparison_table.item(2, 6).text(), "当前采用"
                )
                self.assertEqual(
                    page.laser_model_comparison_table.item(1, 2).text(), "0.2000"
                )
                self.assertIn("3/3", page.laser_model_comparison_status.text())
                self.assertIn("3 张", page.laser_error_plots_status.text())
                self.assertFalse(page.laser_parameters_panel.isVisible())
                page.laser_parameters_toggle.setChecked(True)
                self.app.processEvents()
                self.assertFalse(page.laser_parameters_panel.isHidden())
                self.assertGreater(page.laser_model_parameters_table.rowCount(), 0)
                self.assertFalse(page.laser_error_vs_u.pixmap().isNull())
            finally:
                page.close()

    def test_laser_page_uses_single_vertical_scroll_and_natural_content_height(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._run(
                Path(temporary),
                comparison_models=("global_plane", "quadratic_graph", "circular_cone"),
                model_files=("global_plane", "quadratic_graph", "circular_cone"),
                plots=True,
            )
            page = self._show(run)
            try:
                page.resize(800, 600)
                page.result_tabs.setCurrentWidget(page.laser_page)
                page.show()
                self.app.processEvents()

                self.assertIsInstance(page.laser_scroll_area, QScrollArea)
                self.assertIs(page.laser_scroll_area.widget(), page.laser_content)
                self.assertTrue(page.laser_scroll_area.widgetResizable())
                self.assertEqual(
                    page.laser_scroll_area.horizontalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                )
                self.assertEqual(
                    page.laser_scroll_area.verticalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAsNeeded,
                )
                self.assertIsInstance(page.laser_content.layout(), QVBoxLayout)
                self.assertEqual(page.laser_content.findChildren(QScrollArea), [])
                self.assertGreaterEqual(
                    page.laser_error_vs_u.minimumHeight(), 260
                )
                self.assertLess(
                    page.laser_error_vs_u_card.y(), page.laser_error_vs_v_card.y()
                )
                self.assertLess(
                    page.laser_error_vs_v_card.y(), page.laser_error_vs_depth_card.y()
                )
                self.assertEqual(
                    page.laser_model_parameters_table.verticalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                )
                page.laser_parameters_toggle.setChecked(True)
                self.app.processEvents()
                table = page.laser_model_parameters_table
                expected_height = (
                    table.horizontalHeader().height()
                    + sum(table.rowHeight(index) for index in range(table.rowCount()))
                    + table.frameWidth() * 2
                )
                self.assertGreaterEqual(table.height(), expected_height)
                self.assertGreater(page.laser_scroll_area.verticalScrollBar().maximum(), 0)

                narrow_width = page.laser_error_vs_u.pixmap().width()
                page.resize(1200, 600)
                self.app.processEvents()
                self.assertGreater(page.laser_error_vs_u.pixmap().width(), narrow_width)
            finally:
                page.close()

    def test_missing_comparison_csv_keeps_page_usable(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._run(Path(temporary))
            page = self._show(run)
            try:
                self.assertIn("model_comparison.csv", page.laser_model_comparison_status.text())
                self.assertEqual(
                    page.laser_model_comparison_table.item(2, 2).text(), "0.1400"
                )
                self.assertEqual(
                    page.laser_model_comparison_table.item(0, 6).text(), "未找到"
                )
            finally:
                page.close()

    def test_partial_models_are_marked_without_breaking_comparison_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._run(
                Path(temporary),
                comparison_models=("global_plane", "circular_cone"),
                model_files=("global_plane", "circular_cone"),
            )
            page = self._show(run)
            try:
                self.assertEqual(
                    page.laser_model_comparison_table.item(1, 6).text(), "未找到"
                )
                self.assertEqual(
                    page.laser_model_comparison_table.item(2, 6).text(), "当前采用"
                )
            finally:
                page.close()

    def test_corrupt_laser_result_keeps_stage_summary_and_shows_detail_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._run(Path(temporary), corrupt=True)
            page = self._show(run)
            try:
                self.assertEqual(page.laser_detail_status.text(), "已完成")
                self.assertIn("读取失败", page.laser_model_comparison_status.text())
                self.assertEqual(
                    page.laser_model_comparison_table.item(2, 6).text(), "当前采用"
                )
            finally:
                page.close()


if __name__ == "__main__":
    unittest.main()
