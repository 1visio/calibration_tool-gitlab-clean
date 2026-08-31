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


class GuiGroundExtrinsicsResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_result(root: Path, *, corrupt: bool = False) -> Path:
        result = root / "camera_ground_extrinsics.yaml"
        if corrupt:
            result.write_text("T_ground_from_camera: [", encoding="utf-8")
            return result
        result.write_text(
            "schema_version: 2\n"
            "method: checkerboard_plane_only\n"
            "T_ground_from_camera:\n"
            "- [1.0, 0.0, 0.0, 10.0]\n"
            "- [0.0, 1.0, 0.0, 20.0]\n"
            "- [0.0, 0.0, 1.0, 30.0]\n"
            "- [0.0, 0.0, 0.0, 1.0]\n"
            "quality_checks:\n"
            "  validation_zg: {rmse_mm: 0.06, p95_abs_mm: 0.12}\n"
            "validation:\n"
            "  detected_frame_count: 1\n"
            "  frames:\n"
            "  - image: valid.tif\n"
            "    detection_method: SB\n"
            "    pnp_rmse_px: 0.07\n"
            "    zg_rmse_mm: 0.05\n"
            "    zg_p95_abs_mm: 0.09\n"
            "    zg_max_abs_mm: 0.11\n"
            "    zg_std_mm: 0.03\n"
            "    zg_point_count: 88\n",
            encoding="utf-8",
        )
        return result

    @classmethod
    def _run(
        cls,
        root: Path,
        *,
        include_ground: bool = True,
        result_file: Path | None = None,
        corrupt: bool = False,
        plot: bool = False,
    ) -> CalibrationRun:
        stages: list[CalibrationStage] = []
        if include_ground:
            ground_dir = root / "ground_extrinsics"
            ground_dir.mkdir(parents=True, exist_ok=True)
            source = result_file or cls._write_result(ground_dir, corrupt=corrupt)
            if plot:
                (ground_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
                image = QImage(900, 540, QImage.Format.Format_RGB32)
                image.fill(0x00FFFFFF)
                image.save(str(ground_dir / "diagnostics" / "validation_zg_residual.png"))
            stages.append(
                CalibrationStage(
                    stage="ground_extrinsics_board_only",
                    status="completed",
                    output_dir=ground_dir,
                    result_file=source,
                    metrics={
                        "validation_rmse_mm": 0.06,
                        "validation_p95_mm": 0.12,
                    },
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

    def _show(self, run: CalibrationRun) -> ResultsPage:
        page = ResultsPage(QThreadPool())
        page.show_calibration_run(run)
        page.resize(800, 600)
        page.result_tabs.setCurrentWidget(page.ground_page)
        page.show()
        self.app.processEvents()
        return page

    def test_complete_run_shows_pose_counts_validation_rows_and_existing_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ground_dir = root / "ground_extrinsics"
            ground_dir.mkdir(parents=True)
            result = self._write_result(ground_dir)
            (ground_dir / "validation_frames.csv").write_text(
                "image,pnp_rmse_px,zg_rmse_mm,zg_p95_abs_mm,zg_max_abs_mm,zg_std_mm,zg_point_count\n"
                "valid.tif,0.07,0.05,0.09,0.11,0.03,88\n",
                encoding="utf-8",
            )
            (ground_dir / "fit_frames.csv").write_text(
                "image,zg_rmse_mm\nfit.tif,0.07\n", encoding="utf-8"
            )
            diagnostics = ground_dir / "diagnostics"
            diagnostics.mkdir()
            image = QImage(900, 540, QImage.Format.Format_RGB32)
            image.fill(0x00FFFFFF)
            image.save(str(diagnostics / "validation_zg_residual.png"))
            page = self._show(
                self._run(root, result_file=result, plot=True)
            )
            try:
                self.assertEqual(page.ground_detail_status.text(), "已完成")
                self.assertEqual(page.ground_detail_validation_rmse.text(), "0.0600 mm")
                self.assertEqual(page.ground_detail_validation_p95.text(), "0.1200 mm")
                self.assertEqual(page.ground_detail_fit_frames.text(), "1")
                self.assertEqual(page.ground_detail_validation_frames.text(), "1")
                self.assertEqual(page.ground_transform_name.text(), "T_ground_from_camera")
                self.assertEqual(page.ground_rotation_table.item(0, 0).text(), "1.000000")
                self.assertEqual(
                    page.ground_translation.text(), "[10.000000, 20.000000, 30.000000] mm"
                )
                self.assertEqual(page.ground_validation_table.rowCount(), 1)
                self.assertEqual(page.ground_validation_table.item(0, 3).text(), "0.050000")
                self.assertIn("1 条", page.ground_validation_status.text())
                self.assertFalse(page.ground_validation_error_plot.pixmap().isNull())
                self.assertIn("未自行换算", page.ground_euler_status.text())
            finally:
                page.close()

    def test_ground_page_uses_single_vertical_scroll_area(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = self._show(self._run(root, plot=True))
            try:
                self.assertIsInstance(page.ground_scroll_area, QScrollArea)
                self.assertIs(page.ground_scroll_area.widget(), page.ground_content)
                self.assertTrue(page.ground_scroll_area.widgetResizable())
                self.assertEqual(
                    page.ground_scroll_area.horizontalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                )
                self.assertEqual(
                    page.ground_scroll_area.verticalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAsNeeded,
                )
                self.assertIsInstance(page.ground_content.layout(), QVBoxLayout)
                self.assertEqual(page.ground_content.findChildren(QScrollArea), [])
                self.assertEqual(
                    page.ground_validation_table.verticalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                )
                self.assertGreater(page.ground_scroll_area.verticalScrollBar().maximum(), 0)
            finally:
                page.close()

    def test_missing_stage_keeps_ground_page_in_empty_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            page = self._show(self._run(Path(temporary), include_ground=False))
            try:
                self.assertEqual(page.ground_detail_status.text(), "未执行")
                self.assertEqual(page.ground_detail_fit_frames.text(), "未执行")
                self.assertEqual(page.ground_validation_table.rowCount(), 0)
                self.assertEqual(page.ground_validation_status.text(), "未执行")
            finally:
                page.close()

    def test_missing_or_corrupt_result_does_not_break_ground_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "ground_extrinsics" / "missing.yaml"
            page = self._show(self._run(root, result_file=missing))
            try:
                self.assertEqual(page.ground_detail_status.text(), "已完成")
                self.assertIn("不存在", page.ground_pose_status.text())
            finally:
                page.close()

            corrupt_root = root / "corrupt"
            corrupt_root.mkdir()
            corrupt_dir = corrupt_root / "ground_extrinsics"
            corrupt_dir.mkdir()
            corrupt = self._write_result(corrupt_dir, corrupt=True)
            corrupt_page = self._show(
                self._run(corrupt_root, result_file=corrupt, corrupt=True)
            )
            try:
                self.assertEqual(corrupt_page.ground_detail_status.text(), "已完成")
                self.assertIn("读取失败", corrupt_page.ground_pose_status.text())
            finally:
                corrupt_page.close()


if __name__ == "__main__":
    unittest.main()
