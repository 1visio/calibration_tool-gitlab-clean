import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtWidgets import QApplication, QScrollArea

from calibration_tool.calibration_run import CalibrationRun, CalibrationStage
from calibration_tool.gui.pages import ResultsPage


class GuiResultsPageRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _complete_run(root: Path) -> CalibrationRun:
        return CalibrationRun(
            run_id="20260828",
            project_id="demo",
            workflow_path=root / "workflow.yaml",
            started_utc=datetime(2026, 8, 28, 3, 24, tzinfo=timezone.utc),
            completed_utc=datetime(2026, 8, 28, 3, 26, tzinfo=timezone.utc),
            status="completed",
            laser_orientation="horizontal",
            stages=[
                CalibrationStage(
                    stage="intrinsics",
                    status="completed",
                    metrics={"fit_rmse_px": 0.12, "test_rmse_px": 0.14},
                ),
                CalibrationStage(
                    stage="laser_surface_models",
                    status="completed",
                    metrics={
                        "model_type": "circular_cone",
                        "validation_rmse_mm": 0.13,
                        "validation_p95_mm": 0.25,
                        "validation_valid_rate": 1.0,
                    },
                ),
                CalibrationStage(
                    stage="ground_extrinsics_board_only",
                    status="completed",
                    metrics={
                        "validation_rmse_mm": 0.06,
                        "validation_p95_mm": 0.12,
                    },
                ),
            ],
            overall="pass",
        )

    def test_result_tabs_use_shared_layout_and_status_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            page = ResultsPage(QThreadPool())
            try:
                page.show_calibration_run(self._complete_run(Path(temporary)))
                page.resize(900, 600)
                page.show()
                self.app.processEvents()

                self.assertEqual(page.overview_status.text(), "已完成")
                self.assertEqual(page.overview_overall.text(), "通过")
                self.assertEqual(page.intrinsics_status.text(), "已完成")
                self.assertEqual(page.laser_status.text(), "已完成")
                self.assertEqual(page.ground_status.text(), "已完成")
                self.assertIn("overall：通过", page.run_status.text())
                self.assertLess(page.intrinsics_reprojection_table.height(), 100)

                margins = (12, 12, 12, 12)
                for index in range(page.result_tabs.count()):
                    layout = page.result_tabs.widget(index).layout()
                    self.assertIsNotNone(layout)
                    actual = layout.contentsMargins()
                    self.assertEqual(
                        (actual.left(), actual.top(), actual.right(), actual.bottom()),
                        margins,
                    )
                    self.assertEqual(layout.spacing(), 10)

                self.assertIsInstance(page.laser_scroll_area, QScrollArea)
                self.assertIsInstance(page.ground_scroll_area, QScrollArea)
                self.assertEqual(
                    page.intrinsics_reprojection_table.horizontalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                )
                self.assertFalse(page.acceptance_button.isVisible())
                self.assertFalse(page.artifact_tree.isVisible())
                self.assertFalse(page.plot.isVisible())
            finally:
                page.close()


if __name__ == "__main__":
    unittest.main()
