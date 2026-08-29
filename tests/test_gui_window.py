import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from calibration_tool.gui.main_window import CalibrationWizardWindow


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


if __name__ == "__main__":
    unittest.main()
