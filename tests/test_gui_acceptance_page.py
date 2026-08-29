import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from calibration_tool.gui.pages import ResultsPage
from calibration_tool.gui.project import WizardProject
from calibration_tool.io_utils import load_document


ROOT = Path(__file__).resolve().parents[1]


class GuiAcceptancePageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_page_prepares_project_default_plan_and_updates_after_workflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = WizardProject(
                "demo",
                root / "project",
                ROOT / "configs" / "camera.example.yaml",
            )
            page = ResultsPage(QThreadPool())
            try:
                page.set_project(project)
                plan_path = page.prepare_default_plan()
                self.assertIsNotNone(plan_path)
                assert plan_path is not None
                self.assertTrue(plan_path.is_file())
                self.assertEqual(Path(page.acceptance_plan.text()), plan_path)

                workflow_path = root / "workflow.yaml"
                workflow_path.write_text(
                    "schema_version: 1\n"
                    "report: reports/workflow_run.yaml\n"
                    "stages: []\n",
                    encoding="utf-8",
                )
                page.update_acceptance_from_workflow(
                    {
                        "workflow": str(workflow_path),
                        "status": "completed",
                        "stages": [],
                        "overall": "pass",
                        "counts": {"pass": 0, "warn": 0, "fail": 0},
                        "gates": [],
                    }
                )
                document = load_document(plan_path)
                self.assertIsNotNone(document["inputs"]["workflow_report"])
                self.assertEqual(
                    document["inputs"]["quality_reports"],
                    [document["inputs"]["workflow_report"]],
                )
            finally:
                page.deleteLater()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
