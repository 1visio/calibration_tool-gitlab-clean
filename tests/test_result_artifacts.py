import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from calibration_tool.gui.pages import ResultsPage
from calibration_tool.gui.result_artifacts import discover_result_artifacts


class ResultArtifactDiscoveryTests(unittest.TestCase):
    def test_discovers_pose_previews_and_validation_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previews = root / "previews"
            previews.mkdir()
            for name in (
                "train_001_extraction.png",
                "validation_019_extraction.png",
                "validation_error_vs_u.png",
                "validation_error_vs_v.png",
                "validation_error_vs_depth.png",
            ):
                cv2.imwrite(str(previews / name), np.zeros((8, 10), dtype=np.uint8))
            reprojection = root / "fit" / "reprojection"
            reprojection.mkdir(parents=True)
            cv2.imwrite(str(reprojection / "chess 001.png"), np.zeros((8, 10), dtype=np.uint8))
            (root / "per_image_metrics.csv").write_text(
                "split,pose_id,pnp_rmse_px,quality_warnings\nfit,001,0.2,blur\n",
                encoding="utf-8",
            )
            records = discover_result_artifacts(
                {"stages": [{"stage": "laser_surface_models", "output_dir": str(root)}]}
            )
            by_name = {record.display_path.name: record for record in records}
            self.assertEqual(by_name["train_001_extraction.png"].split, "fit")
            self.assertEqual(by_name["train_001_extraction.png"].pose_id, "001")
            self.assertEqual(by_name["validation_019_extraction.png"].split, "validation")
            self.assertEqual(by_name["validation_019_extraction.png"].pose_id, "019")
            self.assertEqual(by_name["validation_error_vs_u.png"].artifact_type, "validation_error_vs_u")
            self.assertEqual(by_name["chess 001.png"].artifact_type, "reprojection_preview")
            self.assertEqual(by_name["train_001_extraction.png"].status, "warning")
            self.assertEqual(
                sum(name.startswith("validation_error_vs_") for name in by_name),
                3,
            )


class ResultArtifactPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_tree_groups_and_damaged_image_are_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previews = root / "previews"
            previews.mkdir()
            (previews / "train_001_extraction.png").write_bytes(b"not-an-image")
            page = ResultsPage(QThreadPool())
            try:
                page.show_result({"stages": [{"stage": "intrinsics", "output_dir": str(root)}]})
                self.assertEqual(page.artifact_tree.topLevelItemCount(), 1)
                self.assertEqual(page.artifact_tree.topLevelItem(0).text(0), "intrinsics")
                page._display_artifact(0)
                self.assertIn("损坏", page.image_status.text())
            finally:
                page.close()

    def test_empty_result_directory_has_friendly_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            page = ResultsPage(QThreadPool())
            try:
                page.show_result({"stages": [{"stage": "intrinsics", "output_dir": temporary}]})
                self.assertIn("没有发现", page.artifact_tree.topLevelItem(0).text(0))
                self.assertIn("没有结果", page.image_status.text())
            finally:
                page.close()

    def test_missing_explicit_image_is_recorded_without_crashing(self):
        missing = Path(tempfile.gettempdir()) / "calibration-tool-missing-result.png"
        missing.unlink(missing_ok=True)
        records = discover_result_artifacts({"artifacts": [{"path": str(missing), "type": "preview"}]})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "missing")

    def test_old_acceptance_report_with_nested_workflow_is_discoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "previews").mkdir()
            image = root / "previews" / "validation_019_extraction.png"
            cv2.imwrite(str(image), np.zeros((4, 4), dtype=np.uint8))
            records = discover_result_artifacts(
                {
                    "workflow": {
                        "workflow": str(root / "workflow.yaml"),
                        "stages": [{"stage": "laser", "output_dir": str(root)}],
                    }
                }
            )
            self.assertIn(image.resolve(), {record.display_path for record in records})


if __name__ == "__main__":
    unittest.main()
