import tempfile
import unittest
from pathlib import Path

from calibration_tool.gui.workflow_inputs import (
    build_workflow_update_preview,
    save_workflow_update,
)
from calibration_tool.io_utils import dump_yaml, load_document


class WorkflowInputUpdateTests(unittest.TestCase):
    def test_preview_and_atomic_save_preserve_algorithm_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow = root / "workflow.yaml"
            fit = root / "dataset" / "fit"
            validation = root / "dataset" / "validation"
            fit.mkdir(parents=True)
            validation.mkdir()
            dump_yaml(
                workflow,
                {
                    "schema_version": 1,
                    "calibration_src": ".",
                    "stages": [
                        {
                            "name": "intrinsics",
                            "enabled": True,
                            "options": {
                                "fit_dir": "old/fit",
                                "test_dir": "old/validation",
                                "pattern_cols": 11,
                                "square_size_mm": 20,
                            },
                        }
                    ],
                },
            )
            preview = build_workflow_update_preview(
                workflow,
                {"fit_dir": str(fit), "validation_dir": str(validation)},
            )
            self.assertTrue(preview.changed)
            self.assertEqual(preview.updated["stages"][0]["options"]["pattern_cols"], 11)
            self.assertEqual(preview.updated["stages"][0]["options"]["square_size_mm"], 20)
            backup = save_workflow_update(preview)
            self.assertIsNotNone(backup)
            self.assertTrue(Path(backup).is_file())
            loaded = load_document(workflow)
            self.assertEqual(loaded["stages"][0]["options"]["fit_dir"], "dataset/fit")
            self.assertEqual(loaded["stages"][0]["options"]["test_dir"], "dataset/validation")
            self.assertEqual(load_document(backup)["stages"][0]["options"]["fit_dir"], "old/fit")

    def test_missing_validation_directory_does_not_block_fit_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow = root / "workflow.yaml"
            fit = root / "fit"
            fit.mkdir()
            dump_yaml(
                workflow,
                {
                    "schema_version": 1,
                    "stages": [{"name": "intrinsics", "options": {"fit_dir": "old", "test_dir": "old-test"}}],
                },
            )
            preview = build_workflow_update_preview(
                workflow,
                {"fit_dir": str(fit), "validation_dir": str(root / "missing")},
            )
            self.assertEqual(len(preview.changes), 1)
            self.assertEqual(preview.changes[0].option, "fit_dir")


if __name__ == "__main__":
    unittest.main()
