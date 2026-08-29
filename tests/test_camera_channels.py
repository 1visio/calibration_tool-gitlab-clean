import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from calibration_tool.camera import (
    load_camera_channel_registry,
    load_camera_config,
)
from calibration_tool.errors import ConfigError
from calibration_tool.gui.pages import ProjectPage
from calibration_tool.gui.project import WizardProject


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "camera_channels.example.yaml"


class CameraChannelConfigTests(unittest.TestCase):
    def test_registry_selects_both_real_camera_systems(self):
        registry = load_camera_channel_registry(REGISTRY)
        self.assertEqual(registry.default_channel, "hikrobot")
        self.assertEqual(
            [channel.name for channel in registry.channels],
            ["hikrobot", "daheng", "synthetic"],
        )

        hikrobot = load_camera_config(REGISTRY, channel="hikrobot")
        daheng = load_camera_config(REGISTRY, channel="daheng")
        self.assertEqual(hikrobot["backend"], "mvs")
        self.assertEqual(hikrobot["laser"].orientation, "horizontal")
        self.assertEqual(daheng["backend"], "daheng")
        self.assertEqual(daheng["laser"].orientation, "vertical")
        self.assertEqual(daheng["workflow_plan"].name, "workflow.example.yaml")

    def test_unknown_channel_and_channel_on_legacy_config_are_rejected(self):
        with self.assertRaisesRegex(ConfigError, "未知相机通道"):
            load_camera_config(REGISTRY, channel="missing")
        with self.assertRaisesRegex(ConfigError, "不是通道注册表"):
            load_camera_config(
                ROOT / "configs" / "camera.example.yaml",
                channel="synthetic",
            )

    def test_wizard_project_round_trip_preserves_selected_channel(self):
        project = WizardProject.load(ROOT / "configs" / "wizard_project.example.yaml")
        self.assertEqual(project.camera_channel, "hikrobot")
        with tempfile.TemporaryDirectory() as temporary:
            saved = project.save(Path(temporary) / "project.yaml")
            loaded = WizardProject.load(saved)
            self.assertEqual(loaded.camera_channel, "hikrobot")
            self.assertEqual(loaded.camera_config, REGISTRY.resolve())


class CameraChannelProjectPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_one_selector_switches_camera_laser_and_workflow(self):
        page = ProjectPage(REGISTRY, "hikrobot")
        try:
            index = page.camera_channel.findData("daheng")
            self.assertGreaterEqual(index, 0)
            page.camera_channel.setCurrentIndex(index)
            self.assertEqual(page.laser_orientation.currentText(), "vertical")
            self.assertTrue(page.workflow_plan.text().endswith("workflow.example.yaml"))
            project = page._from_fields()
            self.assertEqual(project.camera_channel, "daheng")
        finally:
            page.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
