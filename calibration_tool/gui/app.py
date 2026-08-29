from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from PySide6.QtWidgets import QApplication

from .main_window import CalibrationWizardWindow


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def launch_gui(
    *,
    project: Path | None = None,
    simulate: bool = False,
    camera_channel: str | None = None,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    camera_config = PACKAGE_ROOT / "configs" / "camera_channels.example.yaml"
    selected_channel = "synthetic" if simulate else camera_channel
    window = CalibrationWizardWindow(
        project_path=project,
        default_camera_config=camera_config,
        default_camera_channel=selected_channel,
    )
    window.show()
    return app.exec()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="线激光 PySide6 标定向导")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--channel", help="相机通道名称，例如 hikrobot 或 daheng")
    args = parser.parse_args(argv)
    return launch_gui(
        project=args.project,
        simulate=args.simulate,
        camera_channel=args.channel,
    )


if __name__ == "__main__":
    raise SystemExit(main())
