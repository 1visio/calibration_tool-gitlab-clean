"""统一 workflow 使用的三种激光面模型标定入口。

实际拟合实现位于 calibration_tool/scripts，保留一个位于 calibration/src
的薄入口是为了让既有 ``calibration_src`` stage 注册机制无需改变。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence


FIT_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "calibration_tool"
    / "scripts"
    / "fit_laser_models_from_triplets.py"
)


def _load_fit_script() -> ModuleType:
    if not FIT_SCRIPT.is_file():
        raise FileNotFoundError(f"三模型拟合脚本不存在：{FIT_SCRIPT}")
    spec = importlib.util.spec_from_file_location("calibration_tool_fit_laser_models", FIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载三模型拟合脚本：{FIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    return int(_load_fit_script().main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
