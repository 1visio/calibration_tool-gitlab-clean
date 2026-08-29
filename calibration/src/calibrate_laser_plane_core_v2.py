#!/usr/bin/env python3
"""Laser-plane calibration with 2-D Steger subpixel ridge extraction.

The geometric calibration, quality filters and output generation are reused
from ``calibrate_laser_plane_core.py``.  Only the laser-centre extractor is
replaced, which keeps V1/V2 comparisons attributable to one algorithmic change.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import math
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import yaml
import realtime_steger as realtime
from scipy.ndimage import gaussian_filter


LOGGER = logging.getLogger("laser_plane_core_v2")
COMMON_MODULE_NAME = "calibrate_laser_plane_opencv"
DEFAULT_CONFIG = Path(__file__).with_name("laser_plane_core_config.yaml")
PLOTLY_AVAILABLE = True


def _load_common_dependency() -> None:
    """Load the common helper, including the current source-less workspace case."""
    try:
        __import__(COMMON_MODULE_NAME)
        return
    except ModuleNotFoundError as exc:
        if exc.name != COMMON_MODULE_NAME:
            raise

    cache_tag = sys.implementation.cache_tag
    pyc_path = Path(__file__).with_name("__pycache__") / (
        f"{COMMON_MODULE_NAME}.{cache_tag}.pyc"
    )
    if not pyc_path.is_file():
        raise ModuleNotFoundError(
            f"缺少 {COMMON_MODULE_NAME}.py，且没有兼容当前 Python 的缓存文件：{pyc_path}"
        )
    spec = importlib.util.spec_from_file_location(COMMON_MODULE_NAME, pyc_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"无法加载公共标定模块缓存：{pyc_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[COMMON_MODULE_NAME] = module
    spec.loader.exec_module(module)
    LOGGER.warning(
        "公共源码 %s.py 缺失；临时使用版本相关缓存 %s。请尽快恢复源码。",
        COMMON_MODULE_NAME,
        pyc_path,
    )


def _import_core() -> Any:
    global PLOTLY_AVAILABLE
    if "calibrate_laser_plane_core" in sys.modules:
        return sys.modules["calibrate_laser_plane_core"]
    _load_common_dependency()
    if importlib.util.find_spec("plotly") is None:
        PLOTLY_AVAILABLE = False
        plotly_module = types.ModuleType("plotly")
        graph_objects_module = types.ModuleType("plotly.graph_objects")
        plotly_module.graph_objects = graph_objects_module
        sys.modules.setdefault("plotly", plotly_module)
        sys.modules.setdefault("plotly.graph_objects", graph_objects_module)
    core = importlib.import_module("calibrate_laser_plane_core")
    if not PLOTLY_AVAILABLE:
        core.save_interactive_feature_plane = lambda *args, **kwargs: None
        LOGGER.warning("当前环境缺少 plotly；将跳过可旋转 3D HTML，不影响拟合和数值结果。")
    return core


@dataclass(frozen=True)
class StegerSettings:
    sigma: float = 1.5
    threshold: float = 30.0
    deriv_thresh: float = 0.5
    roi_margin: int = 120
    roi_max_height: int = 512
    scan_axis: str = "column"


def unit_interval(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("必须位于 [0, 1]")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0.0:
        raise argparse.ArgumentTypeError("必须大于或等于 0")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return result


def nested(config: Mapping[str, Any], key: str) -> Any:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"配置缺少字段：{key}")
        value = value[part]
    return value


def build_parser() -> argparse.ArgumentParser:
    core = _import_core()
    parser = core.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--steger-config",
        type=Path,
        default=None,
        help=f"统一实时 Steger 配置；默认 {realtime.DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--steger-sigma",
        type=positive_float,
        default=None,
        help="覆盖统一配置中的高斯导数尺度（像素）",
    )
    parser.add_argument(
        "--steger-threshold",
        type=nonnegative_float,
        default=None,
        help="覆盖实时 Steger 的原始灰度阈值",
    )
    parser.add_argument(
        "--steger-deriv-thresh",
        type=positive_float,
        default=None,
        help="覆盖实时 Steger 的法向二阶导数阈值",
    )
    parser.add_argument(
        "--steger-roi-margin",
        type=int,
        default=None,
        help="覆盖自动条纹带扩展像素",
    )
    parser.add_argument(
        "--steger-roi-max-height",
        type=int,
        default=None,
        help="覆盖 Hessian 计算带最大高度",
    )
    parser.add_argument(
        "--steger-scan-axis",
        choices=("column", "row"),
        default=None,
    )
    return parser


# 下面两组旧的逐点函数保留给历史脚本导入；本 V2 的 active path 在
# ``extract_centres`` 中只调用 realtime_steger.extract_steger_columns。
def derivative_images(
    signal: np.ndarray, sigma_px: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(signal, dtype=np.float32)
    kwargs = {"sigma": sigma_px, "mode": "nearest"}
    gx = gaussian_filter(source, order=(0, 1), **kwargs)
    gy = gaussian_filter(source, order=(1, 0), **kwargs)
    gxx = gaussian_filter(source, order=(0, 2), **kwargs)
    gxy = gaussian_filter(source, order=(1, 1), **kwargs)
    gyy = gaussian_filter(source, order=(2, 0), **kwargs)
    return gx, gy, gxx, gxy, gyy


def steger_point(
    column: int,
    row: int,
    derivatives: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float, float, float, float] | None:
    gx, gy, gxx, gxy, gyy = derivatives
    hessian = np.asarray(
        [[gxx[row, column], gxy[row, column]], [gxy[row, column], gyy[row, column]]],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalue = float(eigenvalues[0])
    if not np.isfinite(eigenvalue) or eigenvalue >= -np.finfo(float).eps:
        return None
    normal = eigenvectors[:, 0]
    gradient = np.asarray([gx[row, column], gy[row, column]], dtype=np.float64)
    offset = -float(gradient @ normal) / eigenvalue
    if not np.isfinite(offset):
        return None
    x = float(column + offset * normal[0])
    y = float(row + offset * normal[1])
    return x, y, -eigenvalue, offset, float(normal[0]), float(normal[1])


def extract_centres(
    prepared: Any,
    config: Mapping[str, Any],
    settings: StegerSettings | None = None,
) -> pd.DataFrame:
    settings = settings or StegerSettings()
    corrected = getattr(prepared, "corrected", None)
    if corrected is None:
        corrected = prepared.background_subtracted
    extracted = realtime.extract_steger_columns(corrected, asdict(settings))
    height, width = corrected.shape
    records: list[dict[str, Any]] = []

    for column in range(width):
        x = float(extracted.u_px[column])
        y = float(extracted.v_px[column])
        valid = bool(extracted.valid[column]) and np.isfinite(y)
        record: dict[str, Any] = {
            "column_px": column,
            "x_px": x,
            "peak_row_px": float(round(y)) if valid else math.nan,
            "y_px": y if valid else math.nan,
            "prominence": math.nan,
            "steger_response": float(extracted.response[column]),
            "steger_offset_px": float(extracted.offset_px[column]),
            "steger_normal_x": math.nan,
            "steger_normal_y": float(extracted.normal_y_abs[column]),
            "line_residual_px": math.nan,
            "status": "rejected_steger",
        }
        if not valid:
            records.append(record)
            continue
        if not (0.0 <= x < width and 0.0 <= y < height):
            record["status"] = "rejected_image_bounds"
        elif cv2.pointPolygonTest(
            np.asarray(prepared.pose.roi_polygon, dtype=np.float32), (x, y), False
        ) < 0:
            record["status"] = "rejected_roi"
        elif prepared.chess_boundary_mask[
            int(np.clip(round(y), 0, height - 1)),
            int(np.clip(round(x), 0, width - 1)),
        ]:
            record["status"] = "rejected_chess_boundary"
        else:
            record["status"] = "candidate"
        records.append(record)
    return pd.DataFrame.from_records(records)


def annotate_outputs(output_dir: Path, settings: StegerSettings) -> None:
    metadata = {
        "method": "steger_realtime",
        **asdict(settings),
        "interactive_plotly_written": PLOTLY_AVAILABLE,
    }
    for name in ("laser_plane.yaml", "laser_plane_core_config_used.yaml"):
        path = output_dir / name
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["centre_extraction"] = metadata
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    report_path = output_dir / "calibration_report.md"
    with report_path.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## 激光中心提取\n\n"
            "- 方法：实时 Steger/Hessian 亚像素脊线定位\n"
            f"- 高斯导数尺度：{settings.sigma:.3f} px\n"
            f"- 原始灰度阈值：{settings.threshold:.3f}\n"
            f"- 法向二阶导数阈值：{settings.deriv_thresh:.3f}\n"
            f"- 条纹带：margin={settings.roi_margin} px，max_height={settings.roi_max_height} px\n"
            f"- Plotly 交互图已生成：{'是' if PLOTLY_AVAILABLE else '否（环境未安装 plotly）'}\n"
        )


def run(args: argparse.Namespace) -> np.ndarray:
    core = _import_core()
    core.plt.switch_backend("Agg")
    settings = StegerSettings(**realtime.options_from_args(args))
    original_extractor = core.extract_centres

    def runtime_extractor(prepared: Any, config: Mapping[str, Any]) -> pd.DataFrame:
        return extract_centres(prepared, config, settings)

    core.extract_centres = runtime_extractor
    try:
        plane = core.run(args)
    finally:
        core.extract_centres = original_extractor
    annotate_outputs(args.output_dir, settings)
    return plane


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run(args)
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        ModuleNotFoundError,
        RuntimeError,
        ValueError,
        cv2.error,
    ) as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        LOGGER.exception("标定失败：%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
