#!/usr/bin/env python3
"""Laser-plane calibration with cumulative and leave-one-step-out ablations.

All coordinates are in millimetres.  Each experiment independently extracts
laser centres and fits its own plane.  Validation images are processed only
after every experiment's training plane has been frozen.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks, peak_widths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("laser_plane_calibration")
DEFAULT_CONFIG = Path(__file__).with_name("laser_plane_config.yaml")

# OpenCV uses BGR.  Requested rejection colours are kept stable in every plot.
STATUS_COLOURS: dict[str, tuple[int, int, int]] = {
    "accepted_final": (0, 255, 0),
    "rejected_saturation": (0, 0, 255),
    "rejected_snr": (0, 255, 255),
    "rejected_fwhm": (255, 0, 0),
    "rejected_chess_boundary": (180, 0, 180),
    "rejected_continuity": (0, 165, 255),
    "rejected_roi": (128, 128, 128),
    "rejected_prominence": (220, 220, 220),
    "rejected_centroid": (255, 255, 255),
    "rejected_intersection": (255, 255, 0),
    "rejected_ransac": (255, 128, 255),
}

# The chess-image overlay intentionally uses two distinct colours: a green
# centreline and magenta point markers.  OpenCV colour tuples are BGR.
CHESS_OVERLAY_LINE_COLOUR = (0, 255, 0)
CHESS_OVERLAY_POINT_COLOUR = (255, 0, 255)


class CalibrationError(RuntimeError):
    """Expected calibration failure with a user-facing message."""


@dataclass(frozen=True)
class Intrinsics:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True)
class BoardPose:
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    plane: np.ndarray
    roi_polygon: np.ndarray
    reprojection_rmse_px: float


@dataclass(frozen=True)
class PreparedFrame:
    image_id: int
    split: str
    chess_path: Path
    laser_path: Path
    laser_gray: np.ndarray
    full_scale: float
    pose: BoardPose
    background: np.ndarray
    background_subtracted: np.ndarray
    raw_smoothed: np.ndarray
    corrected_smoothed: np.ndarray
    chess_boundary_mask: np.ndarray


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    label: str
    category: str
    reference_name: str | None
    changed_step: str
    use_roi: bool
    use_background: bool
    use_prominence: bool
    use_subpixel: bool
    use_quality: bool
    use_chess_boundary: bool
    use_continuity: bool
    use_ransac: bool


@dataclass
class FrameResult:
    image_id: int
    split: str
    chess_path: Path
    laser_path: Path
    image_width: int
    success: bool
    message: str
    reprojection_rmse_px: float | None
    points_3d: np.ndarray
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class PlaneFit:
    coefficients: np.ndarray
    ransac_inliers: np.ndarray


@dataclass
class ExperimentRun:
    spec: ExperimentSpec
    train_results: list[FrameResult] = field(default_factory=list)
    validation_results: list[FrameResult] = field(default_factory=list)
    plane: np.ndarray | None = None
    fit_error: str | None = None


class OpenCVSafeLoader(yaml.SafeLoader):
    """PyYAML loader accepting OpenCV's ``!!opencv-matrix`` tag."""


def _opencv_matrix(loader: yaml.Loader, node: yaml.Node) -> np.ndarray:
    mapping = loader.construct_mapping(node, deep=True)
    return np.asarray(mapping["data"], dtype=np.float64).reshape(
        int(mapping["rows"]), int(mapping["cols"])
    )


OpenCVSafeLoader.add_constructor(
    "tag:yaml.org,2002:opencv-matrix",
    _opencv_matrix,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return parsed


def parse_id_spec(spec: str) -> list[int]:
    """Parse comma-separated IDs and inclusive ranges such as ``1-18,21``."""
    ids: list[int] = []
    for token in (item.strip() for item in spec.split(",")):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", token)
        if match is None:
            raise argparse.ArgumentTypeError(f"无效编号范围：{token!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start <= 0 or end < start:
            raise argparse.ArgumentTypeError(f"无效编号范围：{token!r}")
        ids.extend(range(start, end + 1))
    if not ids:
        raise argparse.ArgumentTypeError("编号范围不能为空")
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError(f"编号范围包含重复值：{spec!r}")
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="棋盘格配对图像线激光平面标定与累计式消融实验。",
    )
    parser.add_argument("--intrinsics", type=Path, required=True, help="相机内参 YAML")
    parser.add_argument("--image-dir", type=Path, required=True, help="配对图像目录")
    parser.add_argument("--chess-pattern", default="chess {id:03d}.tif")
    parser.add_argument("--laser-pattern", default="laser {id:03d}.tif")
    parser.add_argument("--train-ids", type=parse_id_spec, default=parse_id_spec("1-18"))
    parser.add_argument("--val-ids", type=parse_id_spec, default=parse_id_spec("19-24"))
    parser.add_argument("--pattern-cols", type=positive_int, default=6)
    parser.add_argument("--pattern-rows", type=positive_int, default=5)
    parser.add_argument("--square-size-mm", type=positive_float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"算法阈值 YAML（默认：{DEFAULT_CONFIG.name}）",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="允许覆盖输出目录中的同名结果文件",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def build_experiment_specs() -> list[ExperimentSpec]:
    """Return the fixed cumulative and leave-one-step-out experiment contract."""
    cumulative = [
        ExperimentSpec("A0", "raw_integer", "cumulative", None, "raw_integer", False, False, False, False, False, False, False, False),
        ExperimentSpec("A1", "add_board_roi", "cumulative", "A0", "board_roi", True, False, False, False, False, False, False, False),
        ExperimentSpec("A2", "add_background_prominence", "cumulative", "A1", "background_prominence", True, True, True, False, False, False, False, False),
        ExperimentSpec("A3", "add_subpixel", "cumulative", "A2", "subpixel", True, True, True, True, False, False, False, False),
        ExperimentSpec("A4", "add_quality_filters", "cumulative", "A3", "quality", True, True, True, True, True, False, False, False),
        ExperimentSpec("A5", "add_chess_boundary", "cumulative", "A4", "chess_boundary", True, True, True, True, True, True, False, False),
        ExperimentSpec("A6", "add_continuity", "cumulative", "A5", "continuity", True, True, True, True, True, True, True, False),
        ExperimentSpec("A7", "add_ransac", "cumulative", "A6", "ransac", True, True, True, True, True, True, True, True),
    ]
    removals = [
        ExperimentSpec("R_no_background", "remove_background", "single_removal", "A7", "background", True, False, True, True, True, True, True, True),
        ExperimentSpec("R_no_subpixel", "remove_subpixel", "single_removal", "A7", "subpixel", True, True, True, False, True, True, True, True),
        ExperimentSpec("R_no_quality", "remove_quality", "single_removal", "A7", "quality", True, True, True, True, False, True, True, True),
        ExperimentSpec("R_no_chess_boundary", "remove_chess_boundary", "single_removal", "A7", "chess_boundary", True, True, True, True, True, False, True, True),
        ExperimentSpec("R_no_continuity", "remove_continuity", "single_removal", "A7", "continuity", True, True, True, True, True, True, False, True),
        ExperimentSpec("R_no_ransac", "remove_ransac", "single_removal", "A7", "ransac", True, True, True, True, True, True, True, False),
    ]
    return [*cumulative, *removals]


def _cfg(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise CalibrationError(f"配置缺少字段：{dotted_key}")
        value = value[key]
    return value


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"无法读取配置文件 {path}：{exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError(f"配置文件根节点必须是映射：{path}")
    validate_config(data)
    return data


def validate_config(config: Mapping[str, Any]) -> None:
    positive_keys = (
        "chessboard.corner_window_px",
        "chessboard.corner_max_iterations",
        "chessboard.corner_epsilon",
        "chessboard.max_reprojection_rmse_px",
        "chessboard.boundary_samples_per_line",
        "laser.background_window_px",
        "laser.min_peak_prominence_ratio",
        "laser.centroid_half_window_px",
        "laser.noise_floor_ratio",
        "laser.min_snr",
        "laser.min_fwhm_px",
        "laser.max_fwhm_px",
        "laser.chess_boundary_exclusion_px",
        "laser.continuity_median_window_cols",
        "laser.continuity_max_deviation_px",
        "laser.continuity_max_adjacent_jump_px",
        "laser.continuity_max_gap_cols",
        "intersection.parallel_epsilon",
        "intersection.min_depth_mm",
        "intersection.max_depth_mm",
        "ransac.iterations",
        "ransac.distance_threshold_mm",
        "ransac.min_inliers",
        "ransac.min_inlier_ratio",
        "reporting.max_plot_points",
        "reporting.overlay_point_radius_px",
        "reporting.display_high_percentile",
    )
    for key in positive_keys:
        value = _cfg(config, key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise CalibrationError(f"配置字段 {key} 必须是有限正数")

    integer_keys = (
        "chessboard.corner_window_px",
        "chessboard.corner_max_iterations",
        "chessboard.boundary_samples_per_line",
        "laser.background_window_px",
        "laser.centroid_half_window_px",
        "laser.max_saturated_pixels_in_window",
        "laser.continuity_median_window_cols",
        "laser.continuity_max_gap_cols",
        "ransac.iterations",
        "ransac.min_inliers",
        "ransac.random_seed",
        "reporting.max_plot_points",
        "reporting.overlay_point_radius_px",
    )
    for key in integer_keys:
        if not isinstance(_cfg(config, key), int):
            raise CalibrationError(f"配置字段 {key} 必须是整数")

    for key in ("laser.background_window_px", "laser.continuity_median_window_cols"):
        if int(_cfg(config, key)) % 2 == 0:
            raise CalibrationError(f"配置字段 {key} 必须是奇数")

    for key in ("laser.background_percentile", "laser.saturation_ratio", "ransac.min_inlier_ratio"):
        value = float(_cfg(config, key))
        if not 0 < value <= 1:
            raise CalibrationError(f"配置字段 {key} 必须位于 (0, 1]")
    for key in ("laser.min_peak_prominence_ratio", "laser.noise_floor_ratio"):
        if float(_cfg(config, key)) > 1:
            raise CalibrationError(f"配置字段 {key} 必须不大于 1")

    if float(_cfg(config, "laser.min_fwhm_px")) >= float(_cfg(config, "laser.max_fwhm_px")):
        raise CalibrationError("laser.min_fwhm_px 必须小于 laser.max_fwhm_px")
    if float(_cfg(config, "intersection.min_depth_mm")) >= float(_cfg(config, "intersection.max_depth_mm")):
        raise CalibrationError("intersection.min_depth_mm 必须小于 max_depth_mm")

    nonnegative_keys = (
        "chessboard.outer_border_squares",
        "chessboard.boundary_inset_mm",
        "laser.profile_smoothing_sigma_px",
        "reporting.display_low_percentile",
    )
    for key in nonnegative_keys:
        value = _cfg(config, key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise CalibrationError(f"配置字段 {key} 必须是有限非负数")
    if int(_cfg(config, "laser.max_saturated_pixels_in_window")) < 0:
        raise CalibrationError("laser.max_saturated_pixels_in_window 必须是非负整数")
    if int(_cfg(config, "ransac.random_seed")) < 0:
        raise CalibrationError("ransac.random_seed 必须是非负整数")
    sensor_max = _cfg(config, "laser.sensor_max_value")
    if sensor_max is not None and (
        not isinstance(sensor_max, (int, float))
        or not math.isfinite(float(sensor_max))
        or float(sensor_max) <= 0
    ):
        raise CalibrationError("laser.sensor_max_value 必须为 null 或有限正数")
    low = float(_cfg(config, "reporting.display_low_percentile"))
    high = float(_cfg(config, "reporting.display_high_percentile"))
    if not 0 <= low < high <= 100:
        raise CalibrationError("显示百分位必须满足 0 <= low < high <= 100")


def _normalise_opencv_yaml(text: str) -> str:
    return re.sub(r"^\s*%YAML:\d+(?:\.\d+)?\s*$", "", text, flags=re.MULTILINE)


def _matrix_from_yaml_value(value: Any, key: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, Mapping) and {"rows", "cols", "data"} <= set(value):
        return np.asarray(value["data"], dtype=np.float64).reshape(
            int(value["rows"]), int(value["cols"])
        )
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"内参字段 {key} 不是数值矩阵") from exc


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> tuple[str, Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    raise KeyError(tuple(keys))


def _load_intrinsics_with_pyyaml(path: Path) -> Intrinsics:
    try:
        text = _normalise_opencv_yaml(path.read_text(encoding="utf-8-sig"))
        loaded = yaml.load(text, Loader=OpenCVSafeLoader)
    except (OSError, yaml.YAMLError, KeyError, ValueError) as exc:
        raise CalibrationError(f"PyYAML 无法解析内参文件 {path}：{exc}") from exc
    if not isinstance(loaded, Mapping):
        raise CalibrationError(f"内参 YAML 根节点必须是映射：{path}")
    data = loaded.get("opencv_storage", loaded)
    if not isinstance(data, Mapping):
        raise CalibrationError("opencv_storage 必须是映射")
    try:
        matrix_key, matrix_value = _first_present(
            data, ("camera_matrix", "cameraMatrix", "K", "intrinsic_matrix")
        )
        dist_key, dist_value = _first_present(
            data,
            ("dist_coeffs", "distortion_coeffs", "distortion_coefficients", "distCoeffs", "D"),
        )
    except KeyError as exc:
        raise CalibrationError("内参 YAML 缺少相机矩阵或畸变参数") from exc

    image_size: tuple[int, int] | None = None
    for width_key, height_key in (
        ("image_width", "image_height"),
        ("imageWidth", "imageHeight"),
        ("width", "height"),
    ):
        if width_key in data and height_key in data:
            image_size = (int(data[width_key]), int(data[height_key]))
            break
    if image_size is None:
        try:
            _, size_value = _first_present(data, ("image_size", "imageSize"))
        except KeyError as exc:
            raise CalibrationError("内参 YAML 缺少图像尺寸") from exc
        size = np.asarray(size_value, dtype=np.int64).reshape(-1)
        if size.size != 2:
            raise CalibrationError("image_size 必须包含 [width, height]")
        image_size = (int(size[0]), int(size[1]))
    return Intrinsics(
        image_size,
        _matrix_from_yaml_value(matrix_value, matrix_key),
        _matrix_from_yaml_value(dist_value, dist_key).reshape(-1),
    )


def _fs_array(storage: cv2.FileStorage, keys: Iterable[str]) -> np.ndarray | None:
    for key in keys:
        node = storage.getNode(key)
        if node.empty():
            continue
        matrix = node.mat()
        if matrix is not None:
            return np.asarray(matrix, dtype=np.float64)
        if node.isSeq():
            return np.asarray([node.at(index).real() for index in range(node.size())])
    return None


def _fs_scalar(storage: cv2.FileStorage, keys: Iterable[str]) -> float | None:
    for key in keys:
        node = storage.getNode(key)
        if not node.empty() and (node.isInt() or node.isReal()):
            return float(node.real())
    return None


def _load_intrinsics_with_filestorage(path: Path) -> Intrinsics | None:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        return None
    try:
        matrix = _fs_array(storage, ("camera_matrix", "cameraMatrix", "K", "intrinsic_matrix"))
        distortion = _fs_array(
            storage,
            ("dist_coeffs", "distortion_coeffs", "distortion_coefficients", "distCoeffs", "D"),
        )
        width = _fs_scalar(storage, ("image_width", "imageWidth", "width"))
        height = _fs_scalar(storage, ("image_height", "imageHeight", "height"))
        size = _fs_array(storage, ("image_size", "imageSize"))
    except cv2.error:
        return None
    finally:
        storage.release()
    if matrix is None or distortion is None:
        return None
    if width is None or height is None:
        if size is None or size.size != 2:
            return None
        width, height = size.reshape(-1).tolist()
    return Intrinsics((int(width), int(height)), matrix, distortion.reshape(-1))


def validate_intrinsics(intrinsics: Intrinsics) -> Intrinsics:
    width, height = intrinsics.image_size
    matrix = np.asarray(intrinsics.camera_matrix, dtype=np.float64)
    distortion = np.asarray(intrinsics.dist_coeffs, dtype=np.float64).reshape(-1)
    if width <= 0 or height <= 0:
        raise CalibrationError("内参图像尺寸必须为正数")
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise CalibrationError("相机矩阵必须是有限的 3×3 矩阵")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise CalibrationError("相机焦距 fx、fy 必须为正数")
    if distortion.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(distortion)):
        raise CalibrationError("畸变参数长度必须是 OpenCV 支持的 4/5/8/12/14")
    return Intrinsics((width, height), matrix, distortion)


def load_intrinsics(path: Path) -> Intrinsics:
    if not path.is_file():
        raise FileNotFoundError(f"内参文件不存在：{path}")
    try:
        intrinsics = _load_intrinsics_with_filestorage(path)
    except (cv2.error, OSError, SystemError):
        intrinsics = None
    if intrinsics is None:
        intrinsics = _load_intrinsics_with_pyyaml(path)
    return validate_intrinsics(intrinsics)


def format_image_path(image_dir: Path, pattern: str, image_id: int) -> Path:
    try:
        relative = Path(pattern.format(id=image_id))
    except (KeyError, ValueError, IndexError) as exc:
        raise CalibrationError(f"图像文件模式无效 {pattern!r}：{exc}") from exc
    if relative.is_absolute():
        raise CalibrationError("图像文件模式必须是相对于 --image-dir 的路径")
    candidate = image_dir / relative
    try:
        candidate.resolve().relative_to(image_dir.resolve())
    except ValueError as exc:
        raise CalibrationError("图像文件模式不得跳出 --image-dir") from exc
    return candidate


def validate_inputs(args: argparse.Namespace) -> list[tuple[int, str, Path, Path]]:
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"图像目录不存在或不是目录：{args.image_dir}")
    overlap = sorted(set(args.train_ids) & set(args.val_ids))
    if overlap:
        raise CalibrationError(f"训练集和验证集编号重叠：{overlap}")
    pairs: list[tuple[int, str, Path, Path]] = []
    missing: list[Path] = []
    for split, ids in (("train", args.train_ids), ("validation", args.val_ids)):
        for image_id in ids:
            chess_path = format_image_path(args.image_dir, args.chess_pattern, image_id)
            laser_path = format_image_path(args.image_dir, args.laser_pattern, image_id)
            pairs.append((image_id, split, chess_path, laser_path))
            missing.extend(path for path in (chess_path, laser_path) if not path.is_file())
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        suffix = f"\n  ... 另有 {len(missing) - 20} 个" if len(missing) > 20 else ""
        raise FileNotFoundError(f"缺少配对图像：\n{preview}{suffix}")
    return pairs


def read_image(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error) as exc:
        raise CalibrationError(f"无法读取图像 {path}：{exc}") from exc
    if image is None:
        raise CalibrationError(f"OpenCV 无法解码图像：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise CalibrationError(f"OpenCV 无法编码图像：{path}")
    encoded.tofile(path)


def to_gray(image: np.ndarray, path: Path) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise CalibrationError(f"不支持的图像形状 {image.shape}：{path}")
    if not np.all(np.isfinite(gray)):
        raise CalibrationError(f"图像包含 NaN/Inf：{path}")
    return gray


def chessboard_detection_image(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    minimum = float(np.min(gray))
    maximum = float(np.max(gray))
    if maximum == minimum:
        raise CalibrationError("棋盘图没有可用灰度对比度")
    scaled = (gray.astype(np.float64) - minimum) * (255.0 / (maximum - minimum))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _full_scale(gray: np.ndarray, config: Mapping[str, Any]) -> float:
    configured = _cfg(config, "laser.sensor_max_value")
    if configured is not None:
        return float(configured)
    if np.issubdtype(gray.dtype, np.integer):
        return float(np.iinfo(gray.dtype).max)
    return 1.0


def object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return points


def detect_board_pose(
    gray: np.ndarray,
    cols: int,
    rows: int,
    square_size_mm: float,
    intrinsics: Intrinsics,
    config: Mapping[str, Any],
) -> BoardPose:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found or corners is None:
        raise CalibrationError("未检测到完整棋盘格内角点")
    window = int(_cfg(config, "chessboard.corner_window_px"))
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        int(_cfg(config, "chessboard.corner_max_iterations")),
        float(_cfg(config, "chessboard.corner_epsilon")),
    )
    refined = cv2.cornerSubPix(
        gray, corners.astype(np.float32), (window, window), (-1, -1), criteria
    )
    board_points = object_points(cols, rows, square_size_mm)
    solved, rvec, tvec = cv2.solvePnP(
        board_points,
        refined,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise CalibrationError("solvePnP 未能求得标定板位姿")
    projected, _ = cv2.projectPoints(
        board_points, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    residual = refined.reshape(-1, 2) - projected.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    max_rmse = float(_cfg(config, "chessboard.max_reprojection_rmse_px"))
    if rmse > max_rmse:
        raise CalibrationError(f"PnP 重投影 RMSE {rmse:.3f}px 超过 {max_rmse:.3f}px")

    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2].astype(np.float64)
    normal /= np.linalg.norm(normal)
    point_on_plane = tvec.reshape(3).astype(np.float64)
    plane = np.append(normal, -float(normal @ point_on_plane))

    border = float(_cfg(config, "chessboard.outer_border_squares"))
    inset = float(_cfg(config, "chessboard.boundary_inset_mm"))
    x0 = -border * square_size_mm + inset
    y0 = -border * square_size_mm + inset
    x1 = (cols - 1 + border) * square_size_mm - inset
    y1 = (rows - 1 + border) * square_size_mm - inset
    if x0 >= x1 or y0 >= y1:
        raise CalibrationError("棋盘边界内缩量过大，未留下有效板面")
    outer = np.asarray(
        [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]],
        dtype=np.float32,
    )
    polygon, _ = cv2.projectPoints(
        outer, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    return BoardPose(
        refined.reshape(-1, 2),
        rvec,
        tvec,
        plane,
        polygon.reshape(-1, 2).astype(np.float32),
        rmse,
    )


def build_chess_boundary_mask(
    image_shape: tuple[int, int],
    pose: BoardPose,
    args: argparse.Namespace,
    intrinsics: Intrinsics,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Rasterise projected internal black/white grid boundaries with tolerance."""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    samples = int(_cfg(config, "chessboard.boundary_samples_per_line"))
    border = float(_cfg(config, "chessboard.outer_border_squares"))
    square = float(args.square_size_mm)
    x_min = -border * square
    x_max = (args.pattern_cols - 1 + border) * square
    y_min = -border * square
    y_max = (args.pattern_rows - 1 + border) * square
    lines: list[np.ndarray] = []
    for column in range(args.pattern_cols):
        y_values = np.linspace(y_min, y_max, samples)
        x_values = np.full(samples, column * square)
        lines.append(np.column_stack([x_values, y_values, np.zeros(samples)]))
    for row in range(args.pattern_rows):
        x_values = np.linspace(x_min, x_max, samples)
        y_values = np.full(samples, row * square)
        lines.append(np.column_stack([x_values, y_values, np.zeros(samples)]))
    radius = float(_cfg(config, "laser.chess_boundary_exclusion_px"))
    thickness = max(1, int(math.ceil(2.0 * radius + 1.0)))
    for line in lines:
        projected, _ = cv2.projectPoints(
            line.astype(np.float32),
            pose.rvec,
            pose.tvec,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
        )
        polyline = np.rint(projected.reshape(-1, 2)).astype(np.int32)
        cv2.polylines(mask, [polyline], False, 255, thickness, lineType=cv2.LINE_AA)
    return mask


def prepare_frame(
    pair: tuple[int, str, Path, Path],
    args: argparse.Namespace,
    intrinsics: Intrinsics,
    config: Mapping[str, Any],
) -> PreparedFrame:
    image_id, split, chess_path, laser_path = pair
    chess = to_gray(read_image(chess_path), chess_path)
    laser = to_gray(read_image(laser_path), laser_path)
    expected = (intrinsics.image_size[1], intrinsics.image_size[0])
    if chess.shape != expected or laser.shape != expected:
        raise CalibrationError(
            f"ID {image_id:03d} 尺寸不匹配：期望 {expected[::-1]}，"
            f"chess={chess.shape[::-1]}，laser={laser.shape[::-1]}"
        )
    pose = detect_board_pose(
        chessboard_detection_image(chess),
        args.pattern_cols,
        args.pattern_rows,
        args.square_size_mm,
        intrinsics,
        config,
    )
    image = laser.astype(np.float64)
    full_scale = _full_scale(laser, config)
    if float(np.max(image)) > full_scale:
        raise CalibrationError(
            f"ID {image_id:03d} 最大值超过 laser.sensor_max_value={full_scale:g}"
        )
    window = int(_cfg(config, "laser.background_window_px"))
    percentile = float(_cfg(config, "laser.background_percentile")) * 100.0
    background = percentile_filter(
        image, percentile=percentile, size=(window, 1), mode="nearest"
    )
    corrected = np.maximum(image - background, 0.0)
    sigma = float(_cfg(config, "laser.profile_smoothing_sigma_px"))
    raw_smoothed = (
        gaussian_filter1d(image, sigma=sigma, axis=0, mode="nearest")
        if sigma > 0
        else image.copy()
    )
    corrected_smoothed = (
        gaussian_filter1d(corrected, sigma=sigma, axis=0, mode="nearest")
        if sigma > 0
        else corrected.copy()
    )
    boundary_mask = build_chess_boundary_mask(
        laser.shape, pose, args, intrinsics, config
    )
    return PreparedFrame(
        image_id,
        split,
        chess_path,
        laser_path,
        laser,
        full_scale,
        pose,
        background,
        corrected,
        raw_smoothed,
        corrected_smoothed,
        boundary_mask,
    )


def _inside_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def _peak_for_column(
    raw_profile: np.ndarray,
    signal_profile: np.ndarray,
    use_prominence: bool,
    minimum_prominence: float,
) -> tuple[int | None, float]:
    if not use_prominence:
        return int(np.argmax(raw_profile)), float("nan")
    peaks, properties = find_peaks(signal_profile, prominence=minimum_prominence)
    if peaks.size == 0:
        return None, float("nan")
    selected = int(np.argmax(signal_profile[peaks]))
    return int(peaks[selected]), float(properties["prominences"][selected])


def extract_laser_centres(
    prepared: PreparedFrame,
    spec: ExperimentSpec,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Independently regenerate one centre candidate per column for a stage."""
    raw = prepared.laser_gray.astype(np.float64)
    if spec.use_background:
        signal = prepared.corrected_smoothed
    elif spec.use_prominence:
        signal = prepared.raw_smoothed
    else:
        signal = raw  # A0/A1 are exactly the raw integer maximum.

    full_scale = prepared.full_scale
    minimum_prominence = float(_cfg(config, "laser.min_peak_prominence_ratio")) * full_scale
    half_window = int(_cfg(config, "laser.centroid_half_window_px"))
    saturation_level = float(_cfg(config, "laser.saturation_ratio")) * full_scale
    max_saturated = int(_cfg(config, "laser.max_saturated_pixels_in_window"))
    noise_floor = float(_cfg(config, "laser.noise_floor_ratio")) * full_scale
    min_snr = float(_cfg(config, "laser.min_snr"))
    min_fwhm = float(_cfg(config, "laser.min_fwhm_px"))
    max_fwhm = float(_cfg(config, "laser.max_fwhm_px"))

    height, width = raw.shape
    row_coordinates = np.arange(height, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for column in range(width):
        raw_profile = raw[:, column]
        profile = signal[:, column]
        peak_row, prominence = _peak_for_column(
            raw_profile, profile, spec.use_prominence, minimum_prominence
        )
        record: dict[str, Any] = {
            "column_px": column,
            "x_px": float(column),
            "peak_row_px": np.nan,
            "integer_y_px": np.nan,
            "subpixel_y_px": np.nan,
            "y_px": np.nan,
            "peak_value": np.nan,
            "prominence": prominence,
            "snr": np.nan,
            "fwhm_px": np.nan,
            "saturated_pixels": 0,
            "continuity_residual_px": np.nan,
            "status": "rejected_prominence",
        }
        if peak_row is None:
            fallback_row = int(np.argmax(raw_profile))
            record.update(
                peak_row_px=float(fallback_row),
                integer_y_px=float(fallback_row),
                y_px=float(fallback_row),
                peak_value=float(profile[fallback_row]),
            )
            records.append(record)
            continue

        start = max(0, peak_row - half_window)
        stop = min(height, peak_row + half_window + 1)
        weights = np.maximum(profile[start:stop], 0.0)
        weight_sum = float(np.sum(weights))
        centroid = (
            float(np.sum(row_coordinates[start:stop] * weights) / weight_sum)
            if weight_sum > 0
            else float("nan")
        )
        selected_y = centroid if spec.use_subpixel else float(peak_row)
        saturated = 0
        snr = float("nan")
        fwhm = float("nan")
        if spec.use_quality:
            raw_window = raw[start:stop, column]
            saturated = int(np.count_nonzero(raw_window >= saturation_level))
            median = float(np.median(profile))
            mad = float(np.median(np.abs(profile - median)))
            noise = max(1.4826 * mad, noise_floor)
            snr = float(profile[peak_row] / noise)
            if 0 < peak_row < height - 1:
                fwhm = float(
                    peak_widths(profile, np.asarray([peak_row]), rel_height=0.5)[0][0]
                )
            else:
                fwhm = 0.0
        record.update(
            peak_row_px=float(peak_row),
            integer_y_px=float(peak_row),
            subpixel_y_px=centroid if spec.use_subpixel else np.nan,
            y_px=selected_y,
            peak_value=float(profile[peak_row]),
            snr=snr,
            fwhm_px=fwhm,
            saturated_pixels=saturated,
            status="candidate",
        )

        if spec.use_roi and not _inside_polygon(
            (float(column), selected_y), prepared.pose.roi_polygon
        ):
            record["status"] = "rejected_roi"
        elif spec.use_quality and saturated > max_saturated:
            record["status"] = "rejected_saturation"
        elif spec.use_quality and snr < min_snr:
            record["status"] = "rejected_snr"
        elif spec.use_quality and not min_fwhm <= fwhm <= max_fwhm:
            record["status"] = "rejected_fwhm"
        elif spec.use_subpixel and not np.isfinite(centroid):
            record["status"] = "rejected_centroid"
        elif spec.use_chess_boundary:
            rounded_y = int(np.clip(round(selected_y), 0, height - 1))
            if prepared.chess_boundary_mask[rounded_y, column] != 0:
                record["status"] = "rejected_chess_boundary"
        records.append(record)

    diagnostics = pd.DataFrame.from_records(records)
    if spec.use_continuity:
        _apply_continuity_filter(diagnostics, config)
    diagnostics.loc[diagnostics["status"] == "candidate", "status"] = "accepted_2d"
    return diagnostics


def _apply_continuity_filter(
    diagnostics: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    indices = diagnostics.index[diagnostics["status"] == "candidate"].to_numpy()
    if indices.size == 0:
        return
    candidates = diagnostics.loc[indices].copy()
    max_gap = int(_cfg(config, "laser.continuity_max_gap_cols"))
    window = int(_cfg(config, "laser.continuity_median_window_cols"))
    segment = candidates["x_px"].diff().gt(max_gap).cumsum()
    local_median = candidates.groupby(segment, sort=False)["y_px"].transform(
        lambda values: values.rolling(window, center=True, min_periods=1).median()
    )
    residual = (candidates["y_px"] - local_median).abs()
    bad = residual > float(_cfg(config, "laser.continuity_max_deviation_px"))
    max_jump = float(_cfg(config, "laser.continuity_max_adjacent_jump_px"))
    for _, group in candidates.groupby(segment, sort=False):
        y_values = group["y_px"].to_numpy(dtype=np.float64)
        if len(y_values) < 3:
            continue
        previous_jump = np.r_[0.0, np.abs(np.diff(y_values))]
        next_jump = np.r_[np.abs(np.diff(y_values)), 0.0]
        isolated = (
            (np.arange(len(y_values)) > 0)
            & (previous_jump > max_jump)
            & (np.arange(len(y_values)) < len(y_values) - 1)
            & (next_jump > max_jump)
        )
        bad.loc[group.index] = bad.loc[group.index].to_numpy(dtype=bool) | isolated
    diagnostics.loc[candidates.index, "continuity_residual_px"] = residual
    diagnostics.loc[candidates.index[bad], "status"] = "rejected_continuity"


def intersect_board_plane(
    diagnostics: pd.DataFrame,
    plane: np.ndarray,
    intrinsics: Intrinsics,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    indices = diagnostics.index[diagnostics["status"] == "accepted_2d"].to_numpy()
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float64), diagnostics
    pixels = diagnostics.loc[indices, ["x_px", "y_px"]].to_numpy(dtype=np.float64)
    normalised = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    ).reshape(-1, 2)
    rays = np.column_stack([normalised, np.ones(len(normalised))])
    denominator = rays @ plane[:3]
    epsilon = float(_cfg(config, "intersection.parallel_epsilon"))
    scales = np.full(len(rays), np.nan)
    nonparallel = np.abs(denominator) > epsilon
    scales[nonparallel] = -float(plane[3]) / denominator[nonparallel]
    minimum_depth = float(_cfg(config, "intersection.min_depth_mm"))
    maximum_depth = float(_cfg(config, "intersection.max_depth_mm"))
    valid = (
        nonparallel
        & np.isfinite(scales)
        & (scales >= minimum_depth)
        & (scales <= maximum_depth)
    )
    points = rays[valid] * scales[valid, None]
    diagnostics.loc[indices[~valid], "status"] = "rejected_intersection"
    valid_indices = indices[valid]
    diagnostics.loc[valid_indices, "status"] = "accepted_3d"
    diagnostics.loc[valid_indices, ["x_mm", "y_mm", "z_mm"]] = points
    return points, diagnostics


def process_prepared_frame(
    prepared: PreparedFrame,
    spec: ExperimentSpec,
    intrinsics: Intrinsics,
    config: Mapping[str, Any],
) -> FrameResult:
    try:
        diagnostics = extract_laser_centres(prepared, spec, config)
        points, diagnostics = intersect_board_plane(
            diagnostics, prepared.pose.plane, intrinsics, config
        )
        diagnostics.insert(0, "experiment", spec.name)
        diagnostics.insert(0, "split", prepared.split)
        diagnostics.insert(0, "image_id", prepared.image_id)
        success = len(points) > 0
        message = "ok" if success else "没有有效三维点"
        return FrameResult(
            prepared.image_id,
            prepared.split,
            prepared.chess_path,
            prepared.laser_path,
            prepared.laser_gray.shape[1],
            success,
            message,
            prepared.pose.reprojection_rmse_px,
            points,
            diagnostics,
        )
    except (CalibrationError, cv2.error, ValueError, FloatingPointError) as exc:
        LOGGER.warning("%s ID %03d 处理失败：%s", spec.name, prepared.image_id, exc)
        return failed_frame_result(prepared, str(exc))


def failed_frame_result(prepared: PreparedFrame, message: str) -> FrameResult:
    return FrameResult(
        prepared.image_id,
        prepared.split,
        prepared.chess_path,
        prepared.laser_path,
        prepared.laser_gray.shape[1],
        False,
        message,
        prepared.pose.reprojection_rmse_px,
        np.empty((0, 3), dtype=np.float64),
        pd.DataFrame(),
    )


def plane_from_three(points: np.ndarray) -> np.ndarray | None:
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    norm = float(np.linalg.norm(normal))
    if norm == 0:
        return None
    normal /= norm
    return np.append(normal, -float(normal @ points[0]))


def fit_plane_svd(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3 or points.shape[1] != 3:
        raise CalibrationError("SVD 平面拟合至少需要 3 个三维点")
    centroid = np.mean(points, axis=0)
    _, singular_values, right_vectors = np.linalg.svd(points - centroid, full_matrices=False)
    if singular_values[1] == 0:
        raise CalibrationError("三维点退化为直线，无法拟合平面")
    normal = right_vectors[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0 or (normal[2] == 0 and normal[np.argmax(np.abs(normal))] < 0):
        normal = -normal
    return np.append(normal, -float(normal @ centroid))


def point_plane_distances(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty(0, dtype=np.float64)
    normal_norm = float(np.linalg.norm(plane[:3]))
    if normal_norm == 0:
        raise CalibrationError("平面法向量不能为零")
    return np.abs(points @ plane[:3] + plane[3]) / normal_norm


def fit_plane_ransac(points: np.ndarray, config: Mapping[str, Any]) -> PlaneFit:
    if len(points) < 3:
        raise CalibrationError("RANSAC 平面拟合至少需要 3 个训练点")
    iterations = int(_cfg(config, "ransac.iterations"))
    threshold = float(_cfg(config, "ransac.distance_threshold_mm"))
    rng = np.random.default_rng(int(_cfg(config, "ransac.random_seed")))
    best_inliers: np.ndarray | None = None
    best_median = np.inf
    for _ in range(iterations):
        candidate = plane_from_three(points[rng.choice(len(points), size=3, replace=False)])
        if candidate is None:
            continue
        distances = point_plane_distances(points, candidate)
        inliers = distances <= threshold
        count = int(np.count_nonzero(inliers))
        median = float(np.median(distances[inliers])) if count else np.inf
        previous_count = int(np.count_nonzero(best_inliers)) if best_inliers is not None else -1
        if count > previous_count or (count == previous_count and median < best_median):
            best_inliers = inliers
            best_median = median
    if best_inliers is None:
        raise CalibrationError("RANSAC 未找到有效平面")
    count = int(np.count_nonzero(best_inliers))
    minimum = int(_cfg(config, "ransac.min_inliers"))
    minimum_ratio = float(_cfg(config, "ransac.min_inlier_ratio"))
    if count < minimum or count / len(points) < minimum_ratio:
        raise CalibrationError(
            f"RANSAC 内点不足：{count}/{len(points)}，要求至少 {minimum} 且比例不低于 {minimum_ratio:.3f}"
        )
    return PlaneFit(fit_plane_svd(points[best_inliers]), best_inliers)


def stack_points(results: Sequence[FrameResult]) -> tuple[np.ndarray, np.ndarray]:
    usable = [result for result in results if len(result.points_3d)]
    if not usable:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    return (
        np.vstack([result.points_3d for result in usable]),
        np.concatenate(
            [np.full(len(result.points_3d), result.image_id, dtype=np.int64) for result in usable]
        ),
    )


def _mark_training_points(
    results: Sequence[FrameResult],
    global_inliers: np.ndarray,
) -> None:
    offset = 0
    for result in results:
        count = len(result.points_3d)
        if count == 0:
            continue
        local = global_inliers[offset : offset + count]
        accepted_indices = result.diagnostics.index[
            result.diagnostics["status"] == "accepted_3d"
        ].to_numpy()
        if len(accepted_indices) != count:
            raise CalibrationError("内部错误：三维点与诊断行数量不一致")
        result.diagnostics.loc[accepted_indices[~local], "status"] = "rejected_ransac"
        result.diagnostics.loc[accepted_indices[local], "status"] = "accepted_final"
        result.points_3d = result.points_3d[local]
        result.success = len(result.points_3d) > 0
        offset += count


def fit_experiment(run: ExperimentRun, config: Mapping[str, Any]) -> None:
    points, _ = stack_points(run.train_results)
    try:
        if run.spec.use_ransac:
            fit = fit_plane_ransac(points, config)
            run.plane = fit.coefficients
            _mark_training_points(run.train_results, fit.ransac_inliers)
        else:
            run.plane = fit_plane_svd(points)
            _mark_training_points(run.train_results, np.ones(len(points), dtype=bool))
        LOGGER.info(
            "%s 训练平面冻结：[%.9g, %.9g, %.9g, %.9g]",
            run.spec.name,
            *run.plane,
        )
    except CalibrationError as exc:
        run.fit_error = str(exc)
        for result in run.train_results:
            if not result.diagnostics.empty:
                result.diagnostics.loc[
                    result.diagnostics["status"] == "accepted_3d", "status"
                ] = "accepted_final"
        LOGGER.error("%s 平面拟合失败：%s", run.spec.name, exc)


def finalise_validation(run: ExperimentRun) -> None:
    for result in run.validation_results:
        if not result.diagnostics.empty:
            result.diagnostics.loc[
                result.diagnostics["status"] == "accepted_3d", "status"
            ] = "accepted_final"


def _display_u8(image: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    low = float(np.percentile(values, _cfg(config, "reporting.display_low_percentile")))
    high = float(np.percentile(values, _cfg(config, "reporting.display_high_percentile")))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - low) * (255.0 / (high - low))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _overlay_base(gray: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    return cv2.cvtColor(_display_u8(gray, config), cv2.COLOR_GRAY2BGR)


def _draw_points(
    image: np.ndarray,
    frame: pd.DataFrame,
    colour: tuple[int, int, int],
    radius: int,
) -> None:
    for row in frame.itertuples(index=False):
        y = getattr(row, "y_px", np.nan)
        x = getattr(row, "x_px", np.nan)
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(image, (int(round(x)), int(round(y))), radius, colour, -1, cv2.LINE_AA)


def diagnostic_dir(output_dir: Path, spec: ExperimentSpec, result: FrameResult) -> Path:
    return (
        output_dir
        / "experiments"
        / spec.name
        / "images"
        / result.split
        / f"{result.image_id:03d}"
    )


def save_pre_fit_diagnostics(
    prepared: PreparedFrame,
    result: FrameResult,
    spec: ExperimentSpec,
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    directory = diagnostic_dir(output_dir, spec, result)
    raw = _display_u8(prepared.laser_gray, config)
    background = _display_u8(prepared.background, config)
    corrected = _display_u8(prepared.background_subtracted, config)
    write_image(directory / "01_original.png", raw)
    write_image(directory / "02_background.png", background)
    write_image(directory / "03_background_subtracted.png", corrected)

    candidates = _overlay_base(prepared.laser_gray, config)
    centres = candidates.copy()
    radius = int(_cfg(config, "reporting.overlay_point_radius_px"))
    if not result.diagnostics.empty:
        passed_prominence = result.diagnostics["status"] != "rejected_prominence"
        peak_rows = result.diagnostics[
            np.isfinite(result.diagnostics["peak_row_px"]) & passed_prominence
        ].copy()
        peak_rows["y_px"] = peak_rows["peak_row_px"]
        _draw_points(candidates, peak_rows, (255, 255, 255), radius)
        centre_rows = result.diagnostics[
            np.isfinite(result.diagnostics["y_px"]) & passed_prominence
        ]
        _draw_points(centres, centre_rows, (255, 255, 0), radius)
    write_image(directory / "04_peak_candidates.png", candidates)
    write_image(directory / "05_subpixel_centres.png", centres)


def save_post_fit_diagnostics(
    result: FrameResult,
    spec: ExperimentSpec,
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    gray = to_gray(read_image(result.laser_path), result.laser_path)
    rejected_overlay = _overlay_base(gray, config)
    final_overlay = rejected_overlay.copy()
    radius = int(_cfg(config, "reporting.overlay_point_radius_px"))
    valid = result.diagnostics.iloc[0:0]
    if not result.diagnostics.empty:
        for status, colour in STATUS_COLOURS.items():
            rows = result.diagnostics[result.diagnostics["status"] == status]
            if status != "accepted_final":
                _draw_points(rejected_overlay, rows, colour, radius)
        valid = result.diagnostics[result.diagnostics["status"] == "accepted_final"]
        _draw_points(rejected_overlay, valid, STATUS_COLOURS["accepted_final"], radius)
        _draw_points(final_overlay, valid, STATUS_COLOURS["accepted_final"], radius)
        _draw_continuous_line(final_overlay, valid, config)
    chess_gray = to_gray(read_image(result.chess_path), result.chess_path)
    chess_overlay = create_chess_laser_overlay(chess_gray, valid, config)
    directory = diagnostic_dir(output_dir, spec, result)
    write_image(directory / "06_rejections.png", rejected_overlay)
    write_image(directory / "07_final_centerline.png", final_overlay)
    write_image(directory / "08_chess_laser_overlay.png", chess_overlay)


def create_chess_laser_overlay(
    chess_gray: np.ndarray,
    valid: pd.DataFrame,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Draw the extracted centreline and its samples on the paired chess image."""
    overlay = _overlay_base(chess_gray, config)
    radius = max(3, int(_cfg(config, "reporting.overlay_point_radius_px")) + 1)
    _draw_points(overlay, valid, CHESS_OVERLAY_POINT_COLOUR, radius)
    _draw_continuous_line(
        overlay,
        valid,
        config,
        colour=CHESS_OVERLAY_LINE_COLOUR,
        thickness=2,
    )
    return overlay


def _draw_continuous_line(
    image: np.ndarray,
    valid: pd.DataFrame,
    config: Mapping[str, Any],
    colour: tuple[int, int, int] = STATUS_COLOURS["accepted_final"],
    thickness: int = 1,
) -> None:
    if len(valid) < 2:
        return
    ordered = valid.sort_values("x_px")
    max_gap = int(_cfg(config, "laser.continuity_max_gap_cols"))
    segment = ordered["x_px"].diff().gt(max_gap).cumsum()
    for _, group in ordered.groupby(segment, sort=False):
        if len(group) < 2:
            continue
        points = np.rint(group[["x_px", "y_px"]].to_numpy()).astype(np.int32)
        cv2.polylines(
            image,
            [points],
            False,
            colour,
            thickness,
            cv2.LINE_AA,
        )


def _distance_summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mae_mm": None, "rmse_mm": None, "p95_mm": None, "max_mm": None}
    return {
        "mae_mm": float(np.mean(values)),
        "rmse_mm": float(np.sqrt(np.mean(values * values))),
        "p95_mm": float(np.percentile(values, 95)),
        "max_mm": float(np.max(values)),
    }


def _centre_frame(results: Sequence[FrameResult]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        if result.diagnostics.empty:
            continue
        valid = result.diagnostics[result.diagnostics["status"] == "accepted_final"]
        if valid.empty:
            continue
        current = valid[["image_id", "column_px", "y_px"]].copy()
        frames.append(current)
    if not frames:
        return pd.DataFrame(columns=["image_id", "column_px", "y_px"])
    return pd.concat(frames, ignore_index=True)


def centre_change_summary(
    current: Sequence[FrameResult],
    reference: Sequence[FrameResult] | None,
) -> dict[str, float | int | None]:
    if reference is None:
        return {
            "center_common_points": 0,
            "center_change_mae_px": None,
            "center_change_rmse_px": None,
            "center_change_p95_px": None,
            "center_change_max_px": None,
        }
    current_frame = _centre_frame(current).rename(columns={"y_px": "current_y"})
    reference_frame = _centre_frame(reference).rename(columns={"y_px": "reference_y"})
    merged = current_frame.merge(
        reference_frame, on=["image_id", "column_px"], how="inner"
    )
    if merged.empty:
        return {
            "center_common_points": 0,
            "center_change_mae_px": None,
            "center_change_rmse_px": None,
            "center_change_p95_px": None,
            "center_change_max_px": None,
        }
    changes = np.abs(merged["current_y"].to_numpy() - merged["reference_y"].to_numpy())
    return {
        "center_common_points": int(len(changes)),
        "center_change_mae_px": float(np.mean(changes)),
        "center_change_rmse_px": float(np.sqrt(np.mean(changes * changes))),
        "center_change_p95_px": float(np.percentile(changes, 95)),
        "center_change_max_px": float(np.max(changes)),
    }


def split_metrics(
    run: ExperimentRun,
    split: str,
    reference_run: ExperimentRun | None,
) -> dict[str, Any]:
    results = run.train_results if split == "train" else run.validation_results
    reference_results = None
    if reference_run is not None:
        reference_results = (
            reference_run.train_results
            if split == "train"
            else reference_run.validation_results
        )
    points, _ = stack_points(results)
    total_columns = sum(result.image_width for result in results)
    distances = (
        point_plane_distances(points, run.plane)
        if run.plane is not None
        else np.empty(0)
    )
    return {
        "experiment": run.spec.name,
        "label": run.spec.label,
        "category": run.spec.category,
        "reference_experiment": run.spec.reference_name,
        "changed_step": run.spec.changed_step,
        "split": split,
        "successful_poses": sum(result.success for result in results),
        "total_poses": len(results),
        "valid_points": int(len(points)),
        "coverage": float(len(points) / total_columns) if total_columns else None,
        **centre_change_summary(results, reference_results),
        **_distance_summary(distances),
        "fit_error": run.fit_error,
    }


def validation_pose_metrics(
    run: ExperimentRun,
    reference_run: ExperimentRun | None,
) -> pd.DataFrame:
    reference_lookup = (
        {result.image_id: result for result in reference_run.validation_results}
        if reference_run is not None
        else {}
    )
    rows: list[dict[str, Any]] = []
    for result in run.validation_results:
        distances = (
            point_plane_distances(result.points_3d, run.plane)
            if run.plane is not None
            else np.empty(0)
        )
        reference = reference_lookup.get(result.image_id)
        rows.append(
            {
                "experiment": run.spec.name,
                "image_id": result.image_id,
                "success": result.success,
                "message": result.message,
                "valid_points": len(result.points_3d),
                "coverage": len(result.points_3d) / result.image_width,
                "reprojection_rmse_px": result.reprojection_rmse_px,
                **centre_change_summary(
                    [result], [reference] if reference is not None else None
                ),
                **_distance_summary(distances),
            }
        )
    return pd.DataFrame(rows)


def experiment_directory(output_dir: Path, name: str) -> Path:
    directory = output_dir / "experiments" / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_experiment_plots(
    run: ExperimentRun,
    directory: Path,
    config: Mapping[str, Any],
) -> None:
    if run.plane is None:
        return
    train_points, _ = stack_points(run.train_results)
    validation_points, _ = stack_points(run.validation_results)
    if len(train_points) < 3:
        return
    limit = int(_cfg(config, "reporting.max_plot_points"))
    plotted_train = _subsample(train_points, limit)
    plotted_validation = _subsample(validation_points, limit)

    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(
        plotted_train[:, 0], plotted_train[:, 1], plotted_train[:, 2],
        s=2, alpha=0.25, label="train"
    )
    if len(plotted_validation):
        axis.scatter(
            plotted_validation[:, 0], plotted_validation[:, 1], plotted_validation[:, 2],
            s=3, alpha=0.35, label="validation"
        )
    _draw_plane_surface(axis, plotted_train, run.plane)
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(f"{run.spec.name}: points and fitted plane")
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "point_cloud_plane.png", dpi=160)
    plt.close(fig)

    train_distance = point_plane_distances(train_points, run.plane)
    validation_distance = point_plane_distances(validation_points, run.plane)
    fig, axis2d = plt.subplots(figsize=(8, 5))
    axis2d.hist(train_distance, bins=60, alpha=0.6, label="train")
    if len(validation_distance):
        axis2d.hist(validation_distance, bins=60, alpha=0.6, label="validation")
    axis2d.set_xlabel("Absolute point-to-plane distance (mm)")
    axis2d.set_ylabel("Point count")
    axis2d.set_title(f"{run.spec.name}: residual distribution")
    axis2d.legend()
    fig.tight_layout()
    fig.savefig(directory / "residual_distribution.png", dpi=160)
    plt.close(fig)


def _subsample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def _draw_plane_surface(axis: Any, points: np.ndarray, plane: np.ndarray) -> None:
    solved_axis = int(np.argmax(np.abs(plane[:3])))
    first, second = [item for item in range(3) if item != solved_axis]
    first_values = np.linspace(np.min(points[:, first]), np.max(points[:, first]), 25)
    second_values = np.linspace(np.min(points[:, second]), np.max(points[:, second]), 25)
    grid_first, grid_second = np.meshgrid(first_values, second_values)
    coordinates: list[np.ndarray | None] = [None, None, None]
    coordinates[first] = grid_first
    coordinates[second] = grid_second
    coordinates[solved_axis] = -(
        plane[first] * grid_first + plane[second] * grid_second + plane[3]
    ) / plane[solved_axis]
    axis.plot_surface(
        coordinates[0], coordinates[1], coordinates[2],
        color="tomato", alpha=0.25, linewidth=0
    )


def _serialisable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _serialisable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(item) for item in value]
    if pd.isna(value):
        return None
    return value


def experiment_report_row(
    run: ExperimentRun,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    reference_validation_rmse: float | None,
) -> dict[str, Any]:
    current_rmse = validation_metrics.get("rmse_mm")
    delta = None
    note = "无可比较的验证 RMSE"
    if current_rmse is not None and reference_validation_rmse is not None:
        delta = float(current_rmse) - float(reference_validation_rmse)
        if run.spec.category == "cumulative":
            if delta > 0:
                note = (
                    f"警告：启用 {run.spec.changed_step} 后验证 RMSE 增加 {delta:.6g} mm"
                )
            elif delta < 0:
                note = f"启用 {run.spec.changed_step} 后验证 RMSE 降低 {-delta:.6g} mm"
            else:
                note = f"启用 {run.spec.changed_step} 后验证 RMSE 不变"
        else:
            if delta < 0:
                note = (
                    f"警告：移除 {run.spec.changed_step} 后验证 RMSE 降低 {-delta:.6g} mm；"
                    "当前该步骤可能使验证误差变大"
                )
            elif delta > 0:
                note = f"移除 {run.spec.changed_step} 后验证 RMSE 增加 {delta:.6g} mm；支持保留该步骤"
            else:
                note = f"移除 {run.spec.changed_step} 后验证 RMSE 不变"
    plane = run.plane if run.plane is not None else np.full(4, np.nan)
    return {
        "experiment": run.spec.name,
        "label": run.spec.label,
        "category": run.spec.category,
        "reference_experiment": run.spec.reference_name,
        "changed_step": run.spec.changed_step,
        "use_roi": run.spec.use_roi,
        "use_background": run.spec.use_background,
        "use_prominence": run.spec.use_prominence,
        "use_subpixel": run.spec.use_subpixel,
        "use_quality": run.spec.use_quality,
        "use_chess_boundary": run.spec.use_chess_boundary,
        "use_continuity": run.spec.use_continuity,
        "use_ransac": run.spec.use_ransac,
        "plane_a": plane[0],
        "plane_b": plane[1],
        "plane_c": plane[2],
        "plane_d_mm": plane[3],
        **{f"train_{key}": value for key, value in train_metrics.items() if key not in {"experiment", "label", "category", "reference_experiment", "changed_step", "split"}},
        **{f"validation_{key}": value for key, value in validation_metrics.items() if key not in {"experiment", "label", "category", "reference_experiment", "changed_step", "split"}},
        "validation_rmse_delta_vs_reference_mm": delta,
        "effectiveness_note": note,
    }


def save_summary_plot(summary: pd.DataFrame, output_path: Path) -> None:
    cumulative = summary[summary["category"] == "cumulative"]
    removals = summary[summary["category"] == "single_removal"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    cumulative_rmse = pd.to_numeric(cumulative["validation_rmse_mm"], errors="coerce")
    cumulative_mae = pd.to_numeric(cumulative["validation_mae_mm"], errors="coerce")
    cumulative_p95 = pd.to_numeric(cumulative["validation_p95_mm"], errors="coerce")
    axes[0].plot(
        cumulative["experiment"], cumulative_rmse, marker="o", label="RMSE"
    )
    axes[0].plot(
        cumulative["experiment"], cumulative_mae, marker="s", label="MAE"
    )
    axes[0].plot(
        cumulative["experiment"], cumulative_p95, marker="^", label="P95"
    )
    axes[0].set_title("Cumulative ablation: validation error")
    axes[0].set_xlabel("Stage")
    axes[0].set_ylabel("Point-to-plane error (mm)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    removal_rmse = pd.to_numeric(removals["validation_rmse_mm"], errors="coerce")
    axes[1].bar(removals["experiment"], removal_rmse)
    full_rows = summary[summary["experiment"] == "A7"]
    if not full_rows.empty and pd.notna(full_rows.iloc[0]["validation_rmse_mm"]):
        axes[1].axhline(
            float(full_rows.iloc[0]["validation_rmse_mm"]),
            color="red", linestyle="--", label="A7 full"
        )
    axes[1].set_title("Single-step removal: validation RMSE")
    axes[1].set_xlabel("Removal experiment")
    axes[1].set_ylabel("RMSE (mm)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_colour_legend(output_dir: Path) -> None:
    labels = {
        "accepted_final": "有效点/最终中心线（绿色）",
        "rejected_saturation": "饱和点（红色）",
        "rejected_snr": "低 SNR 点（黄色）",
        "rejected_fwhm": "FWHM 异常点（蓝色）",
        "rejected_chess_boundary": "棋盘黑白边界点（紫色）",
        "rejected_continuity": "连续性异常点（橙色）",
        "rejected_roi": "ROI 外点（灰色）",
        "rejected_prominence": "主峰显著性不足（浅灰色）",
        "rejected_centroid": "重心失败（白色）",
        "rejected_intersection": "三维求交失败（青色）",
        "rejected_ransac": "RANSAC 外点（粉色）",
    }
    rows = []
    for status, bgr in STATUS_COLOURS.items():
        rows.append(
            {
                "status": status,
                "description": labels[status],
                "b": bgr[0],
                "g": bgr[1],
                "r": bgr[2],
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "colour_legend.csv", index=False, encoding="utf-8-sig"
    )


def write_all_outputs(
    runs: Mapping[str, ExperimentRun],
    specs: Sequence[ExperimentSpec],
    args: argparse.Namespace,
    config: Mapping[str, Any],
    intrinsics: Intrinsics,
) -> None:
    output_dir = args.output_dir
    metrics_by_experiment: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    report_rows: list[dict[str, Any]] = []
    validation_pose_frames: list[pd.DataFrame] = []

    for spec in specs:
        run = runs[spec.name]
        reference = runs.get(spec.reference_name) if spec.reference_name else None
        train = split_metrics(run, "train", reference)
        validation = split_metrics(run, "validation", reference)
        metrics_by_experiment[spec.name] = (train, validation)
        reference_rmse = None
        if reference is not None:
            reference_rmse = split_metrics(reference, "validation", None).get("rmse_mm")
        report = experiment_report_row(run, train, validation, reference_rmse)
        report_rows.append(report)
        if str(report["effectiveness_note"]).startswith("警告"):
            LOGGER.warning("%s：%s", spec.name, report["effectiveness_note"])

        directory = experiment_directory(output_dir, spec.name)
        pd.DataFrame([train, validation]).to_csv(
            directory / "ablation_metrics.csv", index=False, encoding="utf-8-sig"
        )
        pose_frame = validation_pose_metrics(run, reference)
        pose_frame.to_csv(
            directory / "validation_pose_metrics.csv", index=False, encoding="utf-8-sig"
        )
        validation_pose_frames.append(pose_frame)
        if run.plane is not None:
            plane_document = {
                "experiment": spec.name,
                "coordinate_system": "camera",
                "coordinate_unit": "mm",
                "equation": "a*x + b*y + c*z + d = 0",
                "normal_is_unit_length": True,
                "coefficients": dict(zip(("a", "b", "c", "d"), run.plane.tolist(), strict=True)),
                "training_ids": list(args.train_ids),
                "validation_ids": list(args.val_ids),
                "validation_used_for_fitting": False,
            }
            (directory / "plane.yaml").write_text(
                yaml.safe_dump(plane_document, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        save_experiment_plots(run, directory, config)

    summary = pd.DataFrame(report_rows)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "laser_plane_report.csv", index=False, encoding="utf-8-sig")
    if validation_pose_frames:
        pd.concat(validation_pose_frames, ignore_index=True).to_csv(
            output_dir / "all_validation_pose_metrics.csv", index=False, encoding="utf-8-sig"
        )
    save_summary_plot(summary, output_dir / "ablation_summary.png")
    write_colour_legend(output_dir)

    final_run = runs["A7"]
    if final_run.plane is None:
        raise CalibrationError(f"A7 最终平面拟合失败：{final_run.fit_error}")
    train_metrics, validation_metrics = metrics_by_experiment["A7"]
    warnings = [
        row["effectiveness_note"]
        for row in report_rows
        if str(row["effectiveness_note"]).startswith("警告")
    ]
    final_document = {
        "model": "normalized_plane_ax_by_cz_d_equals_0",
        "coordinate_system": "camera",
        "coordinate_unit": "mm",
        "normal_is_unit_length": True,
        "coefficients": dict(
            zip(("a", "b", "c", "d"), final_run.plane.tolist(), strict=True)
        ),
        "training_ids": list(args.train_ids),
        "validation_ids": list(args.val_ids),
        "validation_used_for_fitting_or_thresholds": False,
        "training_metrics": _serialisable(train_metrics),
        "validation_metrics": _serialisable(validation_metrics),
        "ablation_warnings": warnings,
    }
    (output_dir / "laser_plane.yaml").write_text(
        yaml.safe_dump(final_document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "laser_plane_config_used.yaml").write_text(
        yaml.safe_dump(_serialisable(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "intrinsics": str(args.intrinsics),
        "image_size": list(intrinsics.image_size),
        "image_dir": str(args.image_dir),
        "chess_pattern": args.chess_pattern,
        "laser_pattern": args.laser_pattern,
        "train_ids": list(args.train_ids),
        "validation_ids": list(args.val_ids),
        "pattern_cols": args.pattern_cols,
        "pattern_rows": args.pattern_rows,
        "square_size_mm": args.square_size_mm,
        "config": str(args.config),
        "experiments": [spec.name for spec in specs],
        "validation_processed_after_training_planes_frozen": True,
    }
    (output_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"输出路径存在但不是目录：{path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise CalibrationError(
            f"输出目录非空：{path}；如需覆盖同名结果，请添加 --overwrite-output"
        )
    path.mkdir(parents=True, exist_ok=True)


def configure_logging(level: str, output_dir: Path | None = None) -> None:
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    if output_dir is not None:
        file_handler = logging.FileHandler(output_dir / "processing_log.txt", encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def _failed_results_for_pair(
    pair: tuple[int, str, Path, Path],
    specs: Sequence[ExperimentSpec],
    image_width: int,
    message: str,
) -> dict[str, FrameResult]:
    image_id, split, chess_path, laser_path = pair
    return {
        spec.name: FrameResult(
            image_id,
            split,
            chess_path,
            laser_path,
            image_width,
            False,
            message,
            None,
            np.empty((0, 3)),
            pd.DataFrame(),
        )
        for spec in specs
    }


def process_pairs_for_all_experiments(
    pairs: Sequence[tuple[int, str, Path, Path]],
    specs: Sequence[ExperimentSpec],
    runs: Mapping[str, ExperimentRun],
    args: argparse.Namespace,
    intrinsics: Intrinsics,
    config: Mapping[str, Any],
) -> None:
    for pair in pairs:
        image_id, split, _, _ = pair
        LOGGER.info("准备 %s ID %03d，并独立运行 %d 个实验", split, image_id, len(specs))
        try:
            prepared = prepare_frame(pair, args, intrinsics, config)
            frame_results: dict[str, FrameResult] = {}
            for spec in specs:
                result = process_prepared_frame(prepared, spec, intrinsics, config)
                frame_results[spec.name] = result
                save_pre_fit_diagnostics(prepared, result, spec, args.output_dir, config)
        except (CalibrationError, cv2.error, OSError, ValueError) as exc:
            LOGGER.error("%s ID %03d 公共准备失败：%s", split, image_id, exc)
            frame_results = _failed_results_for_pair(
                pair, specs, intrinsics.image_size[0], str(exc)
            )
        for spec in specs:
            target = (
                runs[spec.name].train_results
                if split == "train"
                else runs[spec.name].validation_results
            )
            target.append(frame_results[spec.name])


def run(args: argparse.Namespace) -> np.ndarray:
    configure_logging(args.log_level)
    config = load_config(args.config)
    intrinsics = load_intrinsics(args.intrinsics)
    pairs = validate_inputs(args)
    prepare_output_dir(args.output_dir, args.overwrite_output)
    configure_logging(args.log_level, args.output_dir)
    specs = build_experiment_specs()
    runs = {spec.name: ExperimentRun(spec) for spec in specs}
    train_pairs = [pair for pair in pairs if pair[1] == "train"]
    validation_pairs = [pair for pair in pairs if pair[1] == "validation"]

    LOGGER.info("开始训练集累计/移除式消融：%d 个实验", len(specs))
    process_pairs_for_all_experiments(
        train_pairs, specs, runs, args, intrinsics, config
    )
    for spec in specs:
        fit_experiment(runs[spec.name], config)

    # Validation extraction starts only after every training plane is frozen.
    LOGGER.info("全部训练平面已冻结；开始独立验证集处理")
    process_pairs_for_all_experiments(
        validation_pairs, specs, runs, args, intrinsics, config
    )
    for run_item in runs.values():
        finalise_validation(run_item)

    LOGGER.info("生成逐图最终剔除图和中心线图")
    for spec in specs:
        run_item = runs[spec.name]
        for result in [*run_item.train_results, *run_item.validation_results]:
            if result.laser_path.is_file():
                save_post_fit_diagnostics(
                    result, spec, args.output_dir, config
                )
    write_all_outputs(runs, specs, args, config, intrinsics)
    LOGGER.info("全部消融结果已写入：%s", args.output_dir)
    final_plane = runs["A7"].plane
    if final_plane is None:
        raise CalibrationError("A7 未生成最终平面")
    return final_plane


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (CalibrationError, FileNotFoundError, NotADirectoryError, OSError) as exc:
        LOGGER.error("标定失败：%s", exc)
        return 2
    except Exception:
        LOGGER.exception("未预期的标定错误")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
