#!/usr/bin/env python3
"""标定相机坐标系到基准地面坐标系的外参。

空间单位统一为 mm。棋盘格仅估计地面法向；地面位置由无障碍地面激光点
沿该法向的中位数确定，避免使用近似共线的激光点单独拟合平面。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from line_laser_static.algorithms import MonoStripeExtractor  # noqa: E402
from line_laser_static.models import Frame, FrameMetadata  # noqa: E402


LOGGER = logging.getLogger("ground_extrinsics")
DEFAULT_INTRINSICS = (
    SCRIPT_DIR
    / "calibration"
    / "calib02"
    / "calibration_fix_k3_exclude_018_026_030_031_matlab"
    / "calibration_result.yaml"
)
DEFAULT_CALIBRATION_DIR = SCRIPT_DIR / "calibration" / "外参标定"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class CalibrationError(RuntimeError):
    """可预期、适合直接展示给用户的标定错误。"""


@dataclass(frozen=True)
class Intrinsics:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True)
class ChessObservation:
    path: Path
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    normal: np.ndarray
    reprojection_rmse_px: float


@dataclass(frozen=True)
class NormalEstimate:
    normal: np.ndarray
    aligned_normals: np.ndarray
    angular_errors_deg: np.ndarray
    inlier_mask: np.ndarray
    threshold_deg: float


@dataclass(frozen=True)
class LaserObservation:
    path: Path
    pixels: np.ndarray
    points_camera: np.ndarray
    total_columns: int
    quality_valid_count: int


class OpenCVSafeLoader(yaml.SafeLoader):
    """支持 OpenCV ``!!opencv-matrix`` 标签的安全 YAML loader。"""


def _opencv_matrix(loader: yaml.Loader, node: yaml.Node) -> np.ndarray:
    value = loader.construct_mapping(node, deep=True)
    return np.asarray(value["data"], dtype=np.float64).reshape(
        int(value["rows"]), int(value["cols"])
    )


OpenCVSafeLoader.add_constructor(
    "tag:yaml.org,2002:opencv-matrix", _opencv_matrix
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("必须是有限非负数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="由平行棋盘格和无障碍地面激光线标定相机到地面的外参。"
    )
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument(
        "--laser-plane",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR / "laser_plane.yaml",
        help="相机坐标系激光平面 YAML，方程 AX+BY+CZ+D=0",
    )
    parser.add_argument(
        "--chessboard-dir", type=Path, default=DEFAULT_CALIBRATION_DIR / "chessboard"
    )
    parser.add_argument(
        "--laser-dir", type=Path, default=DEFAULT_CALIBRATION_DIR / "laser_board"
    )
    parser.add_argument("--image-glob", default="*.tif")
    parser.add_argument("--pattern-cols", type=positive_int, default=6)
    parser.add_argument("--pattern-rows", type=positive_int, default=5)
    parser.add_argument("--square-size-mm", type=positive_float, default=30.0)
    parser.add_argument("--max-pnp-rmse-px", type=positive_float, default=1.0)
    parser.add_argument(
        "--normal-outlier-cap-deg",
        type=positive_float,
        default=3.0,
        help="棋盘法向异常剔除角度的绝对上限",
    )
    parser.add_argument("--centroid-window-radius", type=positive_int, default=4)
    parser.add_argument(
        "--background-percentile", type=nonnegative_float, default=20.0
    )
    parser.add_argument("--min-contrast", type=nonnegative_float, default=10.0)
    parser.add_argument("--min-snr", type=nonnegative_float, default=3.0)
    parser.add_argument("--noise-floor", type=positive_float, default=1.0)
    parser.add_argument("--min-fwhm-px", type=positive_float, default=0.5)
    parser.add_argument("--max-fwhm-px", type=positive_float, default=20.0)
    parser.add_argument(
        "--sensor-max-value",
        type=positive_float,
        default=None,
        help="传感器满量程；不指定时按图像实际位深推断",
    )
    parser.add_argument(
        "--reject-saturated",
        action="store_true",
        help="剔除峰值达到传感器满量程的列",
    )
    parser.add_argument("--continuity-window", type=positive_int, default=11)
    parser.add_argument(
        "--continuity-max-deviation-px", type=positive_float, default=2.5
    )
    parser.add_argument(
        "--line-ransac-threshold-px",
        type=positive_float,
        default=2.0,
        help="去畸变图像坐标中的激光直线 RANSAC 距离阈值",
    )
    parser.add_argument("--min-depth-mm", type=positive_float, default=1.0)
    parser.add_argument("--max-depth-mm", type=positive_float, default=10000.0)
    parser.add_argument(
        "--max-ground-z-std-mm", type=positive_float, default=1.0
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "ground_extrinsics",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录中的同名文件（不会删除其他文件）",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _normalise_opencv_yaml(text: str) -> str:
    return re.sub(r"^\s*%YAML:\d+(?:\.\d+)?\s*$", "", text, flags=re.MULTILINE)


def _as_array(value: Any, label: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, Mapping) and {"rows", "cols", "data"} <= set(value):
        return np.asarray(value["data"], dtype=np.float64).reshape(
            int(value["rows"]), int(value["cols"])
        )
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label} 不是数值数组") from exc


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    try:
        text = _normalise_opencv_yaml(path.read_text(encoding="utf-8-sig"))
        loaded = yaml.load(text, Loader=OpenCVSafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"无法读取 YAML {path}：{exc}") from exc
    if not isinstance(loaded, Mapping):
        raise CalibrationError(f"YAML 根节点必须是映射：{path}")
    data = loaded.get("opencv_storage", loaded)
    if not isinstance(data, Mapping):
        raise CalibrationError(f"opencv_storage 必须是映射：{path}")
    return data


def _first_value(data: Mapping[str, Any], keys: Sequence[str], label: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise CalibrationError(f"缺少 {label}，支持字段：{', '.join(keys)}")


def load_intrinsics(path: Path) -> Intrinsics:
    data = _read_yaml(path)
    camera_matrix = _as_array(
        _first_value(data, ("camera_matrix", "cameraMatrix", "K"), "相机矩阵"),
        "camera_matrix",
    )
    dist_coeffs = _as_array(
        _first_value(
            data,
            ("dist_coeffs", "distortion_coeffs", "distCoeffs", "D"),
            "畸变系数",
        ),
        "dist_coeffs",
    ).reshape(-1)
    if "image_width" in data and "image_height" in data:
        image_size = (int(data["image_width"]), int(data["image_height"]))
    else:
        size = _as_array(
            _first_value(data, ("image_size", "imageSize"), "图像尺寸"),
            "image_size",
        ).reshape(-1)
        if size.size != 2:
            raise CalibrationError("image_size 必须为 [width, height]")
        image_size = (int(size[0]), int(size[1]))
    if camera_matrix.shape != (3, 3):
        raise CalibrationError("camera_matrix 必须是 3x3")
    if dist_coeffs.size not in (4, 5, 8, 12, 14):
        raise CalibrationError("OpenCV 畸变系数数量必须为 4/5/8/12/14")
    if not np.isfinite(camera_matrix).all() or not np.isfinite(dist_coeffs).all():
        raise CalibrationError("相机内参包含 NaN/Inf")
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise CalibrationError("相机焦距必须为正数")
    return Intrinsics(image_size, camera_matrix, dist_coeffs)


def load_laser_plane(path: Path) -> np.ndarray:
    data = _read_yaml(path)
    source: Any = data.get(
        "coefficients",
        data.get("laser_plane", data.get("laser_plane_in_camera", data.get("plane", data))),
    )
    if isinstance(source, Mapping):
        lower = {str(key).lower(): value for key, value in source.items()}
        if "coefficients" in lower:
            source = lower["coefficients"]
            if isinstance(source, Mapping):
                lower = {str(key).lower(): value for key, value in source.items()}
            else:
                lower = {}
        if {"a", "b", "c", "d"} <= set(lower):
            plane = np.asarray([lower[key] for key in ("a", "b", "c", "d")])
        elif "normal" in lower and ("offset" in lower or "d" in lower):
            normal = _as_array(lower["normal"], "laser_plane.normal").reshape(-1)
            offset = lower.get("offset", lower.get("d"))
            plane = np.r_[normal, float(offset)]
        else:
            plane = _as_array(source, "laser_plane").reshape(-1)
    else:
        plane = _as_array(source, "laser_plane").reshape(-1)
    plane = np.asarray(plane, dtype=np.float64).reshape(-1)
    if plane.size != 4 or not np.isfinite(plane).all():
        raise CalibrationError("激光平面必须包含 4 个有限系数 [A,B,C,D]")
    normal_length = float(np.linalg.norm(plane[:3]))
    if normal_length <= np.finfo(float).eps:
        raise CalibrationError("激光平面法向长度为零")
    return plane / normal_length


def list_images(directory: Path, pattern: str) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"图像目录不存在：{directory}")
    images = sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise CalibrationError(f"目录中没有匹配 {pattern!r} 的图像：{directory}")
    return images


def read_image(path: Path) -> np.ndarray:
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
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
        raise CalibrationError(f"不支持图像形状 {image.shape}：{path}")
    if gray.dtype not in (np.uint8, np.uint16):
        raise CalibrationError(f"只支持 uint8/uint16 灰度图：{path}")
    return gray


def detection_u8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    low, high = np.percentile(gray, (0.5, 99.5))
    if high <= low:
        raise CalibrationError("图像没有可用灰度对比度")
    return np.clip((gray.astype(np.float64) - low) * 255.0 / (high - low), 0, 255).astype(
        np.uint8
    )


def display_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(detection_u8(gray), cv2.COLOR_GRAY2BGR)


def chessboard_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return points


def detect_chessboard(
    path: Path, intrinsics: Intrinsics, args: argparse.Namespace
) -> ChessObservation:
    gray = to_gray(read_image(path), path)
    expected_shape = (intrinsics.image_size[1], intrinsics.image_size[0])
    if gray.shape != expected_shape:
        raise CalibrationError(
            f"图像尺寸 {gray.shape[::-1]} 与内参 {intrinsics.image_size} 不一致"
        )
    image = detection_u8(gray)
    pattern = (args.pattern_cols, args.pattern_rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(image, pattern, flags)
    if found and corners is not None:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.001,
        )
        corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    else:
        sb_flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        found, corners = cv2.findChessboardCornersSB(image, pattern, sb_flags)
    if not found or corners is None:
        raise CalibrationError("未检测到完整棋盘格内角点")

    object_points = chessboard_object_points(
        args.pattern_cols, args.pattern_rows, args.square_size_mm
    )
    solved, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise CalibrationError("solvePnP 求解失败")
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    residual = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if rmse > args.max_pnp_rmse_px:
        raise CalibrationError(
            f"PnP 重投影 RMSE {rmse:.3f}px 超过 {args.max_pnp_rmse_px:.3f}px"
        )
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2].astype(np.float64)
    normal /= np.linalg.norm(normal)
    return ChessObservation(
        path, corners.reshape(-1, 2), rvec, tvec, normal, rmse
    )


def robust_average_normal(normals: np.ndarray, cap_deg: float) -> NormalEstimate:
    normals = np.asarray(normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) < 3:
        raise CalibrationError("至少需要 3 帧有效棋盘格法向")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= np.finfo(float).eps):
        raise CalibrationError("棋盘格法向包含零向量")
    unit = normals / lengths[:, None]

    # 外积对法向正负号不敏感，主特征向量可作为统一方向的初始参考。
    eigenvalues, eigenvectors = np.linalg.eigh(unit.T @ unit)
    reference = eigenvectors[:, int(np.argmax(eigenvalues))]
    aligned = unit.copy()
    aligned[(aligned @ reference) < 0.0] *= -1.0
    pairwise_angles = np.degrees(
        np.arccos(np.clip(aligned @ aligned.T, -1.0, 1.0))
    )
    medoid_index = int(np.argmin(np.median(pairwise_angles, axis=1)))
    angles = pairwise_angles[medoid_index]
    median = float(np.median(angles))
    mad = float(np.median(np.abs(angles - median)))
    robust_limit = max(0.1, median + 3.5 * 1.4826 * mad)
    threshold = min(float(cap_deg), robust_limit)
    inliers = angles <= threshold
    if int(np.count_nonzero(inliers)) < 3:
        raise CalibrationError(
            f"法向异常剔除后仅剩 {np.count_nonzero(inliers)} 帧，请检查棋盘图像"
        )
    mean = np.mean(aligned[inliers], axis=0)
    mean /= np.linalg.norm(mean)
    final_angles = np.degrees(np.arccos(np.clip(aligned @ mean, -1.0, 1.0)))
    return NormalEstimate(mean, aligned, final_angles, inliers, threshold)


def save_chessboard_overlay(
    observation: ChessObservation,
    accepted: bool,
    pattern: tuple[int, int],
    output_path: Path,
) -> None:
    gray = to_gray(read_image(observation.path), observation.path)
    overlay = display_bgr(gray)
    colour = (0, 200, 0) if accepted else (0, 0, 255)
    corners = observation.corners.reshape(-1, 1, 2).astype(np.float32)
    cv2.drawChessboardCorners(overlay, pattern, corners, True)
    for point in observation.corners:
        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 3, colour, -1, cv2.LINE_AA)
    label = f"{'INLIER' if accepted else 'NORMAL OUTLIER'}  PnP RMSE={observation.reprojection_rmse_px:.3f}px"
    cv2.putText(overlay, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    write_image(output_path, overlay)


def _infer_pixel_format(gray: np.ndarray, sensor_max_value: float | None) -> str:
    if sensor_max_value is not None:
        bits = max(1, int(math.ceil(math.log2(sensor_max_value + 1.0))))
    elif gray.dtype == np.uint8:
        bits = 8
    else:
        maximum = int(np.max(gray))
        bits = 12 if maximum <= 4095 else 16
    return f"Mono{bits}"


def _make_frame(gray: np.ndarray, pixel_format: str) -> Frame:
    height, width = gray.shape
    return Frame(
        image=gray,
        metadata=FrameMetadata(
            frame_id=0,
            timestamp_ns=0,
            width=width,
            height=height,
            exposure_us=0.0,
            gain_db=0.0,
            pixel_format=pixel_format,
            camera_model="calibration-image",
            serial_number="unknown",
            sdk_version="offline",
        ),
    )


def continuity_filter(
    u_px: np.ndarray,
    v_px: np.ndarray,
    valid: np.ndarray,
    window: int,
    max_deviation_px: float,
) -> np.ndarray:
    if window % 2 == 0:
        raise CalibrationError("--continuity-window 必须是奇数")
    result = valid.copy()
    radius = window // 2
    valid_indices = np.flatnonzero(valid)
    for index in valid_indices:
        left = max(0, index - radius)
        right = min(len(valid), index + radius + 1)
        neighbours = v_px[left:right][valid[left:right]]
        if neighbours.size >= 3 and abs(v_px[index] - np.median(neighbours)) > max_deviation_px:
            result[index] = False
    return result


def intersect_laser_plane(
    pixels: np.ndarray,
    intrinsics: Intrinsics,
    laser_plane: np.ndarray,
    min_depth_mm: float,
    max_depth_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)
    normalised = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    ).reshape(-1, 2)
    rays = np.column_stack([normalised, np.ones(len(normalised))])
    denominator = rays @ laser_plane[:3]
    scales = np.full(len(rays), np.nan, dtype=np.float64)
    stable = np.abs(denominator) > 1.0e-9
    scales[stable] = -laser_plane[3] / denominator[stable]
    valid = (
        stable
        & np.isfinite(scales)
        & (scales >= min_depth_mm)
        & (scales <= max_depth_mm)
    )
    return rays[valid] * scales[valid, None], valid


def line_ransac_filter(
    pixels: np.ndarray,
    intrinsics: Intrinsics,
    threshold_px: float,
    iterations: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """在去畸变像素坐标中剔除不属于主激光线的结构外点。"""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(pixels) < 2:
        return np.zeros(len(pixels), dtype=bool), np.full(len(pixels), np.inf)
    ideal = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        P=intrinsics.camera_matrix,
    ).reshape(-1, 2)
    generator = np.random.default_rng(20260722)
    best_mask = np.zeros(len(ideal), dtype=bool)
    best_median = math.inf
    for _ in range(iterations):
        first, second = generator.choice(len(ideal), size=2, replace=False)
        direction = ideal[second] - ideal[first]
        length = float(np.linalg.norm(direction))
        if length <= np.finfo(float).eps:
            continue
        direction /= length
        residual = np.abs(
            direction[0] * (ideal[:, 1] - ideal[first, 1])
            - direction[1] * (ideal[:, 0] - ideal[first, 0])
        )
        mask = residual <= threshold_px
        count = int(np.count_nonzero(mask))
        median = float(np.median(residual[mask])) if count else math.inf
        if count > np.count_nonzero(best_mask) or (
            count == np.count_nonzero(best_mask) and median < best_median
        ):
            best_mask = mask
            best_median = median
    minimum = max(30, int(math.ceil(0.5 * len(ideal))))
    if int(np.count_nonzero(best_mask)) < minimum:
        raise CalibrationError(
            f"激光直线 RANSAC 内点过少：{np.count_nonzero(best_mask)}/{len(ideal)}"
        )
    centred = ideal[best_mask] - np.mean(ideal[best_mask], axis=0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    direction = right[0]
    centre = np.mean(ideal[best_mask], axis=0)
    residual = np.abs(
        direction[0] * (ideal[:, 1] - centre[1])
        - direction[1] * (ideal[:, 0] - centre[0])
    )
    return residual <= threshold_px, residual


def process_laser_image(
    path: Path,
    intrinsics: Intrinsics,
    laser_plane: np.ndarray,
    args: argparse.Namespace,
    overlay_path: Path,
) -> LaserObservation:
    gray = to_gray(read_image(path), path)
    expected_shape = (intrinsics.image_size[1], intrinsics.image_size[0])
    if gray.shape != expected_shape:
        raise CalibrationError(
            f"图像尺寸 {gray.shape[::-1]} 与内参 {intrinsics.image_size} 不一致"
        )
    extractor = MonoStripeExtractor(
        window_radius=args.centroid_window_radius,
        background_percentile=args.background_percentile,
        min_contrast=args.min_contrast,
        min_snr=args.min_snr,
        noise_floor=args.noise_floor,
        min_fwhm_px=args.min_fwhm_px,
        max_fwhm_px=args.max_fwhm_px,
        reject_saturated=args.reject_saturated,
        sensor_max_value=args.sensor_max_value,
    )
    pixel_format = _infer_pixel_format(gray, args.sensor_max_value)
    profile = extractor.extract(_make_frame(gray, pixel_format))
    quality_valid = continuity_filter(
        profile.u_px,
        profile.v_px,
        profile.valid,
        args.continuity_window,
        args.continuity_max_deviation_px,
    )
    quality_indices = np.flatnonzero(quality_valid)
    candidate_pixels = np.column_stack(
        [profile.u_px[quality_indices], profile.v_px[quality_indices]]
    )
    line_inliers, _ = line_ransac_filter(
        candidate_pixels,
        intrinsics,
        args.line_ransac_threshold_px,
    )
    quality_valid[quality_indices[~line_inliers]] = False
    candidate_pixels = candidate_pixels[line_inliers]
    points, intersection_valid = intersect_laser_plane(
        candidate_pixels,
        intrinsics,
        laser_plane,
        args.min_depth_mm,
        args.max_depth_mm,
    )
    accepted_pixels = candidate_pixels[intersection_valid]
    if len(accepted_pixels) < 30:
        raise CalibrationError(f"有效地面激光点过少：{len(accepted_pixels)}")

    overlay = display_bgr(gray)
    rejected = np.flatnonzero(~quality_valid)
    for index in rejected[::4]:
        if np.isfinite(profile.v_px[index]):
            point = (int(round(profile.u_px[index])), int(round(profile.v_px[index])))
            cv2.circle(overlay, point, 1, (0, 0, 255), -1)
    for point in accepted_pixels:
        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 2, (0, 255, 0), -1)
    label = f"accepted={len(accepted_pixels)}/{len(profile.u_px)}"
    cv2.putText(overlay, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    write_image(overlay_path, overlay)
    return LaserObservation(
        path,
        accepted_pixels,
        points,
        len(profile.u_px),
        int(np.count_nonzero(quality_valid)),
    )


def build_ground_transform(
    normal_camera: np.ndarray, points_camera: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    normal = np.asarray(normal_camera, dtype=np.float64).reshape(3)
    normal /= np.linalg.norm(normal)
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3 or not np.isfinite(points).all():
        raise CalibrationError("确定地面位置至少需要 3 个有限三维点")

    # 严格使用已知法向上的中位数确定位置，不对近似共线点拟合平面。
    plane_d = -float(np.median(points @ normal))
    if plane_d < 0.0:
        normal = -normal
        plane_d = -plane_d
    if abs(normal[2]) < 1.0e-6:
        raise CalibrationError("主光轴与地面平面近似平行，无法定义所需原点")
    origin_camera = np.asarray([0.0, 0.0, -plane_d / normal[2]])

    camera_x = np.asarray([1.0, 0.0, 0.0])
    ground_x = camera_x - normal * float(camera_x @ normal)
    x_length = float(np.linalg.norm(ground_x))
    if x_length < 1.0e-6:
        raise CalibrationError("相机 +X 轴与地面法向近似平行，无法定义 +Xg")
    ground_x /= x_length
    ground_z = normal
    ground_y = np.cross(ground_z, ground_x)
    ground_y /= np.linalg.norm(ground_y)
    ground_x = np.cross(ground_y, ground_z)
    ground_x /= np.linalg.norm(ground_x)

    rotation = np.vstack([ground_x, ground_y, ground_z])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin_camera
    inverse = np.linalg.inv(transform)
    return transform, inverse, np.r_[normal, plane_d], plane_d


def transform_camera_to_ground(
    points_camera: np.ndarray, T_ground_from_camera: np.ndarray
) -> np.ndarray:
    """将最后一维为 XYZ 的相机坐标点转换为地面坐标点。"""
    points = np.asarray(points_camera, dtype=np.float64)
    transform = np.asarray(T_ground_from_camera, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError("points_camera 最后一维必须为 3")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_ground_from_camera 必须是有限的 4x4 矩阵")
    return points @ transform[:3, :3].T + transform[:3, 3]


def save_ground_residual_plot(points_ground: np.ndarray, output_path: Path) -> None:
    limit = 30000
    step = max(1, int(math.ceil(len(points_ground) / limit)))
    sampled = points_ground[::step]
    residual = points_ground[:, 2]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    scatter = axes[0].scatter(
        sampled[:, 0], sampled[:, 1], c=sampled[:, 2], s=3, cmap="coolwarm"
    )
    axes[0].set_title("Ground laser Z residual")
    axes[0].set_xlabel("Xg (mm)")
    axes[0].set_ylabel("Yg (mm)")
    axes[0].set_aspect("equal", adjustable="box")
    figure.colorbar(scatter, ax=axes[0], label="Zg (mm)")
    axes[1].hist(residual, bins=80, color="#2563eb", alpha=0.85)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_title(
        f"median={np.median(residual):.4f} mm, std={np.std(residual):.4f} mm"
    )
    axes[1].set_xlabel("Zg (mm)")
    axes[1].set_ylabel("Count")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _serialisable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _serialisable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(item) for item in value]
    return value


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise CalibrationError(f"输出路径不是目录：{path}")
    if path.is_dir() and any(path.iterdir()) and not overwrite:
        raise CalibrationError(f"输出目录非空；如需覆盖同名文件请加 --overwrite：{path}")
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.background_percentile <= 100.0:
        raise CalibrationError("--background-percentile 必须位于 [0, 100]")
    if args.min_fwhm_px >= args.max_fwhm_px:
        raise CalibrationError("--min-fwhm-px 必须小于 --max-fwhm-px")
    if args.min_depth_mm >= args.max_depth_mm:
        raise CalibrationError("--min-depth-mm 必须小于 --max-depth-mm")
    prepare_output_dir(args.output_dir, args.overwrite)
    intrinsics = load_intrinsics(args.intrinsics)
    laser_plane = load_laser_plane(args.laser_plane)
    chess_paths = list_images(args.chessboard_dir, args.image_glob)
    laser_paths = list_images(args.laser_dir, args.image_glob)
    LOGGER.info("读取 %d 张棋盘图、%d 张地面激光图", len(chess_paths), len(laser_paths))

    chess_observations: list[ChessObservation] = []
    chess_failures: list[dict[str, str]] = []
    for path in chess_paths:
        try:
            chess_observations.append(detect_chessboard(path, intrinsics, args))
        except (CalibrationError, cv2.error, ValueError) as exc:
            chess_failures.append({"image": str(path.resolve()), "reason": str(exc)})
            LOGGER.warning("棋盘图跳过 %s：%s", path.name, exc)
    if len(chess_observations) < 3:
        raise CalibrationError(
            f"仅 {len(chess_observations)} 张棋盘图成功，至少需要 3 张"
        )
    normal_estimate = robust_average_normal(
        np.vstack([item.normal for item in chess_observations]),
        args.normal_outlier_cap_deg,
    )
    chess_dir = args.output_dir / "diagnostics" / "chessboard"
    for observation, accepted in zip(
        chess_observations, normal_estimate.inlier_mask, strict=True
    ):
        save_chessboard_overlay(
            observation,
            bool(accepted),
            (args.pattern_cols, args.pattern_rows),
            chess_dir / f"{observation.path.stem}.png",
        )

    laser_observations: list[LaserObservation] = []
    laser_failures: list[dict[str, str]] = []
    laser_dir = args.output_dir / "diagnostics" / "laser_overlay"
    for path in laser_paths:
        try:
            laser_observations.append(
                process_laser_image(
                    path,
                    intrinsics,
                    laser_plane,
                    args,
                    laser_dir / f"{path.stem}.png",
                )
            )
        except (CalibrationError, cv2.error, ValueError, FloatingPointError) as exc:
            laser_failures.append({"image": str(path.resolve()), "reason": str(exc)})
            LOGGER.warning("地面激光图跳过 %s：%s", path.name, exc)
    if not laser_observations:
        raise CalibrationError("所有地面激光图均处理失败")
    points_camera = np.vstack([item.points_camera for item in laser_observations])
    transform, inverse, ground_plane, camera_height = build_ground_transform(
        normal_estimate.normal, points_camera
    )
    points_ground = transform_camera_to_ground(points_camera, transform)

    rotation = transform[:3, :3]
    orthogonality_error = float(np.linalg.norm(rotation @ rotation.T - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    inverse_error = float(np.linalg.norm(transform @ inverse - np.eye(4)))
    ground_z_std = float(np.std(points_ground[:, 2]))
    ground_z_median = float(np.median(points_ground[:, 2]))
    if orthogonality_error > 1.0e-9 or abs(determinant - 1.0) > 1.0e-9:
        raise CalibrationError("构造的旋转矩阵未通过正交性/行列式检查")
    z_std_passed = ground_z_std <= args.max_ground_z_std_mm
    if not z_std_passed:
        LOGGER.warning(
            "地面点 Z 标准差 %.4f mm 超过阈值 %.4f mm",
            ground_z_std,
            args.max_ground_z_std_mm,
        )

    optical_axis = np.asarray([0.0, 0.0, 1.0])
    tilt_from_normal = math.degrees(
        math.acos(float(np.clip(abs(ground_plane[:3] @ optical_axis), 0.0, 1.0)))
    )
    tilt_from_plane = 90.0 - tilt_from_normal
    save_ground_residual_plot(
        points_ground, args.output_dir / "diagnostics" / "ground_z_residual.png"
    )

    chess_frames = []
    for item, angle, accepted in zip(
        chess_observations,
        normal_estimate.angular_errors_deg,
        normal_estimate.inlier_mask,
        strict=True,
    ):
        chess_frames.append(
            {
                "image": item.path,
                "reprojection_rmse_px": item.reprojection_rmse_px,
                "normal_camera": item.normal,
                "angular_error_deg": float(angle),
                "accepted": bool(accepted),
            }
        )
    laser_frames = [
        {
            "image": item.path,
            "total_columns": item.total_columns,
            "quality_valid_count": item.quality_valid_count,
            "reconstructed_count": len(item.points_camera),
        }
        for item in laser_observations
    ]
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "units": "mm",
        "coordinate_convention": {
            "Xg": "camera +X projected onto ground; approximately image right",
            "Zg": "ground normal toward camera; positive for protrusions",
            "Yg": "Zg cross Xg; approximately image up",
            "origin": "intersection of camera principal axis and ground plane",
        },
        "T_ground_from_camera": transform,
        "T_camera_from_ground": inverse,
        "ground_plane_in_camera": {
            "equation": "n_x*Xc + n_y*Yc + n_z*Zc + D = 0",
            "coefficients": ground_plane,
            "normal": ground_plane[:3],
            "D_mm": ground_plane[3],
        },
        "ground_origin_in_camera_mm": inverse[:3, 3],
        "ground_axes_in_camera": {
            "Xg": rotation[0],
            "Yg": rotation[1],
            "Zg": rotation[2],
        },
        "camera_installation": {
            "optical_axis_to_ground_normal_deg": tilt_from_normal,
            "optical_axis_to_ground_plane_deg": tilt_from_plane,
            "camera_height_above_ground_mm": camera_height,
        },
        "laser_plane_in_camera": {
            "equation": "A*Xc + B*Yc + C*Zc + D = 0",
            "coefficients": laser_plane,
        },
        "quality_checks": {
            "rotation_orthogonality_frobenius": orthogonality_error,
            "rotation_determinant": determinant,
            "transform_inverse_frobenius": inverse_error,
            "ground_z_median_mm": ground_z_median,
            "ground_z_std_mm": ground_z_std,
            "max_ground_z_std_mm": args.max_ground_z_std_mm,
            "ground_z_std_passed": z_std_passed,
        },
        "normal_estimation": {
            "detected_frame_count": len(chess_observations),
            "accepted_frame_count": int(np.count_nonzero(normal_estimate.inlier_mask)),
            "outlier_threshold_deg": normal_estimate.threshold_deg,
            "frames": chess_frames,
            "failures": chess_failures,
        },
        "ground_laser": {
            "processed_frame_count": len(laser_observations),
            "point_count": len(points_camera),
            "frames": laser_frames,
            "failures": laser_failures,
        },
        "inputs": {
            "intrinsics": args.intrinsics,
            "laser_plane": args.laser_plane,
            "chessboard_dir": args.chessboard_dir,
            "laser_dir": args.laser_dir,
            "pattern_cols": args.pattern_cols,
            "pattern_rows": args.pattern_rows,
            "square_size_mm": args.square_size_mm,
        },
    }
    serialised = _serialisable(result)
    (args.output_dir / "camera_ground_extrinsics.yaml").write_text(
        yaml.safe_dump(serialised, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (args.output_dir / "camera_ground_extrinsics.json").write_text(
        json.dumps(serialised, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("外参标定完成：%s", args.output_dir.resolve())
    LOGGER.info(
        "det(R)=%.12f, 正交误差=%.3e, 地面 Z std=%.4f mm",
        determinant,
        orthogonality_error,
        ground_z_std,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    try:
        run(args)
    except (CalibrationError, FileNotFoundError, NotADirectoryError, OSError, cv2.error) as exc:
        LOGGER.error("标定失败：%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
