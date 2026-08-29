#!/usr/bin/env python3
"""可复用的 Mono8 单图/目录线激光三维截面重建入口。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

try:
    import cv2
    import matplotlib
    import numpy as np
    import pandas as pd
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - 仅在依赖缺失时触发
    raise SystemExit(
        "缺少运行依赖。请执行：python -m pip install numpy opencv-python "
        "pyyaml pandas matplotlib"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".bmp"}
CSV_COLUMNS = [
    "u",
    "v",
    "Xc_mm",
    "Yc_mm",
    "Zc_mm",
    "Xg_mm",
    "Yg_mm",
    "Zg_mm",
]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "reconstruct_ground_pointcloud_v3.yaml"


class ReconstructionError(RuntimeError):
    """输入、标定或重建过程不满足约束。"""


@dataclass(frozen=True)
class Calibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int] | None
    laser_plane: np.ndarray
    T_ground_from_camera: np.ndarray


@dataclass(frozen=True)
class ExtractionParams:
    background_kernel: int = 51
    min_local_contrast_dn: float = 20.0
    centroid_window_radius: int = 5
    segment_min_columns: int = 42
    continuity_max_column_gap: int = 2
    continuity_max_vertical_jump: float = 14.0
    correction_window: int = 7
    correction_max_shift: float = 3.5


@dataclass(frozen=True)
class ReconstructionParams:
    parallel_epsilon: float = 1.0e-9
    min_camera_depth_mm: float = 100.0
    max_camera_depth_mm: float = 1500.0


@dataclass(frozen=True)
class ObstacleFitParams:
    ground_quantile: float = 0.40
    min_height_mm: float = 5.0
    noise_sigma_multiplier: float = 6.0
    min_points: int = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="重建单张或一批 Mono8 线激光图像的地面坐标三维截面。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"统一配置 YAML（默认：{DEFAULT_CONFIG}）",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image",
        type=Path,
        help="单张待测 Mono8 激光图像（tif/tiff/png/bmp）",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="批处理 Mono8 激光图像目录；未指定输入时使用配置值",
    )
    parser.add_argument("--intrinsics", type=Path, help="camera_intrinsics.yaml")
    parser.add_argument("--laser-plane", type=Path, help="laser_plane.yaml")
    parser.add_argument("--extrinsics", type=Path, help="camera_ground_extrinsics.yaml")
    parser.add_argument("--output-dir", type=Path, help="结果输出目录")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReconstructionError(f"YAML 文件不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ReconstructionError(f"无法读取 YAML：{path}：{exc}") from exc
    if not isinstance(data, dict):
        raise ReconstructionError(f"YAML 顶层必须是映射：{path}")
    return data


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReconstructionError(f"{name} 必须是数值矩阵") from exc
    if array.shape != shape:
        raise ReconstructionError(f"{name} 尺寸必须为 {shape}，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ReconstructionError(f"{name} 包含 NaN 或无穷值")
    return array


def load_calibration(
    intrinsics_path: Path, laser_plane_path: Path, extrinsics_path: Path
) -> Calibration:
    intrinsics = load_yaml(intrinsics_path)
    camera_matrix = _finite_array(intrinsics.get("camera_matrix"), (3, 3), "camera_matrix")
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise ReconstructionError("camera_matrix 的 fx、fy 必须为正数")
    if not np.allclose(camera_matrix[2], [0.0, 0.0, 1.0], atol=1.0e-9):
        raise ReconstructionError("camera_matrix 最后一行必须为 [0, 0, 1]")

    try:
        dist_coeffs = np.asarray(intrinsics.get("dist_coeffs"), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ReconstructionError("dist_coeffs 必须是一维数值数组") from exc
    if dist_coeffs.size not in {4, 5, 8, 12, 14} or not np.isfinite(dist_coeffs).all():
        raise ReconstructionError("dist_coeffs 长度必须为 4/5/8/12/14，且全部有限")

    width = intrinsics.get("image_width")
    height = intrinsics.get("image_height")
    image_size: tuple[int, int] | None = None
    if width is not None or height is not None:
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ReconstructionError("image_width 和 image_height 必须同时为正整数")
        image_size = (width, height)

    plane_yaml = load_yaml(laser_plane_path)
    if str(plane_yaml.get("coordinate_system", "")).lower() != "camera":
        raise ReconstructionError("laser_plane.yaml 的 coordinate_system 必须为 camera")
    if str(plane_yaml.get("coordinate_unit", "")).lower() != "mm":
        raise ReconstructionError("laser_plane.yaml 的 coordinate_unit 必须为 mm")
    coeffs = plane_yaml.get("coefficients")
    if not isinstance(coeffs, dict):
        raise ReconstructionError("laser_plane.yaml 缺少 coefficients 映射")
    try:
        laser_plane = np.asarray(
            [coeffs[key] for key in ("a", "b", "c", "d")], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionError("激光平面必须包含数值 a、b、c、d") from exc
    if not np.isfinite(laser_plane).all():
        raise ReconstructionError("激光平面系数包含 NaN 或无穷值")
    normal_norm = float(np.linalg.norm(laser_plane[:3]))
    if normal_norm <= 1.0e-12:
        raise ReconstructionError("激光平面法向量长度不能为零")
    laser_plane = laser_plane / normal_norm

    extrinsics = load_yaml(extrinsics_path)
    if str(extrinsics.get("units", "")).lower() != "mm":
        raise ReconstructionError("camera_ground_extrinsics.yaml 的 units 必须为 mm")
    transform = _finite_array(
        extrinsics.get("T_ground_from_camera"), (4, 4), "T_ground_from_camera"
    )
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ReconstructionError("T_ground_from_camera 最后一行必须为 [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1.0e-6 or abs(determinant - 1.0) > 1.0e-6:
        raise ReconstructionError(
            "T_ground_from_camera 的旋转矩阵无效："
            f"正交误差={orthogonality_error:.3e}，det={determinant:.9f}"
        )

    inverse_value = extrinsics.get("T_camera_from_ground")
    if inverse_value is not None:
        inverse = _finite_array(inverse_value, (4, 4), "T_camera_from_ground")
        inverse_error = float(np.linalg.norm(inverse @ transform - np.eye(4), ord="fro"))
        if inverse_error > 1.0e-6:
            raise ReconstructionError(
                "T_camera_from_ground 与 T_ground_from_camera 不互逆："
                f"误差={inverse_error:.3e}"
            )

    convention = extrinsics.get("coordinate_convention")
    if not isinstance(convention, dict) or not all(key in convention for key in ("Xg", "Yg", "Zg")):
        raise ReconstructionError("外参 YAML 必须声明 coordinate_convention.Xg/Yg/Zg")

    return Calibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        laser_plane=laser_plane,
        T_ground_from_camera=transform,
    )


def _load_dataclass(section: Any, cls: type[Any], section_name: str) -> Any:
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ReconstructionError(f"配置项 {section_name} 必须是映射")
    valid_names = {item.name for item in fields(cls)}
    unknown = sorted(set(section) - valid_names)
    if unknown:
        raise ReconstructionError(f"配置项 {section_name} 含未知字段：{', '.join(unknown)}")
    try:
        result = cls(**section)
    except TypeError as exc:
        raise ReconstructionError(f"配置项 {section_name} 无效：{exc}") from exc
    return result


def validate_params(
    extraction: ExtractionParams,
    reconstruction: ReconstructionParams,
    obstacle_fit: ObstacleFitParams,
) -> None:
    if extraction.background_kernel < 3 or extraction.background_kernel % 2 == 0:
        raise ReconstructionError("background_kernel 必须是大于等于 3 的奇数")
    if not 0.0 <= extraction.min_local_contrast_dn <= 255.0:
        raise ReconstructionError("min_local_contrast_dn 必须位于 [0, 255]")
    if extraction.centroid_window_radius < 1:
        raise ReconstructionError("centroid_window_radius 必须大于等于 1")
    if extraction.segment_min_columns < 2:
        raise ReconstructionError("segment_min_columns 必须大于等于 2")
    if extraction.correction_window < 1 or extraction.correction_window % 2 == 0:
        raise ReconstructionError("correction_window 必须为正奇数")
    if extraction.continuity_max_column_gap < 1:
        raise ReconstructionError("continuity_max_column_gap 必须大于等于 1")
    if extraction.continuity_max_vertical_jump <= 0.0:
        raise ReconstructionError("continuity_max_vertical_jump 必须为正数")
    if reconstruction.parallel_epsilon <= 0.0:
        raise ReconstructionError("parallel_epsilon 必须为正数")
    if not 0.0 < reconstruction.min_camera_depth_mm < reconstruction.max_camera_depth_mm:
        raise ReconstructionError("工作距离必须满足 0 < min_camera_depth_mm < max_camera_depth_mm")
    if not 0.0 < obstacle_fit.ground_quantile < 0.5:
        raise ReconstructionError("obstacle_fit.ground_quantile 必须位于 (0, 0.5)")
    if obstacle_fit.min_height_mm <= 0.0:
        raise ReconstructionError("obstacle_fit.min_height_mm 必须为正数")
    if obstacle_fit.noise_sigma_multiplier <= 0.0:
        raise ReconstructionError("obstacle_fit.noise_sigma_multiplier 必须为正数")
    if obstacle_fit.min_points < 2:
        raise ReconstructionError("obstacle_fit.min_points 必须大于等于 2")


def _correct_segment_v(
    values: np.ndarray, contrast: np.ndarray, params: ExtractionParams
) -> np.ndarray:
    if params.correction_window == 1 or len(values) < 3:
        return values
    weights = contrast / max(float(np.max(contrast)), 1.0e-6) + 1.0e-4
    radius = params.correction_window // 2
    corrected = values.copy()
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        estimate = float(
            np.sum(weights[left:right] * values[left:right])
            / np.sum(weights[left:right])
        )
        shift = np.clip(
            estimate - values[index],
            -params.correction_max_shift,
            params.correction_max_shift,
        )
        corrected[index] = values[index] + shift
    return corrected


def extract_laser_centres(
    gray: np.ndarray, params: ExtractionParams
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    """逐列提取亚像素灰度重心；只筛选已有点，不填补分段缺口。"""
    background = cv2.GaussianBlur(
        gray, (params.background_kernel, params.background_kernel), 0
    )
    signal = cv2.subtract(gray, background).astype(np.float32)
    image_height, image_width = gray.shape
    columns = np.arange(image_width)
    peak_rows = np.argmax(signal, axis=0)
    peak_contrast = signal[peak_rows, columns]

    radius = params.centroid_window_radius
    offsets = np.arange(-radius, radius + 1)[:, None]
    raw_rows = peak_rows[None, :] + offsets
    inside = (raw_rows >= 0) & (raw_rows < image_height)
    sample_rows = np.clip(raw_rows, 0, image_height - 1)
    weights = np.take_along_axis(signal, sample_rows, axis=0) * inside
    weight_sum = weights.sum(axis=0)
    centre_v = np.full(image_width, np.nan, dtype=np.float64)
    nonzero = weight_sum > 0.0
    centre_v[nonzero] = (
        (weights[:, nonzero] * raw_rows[:, nonzero]).sum(axis=0)
        / weight_sum[nonzero]
    )
    valid = (
        nonzero
        & np.isfinite(centre_v)
        & (peak_contrast >= params.min_local_contrast_dn)
    )
    candidate_u = columns[valid].astype(np.float64)
    candidate_v = centre_v[valid]
    candidate_contrast = peak_contrast[valid].astype(np.float64)

    if len(candidate_u):
        breaks = np.where(
            (np.diff(candidate_u) > params.continuity_max_column_gap)
            | (np.abs(np.diff(candidate_v)) > params.continuity_max_vertical_jump)
        )[0] + 1
        raw_segments = np.split(np.arange(len(candidate_u)), breaks)
    else:
        raw_segments = []
    accepted = [
        indexes
        for indexes in raw_segments
        if len(indexes) >= params.segment_min_columns
    ]
    points_by_segment: list[np.ndarray] = []
    for indexes in accepted:
        corrected_v = _correct_segment_v(
            candidate_v[indexes], candidate_contrast[indexes], params
        )
        points_by_segment.append(
            np.column_stack([candidate_u[indexes], corrected_v])
        )
    points = (
        np.concatenate(points_by_segment, axis=0)
        if points_by_segment
        else np.empty((0, 2), dtype=np.float64)
    )
    metadata = {
        "min_local_contrast_dn": float(params.min_local_contrast_dn),
        "candidate_point_count": float(len(candidate_u)),
        "raw_segment_count": float(len(raw_segments)),
        "accepted_segment_count": float(len(accepted)),
        "extracted_point_count": float(len(points)),
    }
    return points, metadata, signal.astype(np.uint8)


def reconstruct_points(
    pixels_uv: np.ndarray,
    calibration: Calibration,
    params: ReconstructionParams,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if pixels_uv.size == 0:
        return pd.DataFrame(columns=CSV_COLUMNS), {
            "near_parallel": 0,
            "negative_depth": 0,
            "outside_working_distance": 0,
            "non_finite": 0,
        }
    pixels = pixels_uv.reshape(-1, 1, 2).astype(np.float64)
    normalized = cv2.undistortPoints(
        pixels, calibration.camera_matrix, calibration.dist_coeffs
    ).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
    denominator = rays @ calibration.laser_plane[:3]
    stable = np.abs(denominator) > params.parallel_epsilon
    scale = np.full(len(rays), np.nan, dtype=np.float64)
    scale[stable] = -calibration.laser_plane[3] / denominator[stable]
    points_camera = rays * scale[:, None]

    finite = np.isfinite(points_camera).all(axis=1) & np.isfinite(scale)
    positive = scale > 0.0
    within_distance = (
        (points_camera[:, 2] >= params.min_camera_depth_mm)
        & (points_camera[:, 2] <= params.max_camera_depth_mm)
    )
    valid = stable & finite & positive & within_distance
    filtered = {
        "near_parallel": int(np.count_nonzero(~stable)),
        "negative_depth": int(np.count_nonzero(stable & finite & ~positive)),
        "outside_working_distance": int(
            np.count_nonzero(stable & finite & positive & ~within_distance)
        ),
        "non_finite": int(np.count_nonzero(stable & ~finite)),
    }
    points_camera = points_camera[valid]
    valid_pixels = pixels_uv[valid]
    homogeneous = np.column_stack(
        [points_camera, np.ones(len(points_camera), dtype=np.float64)]
    )
    # 方向严格为 ground <- camera；不得改用逆矩阵，也不对 Zg 取绝对值。
    points_ground = (
        calibration.T_ground_from_camera @ homogeneous.T
    ).T[:, :3]
    final_finite = np.isfinite(points_ground).all(axis=1)
    filtered["non_finite"] += int(np.count_nonzero(~final_finite))
    points_camera = points_camera[final_finite]
    points_ground = points_ground[final_finite]
    valid_pixels = valid_pixels[final_finite]

    values = np.column_stack([valid_pixels, points_camera, points_ground])
    return pd.DataFrame(values, columns=CSV_COLUMNS), filtered


def read_mono8(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error) as exc:
        raise ReconstructionError(f"图像读取失败：{path}：{exc}") from exc
    if image is None:
        raise ReconstructionError(f"OpenCV 无法解码图像：{path}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ReconstructionError(
            f"图像必须是 Mono8（二位 uint8），实际 shape={image.shape}、dtype={image.dtype}"
        )
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower()
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ReconstructionError(f"OpenCV 无法编码图像：{path}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise ReconstructionError(f"图像写入失败：{path}：{exc}") from exc


def save_overlay(gray: np.ndarray, pixels: np.ndarray, path: Path) -> None:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for u, v in pixels:
        point = (int(round(float(u))), int(round(float(v))))
        cv2.circle(overlay, point, 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    write_image(path, overlay)


def save_pointcloud_plot(frame: pd.DataFrame, path: Path, title: str) -> None:
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    if frame.empty:
        axis.text2D(0.5, 0.5, "No valid points", transform=axis.transAxes, ha="center")
    else:
        xyz = frame[["Xg_mm", "Yg_mm", "Zg_mm"]].to_numpy()
        scatter = axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap="turbo", s=5)
        colorbar = figure.colorbar(scatter, ax=axis, pad=0.10)
        colorbar.set_label("Zg (mm)")
        # 仅调整显示盒比例，数据坐标保持原始毫米值，避免狭长截面的标签互相遮挡。
        axis.set_box_aspect((1.8, 1.0, 1.0))
    axis.xaxis.set_major_locator(MaxNLocator(6))
    axis.yaxis.set_major_locator(MaxNLocator(6))
    axis.zaxis.set_major_locator(MaxNLocator(6))
    axis.set_xlabel("Xg (mm)")
    axis.set_ylabel("Yg (mm)")
    axis.set_zlabel("Zg (mm)")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def calculate_top_view_limits(
    x_values: np.ndarray,
    y_values: np.ndarray,
    image_aspect_ratio: float,
    padding_fraction: float = 0.05,
) -> tuple[float, float, float, float]:
    """扩展较短坐标轴，使显示窗口宽高比接近原图，同时覆盖全部点。"""
    if image_aspect_ratio <= 0.0 or not np.isfinite(image_aspect_ratio):
        raise ValueError("image_aspect_ratio 必须是有限正数")
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_span = max(x_max - x_min, 1.0) * (1.0 + 2.0 * padding_fraction)
    y_span = max(y_max - y_min, 1.0) * (1.0 + 2.0 * padding_fraction)
    if x_span / y_span > image_aspect_ratio:
        y_span = x_span / image_aspect_ratio
    else:
        x_span = y_span * image_aspect_ratio
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    return (
        x_center - 0.5 * x_span,
        x_center + 0.5 * x_span,
        y_center - 0.5 * y_span,
        y_center + 0.5 * y_span,
    )


def fit_obstacle_line(
    frame: pd.DataFrame, params: ObstacleFitParams
) -> dict[str, Any]:
    """以稳健地面基准筛出凸起点，并在 X-Y 平面做正交直线拟合。"""
    if len(frame) < params.min_points:
        return {
            "found": False,
            "reason": "有效点不足",
            "obstacle_point_count": 0,
        }
    xyz = frame[["Xg_mm", "Yg_mm", "Zg_mm"]].to_numpy(dtype=np.float64)
    zg = xyz[:, 2]
    ground_limit = float(np.quantile(zg, params.ground_quantile))
    ground_samples = zg[zg <= ground_limit]
    ground_baseline = float(np.median(ground_samples))
    ground_sigma = 1.4826 * float(
        np.median(np.abs(ground_samples - ground_baseline))
    )
    height_threshold = max(
        params.min_height_mm,
        params.noise_sigma_multiplier * ground_sigma,
    )
    obstacle_mask = (zg - ground_baseline) >= height_threshold
    obstacle_xyz = xyz[obstacle_mask]
    if len(obstacle_xyz) < params.min_points:
        return {
            "found": False,
            "reason": "高于阈值的障碍物点不足",
            "obstacle_point_count": int(len(obstacle_xyz)),
            "ground_baseline_zg_mm": ground_baseline,
            "height_threshold_mm": float(height_threshold),
            "obstacle_mask": obstacle_mask,
        }

    obstacle_xy = obstacle_xyz[:, :2]
    centre = np.mean(obstacle_xy, axis=0)
    _, _, right_vectors = np.linalg.svd(obstacle_xy - centre, full_matrices=False)
    direction = right_vectors[0]
    if direction[0] < 0.0:
        direction = -direction
    projections = (obstacle_xy - centre) @ direction
    fitted_xy = centre + projections[:, None] * direction
    orthogonal_residuals = np.linalg.norm(obstacle_xy - fitted_xy, axis=1)
    endpoints = np.vstack(
        [
            centre + float(np.min(projections)) * direction,
            centre + float(np.max(projections)) * direction,
        ]
    )
    relative_heights = obstacle_xyz[:, 2] - ground_baseline
    return {
        "found": True,
        "obstacle_point_count": int(len(obstacle_xyz)),
        "ground_baseline_zg_mm": ground_baseline,
        "ground_noise_sigma_mm": ground_sigma,
        "height_threshold_mm": float(height_threshold),
        "mean_height_mm": float(np.mean(relative_heights)),
        "median_height_mm": float(np.median(relative_heights)),
        "line_centre_xg_mm": float(centre[0]),
        "line_centre_yg_mm": float(centre[1]),
        "line_direction_x": float(direction[0]),
        "line_direction_y": float(direction[1]),
        "line_fit_rmse_mm": float(np.sqrt(np.mean(orthogonal_residuals**2))),
        "line_endpoints_xy": endpoints,
        "obstacle_mask": obstacle_mask,
    }


def save_obstacle_line_fit(
    frame: pd.DataFrame,
    fit: dict[str, Any],
    path: Path,
    title: str,
    image_aspect_ratio: float,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    if frame.empty:
        axis.text(0.5, 0.5, "No valid points", transform=axis.transAxes, ha="center")
    else:
        axis.scatter(
            frame["Xg_mm"],
            frame["Yg_mm"],
            c="#aeb6bf",
            s=5,
            label="All points",
        )
        if bool(fit.get("found")):
            mask = np.asarray(fit["obstacle_mask"], dtype=bool)
            obstacle = frame.loc[mask]
            scatter = axis.scatter(
                obstacle["Xg_mm"],
                obstacle["Yg_mm"],
                c=obstacle["Zg_mm"],
                cmap="turbo",
                s=9,
                label="Obstacle points",
            )
            colorbar = figure.colorbar(scatter, ax=axis)
            colorbar.set_label("Zg (mm)")
            endpoints = np.asarray(fit["line_endpoints_xy"], dtype=np.float64)
            axis.plot(
                endpoints[:, 0],
                endpoints[:, 1],
                color="#0068b5",
                linewidth=2.2,
                linestyle="--",
                label="PCA fitted line",
            )
            annotation = (
                f"Mean height = {fit['mean_height_mm']:.2f} mm\n"
                f"Ground baseline = {fit['ground_baseline_zg_mm']:.2f} mm\n"
                f"Line RMSE = {fit['line_fit_rmse_mm']:.3f} mm\n"
                f"Obstacle points = {fit['obstacle_point_count']}"
            )
        else:
            annotation = f"Obstacle line not found\n{fit.get('reason', 'unknown reason')}"
        axis.text(
            0.02,
            0.98,
            annotation,
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
        )
        x0, x1, y0, y1 = calculate_top_view_limits(
            frame["Xg_mm"].to_numpy(dtype=np.float64),
            frame["Yg_mm"].to_numpy(dtype=np.float64),
            image_aspect_ratio,
        )
        axis.set_xlim(x0, x1)
        axis.set_ylim(y0, y1)
        axis.legend(loc="lower right")
    axis.set_xlabel("Xg (mm)")
    axis.set_ylabel("Yg (mm)")
    axis.set_title(f"Obstacle line fit | {title}")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_top_view(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    image_aspect_ratio: float,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    if frame.empty:
        axis.text(0.5, 0.5, "No valid points", transform=axis.transAxes, ha="center")
    else:
        scatter = axis.scatter(
            frame["Xg_mm"], frame["Yg_mm"], c=frame["Zg_mm"], cmap="turbo", s=7
        )
        colorbar = figure.colorbar(scatter, ax=axis)
        colorbar.set_label("Zg (mm)")
        x0, x1, y0, y1 = calculate_top_view_limits(
            frame["Xg_mm"].to_numpy(dtype=np.float64),
            frame["Yg_mm"].to_numpy(dtype=np.float64),
            image_aspect_ratio,
        )
        axis.set_xlim(x0, x1)
        axis.set_ylim(y0, y1)
    axis.set_xlabel("Xg (mm)")
    axis.set_ylabel("Yg (mm)")
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def compute_statistics(
    image_name: str,
    extracted_count: int,
    frame: pd.DataFrame,
    filtered: dict[str, int],
    extraction_meta: dict[str, float],
) -> dict[str, Any]:
    zg = frame["Zg_mm"].to_numpy(dtype=np.float64)
    return {
        "image": image_name,
        "status": "ok",
        "extracted_point_count": extracted_count,
        "valid_point_count": int(len(frame)),
        "filtered_point_count": int(extracted_count - len(frame)),
        "Zg_min_mm": float(np.min(zg)) if len(zg) else None,
        "Zg_max_mm": float(np.max(zg)) if len(zg) else None,
        "Zg_median_mm": float(np.median(zg)) if len(zg) else None,
        "Zg_std_mm": float(np.std(zg)) if len(zg) else None,
        "filter_counts": filtered,
        "extraction": extraction_meta,
    }


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ReconstructionError(f"输入目录不存在：{input_dir}")
    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ReconstructionError(f"输入目录没有支持的图像（{formats}）：{input_dir}")
    stems = [path.stem.casefold() for path in images]
    if len(stems) != len(set(stems)):
        raise ReconstructionError("输入目录存在同名但扩展名不同的图像，无法建立唯一帧输出目录")
    return images


def select_images(paths: dict[str, Path]) -> list[Path]:
    image_path = paths.get("image")
    if image_path is None:
        return discover_images(paths["input_dir"])
    if not image_path.is_file():
        raise ReconstructionError(f"输入图像不存在：{image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ReconstructionError(
            f"不支持的图像格式 {image_path.suffix!r}；支持格式：{formats}"
        )
    return [image_path]


def process_frame(
    image_path: Path,
    output_dir: Path,
    calibration: Calibration,
    extraction: ExtractionParams,
    reconstruction: ReconstructionParams,
    obstacle_fit_params: ObstacleFitParams,
) -> dict[str, Any]:
    gray = read_mono8(image_path)
    if calibration.image_size is not None:
        actual_size = (gray.shape[1], gray.shape[0])
        if actual_size != calibration.image_size:
            raise ReconstructionError(
                f"图像尺寸 {actual_size} 与内参尺寸 {calibration.image_size} 不一致"
            )
    frame_dir = output_dir / image_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    pixels, extraction_meta, _ = extract_laser_centres(gray, extraction)
    frame, filtered = reconstruct_points(pixels, calibration, reconstruction)
    frame.to_csv(frame_dir / "points_ground.csv", index=False, float_format="%.9f")
    save_overlay(gray, pixels, frame_dir / "laser_center_overlay.png")
    save_pointcloud_plot(frame, frame_dir / "pointcloud_3d.png", image_path.name)
    save_top_view(
        frame,
        frame_dir / "top_view_xy.png",
        image_path.name,
        image_aspect_ratio=gray.shape[1] / gray.shape[0],
    )
    obstacle_fit = fit_obstacle_line(frame, obstacle_fit_params)
    save_obstacle_line_fit(
        frame,
        obstacle_fit,
        frame_dir / "obstacle_line_fit.png",
        image_path.name,
        image_aspect_ratio=gray.shape[1] / gray.shape[0],
    )
    statistics = compute_statistics(
        image_path.name, len(pixels), frame, filtered, extraction_meta
    )
    statistics["obstacle_mean_height_mm"] = obstacle_fit.get("mean_height_mm")
    statistics["obstacle_line_fit_rmse_mm"] = obstacle_fit.get("line_fit_rmse_mm")
    statistics["obstacle_point_count"] = obstacle_fit.get("obstacle_point_count", 0)
    statistics["obstacle_line_fit"] = {
        key: value
        for key, value in obstacle_fit.items()
        if key not in {"obstacle_mask", "line_endpoints_xy"}
    }
    with (frame_dir / "statistics.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(statistics, stream, allow_unicode=True, sort_keys=False)
    return statistics


def _resolve_config_path(value: Any, config_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReconstructionError(f"统一配置缺少 paths.{name}")
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def resolve_paths(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path]:
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ReconstructionError("统一配置中的 paths 必须是映射")
    config_dir = args.config.resolve().parent
    cli_values = {
        "intrinsics": args.intrinsics,
        "laser_plane": args.laser_plane,
        "extrinsics": args.extrinsics,
        "output_dir": args.output_dir,
    }
    resolved: dict[str, Path] = {}
    for name, cli_value in cli_values.items():
        if cli_value is not None:
            resolved[name] = cli_value.resolve()
        else:
            resolved[name] = _resolve_config_path(paths.get(name), config_dir, name)
    if args.image is not None:
        resolved["image"] = args.image.resolve()
    elif args.input_dir is not None:
        resolved["input_dir"] = args.input_dir.resolve()
    else:
        resolved["input_dir"] = _resolve_config_path(
            paths.get("input_dir"), config_dir, "input_dir"
        )
    return resolved


def run(args: argparse.Namespace) -> int:
    config = load_yaml(args.config.resolve())
    paths = resolve_paths(args, config)
    extraction = _load_dataclass(config.get("extraction"), ExtractionParams, "extraction")
    reconstruction = _load_dataclass(
        config.get("reconstruction"), ReconstructionParams, "reconstruction"
    )
    obstacle_fit = _load_dataclass(
        config.get("obstacle_fit"), ObstacleFitParams, "obstacle_fit"
    )
    validate_params(extraction, reconstruction, obstacle_fit)
    calibration = load_calibration(
        paths["intrinsics"], paths["laser_plane"], paths["extrinsics"]
    )
    images = select_images(paths)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures = 0
    print(f"输入帧数：{len(images)}")
    print(f"输出目录：{paths['output_dir']}")
    for index, image_path in enumerate(images, start=1):
        try:
            statistics = process_frame(
                image_path,
                paths["output_dir"],
                calibration,
                extraction,
                reconstruction,
                obstacle_fit,
            )
            rows.append({key: value for key, value in statistics.items() if not isinstance(value, dict)})
            print(
                f"[{index}/{len(images)}] {image_path.name}: "
                f"有效点={statistics['valid_point_count']}, "
                f"Zg[min,max,median,std]="
                f"[{statistics['Zg_min_mm']}, {statistics['Zg_max_mm']}, "
                f"{statistics['Zg_median_mm']}, {statistics['Zg_std_mm']}] mm"
            )
        except Exception as exc:  # 单帧失败不阻断其余批次
            failures += 1
            rows.append(
                {
                    "image": image_path.name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(images)}] {image_path.name}: 失败：{exc}", file=sys.stderr)
    pd.DataFrame(rows).to_csv(
        paths["output_dir"] / "frame_statistics.csv", index=False, encoding="utf-8-sig"
    )
    print(f"处理完成：成功 {len(images) - failures} 帧，失败 {failures} 帧。")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ReconstructionError, OSError, ValueError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
