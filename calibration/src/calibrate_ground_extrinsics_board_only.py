#!/usr/bin/env python3
"""仅使用位于同一刚性基准平面上的棋盘格图像标定地面外参。

坐标单位统一为 mm。棋盘图案表面直接定义为 Zg=0；脚本不读取激光平面，
也不使用地面激光点，因此地面外参与激光几何模型完全解耦。

坐标约定：
- Xg：相机 +X 轴投影到基准平面，近似图像向右；
- Yg：由右手系确定，近似图像向上；
- Zg：基准平面法向并指向相机，凸起高度为正；
- 原点：相机主光轴与棋盘图案基准平面的交点。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
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


LOGGER = logging.getLogger("ground_extrinsics_board_only")
SCRIPT_DIR = Path(__file__).resolve().parent
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
    rotation: np.ndarray
    normal: np.ndarray
    plane_d_mm: float
    points_camera: np.ndarray
    reprojection_rmse_px: float
    detection_method: str


@dataclass(frozen=True)
class FitSelection:
    reference_normal: np.ndarray
    aligned_normals: np.ndarray
    angular_errors_deg: np.ndarray
    normal_threshold_deg: float
    plane_distance_median_mm: float
    plane_distance_errors_mm: np.ndarray
    plane_distance_threshold_mm: float
    inlier_mask: np.ndarray


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "仅由位于同一刚性基准平面上的棋盘格图像，标定相机坐标系到地面坐标系的外参。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--intrinsics", type=Path, required=True, help="相机内参 YAML")
    parser.add_argument("--fit-dir", type=Path, required=True, help="外参拟合棋盘图目录")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="独立验证棋盘图目录；不提供时只完成拟合",
    )
    parser.add_argument("--fit-glob", default="*.tif", help="拟合图像通配符")
    parser.add_argument(
        "--validation-glob", default="*.tif", help="验证图像通配符"
    )
    parser.add_argument("--pattern-cols", type=positive_int, required=True)
    parser.add_argument("--pattern-rows", type=positive_int, required=True)
    parser.add_argument("--square-size-mm", type=positive_float, required=True)
    parser.add_argument(
        "--max-pnp-rmse-px",
        type=positive_float,
        default=0.25,
        help="单张棋盘 PnP 重投影 RMSE 上限",
    )
    parser.add_argument(
        "--normal-outlier-cap-deg",
        type=positive_float,
        default=0.50,
        help="法向异常剔除阈值的绝对上限",
    )
    parser.add_argument(
        "--plane-distance-outlier-cap-mm",
        type=positive_float,
        default=0.50,
        help="各帧棋盘平面到相机距离异常阈值的绝对上限",
    )
    parser.add_argument(
        "--min-fit-images",
        type=positive_int,
        default=6,
        help="法向和距离联合筛选后最少拟合图数量",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许输出目录非空；覆盖同名文件但不删除其他文件",
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
    return Intrinsics(image_size, camera_matrix, dist_coeffs)


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
    return np.clip(
        (gray.astype(np.float64) - low) * 255.0 / (high - low), 0, 255
    ).astype(np.uint8)


def display_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(detection_u8(gray), cv2.COLOR_GRAY2BGR)


def chessboard_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return points


def detect_chessboard(
    path: Path,
    intrinsics: Intrinsics,
    object_points: np.ndarray,
    pattern: tuple[int, int],
    max_pnp_rmse_px: float,
) -> ChessObservation:
    gray = to_gray(read_image(path), path)
    expected_shape = (intrinsics.image_size[1], intrinsics.image_size[0])
    if gray.shape != expected_shape:
        raise CalibrationError(
            f"图像尺寸 {gray.shape[::-1]} 与内参 {intrinsics.image_size} 不一致"
        )
    image = detection_u8(gray)

    sb_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    found, corners = cv2.findChessboardCornersSB(image, pattern, sb_flags)
    method = "SB"
    if not found or corners is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(image, pattern, flags)
        method = "classic+cornerSubPix"
        if found and corners is not None:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                50,
                1.0e-4,
            )
            corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    if not found or corners is None:
        raise CalibrationError("未检测到完整棋盘格内角点")

    solved, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise CalibrationError("solvePnP 求解失败")
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            corners,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            rvec,
            tvec,
        )

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    residual = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if rmse > max_pnp_rmse_px:
        raise CalibrationError(
            f"PnP 重投影 RMSE {rmse:.4f}px 超过 {max_pnp_rmse_px:.4f}px"
        )

    rotation, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3).astype(np.float64)
    normal = rotation[:, 2].astype(np.float64)
    normal /= np.linalg.norm(normal)

    # 法向统一指向相机：平面点到相机原点的方向与法向同向。
    if float(normal @ t) > 0.0:
        normal = -normal
    plane_d = -float(normal @ t)
    if plane_d <= 0.0:
        raise CalibrationError("棋盘平面距离非正，请检查PnP姿态或坐标方向")

    points_camera = (
        object_points.astype(np.float64) @ rotation.T + t.reshape(1, 3)
    )
    return ChessObservation(
        path=path,
        corners=corners.reshape(-1, 2),
        rvec=rvec.reshape(3),
        tvec=t,
        rotation=rotation,
        normal=normal,
        plane_d_mm=plane_d,
        points_camera=points_camera,
        reprojection_rmse_px=rmse,
        detection_method=method,
    )


def robust_threshold(values: np.ndarray, cap: float, floor: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_limit = max(floor, 3.5 * 1.4826 * mad)
    return median, min(float(cap), robust_limit)


def select_fit_observations(
    observations: list[ChessObservation],
    normal_cap_deg: float,
    distance_cap_mm: float,
    min_fit_images: int,
) -> FitSelection:
    normals = np.vstack([item.normal for item in observations])
    distances = np.asarray([item.plane_d_mm for item in observations], dtype=np.float64)

    # 使用外积主特征向量生成稳定参考方向，再统一符号。
    _, eigenvectors = np.linalg.eigh(normals.T @ normals)
    reference = eigenvectors[:, -1]
    if reference[2] > 0.0:
        reference = -reference
    aligned = normals.copy()
    aligned[(aligned @ reference) < 0.0] *= -1.0

    pairwise_angles = np.degrees(
        np.arccos(np.clip(aligned @ aligned.T, -1.0, 1.0))
    )
    medoid_index = int(np.argmin(np.median(pairwise_angles, axis=1)))
    medoid_angles = pairwise_angles[medoid_index]
    angle_median, angle_threshold = robust_threshold(
        medoid_angles, normal_cap_deg, floor=0.05
    )
    del angle_median
    normal_inliers = medoid_angles <= angle_threshold
    if int(np.count_nonzero(normal_inliers)) < min_fit_images:
        raise CalibrationError(
            "法向筛选后有效图像不足："
            f"{np.count_nonzero(normal_inliers)}/{len(observations)}，"
            f"至少需要 {min_fit_images} 张"
        )

    provisional_normal = np.mean(aligned[normal_inliers], axis=0)
    provisional_normal /= np.linalg.norm(provisional_normal)
    angular_errors = np.degrees(
        np.arccos(np.clip(aligned @ provisional_normal, -1.0, 1.0))
    )
    normal_inliers = angular_errors <= angle_threshold

    distance_median, distance_threshold = robust_threshold(
        distances[normal_inliers], distance_cap_mm, floor=0.05
    )
    distance_errors = np.abs(distances - distance_median)
    distance_inliers = distance_errors <= distance_threshold
    inliers = normal_inliers & distance_inliers
    if int(np.count_nonzero(inliers)) < min_fit_images:
        raise CalibrationError(
            "法向与平面距离联合筛选后有效图像不足："
            f"{np.count_nonzero(inliers)}/{len(observations)}，"
            f"至少需要 {min_fit_images} 张"
        )

    final_normal = np.mean(aligned[inliers], axis=0)
    final_normal /= np.linalg.norm(final_normal)
    final_angles = np.degrees(
        np.arccos(np.clip(aligned @ final_normal, -1.0, 1.0))
    )
    return FitSelection(
        reference_normal=final_normal,
        aligned_normals=aligned,
        angular_errors_deg=final_angles,
        normal_threshold_deg=angle_threshold,
        plane_distance_median_mm=distance_median,
        plane_distance_errors_mm=distance_errors,
        plane_distance_threshold_mm=distance_threshold,
        inlier_mask=inliers,
    )


def fit_plane_from_points(points_camera: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3 or not np.isfinite(points).all():
        raise CalibrationError("拟合基准平面需要至少3个有限三维点")
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    normal = right[-1]
    normal /= np.linalg.norm(normal)
    if float(normal @ centroid) > 0.0:
        normal = -normal
    d = -float(normal @ centroid)
    if d <= 0.0:
        raise CalibrationError("拟合得到的基准平面距离非正")
    return np.r_[normal, d]


def build_ground_transform(ground_plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(ground_plane[:3], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    d = float(ground_plane[3])
    if normal[2] >= -1.0e-6:
        raise CalibrationError(
            "基准面法向未指向相机或与主光轴近似平行；期望 normal_z < 0"
        )

    # 原点：相机主光轴与棋盘图案表面的交点。
    origin_camera = np.asarray([0.0, 0.0, -d / normal[2]], dtype=np.float64)

    camera_x = np.asarray([1.0, 0.0, 0.0])
    ground_x = camera_x - normal * float(camera_x @ normal)
    x_length = float(np.linalg.norm(ground_x))
    if x_length < 1.0e-9:
        raise CalibrationError("相机 +X 轴与基准面法向近似平行，无法定义 Xg")
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
    return transform, inverse


def transform_camera_to_ground(
    points_camera: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def residual_metrics(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "point_count": 0,
            "mean_mm": None,
            "median_mm": None,
            "std_mm": None,
            "rmse_mm": None,
            "p95_abs_mm": None,
            "max_abs_mm": None,
            "min_mm": None,
            "max_mm": None,
        }
    return {
        "point_count": int(array.size),
        "mean_mm": float(np.mean(array)),
        "median_mm": float(np.median(array)),
        "std_mm": float(np.std(array, ddof=0)),
        "rmse_mm": float(np.sqrt(np.mean(array * array))),
        "p95_abs_mm": float(np.percentile(np.abs(array), 95)),
        "max_abs_mm": float(np.max(np.abs(array))),
        "min_mm": float(np.min(array)),
        "max_mm": float(np.max(array)),
    }


def evaluate_observations(
    observations: list[ChessObservation], transform: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, float | int | None], np.ndarray]:
    rows: list[dict[str, Any]] = []
    all_points: list[np.ndarray] = []
    for item in observations:
        points_ground = transform_camera_to_ground(item.points_camera, transform)
        all_points.append(points_ground)
        metrics = residual_metrics(points_ground[:, 2])
        rows.append(
            {
                "image": item.path.name,
                "detection_method": item.detection_method,
                "pnp_rmse_px": item.reprojection_rmse_px,
                "normal_x": item.normal[0],
                "normal_y": item.normal[1],
                "normal_z": item.normal[2],
                "plane_d_mm": item.plane_d_mm,
                **{f"zg_{key}": value for key, value in metrics.items()},
            }
        )
    stacked = np.vstack(all_points) if all_points else np.empty((0, 3), dtype=np.float64)
    overall = residual_metrics(stacked[:, 2] if len(stacked) else np.asarray([]))
    return rows, overall, stacked


def save_overlay(
    observation: ChessObservation,
    accepted: bool | None,
    pattern: tuple[int, int],
    output_path: Path,
    extra_text: str = "",
) -> None:
    gray = to_gray(read_image(observation.path), observation.path)
    overlay = display_bgr(gray)
    if accepted is None:
        colour = (0, 200, 255)
        state = "VALIDATION"
    elif accepted:
        colour = (0, 200, 0)
        state = "INLIER"
    else:
        colour = (0, 0, 255)
        state = "OUTLIER"
    corners = observation.corners.reshape(-1, 1, 2).astype(np.float32)
    cv2.drawChessboardCorners(overlay, pattern, corners, True)
    label = (
        f"{state} PnP={observation.reprojection_rmse_px:.4f}px "
        f"d={observation.plane_d_mm:.3f}mm {extra_text}"
    )
    cv2.putText(
        overlay,
        label,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colour,
        2,
        cv2.LINE_AA,
    )
    write_image(output_path, overlay)


def save_consistency_plot(
    observations: list[ChessObservation],
    selection: FitSelection,
    output_path: Path,
) -> None:
    labels = [item.path.stem for item in observations]
    index = np.arange(len(observations))
    colours = np.where(selection.inlier_mask, "tab:green", "tab:red")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(index, selection.angular_errors_deg, color=colours)
    axes[0].axhline(selection.normal_threshold_deg, color="black", linestyle="--")
    axes[0].set_ylabel("Normal angular error (deg)")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[1].bar(index, selection.plane_distance_errors_mm, color=colours)
    axes[1].axhline(selection.plane_distance_threshold_mm, color="black", linestyle="--")
    axes[1].set_ylabel("Plane-distance error (mm)")
    axes[1].set_xticks(index, labels, rotation=45, ha="right")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def save_z_residual_plot(
    points_ground: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    if len(points_ground) == 0:
        return
    z = points_ground[:, 2]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    scatter = axes[0].scatter(
        points_ground[:, 0], points_ground[:, 1], c=z, s=8, cmap="coolwarm"
    )
    axes[0].set_title(title)
    axes[0].set_xlabel("Xg (mm)")
    axes[0].set_ylabel("Yg (mm)")
    axes[0].set_aspect("equal", adjustable="box")
    figure.colorbar(scatter, ax=axes[0], label="Zg (mm)")
    axes[1].hist(z, bins=60, alpha=0.85)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_title(
        f"mean={np.mean(z):.4f} mm, RMSE={np.sqrt(np.mean(z*z)):.4f} mm"
    )
    axes[1].set_xlabel("Zg (mm)")
    axes[1].set_ylabel("Count")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        raise CalibrationError(f"输出目录非空；如需覆盖请加 --overwrite：{path}")
    path.mkdir(parents=True, exist_ok=True)


def scan_images(
    paths: list[Path],
    intrinsics: Intrinsics,
    object_points: np.ndarray,
    pattern: tuple[int, int],
    max_pnp_rmse_px: float,
) -> tuple[list[ChessObservation], list[dict[str, str]]]:
    observations: list[ChessObservation] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            observations.append(
                detect_chessboard(
                    path,
                    intrinsics,
                    object_points,
                    pattern,
                    max_pnp_rmse_px,
                )
            )
        except (CalibrationError, cv2.error, ValueError, FloatingPointError) as exc:
            failures.append({"image": str(path.resolve()), "reason": str(exc)})
            LOGGER.warning("棋盘图跳过 %s：%s", path.name, exc)
    return observations, failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_dir(args.output_dir, args.overwrite)
    intrinsics = load_intrinsics(args.intrinsics)
    pattern = (args.pattern_cols, args.pattern_rows)
    object_points = chessboard_object_points(
        args.pattern_cols, args.pattern_rows, args.square_size_mm
    )

    fit_paths = list_images(args.fit_dir, args.fit_glob)
    validation_paths: list[Path] = []
    if args.validation_dir is not None:
        validation_paths = list_images(args.validation_dir, args.validation_glob)
        overlap = {path.resolve() for path in fit_paths} & {
            path.resolve() for path in validation_paths
        }
        if overlap:
            names = ", ".join(sorted(path.name for path in overlap))
            raise CalibrationError(f"拟合集和验证集存在重叠：{names}")

    LOGGER.info(
        "读取 %d 张拟合棋盘图、%d 张验证棋盘图",
        len(fit_paths),
        len(validation_paths),
    )
    fit_observations, fit_failures = scan_images(
        fit_paths,
        intrinsics,
        object_points,
        pattern,
        args.max_pnp_rmse_px,
    )
    if len(fit_observations) < args.min_fit_images:
        raise CalibrationError(
            f"成功检测的拟合图仅 {len(fit_observations)} 张，"
            f"少于 --min-fit-images={args.min_fit_images}"
        )

    selection = select_fit_observations(
        fit_observations,
        args.normal_outlier_cap_deg,
        args.plane_distance_outlier_cap_mm,
        args.min_fit_images,
    )
    accepted_observations = [
        item
        for item, accepted in zip(
            fit_observations, selection.inlier_mask, strict=True
        )
        if accepted
    ]
    pooled_points = np.vstack([item.points_camera for item in accepted_observations])
    ground_plane = fit_plane_from_points(pooled_points)
    transform, inverse = build_ground_transform(ground_plane)

    fit_rows, fit_metrics, fit_points_ground = evaluate_observations(
        accepted_observations, transform
    )
    validation_observations: list[ChessObservation] = []
    validation_failures: list[dict[str, str]] = []
    validation_rows: list[dict[str, Any]] = []
    validation_metrics = residual_metrics(np.asarray([]))
    validation_points_ground = np.empty((0, 3), dtype=np.float64)
    if validation_paths:
        validation_observations, validation_failures = scan_images(
            validation_paths,
            intrinsics,
            object_points,
            pattern,
            args.max_pnp_rmse_px,
        )
        if not validation_observations:
            raise CalibrationError("验证目录中没有成功检测到棋盘格的图像")
        (
            validation_rows,
            validation_metrics,
            validation_points_ground,
        ) = evaluate_observations(validation_observations, transform)

    diagnostics = args.output_dir / "diagnostics"
    for item, accepted, angle, distance_error in zip(
        fit_observations,
        selection.inlier_mask,
        selection.angular_errors_deg,
        selection.plane_distance_errors_mm,
        strict=True,
    ):
        save_overlay(
            item,
            bool(accepted),
            pattern,
            diagnostics / "fit" / f"{item.path.stem}.png",
            extra_text=f"angle={angle:.4f}deg dd={distance_error:.4f}mm",
        )
    for item in validation_observations:
        save_overlay(
            item,
            None,
            pattern,
            diagnostics / "validation" / f"{item.path.stem}.png",
        )
    save_consistency_plot(
        fit_observations,
        selection,
        diagnostics / "fit_plane_consistency.png",
    )
    save_z_residual_plot(
        fit_points_ground,
        "Fit checkerboard points: Zg residual",
        diagnostics / "fit_zg_residual.png",
    )
    save_z_residual_plot(
        validation_points_ground,
        "Validation checkerboard points: Zg residual",
        diagnostics / "validation_zg_residual.png",
    )

    write_rows_csv(args.output_dir / "fit_frames.csv", fit_rows)
    write_rows_csv(args.output_dir / "validation_frames.csv", validation_rows)

    rotation = transform[:3, :3]
    orthogonality_error = float(np.linalg.norm(rotation @ rotation.T - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    inverse_error = float(np.linalg.norm(transform @ inverse - np.eye(4)))
    if orthogonality_error > 1.0e-9 or abs(determinant - 1.0) > 1.0e-9:
        raise CalibrationError("构造的旋转矩阵未通过正交性/行列式检查")

    optical_axis = np.asarray([0.0, 0.0, 1.0])
    tilt_from_normal = math.degrees(
        math.acos(float(np.clip(abs(ground_plane[:3] @ optical_axis), 0.0, 1.0)))
    )
    tilt_from_plane = 90.0 - tilt_from_normal
    origin_camera = inverse[:3, 3]
    camera_height = float(ground_plane[3])

    fit_frame_details = []
    for item, angle, distance_error, accepted in zip(
        fit_observations,
        selection.angular_errors_deg,
        selection.plane_distance_errors_mm,
        selection.inlier_mask,
        strict=True,
    ):
        fit_frame_details.append(
            {
                "image": item.path,
                "detection_method": item.detection_method,
                "reprojection_rmse_px": item.reprojection_rmse_px,
                "normal_camera": item.normal,
                "plane_d_mm": item.plane_d_mm,
                "normal_angular_error_deg": float(angle),
                "plane_distance_error_mm": float(distance_error),
                "accepted": bool(accepted),
            }
        )

    result = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "checkerboard_plane_only",
        "units": "mm",
        "coordinate_convention": {
            "Xg": "camera +X projected onto checkerboard reference plane; approximately image right",
            "Yg": "Zg cross Xg; approximately image up",
            "Zg": "checkerboard reference-plane normal toward camera; positive for protrusions",
            "origin": "intersection of camera principal axis and checkerboard pattern plane",
            "zero_surface": "checkerboard pattern surface",
        },
        "T_ground_from_camera": transform,
        "T_camera_from_ground": inverse,
        "ground_plane_in_camera": {
            "equation": "n_x*Xc + n_y*Yc + n_z*Zc + D = 0",
            "coefficients": ground_plane,
            "normal": ground_plane[:3],
            "D_mm": ground_plane[3],
        },
        "ground_origin_in_camera_mm": origin_camera,
        "ground_axes_in_camera": {
            "Xg": rotation[0],
            "Yg": rotation[1],
            "Zg": rotation[2],
        },
        "camera_installation": {
            "optical_axis_to_ground_normal_deg": tilt_from_normal,
            "optical_axis_to_ground_plane_deg": tilt_from_plane,
            "camera_perpendicular_distance_to_pattern_plane_mm": camera_height,
            "principal_axis_intersection_zc_mm": float(origin_camera[2]),
        },
        "quality_checks": {
            "rotation_orthogonality_frobenius": orthogonality_error,
            "rotation_determinant": determinant,
            "transform_inverse_frobenius": inverse_error,
            "fit_zg": fit_metrics,
            "validation_zg": validation_metrics,
        },
        "fit_selection": {
            "detected_frame_count": len(fit_observations),
            "accepted_frame_count": len(accepted_observations),
            "normal_outlier_threshold_deg": selection.normal_threshold_deg,
            "plane_distance_median_mm": selection.plane_distance_median_mm,
            "plane_distance_outlier_threshold_mm": selection.plane_distance_threshold_mm,
            "frames": fit_frame_details,
            "failures": fit_failures,
        },
        "validation": {
            "detected_frame_count": len(validation_observations),
            "metrics": validation_metrics,
            "frames": validation_rows,
            "failures": validation_failures,
        },
        "inputs": {
            "intrinsics": args.intrinsics,
            "fit_dir": args.fit_dir,
            "validation_dir": args.validation_dir,
            "fit_glob": args.fit_glob,
            "validation_glob": args.validation_glob,
            "pattern_cols": args.pattern_cols,
            "pattern_rows": args.pattern_rows,
            "square_size_mm": args.square_size_mm,
            "zero_surface": "checkerboard pattern surface",
        },
    }

    serialised = _serialisable(result)
    (args.output_dir / "camera_ground_extrinsics.yaml").write_text(
        yaml.safe_dump(serialised, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (args.output_dir / "camera_ground_extrinsics.json").write_text(
        json.dumps(serialised, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("棋盘独立地面外参标定完成：%s", args.output_dir.resolve())
    LOGGER.info(
        "拟合图：检测=%d，采用=%d；fit Zg RMSE=%s mm",
        len(fit_observations),
        len(accepted_observations),
        "N/A" if fit_metrics["rmse_mm"] is None else f"{fit_metrics['rmse_mm']:.6f}",
    )
    if validation_observations:
        LOGGER.info(
            "验证图：%d；validation Zg RMSE=%.6f mm，P95=%.6f mm",
            len(validation_observations),
            float(validation_metrics["rmse_mm"]),
            float(validation_metrics["p95_abs_mm"]),
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
    except (
        CalibrationError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        cv2.error,
        yaml.YAMLError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        LOGGER.error("标定失败：%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
