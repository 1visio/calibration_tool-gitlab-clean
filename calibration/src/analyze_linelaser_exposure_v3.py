#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
不同曝光下棋盘格激光线提取质量分析 V3

适用数据
--------
1. 短曝光、激光开启图像：
   linelaser_exposure_test/
   ├─ 150us/pose_01.tif
   ├─ 200us/pose_01.tif
   └─ ...

2. 长曝光、激光关闭棋盘格图像：
   chessboard_exposure_test/
   └─ 20000us/
      ├─ pose_01.tif
      └─ ...

3. 可选：与激光开启图同曝光、同姿态的激光关闭图像：
   linelaser_off_exposure_test/
   ├─ 150us/pose_01.tif
   └─ ...

核心改进
--------
- 长曝光棋盘格图只用于角点、棋盘格范围、黑白格几何分类，不参与灰度相减；
- 若有同曝光 laser-off 图，则执行严格的 on-off 差分；
- 先提取高置信度激光点并用 RANSAC 拟合初始直线；
- 再在初始直线附近逐列/逐行局部搜索，提高弱反射区的检出率；
- 使用激光两侧背景带估计噪声，避免 MAD=0 导致虚假的超大 SNR；
- 排除棋盘格黑白边界附近的点，降低反射率突变造成的灰度重心偏差；
- 使用亚像素半高交点计算 FWHM；
- 覆盖率分母只统计“拟合激光线实际穿过有效棋盘格区域”的扫描线；
- 黑格/白格由棋盘格单应性和格子奇偶性分类，覆盖率不会超过 100%；
- 饱和、线宽异常、峰形不对称的点默认不参与最终中心拟合。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须大于或等于 0")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("必须是大于或等于 0 的有限数")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用两阶段亚像素提取评估棋盘格激光线的最佳曝光。"
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="短曝光、激光开启图像根目录"
    )
    parser.add_argument(
        "--board-root", required=True, type=Path,
        help="长曝光、激光关闭棋盘格图像根目录"
    )
    parser.add_argument(
        "--board-group", default="20000us",
        help="--board-root 下用于定位棋盘格的曝光文件夹，默认 20000us"
    )
    parser.add_argument(
        "--background-root", type=Path,
        help="可选：与激光开启图同曝光、同姿态的激光关闭图像根目录"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("linelaser_exposure_result_v3"),
        help="输出目录"
    )

    parser.add_argument("--cols", type=positive_int, default=6,
                        help="棋盘格横向内角点数，默认 6")
    parser.add_argument("--rows", type=positive_int, default=5,
                        help="棋盘格纵向内角点数，默认 5")
    parser.add_argument(
        "--max-value", type=nonnegative_float, default=0,
        help="图像有效最大灰度；Mono8=255，Mono12=4095，0=自动"
    )
    parser.add_argument(
        "--scan-axis", choices=("auto", "x", "y"), default="x",
        help="x=激光近似水平并逐列提取；y=近似竖直并逐行提取；默认 x"
    )

    parser.add_argument(
        "--board-erode", type=nonnegative_int, default=4,
        help="棋盘格外边界向内收缩像素数，默认 4"
    )
    parser.add_argument(
        "--boundary-exclude", type=nonnegative_int, default=5,
        help="黑白格边界两侧排除宽度，默认各约 5 像素"
    )

    parser.add_argument(
        "--initial-min-snr", type=positive_float, default=5.0,
        help="初始直线高置信度点的最低 SNR，默认 5"
    )
    parser.add_argument(
        "--min-snr", type=positive_float, default=3.0,
        help="精细提取点的最低 SNR，默认 3"
    )
    parser.add_argument(
        "--min-peak-ratio", type=positive_float, default=0.01,
        help="局部激光峰值至少占满量程的比例，默认 0.01"
    )
    parser.add_argument(
        "--noise-floor-dn", type=positive_float, default=1.0,
        help="噪声标准差下限，Mono8 默认 1 DN；Mono12 可设为 4～8"
    )

    parser.add_argument(
        "--search-half-width", type=positive_int, default=8,
        help="精细阶段在预测直线两侧的搜索半宽，默认 8 像素"
    )
    parser.add_argument(
        "--centroid-half-width", type=positive_int, default=5,
        help="灰度重心窗口半宽，默认 5 像素"
    )
    parser.add_argument(
        "--background-inner", type=positive_int, default=8,
        help="背景带距离激光峰值的内边界，默认 8 像素"
    )
    parser.add_argument(
        "--background-outer", type=positive_int, default=20,
        help="背景带距离激光峰值的外边界，默认 20 像素"
    )
    parser.add_argument(
        "--centroid-noise-k", type=nonnegative_float, default=1.0,
        help="重心权重扣除的噪声倍数，默认 1.0"
    )

    parser.add_argument(
        "--min-fwhm", type=nonnegative_float, default=1.0,
        help="允许的最小激光线 FWHM，默认 1 像素"
    )
    parser.add_argument(
        "--max-fwhm", type=positive_float, default=10.0,
        help="允许的最大激光线 FWHM，默认 10 像素"
    )
    parser.add_argument(
        "--max-asymmetry", type=nonnegative_float, default=0.45,
        help="允许的最大峰形左右能量不对称度，默认 0.45"
    )
    parser.add_argument(
        "--sat-ratio", type=positive_float, default=0.98,
        help="达到满量程该比例视为饱和，默认 0.98"
    )
    parser.add_argument(
        "--keep-saturated", action="store_true",
        help="保留饱和候选点；默认会将饱和点排除出最终中心"
    )

    parser.add_argument(
        "--ransac-residual", type=positive_float, default=1.5,
        help="最终直线 RANSAC 最大残差，默认 1.5 像素"
    )
    parser.add_argument(
        "--ransac-iters", type=positive_int, default=500,
        help="RANSAC 迭代次数，默认 500"
    )
    parser.add_argument(
        "--refine-iterations", type=positive_int, default=2,
        help="沿预测直线重复精细提取的次数，默认 2"
    )

    parser.add_argument(
        "--max-mean-saturation", type=positive_float, default=0.05,
        help="自动推荐单曝光时允许的平均候选饱和率，默认 0.05（5%%）"
    )
    parser.add_argument(
        "--save-signal", action="store_true",
        help="保存用于初始检测的激光响应图"
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.background_outer <= args.background_inner:
        raise ValueError("background-outer 必须大于 background-inner")
    if args.background_inner <= args.centroid_half_width:
        raise ValueError(
            "background-inner 应大于 centroid-half-width，"
            "避免背景带包含激光主体"
        )
    if args.max_fwhm <= args.min_fwhm:
        raise ValueError("max-fwhm 必须大于 min-fwhm")
    if args.max_asymmetry > 1:
        raise ValueError("max-asymmetry 应在 0～1 之间")
    if args.sat_ratio > 1:
        raise ValueError("sat-ratio 应在 0～1 之间")


# ---------------------------------------------------------------------------
# 文件和图像
# ---------------------------------------------------------------------------

def read_image(path: Path) -> np.ndarray | None:
    """兼容 Windows 中文路径。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"无法编码图像：{path}")
    encoded.tofile(str(path))


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"不支持的图像形状：{image.shape}")


def infer_max_value(gray: np.ndarray, specified: float) -> float:
    if specified > 0:
        return float(specified)
    if gray.dtype == np.uint8:
        return 255.0
    if gray.dtype == np.uint16:
        return 4095.0
    return max(float(np.nanmax(gray)), 1.0)


def normalize_u8(gray: np.ndarray, maximum: float) -> np.ndarray:
    return np.clip(
        gray.astype(np.float32) * 255.0 / maximum, 0, 255
    ).astype(np.uint8)


def list_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def exposure_group(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else "root"


def parse_exposure(group: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", group)
    return float(match.group(1)) if match else None


def find_same_stem(folder: Path, filename: Path) -> Path | None:
    exact = folder / filename.name
    if exact.is_file():
        return exact
    if not folder.is_dir():
        return None
    matches = [
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.casefold() == filename.stem.casefold()
    ]
    return sorted(matches)[0] if matches else None


def find_board_reference(
    board_root: Path, board_group: str, laser_relative: Path
) -> Path | None:
    return find_same_stem(board_root / board_group, laser_relative)


def find_background(background_root: Path, laser_relative: Path) -> Path | None:
    exact = background_root / laser_relative
    if exact.is_file():
        return exact
    return find_same_stem(background_root / laser_relative.parent, laser_relative)


# ---------------------------------------------------------------------------
# 棋盘格几何
# ---------------------------------------------------------------------------

def detect_chessboard(
    gray8: np.ndarray, pattern: tuple[int, int]
) -> tuple[bool, np.ndarray | None]:
    found = False
    corners = None

    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
            found, corners = cv2.findChessboardCornersSB(gray8, pattern, flags)
        except cv2.error:
            found, corners = False, None

    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray8, pattern, flags)
        if found and corners is not None:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                50,
                1e-3,
            )
            corners = cv2.cornerSubPix(
                gray8, corners, (11, 11), (-1, -1), criteria
            )

    return bool(found), corners


def compute_board_homography(
    corners: np.ndarray, cols: int, rows: int
) -> np.ndarray:
    image_points = corners.reshape(-1, 2).astype(np.float32)
    grid_points = np.array(
        [(x, y) for y in range(rows) for x in range(cols)],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(grid_points, image_points, method=0)
    if homography is None:
        raise RuntimeError("无法计算棋盘格单应矩阵")
    return homography


def transform_points(points: Iterable[tuple[float, float]], H: np.ndarray) -> np.ndarray:
    array = np.asarray(list(points), dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(array, H)[0]


def make_board_geometry(
    shape: tuple[int, int],
    H: np.ndarray,
    cols: int,
    rows: int,
    board_erode: int,
    boundary_exclude: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    内角点坐标范围为 x=0..cols-1, y=0..rows-1。
    完整棋盘格外边界可近似扩展至 x=-1..cols, y=-1..rows。
    """
    h, w = shape
    outer_board = transform_points(
        [(-1, -1), (cols, -1), (cols, rows), (-1, rows)], H
    )
    outer_board[:, 0] = np.clip(outer_board[:, 0], 0, w - 1)
    outer_board[:, 1] = np.clip(outer_board[:, 1], 0, h - 1)

    board_mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(
        board_mask, np.round(outer_board).astype(np.int32), 255
    )

    if board_erode > 0:
        size = 2 * board_erode + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (size, size)
        )
        board_mask = cv2.erode(board_mask, kernel)

    boundary_mask = np.zeros(shape, dtype=np.uint8)
    grid_lines: list[np.ndarray] = []
    thickness = max(1, 2 * boundary_exclude + 1)

    # 内部竖直黑白边界：x = 0, 1, ..., cols-1
    for x in range(cols):
        line = transform_points([(x, -1), (x, rows)], H)
        grid_lines.append(line)
        cv2.line(
            boundary_mask,
            tuple(np.round(line[0]).astype(int)),
            tuple(np.round(line[1]).astype(int)),
            255,
            thickness,
            cv2.LINE_AA,
        )

    # 内部水平黑白边界：y = 0, 1, ..., rows-1
    for y in range(rows):
        line = transform_points([(-1, y), (cols, y)], H)
        grid_lines.append(line)
        cv2.line(
            boundary_mask,
            tuple(np.round(line[0]).astype(int)),
            tuple(np.round(line[1]).astype(int)),
            255,
            thickness,
            cv2.LINE_AA,
        )

    usable_mask = cv2.bitwise_and(
        board_mask, cv2.bitwise_not(boundary_mask)
    )
    return board_mask, boundary_mask, usable_mask, grid_lines


def sample_patch_median(gray: np.ndarray, x: float, y: float, radius: int = 3) -> float:
    xi = int(round(x))
    yi = int(round(y))
    y0 = max(0, yi - radius)
    y1 = min(gray.shape[0], yi + radius + 1)
    x0 = max(0, xi - radius)
    x1 = min(gray.shape[1], xi + radius + 1)
    patch = gray[y0:y1, x0:x1]
    return float(np.median(patch)) if patch.size else float("nan")


def determine_white_parity(
    board_gray: np.ndarray,
    H: np.ndarray,
    cols: int,
    rows: int,
) -> int:
    """
    棋盘格有 (cols+1) × (rows+1) 个方格。
    根据两类奇偶格中心的长曝光灰度，确定哪一种奇偶性是白格。
    """
    parity_values: dict[int, list[float]] = {0: [], 1: []}

    centers: list[tuple[float, float]] = []
    parities: list[int] = []
    for cell_y in range(rows + 1):
        for cell_x in range(cols + 1):
            # 第0个方格范围为[-1,0]，中心为-0.5。
            centers.append((cell_x - 0.5, cell_y - 0.5))
            parities.append((cell_x + cell_y) % 2)

    image_centers = transform_points(centers, H)
    for point, parity in zip(image_centers, parities):
        value = sample_patch_median(board_gray, float(point[0]), float(point[1]))
        if np.isfinite(value):
            parity_values[parity].append(value)

    medians = {
        parity: (
            float(np.median(values)) if values else float("-inf")
        )
        for parity, values in parity_values.items()
    }
    return 0 if medians[0] >= medians[1] else 1


def image_to_board(
    x: float, y: float, H_inv: np.ndarray
) -> tuple[float, float]:
    point = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, H_inv)[0, 0]
    return float(mapped[0]), float(mapped[1])


def board_surface_class(
    x: float,
    y: float,
    H_inv: np.ndarray,
    cols: int,
    rows: int,
    white_parity: int,
) -> int:
    """返回 1=黑格，2=白格，0=棋盘外。"""
    u, v = image_to_board(x, y, H_inv)
    cell_x = math.floor(u + 1.0)
    cell_y = math.floor(v + 1.0)
    if not (0 <= cell_x <= cols and 0 <= cell_y <= rows):
        return 0
    parity = (cell_x + cell_y) % 2
    return 2 if parity == white_parity else 1


# ---------------------------------------------------------------------------
# 激光响应与局部亚像素提取
# ---------------------------------------------------------------------------

def make_initial_signal(
    laser_on: np.ndarray,
    laser_off: np.ndarray | None,
    sigma: float = 7.0,
) -> np.ndarray:
    """
    仅用于初始直线粗定位。
    有同曝光 laser-off 时使用正差分；否则使用局部高通响应。
    """
    on = laser_on.astype(np.float32)
    if laser_off is not None:
        return np.maximum(on - laser_off.astype(np.float32), 0.0)
    background = cv2.GaussianBlur(on, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.maximum(on - background, 0.0)


def make_measurement_image(
    laser_on: np.ndarray,
    laser_off: np.ndarray | None,
) -> np.ndarray:
    """
    精细提取使用的强度图。
    有 laser-off 时使用 on-off；没有时保留原始短曝光灰度，
    再通过局部两侧背景带估计基线。
    """
    on = laser_on.astype(np.float32)
    if laser_off is not None:
        return np.maximum(on - laser_off.astype(np.float32), 0.0)
    return on


def robust_noise(values: np.ndarray, floor_dn: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float(floor_dn)

    median = float(np.median(values))
    mad_sigma = 1.4826 * float(np.median(np.abs(values - median)))
    std_sigma = float(np.std(values))

    # MAD 对大量完全相同的暗像素可能得到0，std作为回退。
    candidates = [floor_dn]
    if np.isfinite(mad_sigma) and mad_sigma > 0:
        candidates.append(mad_sigma)
    if np.isfinite(std_sigma) and std_sigma > 0:
        candidates.append(std_sigma)
    return float(max(candidates))


def interpolate_half_crossing(
    x0: float, y0: float, x1: float, y1: float, half: float
) -> float:
    if abs(y1 - y0) < 1e-12:
        return 0.5 * (x0 + x1)
    alpha = (half - y0) / (y1 - y0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return x0 + alpha * (x1 - x0)


def subpixel_fwhm(
    coordinates: np.ndarray,
    residual: np.ndarray,
    peak_local_index: int,
) -> float:
    peak_value = float(residual[peak_local_index])
    if peak_value <= 0:
        return float("nan")
    half = 0.5 * peak_value

    left_index = peak_local_index
    while left_index > 0 and residual[left_index] >= half:
        left_index -= 1

    right_index = peak_local_index
    while right_index < len(residual) - 1 and residual[right_index] >= half:
        right_index += 1

    if left_index == peak_local_index or right_index == peak_local_index:
        return float("nan")

    left_cross = interpolate_half_crossing(
        coordinates[left_index],
        residual[left_index],
        coordinates[left_index + 1],
        residual[left_index + 1],
        half,
    )
    right_cross = interpolate_half_crossing(
        coordinates[right_index - 1],
        residual[right_index - 1],
        coordinates[right_index],
        residual[right_index],
        half,
    )
    return float(max(0.0, right_cross - left_cross))


def extract_local_center(
    scan: int,
    prediction: float,
    laser_on: np.ndarray,
    measurement: np.ndarray,
    initial_signal: np.ndarray,
    usable_mask: np.ndarray,
    axis: str,
    maximum: float,
    args: argparse.Namespace,
    search_half_width: int,
    minimum_snr: float,
    apply_quality_filters: bool,
) -> dict[str, Any] | None:
    """
    在预测位置附近：
    1. 用初始响应寻找整数峰值；
    2. 用峰值两侧背景带估计左右背景及噪声；
    3. 在线性背景基线扣除后计算灰度重心；
    4. 计算 SNR、FWHM、对称性和饱和状态。
    """
    if axis == "x":
        length = laser_on.shape[0]
        raw_profile = laser_on[:, scan].astype(np.float64)
        measure_profile = measurement[:, scan].astype(np.float64)
        search_profile = initial_signal[:, scan].astype(np.float64)
        mask_profile = usable_mask[:, scan] > 0
    else:
        length = laser_on.shape[1]
        raw_profile = laser_on[scan, :].astype(np.float64)
        measure_profile = measurement[scan, :].astype(np.float64)
        search_profile = initial_signal[scan, :].astype(np.float64)
        mask_profile = usable_mask[scan, :] > 0

    predicted_index = int(round(prediction))
    search_lo = max(0, predicted_index - search_half_width)
    search_hi = min(length - 1, predicted_index + search_half_width)
    search_indices = np.arange(search_lo, search_hi + 1)
    search_indices = search_indices[mask_profile[search_indices]]

    if search_indices.size == 0:
        return None

    peak_index = int(
        search_indices[np.argmax(search_profile[search_indices])]
    )

    left_indices = np.arange(
        max(0, peak_index - args.background_outer),
        max(0, peak_index - args.background_inner) + 1,
    )
    right_indices = np.arange(
        min(length - 1, peak_index + args.background_inner),
        min(length - 1, peak_index + args.background_outer) + 1,
    )
    left_indices = left_indices[mask_profile[left_indices]]
    right_indices = right_indices[mask_profile[right_indices]]

    if left_indices.size + right_indices.size < 6:
        return None

    left_background = (
        float(np.median(measure_profile[left_indices]))
        if left_indices.size else float("nan")
    )
    right_background = (
        float(np.median(measure_profile[right_indices]))
        if right_indices.size else float("nan")
    )

    if not np.isfinite(left_background):
        left_background = right_background
    if not np.isfinite(right_background):
        right_background = left_background
    if not np.isfinite(left_background) or not np.isfinite(right_background):
        return None

    side_residuals = []
    if left_indices.size:
        side_residuals.extend(
            (measure_profile[left_indices] - left_background).tolist()
        )
    if right_indices.size:
        side_residuals.extend(
            (measure_profile[right_indices] - right_background).tolist()
        )
    noise = robust_noise(np.asarray(side_residuals), args.noise_floor_dn)

    centroid_lo = max(0, peak_index - args.centroid_half_width)
    centroid_hi = min(length - 1, peak_index + args.centroid_half_width)
    coordinates = np.arange(centroid_lo, centroid_hi + 1, dtype=np.float64)

    # 左右背景中点之间建立线性基线，适应缓慢背景梯度。
    left_anchor = (
        float(np.mean(left_indices)) if left_indices.size
        else float(peak_index - args.background_inner)
    )
    right_anchor = (
        float(np.mean(right_indices)) if right_indices.size
        else float(peak_index + args.background_inner)
    )
    if abs(right_anchor - left_anchor) < 1e-9:
        baseline = np.full_like(coordinates, 0.5 * (
            left_background + right_background
        ))
    else:
        baseline = left_background + (
            (coordinates - left_anchor)
            / (right_anchor - left_anchor)
            * (right_background - left_background)
        )

    raw_residual = measure_profile[centroid_lo:centroid_hi + 1] - baseline
    residual = np.maximum(raw_residual, 0.0)
    weights = np.maximum(
        raw_residual - args.centroid_noise_k * noise, 0.0
    )
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return None

    center = float(np.dot(coordinates, weights) / weight_sum)
    peak_local_index = int(np.argmax(residual))
    peak_residual = float(residual[peak_local_index])
    snr = peak_residual / noise

    if peak_residual < args.min_peak_ratio * maximum:
        return None
    if snr < minimum_snr:
        return None

    width = subpixel_fwhm(coordinates, residual, peak_local_index)

    left_energy = float(np.sum(weights[coordinates < center]))
    right_energy = float(np.sum(weights[coordinates > center]))
    symmetry_denominator = left_energy + right_energy
    asymmetry = (
        abs(left_energy - right_energy) / symmetry_denominator
        if symmetry_denominator > 0 else 1.0
    )

    saturation_window_lo = max(0, peak_index - 1)
    saturation_window_hi = min(length - 1, peak_index + 1)
    saturated = bool(
        np.any(
            raw_profile[saturation_window_lo:saturation_window_hi + 1]
            >= args.sat_ratio * maximum
        )
    )

    center_index = int(round(center))
    if not (0 <= center_index < length):
        return None
    if axis == "x":
        if usable_mask[center_index, scan] == 0:
            return None
        x, y = float(scan), center
    else:
        if usable_mask[scan, center_index] == 0:
            return None
        x, y = center, float(scan)

    rejection_reasons: list[str] = []
    if not np.isfinite(width):
        rejection_reasons.append("invalid_fwhm")
    elif width < args.min_fwhm:
        rejection_reasons.append("fwhm_too_small")
    elif width > args.max_fwhm:
        rejection_reasons.append("fwhm_too_large")

    if asymmetry > args.max_asymmetry:
        rejection_reasons.append("asymmetric_peak")
    if saturated and not args.keep_saturated:
        rejection_reasons.append("saturated")

    quality_ok = not rejection_reasons if apply_quality_filters else True

    return {
        "scan": int(scan),
        "prediction": float(prediction),
        "x": x,
        "y": y,
        "peak_index": int(peak_index),
        "peak_residual_dn": peak_residual,
        "noise_dn": float(noise),
        "snr": float(snr),
        "fwhm_px": float(width),
        "asymmetry": float(asymmetry),
        "saturated": int(saturated),
        "quality_ok": int(quality_ok),
        "rejection_reason": "|".join(rejection_reasons),
    }


# ---------------------------------------------------------------------------
# 直线拟合和理论有效扫描线
# ---------------------------------------------------------------------------

def point_arrays(
    points: list[dict[str, Any]], axis: str
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([[point["x"], point["y"]] for point in points], dtype=float)
    if axis == "x":
        return xy[:, 0], xy[:, 1]
    return xy[:, 1], xy[:, 0]


def ransac_line(
    points: list[dict[str, Any]],
    axis: str,
    residual_threshold: float,
    iterations: int,
) -> tuple[np.ndarray, float, float]:
    if len(points) < 2:
        return np.zeros(len(points), dtype=bool), float("nan"), float("nan")

    independent, dependent = point_arrays(points, axis)
    rng = np.random.default_rng(20260724)

    best_mask = np.zeros(len(points), dtype=bool)
    best_error = float("inf")

    for _ in range(iterations):
        i, j = rng.choice(len(points), size=2, replace=False)
        delta = independent[j] - independent[i]
        if abs(delta) < 1e-9:
            continue

        slope = (dependent[j] - dependent[i]) / delta
        intercept = dependent[i] - slope * independent[i]
        residuals = np.abs(
            dependent - (slope * independent + intercept)
        )
        mask = residuals <= residual_threshold
        count = int(np.sum(mask))
        if count < 2:
            continue

        error = float(np.mean(residuals[mask] ** 2))
        if count > int(np.sum(best_mask)) or (
            count == int(np.sum(best_mask)) and error < best_error
        ):
            best_mask = mask
            best_error = error

    if int(np.sum(best_mask)) < 2:
        best_mask[:] = True

    slope, intercept = np.polyfit(
        independent[best_mask], dependent[best_mask], 1
    )
    residuals = np.abs(
        dependent - (slope * independent + intercept)
    )
    refined_mask = residuals <= residual_threshold

    if int(np.sum(refined_mask)) >= 2:
        slope, intercept = np.polyfit(
            independent[refined_mask], dependent[refined_mask], 1
        )
        residuals = np.abs(
            dependent - (slope * independent + intercept)
        )
        refined_mask = residuals <= residual_threshold

    return refined_mask, float(slope), float(intercept)


def prediction_point(
    scan: int, slope: float, intercept: float, axis: str
) -> tuple[float, float]:
    dependent = slope * scan + intercept
    return (
        (float(scan), float(dependent))
        if axis == "x"
        else (float(dependent), float(scan))
    )


def scanline_sets(
    slope: float,
    intercept: float,
    axis: str,
    board_mask: np.ndarray,
    usable_mask: np.ndarray,
) -> tuple[list[int], list[int]]:
    """
    返回：
    - 激光拟合线穿过完整棋盘格的扫描线；
    - 排除黑白边界后可用于精确中心提取的扫描线。
    """
    count = board_mask.shape[1] if axis == "x" else board_mask.shape[0]
    board_scans: list[int] = []
    usable_scans: list[int] = []

    for scan in range(count):
        x, y = prediction_point(scan, slope, intercept, axis)
        xi = int(round(x))
        yi = int(round(y))
        if not (
            0 <= xi < board_mask.shape[1]
            and 0 <= yi < board_mask.shape[0]
        ):
            continue
        if board_mask[yi, xi] > 0:
            board_scans.append(scan)
        if usable_mask[yi, xi] > 0:
            usable_scans.append(scan)

    return board_scans, usable_scans


def contiguous_runs(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    sorted_values = sorted(set(int(value) for value in values))
    runs = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs


def longest_run_ratio(valid_scans: list[int], possible_scans: list[int]) -> float:
    possible_runs = contiguous_runs(possible_scans)
    valid_set = set(valid_scans)
    if not possible_runs:
        return 0.0

    best = 0.0
    for possible_run in possible_runs:
        valid_in_segment = [
            value for value in possible_run if value in valid_set
        ]
        valid_runs = contiguous_runs(valid_in_segment)
        longest_valid = max((len(run) for run in valid_runs), default=0)
        best = max(best, longest_valid / len(possible_run))
    return float(best)


def line_residuals(
    points: list[dict[str, Any]],
    slope: float,
    intercept: float,
    axis: str,
) -> np.ndarray:
    if not points:
        return np.array([], dtype=float)
    independent, dependent = point_arrays(points, axis)
    return dependent - (slope * independent + intercept)


def choose_axis(signal: np.ndarray, mask: np.ndarray) -> str:
    values = signal[mask > 0]
    if values.size == 0:
        return "x"
    threshold = float(np.percentile(values, 99.5))
    ys, xs = np.nonzero((signal >= threshold) & (mask > 0))
    if xs.size < 20:
        return "x"
    return "x" if np.ptp(xs) >= np.ptp(ys) else "y"


# ---------------------------------------------------------------------------
# 两阶段提取
# ---------------------------------------------------------------------------

def initial_candidates(
    laser_on: np.ndarray,
    measurement: np.ndarray,
    initial_signal: np.ndarray,
    usable_mask: np.ndarray,
    axis: str,
    maximum: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """
    每条扫描线先在完整有效棋盘区域寻找最强响应。
    这些点只用于初始直线，不作为最终测量结果。
    """
    scan_count = (
        laser_on.shape[1] if axis == "x" else laser_on.shape[0]
    )
    candidates: list[dict[str, Any]] = []

    for scan in range(scan_count):
        if axis == "x":
            valid_positions = np.flatnonzero(usable_mask[:, scan] > 0)
            profile = initial_signal[:, scan]
        else:
            valid_positions = np.flatnonzero(usable_mask[scan, :] > 0)
            profile = initial_signal[scan, :]

        if valid_positions.size == 0:
            continue

        rough_peak = int(
            valid_positions[np.argmax(profile[valid_positions])]
        )
        point = extract_local_center(
            scan=scan,
            prediction=float(rough_peak),
            laser_on=laser_on,
            measurement=measurement,
            initial_signal=initial_signal,
            usable_mask=usable_mask,
            axis=axis,
            maximum=maximum,
            args=args,
            search_half_width=2,
            minimum_snr=args.initial_min_snr,
            apply_quality_filters=False,
        )
        if point is not None:
            candidates.append(point)

    return candidates


def refine_from_line(
    slope: float,
    intercept: float,
    laser_on: np.ndarray,
    measurement: np.ndarray,
    initial_signal: np.ndarray,
    usable_mask: np.ndarray,
    axis: str,
    maximum: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scan_count = (
        laser_on.shape[1] if axis == "x" else laser_on.shape[0]
    )
    all_candidates: list[dict[str, Any]] = []
    quality_candidates: list[dict[str, Any]] = []

    for scan in range(scan_count):
        x, y = prediction_point(scan, slope, intercept, axis)
        xi = int(round(x))
        yi = int(round(y))
        if not (
            0 <= xi < usable_mask.shape[1]
            and 0 <= yi < usable_mask.shape[0]
            and usable_mask[yi, xi] > 0
        ):
            continue

        point = extract_local_center(
            scan=scan,
            prediction=(y if axis == "x" else x),
            laser_on=laser_on,
            measurement=measurement,
            initial_signal=initial_signal,
            usable_mask=usable_mask,
            axis=axis,
            maximum=maximum,
            args=args,
            search_half_width=args.search_half_width,
            minimum_snr=args.min_snr,
            apply_quality_filters=True,
        )
        if point is None:
            continue

        all_candidates.append(point)
        if point["quality_ok"]:
            quality_candidates.append(point)

    return all_candidates, quality_candidates


def extract_two_stage(
    laser_on: np.ndarray,
    laser_off: np.ndarray | None,
    usable_mask: np.ndarray,
    axis: str,
    maximum: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    initial_signal = make_initial_signal(laser_on, laser_off)
    measurement = make_measurement_image(laser_on, laser_off)

    seeds = initial_candidates(
        laser_on, measurement, initial_signal,
        usable_mask, axis, maximum, args
    )
    seed_inliers, slope, intercept = ransac_line(
        seeds, axis, args.ransac_residual * 2.0, args.ransac_iters
    )
    seed_valid = [
        point for point, keep in zip(seeds, seed_inliers) if bool(keep)
    ]

    if len(seed_valid) < 2 or not np.isfinite(slope):
        return {
            "initial_signal": initial_signal,
            "seed_points": seeds,
            "all_candidates": [],
            "valid_points": [],
            "outlier_points": [],
            "slope": float("nan"),
            "intercept": float("nan"),
        }

    all_candidates: list[dict[str, Any]] = []
    quality_candidates: list[dict[str, Any]] = []
    outlier_points: list[dict[str, Any]] = []

    for _ in range(args.refine_iterations):
        all_candidates, quality_candidates = refine_from_line(
            slope, intercept,
            laser_on, measurement, initial_signal,
            usable_mask, axis, maximum, args
        )
        if len(quality_candidates) < 2:
            break

        inliers, new_slope, new_intercept = ransac_line(
            quality_candidates,
            axis,
            args.ransac_residual,
            args.ransac_iters,
        )
        outlier_points = [
            point for point, keep in zip(quality_candidates, inliers)
            if not bool(keep)
        ]
        quality_candidates = [
            point for point, keep in zip(quality_candidates, inliers)
            if bool(keep)
        ]

        if len(quality_candidates) < 2:
            break
        slope, intercept = new_slope, new_intercept

    return {
        "initial_signal": initial_signal,
        "seed_points": seed_valid,
        "all_candidates": all_candidates,
        "valid_points": quality_candidates,
        "outlier_points": outlier_points,
        "slope": float(slope),
        "intercept": float(intercept),
    }


# ---------------------------------------------------------------------------
# 指标、评分和输出
# ---------------------------------------------------------------------------

def finite_stat(
    values: Iterable[float],
    operation: str = "median",
) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    if operation == "mean":
        return float(np.mean(array))
    if operation == "p10":
        return float(np.percentile(array, 10))
    if operation == "p90":
        return float(np.percentile(array, 90))
    if operation == "min":
        return float(np.min(array))
    return float(np.median(array))


def balanced_coverage(black: float, white: float, overall: float) -> float:
    if np.isfinite(black) and np.isfinite(white):
        return float(min(black, white))
    return float(overall)


def quality_score(
    coverage: float,
    continuity: float,
    black_coverage: float,
    white_coverage: float,
    median_snr: float,
    saturation_ratio: float,
    median_fwhm: float,
    median_asymmetry: float,
    residual_rms: float,
) -> float:
    """
    仅用于同一批数据内部排序，不代表物理测量精度。
    """
    coverage_score = float(np.clip(coverage, 0, 1))
    continuity_score = float(np.clip(continuity, 0, 1))
    balance_score = balanced_coverage(
        black_coverage, white_coverage, coverage
    )
    snr_score = (
        float(np.clip((median_snr - 3.0) / 7.0, 0, 1))
        if np.isfinite(median_snr) else 0.0
    )
    saturation_score = float(
        math.exp(-max(0.0, saturation_ratio) / 0.05)
    )
    width_score = (
        float(math.exp(-abs(median_fwhm - 3.0) / 4.0))
        if np.isfinite(median_fwhm) else 0.0
    )
    asymmetry_score = (
        float(math.exp(-median_asymmetry / 0.35))
        if np.isfinite(median_asymmetry) else 0.0
    )
    residual_score = (
        float(math.exp(-residual_rms / 1.0))
        if np.isfinite(residual_rms) else 0.0
    )

    return 100.0 * (
        0.24 * coverage_score
        + 0.12 * continuity_score
        + 0.20 * balance_score
        + 0.12 * snr_score
        + 0.13 * saturation_score
        + 0.06 * width_score
        + 0.05 * asymmetry_score
        + 0.08 * residual_score
    )


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def signal_to_u8(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = signal[mask > 0]
    if values.size == 0:
        return np.zeros(signal.shape, dtype=np.uint8)
    upper = max(float(np.percentile(values, 99.7)), 1.0)
    return np.clip(signal * 255.0 / upper, 0, 255).astype(np.uint8)


def draw_line_segment(
    image: np.ndarray,
    slope: float,
    intercept: float,
    axis: str,
    scans: list[int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    runs = contiguous_runs(scans)
    for run in runs:
        if len(run) < 2:
            continue
        points = [
            prediction_point(scan, slope, intercept, axis)
            for scan in run
        ]
        polyline = np.round(np.asarray(points)).astype(np.int32)
        cv2.polylines(
            image, [polyline], False, color, thickness, cv2.LINE_AA
        )


def make_failed_row(
    group: str,
    exposure: float | None,
    filename: str,
    board_reference_found: int,
    background_found: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "group": group,
        "exposure_us": exposure,
        "filename": filename,
        "board_reference_found": board_reference_found,
        "background_found": background_found,
        "board_detected": 0,
        "failure_reason": reason,
        "scan_axis": "",
        "initial_point_count": 0,
        "board_intersection_scanlines": 0,
        "boundary_excluded_scanlines": 0,
        "possible_scanlines": 0,
        "candidate_count": 0,
        "valid_center_count": 0,
        "coverage": 0.0,
        "longest_run_ratio": 0.0,
        "black_possible": 0,
        "black_valid": 0,
        "black_coverage": float("nan"),
        "white_possible": 0,
        "white_valid": 0,
        "white_coverage": float("nan"),
        "balanced_coverage": 0.0,
        "median_snr": float("nan"),
        "p10_snr": float("nan"),
        "median_noise_dn": float("nan"),
        "median_fwhm_px": float("nan"),
        "p90_fwhm_px": float("nan"),
        "median_asymmetry": float("nan"),
        "candidate_saturation_ratio": float("nan"),
        "line_residual_rms_px": float("nan"),
        "quality_score": 0.0,
    }


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    input_root = args.input.expanduser().resolve()
    board_root = args.board_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    background_root = (
        args.background_root.expanduser().resolve()
        if args.background_root else None
    )

    if not input_root.is_dir():
        print(f"[错误] 激光图目录不存在：{input_root}", file=sys.stderr)
        return 1
    if not board_root.is_dir():
        print(f"[错误] 棋盘格参考目录不存在：{board_root}", file=sys.stderr)
        return 1
    if background_root is not None and not background_root.is_dir():
        print(
            f"[错误] 同曝光激光关闭图目录不存在：{background_root}",
            file=sys.stderr,
        )
        return 1

    laser_files = list_images(input_root)
    if not laser_files:
        print(f"[错误] 未找到激光图像：{input_root}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    for index, laser_path in enumerate(laser_files, start=1):
        relative = laser_path.relative_to(input_root)
        group = exposure_group(input_root, laser_path)
        exposure = parse_exposure(group)

        laser_image = read_image(laser_path)
        if laser_image is None:
            print(f"[跳过] 无法读取：{laser_path}")
            continue
        laser_on = to_gray(laser_image)
        maximum = infer_max_value(laser_on, args.max_value)

        board_path = find_board_reference(
            board_root, args.board_group, relative
        )
        board_reference_found = int(board_path is not None)
        if board_path is None:
            image_rows.append(
                make_failed_row(
                    group, exposure, str(relative), 0, 0,
                    "board_reference_not_found",
                )
            )
            print(
                f"[{index:>3}/{len(laser_files)}] {group:<8} "
                f"未找到长曝光棋盘格图：{relative.name}"
            )
            continue

        board_image = read_image(board_path)
        if board_image is None:
            image_rows.append(
                make_failed_row(
                    group, exposure, str(relative), 1, 0,
                    "board_reference_read_failed",
                )
            )
            continue
        board_gray = to_gray(board_image)

        if board_gray.shape != laser_on.shape:
            image_rows.append(
                make_failed_row(
                    group, exposure, str(relative), 1, 0,
                    "board_reference_shape_mismatch",
                )
            )
            print(
                f"[{index:>3}/{len(laser_files)}] 尺寸不一致："
                f"{relative.name}"
            )
            continue

        laser_off = None
        background_found = 0
        if background_root is not None:
            off_path = find_background(background_root, relative)
            if off_path is not None:
                off_image = read_image(off_path)
                if off_image is not None:
                    candidate_off = to_gray(off_image)
                    if candidate_off.shape == laser_on.shape:
                        laser_off = candidate_off
                        background_found = 1

        board_maximum = infer_max_value(board_gray, 0)
        board_found, corners = detect_chessboard(
            normalize_u8(board_gray, board_maximum),
            (args.cols, args.rows),
        )
        if not board_found or corners is None:
            image_rows.append(
                make_failed_row(
                    group, exposure, str(relative),
                    board_reference_found, background_found,
                    "chessboard_detection_failed",
                )
            )
            print(
                f"[{index:>3}/{len(laser_files)}] {group:<8} "
                f"棋盘格检测失败：{relative.name}"
            )
            continue

        try:
            H = compute_board_homography(corners, args.cols, args.rows)
            H_inv = np.linalg.inv(H)
        except (RuntimeError, np.linalg.LinAlgError):
            image_rows.append(
                make_failed_row(
                    group, exposure, str(relative),
                    board_reference_found, background_found,
                    "homography_failed",
                )
            )
            continue

        board_mask, boundary_mask, usable_mask, grid_lines = (
            make_board_geometry(
                laser_on.shape,
                H,
                args.cols,
                args.rows,
                args.board_erode,
                args.boundary_exclude,
            )
        )
        white_parity = determine_white_parity(
            board_gray, H, args.cols, args.rows
        )

        initial_signal_for_axis = make_initial_signal(laser_on, laser_off)
        axis = (
            choose_axis(initial_signal_for_axis, usable_mask)
            if args.scan_axis == "auto"
            else args.scan_axis
        )

        extracted = extract_two_stage(
            laser_on=laser_on,
            laser_off=laser_off,
            usable_mask=usable_mask,
            axis=axis,
            maximum=maximum,
            args=args,
        )
        slope = extracted["slope"]
        intercept = extracted["intercept"]
        seed_points = extracted["seed_points"]
        candidates = extracted["all_candidates"]
        valid_points = extracted["valid_points"]
        outlier_points = extracted["outlier_points"]
        initial_signal = extracted["initial_signal"]

        if not np.isfinite(slope) or len(valid_points) < 2:
            row = make_failed_row(
                group, exposure, str(relative),
                board_reference_found, background_found,
                "laser_line_fit_failed",
            )
            row["board_detected"] = 1
            row["scan_axis"] = axis
            row["initial_point_count"] = len(seed_points)
            row["candidate_count"] = len(candidates)
            image_rows.append(row)
            print(
                f"[{index:>3}/{len(laser_files)}] {group:<8} "
                f"激光直线拟合失败：{relative.name}"
            )
            continue

        board_scans, possible_scans = scanline_sets(
            slope, intercept, axis, board_mask, usable_mask
        )
        possible_set = set(possible_scans)

        # 最终只保留位于理论有效扫描线中的实测点。
        valid_points = [
            point for point in valid_points
            if int(point["scan"]) in possible_set
        ]

        # 重新拟合一次最终直线，并计算最终残差。
        final_inliers, slope, intercept = ransac_line(
            valid_points,
            axis,
            args.ransac_residual,
            args.ransac_iters,
        )
        final_outliers = [
            point for point, keep in zip(valid_points, final_inliers)
            if not bool(keep)
        ]
        valid_points = [
            point for point, keep in zip(valid_points, final_inliers)
            if bool(keep)
        ]
        outlier_points.extend(final_outliers)

        board_scans, possible_scans = scanline_sets(
            slope, intercept, axis, board_mask, usable_mask
        )
        possible_set = set(possible_scans)
        valid_points = [
            point for point in valid_points
            if int(point["scan"]) in possible_set
        ]

        valid_scan_set = {int(point["scan"]) for point in valid_points}
        coverage = (
            len(valid_scan_set) / len(possible_scans)
            if possible_scans else 0.0
        )
        continuity = longest_run_ratio(
            sorted(valid_scan_set), possible_scans
        )

        black_possible = white_possible = 0
        scan_class: dict[int, int] = {}
        for scan in possible_scans:
            x_pred, y_pred = prediction_point(
                scan, slope, intercept, axis
            )
            surface = board_surface_class(
                x_pred, y_pred, H_inv,
                args.cols, args.rows, white_parity
            )
            scan_class[scan] = surface
            if surface == 1:
                black_possible += 1
            elif surface == 2:
                white_possible += 1

        black_valid = sum(
            scan_class.get(int(point["scan"])) == 1
            for point in valid_points
        )
        white_valid = sum(
            scan_class.get(int(point["scan"])) == 2
            for point in valid_points
        )

        black_coverage = (
            black_valid / black_possible
            if black_possible else float("nan")
        )
        white_coverage = (
            white_valid / white_possible
            if white_possible else float("nan")
        )
        balance = balanced_coverage(
            black_coverage, white_coverage, coverage
        )

        candidate_saturation_ratio = finite_stat(
            [point["saturated"] for point in candidates], "mean"
        )
        median_snr = finite_stat(
            [point["snr"] for point in valid_points]
        )
        p10_snr = finite_stat(
            [point["snr"] for point in valid_points], "p10"
        )
        median_noise = finite_stat(
            [point["noise_dn"] for point in valid_points]
        )
        median_fwhm = finite_stat(
            [point["fwhm_px"] for point in valid_points]
        )
        p90_fwhm = finite_stat(
            [point["fwhm_px"] for point in valid_points], "p90"
        )
        median_asymmetry = finite_stat(
            [point["asymmetry"] for point in valid_points]
        )

        residuals = line_residuals(
            valid_points, slope, intercept, axis
        )
        residual_rms = (
            float(np.sqrt(np.mean(residuals ** 2)))
            if residuals.size else float("nan")
        )

        score = quality_score(
            coverage,
            continuity,
            black_coverage,
            white_coverage,
            median_snr,
            candidate_saturation_ratio,
            median_fwhm,
            median_asymmetry,
            residual_rms,
        )

        row = {
            "group": group,
            "exposure_us": exposure,
            "filename": str(relative),
            "board_reference_found": board_reference_found,
            "background_found": background_found,
            "board_detected": 1,
            "failure_reason": "",
            "scan_axis": axis,
            "initial_point_count": len(seed_points),
            "board_intersection_scanlines": len(board_scans),
            "boundary_excluded_scanlines": (
                len(board_scans) - len(possible_scans)
            ),
            "possible_scanlines": len(possible_scans),
            "candidate_count": len(candidates),
            "valid_center_count": len(valid_points),
            "coverage": coverage,
            "longest_run_ratio": continuity,
            "black_possible": black_possible,
            "black_valid": black_valid,
            "black_coverage": black_coverage,
            "white_possible": white_possible,
            "white_valid": white_valid,
            "white_coverage": white_coverage,
            "balanced_coverage": balance,
            "median_snr": median_snr,
            "p10_snr": p10_snr,
            "median_noise_dn": median_noise,
            "median_fwhm_px": median_fwhm,
            "p90_fwhm_px": p90_fwhm,
            "median_asymmetry": median_asymmetry,
            "candidate_saturation_ratio": candidate_saturation_ratio,
            "line_residual_rms_px": residual_rms,
            "quality_score": score,
        }
        image_rows.append(row)

        # 每个候选点的诊断信息。
        valid_ids = {id(point) for point in valid_points}
        outlier_ids = {id(point) for point in outlier_points}
        for point in candidates:
            scan = int(point["scan"])
            surface = scan_class.get(scan, 0)
            if id(point) in valid_ids:
                final_status = "valid"
            elif id(point) in outlier_ids:
                final_status = "ransac_outlier"
            elif not point["quality_ok"]:
                final_status = "quality_rejected"
            else:
                final_status = "not_final"

            point_rows.append({
                "group": group,
                "exposure_us": exposure,
                "filename": str(relative),
                "scan_axis": axis,
                "scan": scan,
                "x": point["x"],
                "y": point["y"],
                "prediction": point["prediction"],
                "surface": (
                    "black" if surface == 1
                    else "white" if surface == 2
                    else "outside"
                ),
                "peak_residual_dn": point["peak_residual_dn"],
                "noise_dn": point["noise_dn"],
                "snr": point["snr"],
                "fwhm_px": point["fwhm_px"],
                "asymmetry": point["asymmetry"],
                "saturated": point["saturated"],
                "quality_ok": point["quality_ok"],
                "rejection_reason": point["rejection_reason"],
                "final_status": final_status,
            })

        # 可视化
        visual = cv2.cvtColor(
            normalize_u8(laser_on, maximum), cv2.COLOR_GRAY2BGR
        )
        outer_polygon = transform_points(
            [(-1, -1), (args.cols, -1),
             (args.cols, args.rows), (-1, args.rows)],
            H,
        )
        cv2.polylines(
            visual,
            [np.round(outer_polygon).astype(np.int32)],
            True,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 青色细线显示被排除的内部黑白边界。
        for line in grid_lines:
            cv2.line(
                visual,
                tuple(np.round(line[0]).astype(int)),
                tuple(np.round(line[1]).astype(int)),
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # 蓝色为最终拟合线在有效棋盘格中的理论段。
        draw_line_segment(
            visual, slope, intercept, axis,
            possible_scans, (255, 0, 0), 1
        )

        # 橙色：质量筛选未通过；红色：RANSAC离群点；绿色：有效中心。
        for point in candidates:
            center = (
                int(round(point["x"])),
                int(round(point["y"])),
            )
            if not point["quality_ok"]:
                cv2.circle(visual, center, 2, (0, 165, 255), -1)

        for point in outlier_points:
            center = (
                int(round(point["x"])),
                int(round(point["y"])),
            )
            cv2.circle(visual, center, 2, (0, 0, 255), -1)

        for point in valid_points:
            center = (
                int(round(point["x"])),
                int(round(point["y"])),
            )
            cv2.circle(visual, center, 1, (0, 255, 0), -1)

        text = (
            f"{group} cov={coverage*100:.1f}% "
            f"bal={balance*100:.1f}% "
            f"sat={candidate_saturation_ratio*100:.2f}% "
            f"score={score:.1f}"
        )
        cv2.putText(
            visual, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 255), 2, cv2.LINE_AA
        )
        write_image(
            output_root / "annotated" / relative.with_suffix(".png"),
            visual,
        )

        if args.save_signal:
            write_image(
                output_root / "signal" / relative.with_suffix(".png"),
                signal_to_u8(initial_signal, board_mask),
            )

        print(
            f"[{index:>3}/{len(laser_files)}] {group:<8} "
            f"覆盖={coverage*100:5.1f}%  "
            f"黑={black_coverage*100:5.1f}%  "
            f"白={white_coverage*100:5.1f}%  "
            f"饱和={candidate_saturation_ratio*100:5.2f}%  "
            f"FWHM={median_fwhm:4.2f}px  "
            f"得分={score:5.1f}"
        )

    image_fields = [
        "group", "exposure_us", "filename",
        "board_reference_found", "background_found", "board_detected",
        "failure_reason", "scan_axis", "initial_point_count",
        "board_intersection_scanlines", "boundary_excluded_scanlines",
        "possible_scanlines", "candidate_count", "valid_center_count",
        "coverage", "longest_run_ratio",
        "black_possible", "black_valid", "black_coverage",
        "white_possible", "white_valid", "white_coverage",
        "balanced_coverage", "median_snr", "p10_snr",
        "median_noise_dn", "median_fwhm_px", "p90_fwhm_px",
        "median_asymmetry", "candidate_saturation_ratio",
        "line_residual_rms_px", "quality_score",
    ]
    save_csv(
        output_root / "image_metrics.csv",
        image_rows,
        image_fields,
    )

    point_fields = [
        "group", "exposure_us", "filename", "scan_axis", "scan",
        "x", "y", "prediction", "surface",
        "peak_residual_dn", "noise_dn", "snr", "fwhm_px",
        "asymmetry", "saturated", "quality_ok",
        "rejection_reason", "final_status",
    ]
    save_csv(
        output_root / "center_points.csv",
        point_rows,
        point_fields,
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        grouped[str(row["group"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group, rows in grouped.items():
        successful = [
            row for row in rows
            if int(row["board_detected"]) == 1
            and not row["failure_reason"]
        ]
        source = successful if successful else rows

        summary = {
            "group": group,
            "exposure_us": parse_exposure(group),
            "image_count": len(rows),
            "successful_image_count": len(successful),
            "success_rate": len(successful) / len(rows) if rows else 0.0,
            "board_reference_rate": finite_stat(
                [row["board_reference_found"] for row in rows], "mean"
            ),
            "background_pair_rate": finite_stat(
                [row["background_found"] for row in rows], "mean"
            ),
            "median_coverage": finite_stat(
                [row["coverage"] for row in source]
            ),
            "worst_coverage": finite_stat(
                [row["coverage"] for row in source], "p10"
            ),
            "median_longest_run_ratio": finite_stat(
                [row["longest_run_ratio"] for row in source]
            ),
            "median_black_coverage": finite_stat(
                [row["black_coverage"] for row in source]
            ),
            "worst_black_coverage": finite_stat(
                [row["black_coverage"] for row in source], "p10"
            ),
            "median_white_coverage": finite_stat(
                [row["white_coverage"] for row in source]
            ),
            "worst_white_coverage": finite_stat(
                [row["white_coverage"] for row in source], "p10"
            ),
            "median_balanced_coverage": finite_stat(
                [row["balanced_coverage"] for row in source]
            ),
            "worst_balanced_coverage": finite_stat(
                [row["balanced_coverage"] for row in source], "p10"
            ),
            "median_snr": finite_stat(
                [row["median_snr"] for row in source]
            ),
            "median_p10_snr": finite_stat(
                [row["p10_snr"] for row in source]
            ),
            "median_noise_dn": finite_stat(
                [row["median_noise_dn"] for row in source]
            ),
            "median_fwhm_px": finite_stat(
                [row["median_fwhm_px"] for row in source]
            ),
            "median_p90_fwhm_px": finite_stat(
                [row["p90_fwhm_px"] for row in source]
            ),
            "median_asymmetry": finite_stat(
                [row["median_asymmetry"] for row in source]
            ),
            "mean_candidate_saturation_ratio": finite_stat(
                [row["candidate_saturation_ratio"] for row in source],
                "mean",
            ),
            "median_line_residual_rms_px": finite_stat(
                [row["line_residual_rms_px"] for row in source]
            ),
            "median_quality_score": finite_stat(
                [row["quality_score"] for row in source]
            ),
            "worst_quality_score": finite_stat(
                [row["quality_score"] for row in source], "p10"
            ),
        }
        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda row: (
            float("inf")
            if row["exposure_us"] is None else float(row["exposure_us"])
        )
    )

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    save_csv(
        output_root / "exposure_summary.csv",
        summary_rows,
        summary_fields,
    )

    # 自动推荐规则：
    # 1. 成功率尽量高；
    # 2. 平均候选饱和率不超过阈值；
    # 3. 优先困难姿态下的黑白平衡覆盖；
    # 4. 再考虑总体覆盖、连续性和较低饱和。
    eligible = [
        row for row in summary_rows
        if row["success_rate"] >= 0.8
        and np.isfinite(row["mean_candidate_saturation_ratio"])
        and row["mean_candidate_saturation_ratio"]
        <= args.max_mean_saturation
    ]
    recommendation_pool = eligible if eligible else summary_rows
    ranked = sorted(
        recommendation_pool,
        key=lambda row: (
            -float(row["worst_balanced_coverage"])
            if np.isfinite(row["worst_balanced_coverage"]) else float("inf"),
            -float(row["median_balanced_coverage"])
            if np.isfinite(row["median_balanced_coverage"]) else float("inf"),
            -float(row["worst_coverage"])
            if np.isfinite(row["worst_coverage"]) else float("inf"),
            -float(row["median_longest_run_ratio"])
            if np.isfinite(row["median_longest_run_ratio"]) else float("inf"),
            float(row["mean_candidate_saturation_ratio"])
            if np.isfinite(row["mean_candidate_saturation_ratio"])
            else float("inf"),
        ),
    )

    recommendation_path = output_root / "recommendation.txt"
    with recommendation_path.open("w", encoding="utf-8") as handle:
        handle.write("激光线曝光分析 V3\n")
        handle.write("=" * 72 + "\n\n")
        handle.write(
            "自动推荐优先保证困难姿态下黑格与白格的平衡覆盖，"
            "并限制候选点平均饱和率。\n"
        )
        handle.write(
            f"当前允许的平均候选饱和率上限："
            f"{args.max_mean_saturation*100:.2f}%\n\n"
        )

        if ranked:
            handle.write(
                f"推荐优先复核的单曝光：{ranked[0]['group']}\n\n"
            )

        handle.write("候选排序：\n")
        for rank, row in enumerate(ranked, start=1):
            handle.write(
                f"{rank}. {row['group']}: "
                f"最差平衡覆盖={row['worst_balanced_coverage']*100:.2f}%, "
                f"中位平衡覆盖={row['median_balanced_coverage']*100:.2f}%, "
                f"总体覆盖={row['median_coverage']*100:.2f}%, "
                f"连续性={row['median_longest_run_ratio']*100:.2f}%, "
                f"饱和={row['mean_candidate_saturation_ratio']*100:.2f}%, "
                f"FWHM={row['median_fwhm_px']:.3f}px, "
                f"残差={row['median_line_residual_rms_px']:.3f}px\n"
            )

        handle.write(
            "\n注意：推荐结果仍需结合 annotated 标注图人工确认。"
            "绿色点应位于激光条纹中心，橙色/红色异常点不应大量出现。"
        )

    print("\n=== 各曝光汇总 ===")
    print(
        "曝光\t平衡覆盖\t最差平衡\t总体覆盖\t饱和率\t"
        "FWHM\t残差\t综合分"
    )
    for row in summary_rows:
        print(
            f"{row['group']}\t"
            f"{row['median_balanced_coverage']*100:.1f}%\t"
            f"{row['worst_balanced_coverage']*100:.1f}%\t"
            f"{row['median_coverage']*100:.1f}%\t"
            f"{row['mean_candidate_saturation_ratio']*100:.2f}%\t"
            f"{row['median_fwhm_px']:.2f}\t"
            f"{row['median_line_residual_rms_px']:.3f}\t"
            f"{row['median_quality_score']:.1f}"
        )

    if ranked:
        print(
            f"\n建议首先人工复核：{ranked[0]['group']}，"
            "并同时比较其前后相邻曝光。"
        )
    print(f"结果已保存到：{output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
