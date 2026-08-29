#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
棋盘格角点重复性底限分析
==========================

用途
----
对“相机与棋盘格保持固定，同一曝光连续采集多帧”的图像进行角点重复性分析。

脚本同时报告两类结果：

1. raw（原始重复性）
   直接统计同一角点在多帧图像中的坐标波动。
   其中包含：
   - 角点检测噪声；
   - 相机/棋盘格微小振动；
   - 支架漂移；
   - 图像整体平移、旋转或微小尺度变化。

2. registered（配准后重复性）
   每帧角点通过二维相似变换（平移+旋转+统一尺度）配准到公共模板，
   再统计局部残差。它更接近：
   - 角点亚像素定位算法的重复性底限；
   - 局部成像噪声和边缘质量造成的波动。

注意
----
配准后结果不是相机标定的重投影误差，也不是最终三维误差。
它只评价“固定棋盘格在重复采集时，角点坐标能重复到什么程度”。

推荐目录结构（与 metadata.csv 对应）：
dataset_root/
├─ metadata.csv
└─ raw/
   ├─ F4_exp20000_chessboard/
   │  ├─ frame_0001.tiff
   │  └─ ...
   ├─ F4_exp30000_chessboard/
   ├─ F4_exp40000_chessboard/
   └─ F4_exp50000_chessboard/
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于0")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须大于或等于0")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于0的有限数")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分析固定棋盘格在多帧重复采集中的角点坐标重复性底限。"
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="采集数据的 metadata.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "数据根目录。metadata 中的相对 image_dir 将相对于该目录解析；"
            "默认使用 metadata.csv 所在目录"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corner_repeatability_result"),
        help="结果输出目录",
    )
    parser.add_argument(
        "--cols",
        type=positive_int,
        default=6,
        help="棋盘格横向内角点数量，默认6",
    )
    parser.add_argument(
        "--rows",
        type=positive_int,
        default=5,
        help="棋盘格纵向内角点数量，默认5",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=0,
        help="图像有效最大灰度：Mono8填255，Mono12填4095；0表示自动",
    )
    parser.add_argument(
        "--max-frames",
        type=nonnegative_int,
        default=0,
        help="每组最多分析多少帧；0表示全部",
    )
    parser.add_argument(
        "--reference-exposure-us",
        type=float,
        help="跨曝光系统偏移分析的参考曝光；默认选择最短曝光",
    )
    parser.add_argument(
        "--focal-length-px",
        type=positive_float,
        help=(
            "可选：像素焦距。若提供，且 metadata 中有 distance_mm、baseline_mm，"
            "会给出简化三角模型下的等效深度重复性"
        ),
    )
    parser.add_argument(
        "--save-detected-preview",
        action="store_true",
        help="每个曝光保存第一张成功检测的角点标注图",
    )
    return parser


# ---------------------------------------------------------------------------
# 图像读取与棋盘格检测
# ---------------------------------------------------------------------------

def imread_unicode(path: Path) -> np.ndarray | None:
    """兼容Windows中文路径。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"无法保存图像：{path}")
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
        # 大多数Mono12图像以uint16容器保存。
        return 4095.0
    observed = float(np.nanmax(gray))
    return max(observed, 1.0)


def normalize_to_u8(gray: np.ndarray, maximum: float) -> np.ndarray:
    return np.clip(
        gray.astype(np.float32) * 255.0 / maximum,
        0,
        255,
    ).astype(np.uint8)


def detect_chessboard(
    gray8: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None, str]:
    """
    与相机标定流程保持接近：
    1. 优先使用findChessboardCornersSB；
    2. SB失败时使用传统检测+cornerSubPix。
    """
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray8, pattern_size, flags
            )
            if found and corners is not None:
                return True, corners.reshape(-1, 2).astype(np.float64), "SB"
        except cv2.error:
            pass

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    found, corners = cv2.findChessboardCorners(
        gray8, pattern_size, flags
    )
    if found and corners is not None:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            80,
            1e-4,
        )
        corners = cv2.cornerSubPix(
            gray8,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
        return (
            True,
            corners.reshape(-1, 2).astype(np.float64),
            "traditional+cornerSubPix",
        )

    return False, None, "failed"


def normalize_corner_order(
    corners: np.ndarray,
    reference: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    固定姿态实验中，正确的角点编号应与参考帧坐标最接近。
    尝试四种行列翻转，避免偶发的角点顺序翻转。
    """
    grid = corners.reshape(rows, cols, 2)
    candidates = [
        grid,
        grid[::-1, ::-1],
        grid[:, ::-1],
        grid[::-1, :],
    ]
    candidate_arrays = [item.reshape(-1, 2) for item in candidates]
    errors = [
        float(np.sqrt(np.mean(np.sum((item - reference) ** 2, axis=1))))
        for item in candidate_arrays
    ]
    return candidate_arrays[int(np.argmin(errors))]


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# 相似变换配准
# ---------------------------------------------------------------------------

@dataclass
class SimilarityTransform:
    scale: float
    rotation_deg: float
    translation_x: float
    translation_y: float
    matrix: np.ndarray


def fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> SimilarityTransform:
    """
    使用最小二乘Procrustes/Umeyama方法，求：
        target ≈ scale * R * source + t
    不允许镜像。
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = (
        target_centered.T @ source_centered / len(source)
    )
    U, singular_values, Vt = np.linalg.svd(covariance)

    correction = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1

    rotation = U @ correction @ Vt
    source_variance = float(
        np.mean(np.sum(source_centered ** 2, axis=1))
    )
    if source_variance <= 1e-15:
        raise ValueError("角点分布退化，无法拟合相似变换")

    scale = float(
        np.sum(singular_values * np.diag(correction))
        / source_variance
    )
    translation = target_mean - scale * (rotation @ source_mean)

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation

    rotation_deg = math.degrees(
        math.atan2(rotation[1, 0], rotation[0, 0])
    )
    return SimilarityTransform(
        scale=scale,
        rotation_deg=rotation_deg,
        translation_x=float(translation[0]),
        translation_y=float(translation[1]),
        matrix=matrix,
    )


def apply_similarity(
    points: np.ndarray,
    transform: SimilarityTransform,
) -> np.ndarray:
    homogeneous = np.column_stack(
        [points, np.ones(len(points), dtype=np.float64)]
    )
    return homogeneous @ transform.matrix.T


def iterative_register(
    corner_frames: np.ndarray,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[SimilarityTransform]]:
    """
    先以第一帧为模板配准，再用所有配准结果的逐点中位数更新模板。
    这样可以减少第一帧自身噪声对registered结果的影响。
    """
    template = corner_frames[0].copy()
    transforms: list[SimilarityTransform] = []

    for _ in range(iterations):
        aligned_frames = []
        transforms = []
        for corners in corner_frames:
            transform = fit_similarity(corners, template)
            aligned_frames.append(apply_similarity(corners, transform))
            transforms.append(transform)
        aligned_array = np.asarray(aligned_frames)
        template = np.median(aligned_array, axis=0)

    # 最后再严格对齐到最终模板。
    aligned_frames = []
    transforms = []
    for corners in corner_frames:
        transform = fit_similarity(corners, template)
        aligned_frames.append(apply_similarity(corners, transform))
        transforms.append(transform)

    return np.asarray(aligned_frames), template, transforms


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def sample_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    if values.shape[axis] < 2:
        output_shape = list(values.shape)
        del output_shape[axis]
        return np.full(output_shape, np.nan)
    return np.std(values, axis=axis, ddof=1)


def finite_stat(
    values: Iterable[float],
    operation: str,
) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    if operation == "mean":
        return float(np.mean(array))
    if operation == "median":
        return float(np.median(array))
    if operation == "p95":
        return float(np.percentile(array, 95))
    if operation == "max":
        return float(np.max(array))
    if operation == "min":
        return float(np.min(array))
    raise ValueError(operation)


def linear_drift(values: np.ndarray) -> tuple[float, float]:
    """
    返回每帧漂移斜率和从第一帧到最后一帧的拟合总漂移。
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float("nan"), float("nan")
    index = np.arange(len(values), dtype=np.float64)
    slope, _ = np.polyfit(index, values, 1)
    return float(slope), float(slope * (len(values) - 1))


def robust_outlier_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-15:
        return np.zeros(len(values), dtype=bool)
    robust_sigma = 1.4826 * mad
    return values > median + 3.0 * robust_sigma


def interpretation_label(
    detection_rate: float,
    registered_p95_sigma: float,
) -> str:
    """
    经验性分级，仅用于同一设备实验的快速解释，不是行业标准。
    """
    if detection_rate < 0.95:
        return "检测率不足，先改善曝光/清晰度"
    if not np.isfinite(registered_p95_sigma):
        return "有效帧不足"
    if registered_p95_sigma <= 0.03:
        return "角点重复性极佳"
    if registered_p95_sigma <= 0.05:
        return "角点重复性很好"
    if registered_p95_sigma <= 0.10:
        return "角点重复性良好"
    if registered_p95_sigma <= 0.20:
        return "存在可见定位波动"
    return "重复性偏差较大，需要排查"


# ---------------------------------------------------------------------------
# 绘图和报告
# ---------------------------------------------------------------------------

def save_line_plot(
    x: np.ndarray,
    series: list[tuple[np.ndarray, str]],
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for y, label in series:
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def save_frame_rms_plot(
    frame_indices: np.ndarray,
    raw_rms: np.ndarray,
    registered_rms: np.ndarray,
    title: str,
    output: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(frame_indices, raw_rms, label="raw RMS")
    plt.plot(frame_indices, registered_rms, label="registered RMS")
    plt.xlabel("Successful frame index")
    plt.ylabel("Corner RMS / px")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def save_corner_sigma_map(
    mean_corners: np.ndarray,
    registered_sigma_r: np.ndarray,
    title: str,
    output: Path,
) -> None:
    """
    用点尺寸表示角点波动，不依赖固定颜色映射。
    """
    finite = registered_sigma_r[np.isfinite(registered_sigma_r)]
    reference = float(np.median(finite)) if finite.size else 1.0
    reference = max(reference, 1e-6)
    sizes = 40.0 + 180.0 * np.clip(
        registered_sigma_r / reference, 0, 5
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(
        mean_corners[:, 0],
        mean_corners[:, 1],
        s=sizes,
    )
    for index, (x, y) in enumerate(mean_corners):
        plt.text(x, y, str(index), fontsize=7)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("u / px")
    plt.ylabel("v / px")
    plt.title(title + "\n(circle size ∝ registered σ)")
    plt.grid(True)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()

    metadata_path = args.metadata.expanduser().resolve()
    if not metadata_path.is_file():
        print(f"[错误] metadata不存在：{metadata_path}", file=sys.stderr)
        return 1

    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else metadata_path.parent
    )
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[错误] 无法读取metadata：{exc}", file=sys.stderr)
        return 1

    required_columns = {"exp_id", "image_dir", "exposure_us"}
    missing = required_columns - set(metadata.columns)
    if missing:
        print(
            f"[错误] metadata缺少字段：{sorted(missing)}",
            file=sys.stderr,
        )
        return 1

    pattern_size = (args.cols, args.rows)
    expected_corner_count = args.cols * args.rows

    frame_rows: list[dict[str, Any]] = []
    corner_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    exposure_results: dict[float, dict[str, Any]] = {}

    for _, metadata_row in metadata.iterrows():
        exp_id = str(metadata_row["exp_id"])
        exposure_us = float(metadata_row["exposure_us"])
        image_dir_value = Path(str(metadata_row["image_dir"]))
        image_dir = (
            image_dir_value
            if image_dir_value.is_absolute()
            else data_root / image_dir_value
        )
        image_dir = image_dir.resolve()

        images = list_images(image_dir)
        if args.max_frames > 0:
            images = images[: args.max_frames]

        print(
            f"\n[{exp_id}] exposure={exposure_us:g} us, "
            f"folder={image_dir}, images={len(images)}"
        )

        if not images:
            print("[警告] 未找到图像，跳过。")
            continue

        successful_corners: list[np.ndarray] = []
        successful_files: list[Path] = []
        successful_methods: list[str] = []
        first_reference: np.ndarray | None = None
        preview_saved = False

        for frame_index, image_path in enumerate(images):
            image = imread_unicode(image_path)
            if image is None:
                frame_rows.append({
                    "exp_id": exp_id,
                    "exposure_us": exposure_us,
                    "frame_index": frame_index,
                    "filename": str(image_path),
                    "detected": 0,
                    "detector": "read_failed",
                })
                continue

            try:
                gray = to_gray(image)
            except ValueError:
                frame_rows.append({
                    "exp_id": exp_id,
                    "exposure_us": exposure_us,
                    "frame_index": frame_index,
                    "filename": str(image_path),
                    "detected": 0,
                    "detector": "unsupported_image",
                })
                continue

            maximum = infer_max_value(gray, args.max_value)
            gray8 = normalize_to_u8(gray, maximum)
            found, corners, detector = detect_chessboard(
                gray8, pattern_size
            )

            if not found or corners is None or len(corners) != expected_corner_count:
                frame_rows.append({
                    "exp_id": exp_id,
                    "exposure_us": exposure_us,
                    "frame_index": frame_index,
                    "filename": str(image_path),
                    "detected": 0,
                    "detector": detector,
                })
                continue

            if first_reference is None:
                first_reference = corners.copy()
            else:
                corners = normalize_corner_order(
                    corners,
                    first_reference,
                    args.rows,
                    args.cols,
                )

            successful_corners.append(corners)
            successful_files.append(image_path)
            successful_methods.append(detector)

            frame_rows.append({
                "exp_id": exp_id,
                "exposure_us": exposure_us,
                "frame_index": frame_index,
                "filename": str(image_path),
                "detected": 1,
                "detector": detector,
            })

            if args.save_detected_preview and not preview_saved:
                visual = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(
                    visual,
                    pattern_size,
                    corners.reshape(-1, 1, 2).astype(np.float32),
                    True,
                )
                cv2.putText(
                    visual,
                    f"{exp_id}: {detector}",
                    (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                imwrite_unicode(
                    output_root / "previews" / f"{exp_id}.png",
                    visual,
                )
                preview_saved = True

        total_frames = len(images)
        detected_frames = len(successful_corners)
        detection_rate = (
            detected_frames / total_frames if total_frames else 0.0
        )

        if detected_frames < 2:
            print(
                f"[警告] 仅成功检测{detected_frames}帧，"
                "无法计算标准差。"
            )
            continue

        corners_array = np.asarray(successful_corners, dtype=np.float64)

        # raw：直接相对于逐角点均值。
        raw_mean = np.mean(corners_array, axis=0)
        raw_deviation = corners_array - raw_mean
        raw_frame_rms = np.sqrt(
            np.mean(np.sum(raw_deviation ** 2, axis=2), axis=1)
        )

        raw_sigma_x = sample_std(corners_array[:, :, 0], axis=0)
        raw_sigma_y = sample_std(corners_array[:, :, 1], axis=0)
        raw_sigma_r = np.sqrt(raw_sigma_x ** 2 + raw_sigma_y ** 2)

        # registered：去除整板平移、旋转、统一尺度变化。
        aligned, template, transforms = iterative_register(corners_array)
        registered_mean = np.mean(aligned, axis=0)
        registered_deviation = aligned - registered_mean
        registered_frame_rms = np.sqrt(
            np.mean(
                np.sum(registered_deviation ** 2, axis=2),
                axis=1,
            )
        )

        registered_sigma_x = sample_std(aligned[:, :, 0], axis=0)
        registered_sigma_y = sample_std(aligned[:, :, 1], axis=0)
        registered_sigma_r = np.sqrt(
            registered_sigma_x ** 2 + registered_sigma_y ** 2
        )

        transform_tx = np.array(
            [transform.translation_x for transform in transforms]
        )
        transform_ty = np.array(
            [transform.translation_y for transform in transforms]
        )
        transform_rotation = np.array(
            [transform.rotation_deg for transform in transforms]
        )
        transform_scale = np.array(
            [transform.scale for transform in transforms]
        )

        centroid_x = np.mean(corners_array[:, :, 0], axis=1)
        centroid_y = np.mean(corners_array[:, :, 1], axis=1)
        centroid_x_slope, centroid_x_total = linear_drift(centroid_x)
        centroid_y_slope, centroid_y_total = linear_drift(centroid_y)

        outlier_mask = robust_outlier_mask(registered_frame_rms)

        # 更新成功帧的详细指标。
        success_row_indices = [
            index
            for index, row in enumerate(frame_rows)
            if row["exp_id"] == exp_id and int(row["detected"]) == 1
        ]
        for local_index, global_index in enumerate(success_row_indices):
            frame_rows[global_index].update({
                "successful_index": local_index,
                "raw_frame_rms_px": float(raw_frame_rms[local_index]),
                "registered_frame_rms_px": float(
                    registered_frame_rms[local_index]
                ),
                "board_centroid_x_px": float(centroid_x[local_index]),
                "board_centroid_y_px": float(centroid_y[local_index]),
                "registration_tx_px": float(transform_tx[local_index]),
                "registration_ty_px": float(transform_ty[local_index]),
                "registration_rotation_deg": float(
                    transform_rotation[local_index]
                ),
                "registration_scale": float(transform_scale[local_index]),
                "registered_rms_outlier": int(outlier_mask[local_index]),
            })

        for corner_index in range(expected_corner_count):
            corner_row = corner_index // args.cols
            corner_col = corner_index % args.cols
            corner_rows.append({
                "exp_id": exp_id,
                "exposure_us": exposure_us,
                "corner_index": corner_index,
                "corner_row": corner_row,
                "corner_col": corner_col,
                "mean_u_px": float(raw_mean[corner_index, 0]),
                "mean_v_px": float(raw_mean[corner_index, 1]),
                "raw_sigma_u_px": float(raw_sigma_x[corner_index]),
                "raw_sigma_v_px": float(raw_sigma_y[corner_index]),
                "raw_sigma_r_px": float(raw_sigma_r[corner_index]),
                "registered_mean_u_px": float(
                    registered_mean[corner_index, 0]
                ),
                "registered_mean_v_px": float(
                    registered_mean[corner_index, 1]
                ),
                "registered_sigma_u_px": float(
                    registered_sigma_x[corner_index]
                ),
                "registered_sigma_v_px": float(
                    registered_sigma_y[corner_index]
                ),
                "registered_sigma_r_px": float(
                    registered_sigma_r[corner_index]
                ),
            })

        raw_median = finite_stat(raw_sigma_r, "median")
        raw_p95 = finite_stat(raw_sigma_r, "p95")
        registered_median = finite_stat(
            registered_sigma_r, "median"
        )
        registered_p95 = finite_stat(
            registered_sigma_r, "p95"
        )

        # 可选：简化三角测量等效值。
        equivalent_depth_median_mm = float("nan")
        equivalent_depth_p95_mm = float("nan")
        if args.focal_length_px is not None:
            distance_mm = (
                float(metadata_row["distance_mm"])
                if "distance_mm" in metadata.columns
                and pd.notna(metadata_row["distance_mm"])
                else float("nan")
            )
            baseline_mm = (
                float(metadata_row["baseline_mm"])
                if "baseline_mm" in metadata.columns
                and pd.notna(metadata_row["baseline_mm"])
                else float("nan")
            )
            if (
                np.isfinite(distance_mm)
                and np.isfinite(baseline_mm)
                and baseline_mm > 0
            ):
                sensitivity = (
                    distance_mm ** 2
                    / (args.focal_length_px * baseline_mm)
                )
                equivalent_depth_median_mm = (
                    sensitivity * registered_median
                )
                equivalent_depth_p95_mm = (
                    sensitivity * registered_p95
                )

        summary = {
            "exp_id": exp_id,
            "exposure_us": exposure_us,
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate": detection_rate,
            "sb_frame_count": sum(
                method == "SB" for method in successful_methods
            ),
            "traditional_frame_count": sum(
                method == "traditional+cornerSubPix"
                for method in successful_methods
            ),
            "raw_median_sigma_r_px": raw_median,
            "raw_mean_sigma_r_px": finite_stat(raw_sigma_r, "mean"),
            "raw_p95_sigma_r_px": raw_p95,
            "raw_max_sigma_r_px": finite_stat(raw_sigma_r, "max"),
            "registered_median_sigma_r_px": registered_median,
            "registered_mean_sigma_r_px": finite_stat(
                registered_sigma_r, "mean"
            ),
            "registered_p95_sigma_r_px": registered_p95,
            "registered_max_sigma_r_px": finite_stat(
                registered_sigma_r, "max"
            ),
            "raw_median_frame_rms_px": finite_stat(
                raw_frame_rms, "median"
            ),
            "registered_median_frame_rms_px": finite_stat(
                registered_frame_rms, "median"
            ),
            "registered_p95_frame_rms_px": finite_stat(
                registered_frame_rms, "p95"
            ),
            "registered_outlier_frame_count": int(np.sum(outlier_mask)),
            "global_translation_x_std_px": float(
                sample_std(transform_tx)
            ),
            "global_translation_y_std_px": float(
                sample_std(transform_ty)
            ),
            "global_rotation_std_deg": float(
                sample_std(transform_rotation)
            ),
            "global_scale_std_ppm": float(
                sample_std(transform_scale) * 1e6
            ),
            "centroid_x_drift_px_per_frame": centroid_x_slope,
            "centroid_y_drift_px_per_frame": centroid_y_slope,
            "centroid_x_total_drift_px": centroid_x_total,
            "centroid_y_total_drift_px": centroid_y_total,
            "equivalent_depth_median_mm": equivalent_depth_median_mm,
            "equivalent_depth_p95_mm": equivalent_depth_p95_mm,
            "interpretation": interpretation_label(
                detection_rate, registered_p95
            ),
        }
        summary_rows.append(summary)

        exposure_results[exposure_us] = {
            "exp_id": exp_id,
            "template": template,
            "registered_mean": registered_mean,
            "raw_mean": raw_mean,
            "registered_sigma_r": registered_sigma_r,
        }

        save_frame_rms_plot(
            np.arange(detected_frames),
            raw_frame_rms,
            registered_frame_rms,
            f"{exp_id} corner repeatability",
            output_root / "plots" / f"{exp_id}_frame_rms.png",
        )
        save_corner_sigma_map(
            registered_mean,
            registered_sigma_r,
            f"{exp_id} registered corner repeatability",
            output_root / "plots" / f"{exp_id}_corner_sigma_map.png",
        )

        print(
            f"检测率={detection_rate*100:.1f}% | "
            f"raw P95 σ={raw_p95:.4f}px | "
            f"registered P95 σ={registered_p95:.4f}px | "
            f"{summary['interpretation']}"
        )

    if not summary_rows:
        print("[错误] 没有得到可分析的曝光组。", file=sys.stderr)
        return 1

    # -----------------------------------------------------------------------
    # 跨曝光系统性角点偏移
    # -----------------------------------------------------------------------
    available_exposures = sorted(exposure_results.keys())
    if (
        args.reference_exposure_us is not None
        and args.reference_exposure_us in exposure_results
    ):
        reference_exposure = float(args.reference_exposure_us)
    else:
        reference_exposure = available_exposures[0]

    reference_grid = exposure_results[reference_exposure][
        "registered_mean"
    ]
    cross_exposure_rows: list[dict[str, Any]] = []

    summary_by_exposure = {
        float(row["exposure_us"]): row for row in summary_rows
    }

    for exposure_us in available_exposures:
        grid = exposure_results[exposure_us]["registered_mean"]
        transform = fit_similarity(grid, reference_grid)
        aligned_grid = apply_similarity(grid, transform)
        bias_vectors = aligned_grid - reference_grid
        bias_distance = np.sqrt(
            np.sum(bias_vectors ** 2, axis=1)
        )

        mean_bias = finite_stat(bias_distance, "mean")
        p95_bias = finite_stat(bias_distance, "p95")
        max_bias = finite_stat(bias_distance, "max")

        summary_by_exposure[exposure_us].update({
            "reference_exposure_us": reference_exposure,
            "cross_exposure_mean_bias_px": mean_bias,
            "cross_exposure_p95_bias_px": p95_bias,
            "cross_exposure_max_bias_px": max_bias,
        })

        for corner_index in range(expected_corner_count):
            cross_exposure_rows.append({
                "reference_exposure_us": reference_exposure,
                "exposure_us": exposure_us,
                "exp_id": exposure_results[exposure_us]["exp_id"],
                "corner_index": corner_index,
                "bias_u_px": float(bias_vectors[corner_index, 0]),
                "bias_v_px": float(bias_vectors[corner_index, 1]),
                "bias_r_px": float(bias_distance[corner_index]),
            })

    # -----------------------------------------------------------------------
    # 保存CSV
    # -----------------------------------------------------------------------
    frame_df = pd.DataFrame(frame_rows)
    corner_df = pd.DataFrame(corner_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("exposure_us")
    cross_exposure_df = pd.DataFrame(cross_exposure_rows)

    frame_df.to_csv(
        output_root / "frame_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    corner_df.to_csv(
        output_root / "corner_repeatability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        output_root / "exposure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cross_exposure_df.to_csv(
        output_root / "cross_exposure_bias.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------------
    # 汇总图
    # -----------------------------------------------------------------------
    exposure_values = summary_df["exposure_us"].to_numpy(dtype=float)
    save_line_plot(
        exposure_values,
        [
            (
                summary_df["raw_median_sigma_r_px"].to_numpy(dtype=float),
                "raw median σ",
            ),
            (
                summary_df["registered_median_sigma_r_px"].to_numpy(dtype=float),
                "registered median σ",
            ),
            (
                summary_df["registered_p95_sigma_r_px"].to_numpy(dtype=float),
                "registered P95 σ",
            ),
        ],
        "Exposure / μs",
        "Corner repeatability / px",
        "Corner repeatability versus exposure",
        output_root / "plots" / "repeatability_vs_exposure.png",
    )

    save_line_plot(
        exposure_values,
        [
            (
                summary_df["detection_rate"].to_numpy(dtype=float) * 100.0,
                "detection rate",
            ),
        ],
        "Exposure / μs",
        "Detection rate / %",
        "Chessboard detection rate versus exposure",
        output_root / "plots" / "detection_rate_vs_exposure.png",
    )

    if "cross_exposure_p95_bias_px" in summary_df.columns:
        save_line_plot(
            exposure_values,
            [
                (
                    summary_df["cross_exposure_p95_bias_px"].to_numpy(
                        dtype=float
                    ),
                    "P95 bias to reference exposure",
                ),
            ],
            "Exposure / μs",
            "Cross-exposure corner bias / px",
            f"Exposure-dependent corner bias (reference={reference_exposure:g} μs)",
            output_root / "plots" / "cross_exposure_bias.png",
        )

    # -----------------------------------------------------------------------
    # 自动生成报告
    # -----------------------------------------------------------------------
    best_row = summary_df.sort_values(
        by=[
            "registered_p95_sigma_r_px",
            "cross_exposure_p95_bias_px",
            "exposure_us",
        ],
        ascending=[True, True, True],
    ).iloc[0]

    report_path = output_root / "repeatability_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# 棋盘格角点重复性分析报告\n\n")
        handle.write("## 评价定义\n\n")
        handle.write(
            "- **raw σ**：未配准角点的跨帧标准差，包含机械振动与整体漂移。\n"
        )
        handle.write(
            "- **registered σ**：去除整板平移、旋转和统一尺度变化后的标准差，"
            "更接近角点检测的局部重复性底限。\n"
        )
        handle.write(
            "- 每个角点的二维重复性定义为："
            "`σr = sqrt(σu² + σv²)`。\n"
        )
        handle.write(
            "- **P95 σ**：30个角点中95%角点不超过的重复性数值，"
            "建议作为较保守的底限指标。\n\n"
        )

        handle.write("## 曝光汇总\n\n")
        handle.write(
            "|曝光/μs|检测率|raw中位σ/px|registered中位σ/px|"
            "registered P95σ/px|跨曝光P95偏移/px|评价|\n"
        )
        handle.write("|---:|---:|---:|---:|---:|---:|---|\n")
        for _, row in summary_df.iterrows():
            handle.write(
                f"|{row['exposure_us']:.0f}|"
                f"{row['detection_rate']*100:.1f}%|"
                f"{row['raw_median_sigma_r_px']:.5f}|"
                f"{row['registered_median_sigma_r_px']:.5f}|"
                f"{row['registered_p95_sigma_r_px']:.5f}|"
                f"{row['cross_exposure_p95_bias_px']:.5f}|"
                f"{row['interpretation']}|\n"
            )

        handle.write("\n## 自动判断\n\n")
        handle.write(
            f"- 当前重复性最优候选曝光："
            f"**{best_row['exposure_us']:.0f} μs**。\n"
        )
        handle.write(
            f"- 其registered中位σ为"
            f" **{best_row['registered_median_sigma_r_px']:.5f} px**，"
            f"P95 σ为"
            f" **{best_row['registered_p95_sigma_r_px']:.5f} px**。\n"
        )
        handle.write(
            "- 若raw明显大于registered，说明整体机械振动/漂移是主要来源；"
            "若两者接近，则局部角点定位噪声占主导。\n"
        )
        handle.write(
            "- 跨曝光偏移反映曝光变化是否导致角点位置产生系统性偏差。"
            "高曝光下若该值明显增大，应检查白格饱和和边缘扩张。\n"
        )

        if args.focal_length_px is not None:
            handle.write(
                "\n## 简化三角模型参考\n\n"
                "等效深度值使用 `δZ≈Z²/(fB)·δp` 计算，"
                "仅用于量级参考。角点重复性不等于激光中心重复性，"
                "不能直接作为最终三维精度结论。\n"
            )

    print("\n=== 分析完成 ===")
    print(summary_df[
        [
            "exposure_us",
            "detection_rate",
            "raw_median_sigma_r_px",
            "registered_median_sigma_r_px",
            "registered_p95_sigma_r_px",
            "cross_exposure_p95_bias_px",
            "interpretation",
        ]
    ].to_string(index=False))

    print(
        f"\n建议优先复核曝光：{best_row['exposure_us']:.0f} μs"
    )
    print(f"结果目录：{output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
