#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在冻结相机内参和畸变参数的条件下，对同一组棋盘格图像进行重投影误差评价。

核心原则
--------
1. 所有候选内参共用同一批已检测二维角点。
2. 不调用 cv2.calibrateCamera()，不优化 K 和 D。
3. 每张图只用 cv2.solvePnP() 求外参 rvec/tvec。
4. 使用 cv2.projectPoints() 将棋盘三维角点重投影回图像。
5. 输出逐点、逐图、总体误差，并支持多组内参公平对比。

适用场景
--------
- 在同一组独立测试图上比较 F4、F5.6、F8 等不同光圈下得到的内参。
- 比较固定/释放 k3、剔除/不剔除异常图等不同标定方案。
- 检查一组已有内参在当前光学状态和当前测试姿态上的解释能力。

依赖
----
numpy
opencv-python 或 opencv-contrib-python
PyYAML
matplotlib（可选，仅用于生成对比图）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


SCRIPT_VERSION = "1.0.0"
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
    ".png", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
    ".ras", ".sr", ".tif", ".tiff", ".webp",
}


@dataclass
class IntrinsicsModel:
    label: str
    source_path: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    declared_width: int | None = None
    declared_height: int | None = None


@dataclass
class DetectionRecord:
    image: str
    ok: bool
    reason: str
    detection_method: str
    image_width: int
    image_height: int


@dataclass
class ImageMetric:
    model: str
    image: str
    detection_method: str
    pnp_method: str
    point_count: int
    mean_euclidean_error_px: float
    median_euclidean_error_px: float
    p95_euclidean_error_px: float
    max_euclidean_error_px: float
    rmse_px: float
    x_bias_px: float
    y_bias_px: float
    x_rmse_px: float
    y_rmse_px: float
    rvec_x: float
    rvec_y: float
    rvec_z: float
    tvec_x_mm: float
    tvec_y_mm: float
    tvec_z_mm: float


@dataclass
class ModelSummary:
    label: str
    source_path: str
    image_count: int
    point_count: int
    mean_euclidean_error_px: float
    overall_rmse_px: float
    mean_per_image_rmse_px: float
    median_per_image_rmse_px: float
    p95_per_image_rmse_px: float
    max_per_image_rmse_px: float
    x_bias_px: float
    y_bias_px: float
    x_rmse_px: float
    y_rmse_px: float
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]


def natural_key(path: Path) -> list[tuple[int, Any]]:
    parts = re.split(r"(\d+)", path.name)
    return [(1, int(part)) if part.isdigit() else (0, part.casefold()) for part in parts]


def safe_label(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text.strip())
    text = re.sub(r"\s+", "_", text)
    return text or "intrinsics"


def read_image_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """兼容 Windows 中文路径的图像读取。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def write_image_unicode(path: Path, image: np.ndarray) -> bool:
    """兼容 Windows 中文路径的图像写入。"""
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return True


def to_gray8(image: np.ndarray) -> np.ndarray:
    """将 Mono8、Mono12/16、浮点图或彩色图统一转换为8位灰度检测图。"""
    if image.ndim == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if gray.dtype == np.uint8:
        return gray

    arr = gray.astype(np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("图像不含有限灰度值")

    valid = arr[finite]
    low = float(np.percentile(valid, 0.5))
    high = float(np.percentile(valid, 99.5))
    if high <= low + 1e-12:
        low = float(valid.min())
        high = float(valid.max())
    if high <= low + 1e-12:
        return np.zeros(gray.shape, np.uint8)

    arr = np.clip((arr - low) * 255.0 / (high - low), 0.0, 255.0)
    return arr.astype(np.uint8)


class OpenCVYamlLoader(yaml.SafeLoader):
    pass


def _opencv_matrix_constructor(loader: yaml.Loader, node: yaml.Node) -> Any:
    return loader.construct_mapping(node, deep=True)


OpenCVYamlLoader.add_constructor(
    "tag:yaml.org,2002:opencv-matrix",
    _opencv_matrix_constructor,
)


def _prepare_yaml_text(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%YAML:"):
            continue
        if stripped == "---":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _node_to_array(node: Any, name: str) -> np.ndarray:
    if node is None:
        raise KeyError(f"YAML中缺少 {name}")

    if isinstance(node, dict):
        if "data" in node:
            data = np.asarray(node["data"], dtype=np.float64)
            rows = node.get("rows")
            cols = node.get("cols")
            if rows is not None and cols is not None:
                data = data.reshape(int(rows), int(cols))
            return data
        if all(key in node for key in ("fx", "fy", "cx", "cy")):
            return np.array(
                [
                    [float(node["fx"]), 0.0, float(node["cx"])],
                    [0.0, float(node["fy"]), float(node["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

    return np.asarray(node, dtype=np.float64)


def _first_existing(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def load_intrinsics_yaml(label: str, path: Path) -> IntrinsicsModel:
    if not path.is_file():
        raise FileNotFoundError(f"内参文件不存在：{path}")

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    data: dict[str, Any] | None = None

    # 先尝试 PyYAML，支持普通YAML和 !!opencv-matrix。
    try:
        parsed = yaml.load(_prepare_yaml_text(text), Loader=OpenCVYamlLoader)
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        data = None

    # 若普通解析失败，再尝试 OpenCV FileStorage。
    if data is None:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError(f"无法解析内参YAML：{path}")
        try:
            K_node = fs.getNode("camera_matrix")
            if K_node.empty():
                K_node = fs.getNode("K")
            D_node = fs.getNode("dist_coeffs")
            if D_node.empty():
                D_node = fs.getNode("distortion_coefficients")
            if D_node.empty():
                D_node = fs.getNode("D")
            K = K_node.mat()
            D = D_node.mat()
            width_node = fs.getNode("image_width")
            height_node = fs.getNode("image_height")
            width = int(width_node.real()) if not width_node.empty() else None
            height = int(height_node.real()) if not height_node.empty() else None
        finally:
            fs.release()

        if K is None or D is None:
            raise ValueError(f"OpenCV FileStorage未找到camera_matrix/dist_coeffs：{path}")
        return validate_intrinsics(label, path, K, D, width, height)

    K_node = _first_existing(
        data,
        ("camera_matrix", "K", "intrinsic_matrix", "cameraMatrix"),
    )
    D_node = _first_existing(
        data,
        ("dist_coeffs", "distortion_coefficients", "D", "distCoeffs"),
    )

    # 支持直接以 fx/fy/cx/cy 保存。
    if K_node is None and all(key in data for key in ("fx", "fy", "cx", "cy")):
        K_node = {key: data[key] for key in ("fx", "fy", "cx", "cy")}

    K = _node_to_array(K_node, "camera_matrix")
    D = _node_to_array(D_node, "dist_coeffs")

    width_value = _first_existing(data, ("image_width", "width"))
    height_value = _first_existing(data, ("image_height", "height"))
    width = int(width_value) if width_value is not None else None
    height = int(height_value) if height_value is not None else None

    return validate_intrinsics(label, path, K, D, width, height)


def validate_intrinsics(
    label: str,
    path: Path,
    K: np.ndarray,
    D: np.ndarray,
    width: int | None,
    height: int | None,
) -> IntrinsicsModel:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)

    if D.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"{path} 的畸变参数数量为 {D.size}，应为4/5/8/12/14"
        )
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
        raise ValueError(f"{path} 含非有限内参")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[2, 2]) < 1e-12:
        raise ValueError(f"{path} 的相机矩阵不合法")

    return IntrinsicsModel(
        label=label,
        source_path=str(path.resolve()),
        camera_matrix=K,
        dist_coeffs=D,
        declared_width=width,
        declared_height=height,
    )


def parse_intrinsics_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"内参参数格式错误：{spec}")
        return label, Path(raw_path)

    path = Path(spec)
    return path.stem, path


def find_images(directory: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
    paths = [
        p for p in iterator
        if p.is_file() and p.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return sorted(paths, key=natural_key)


def build_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    obj = np.zeros((cols * rows, 3), np.float64)
    obj[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
        * float(square_size_mm)
    )
    return obj


def detect_chessboard(
    gray8: np.ndarray,
    pattern_size: tuple[int, int],
    sb_exhaustive: bool,
    sb_accuracy: bool,
) -> tuple[bool, np.ndarray | None, str]:
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    if sb_exhaustive and hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
        sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
    if sb_accuracy and hasattr(cv2, "CALIB_CB_ACCURACY"):
        sb_flags |= cv2.CALIB_CB_ACCURACY

    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray8, pattern_size, sb_flags)
        if ok and corners is not None:
            return True, corners.reshape(-1, 2).astype(np.float64), "SB"

    legacy_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray8, pattern_size, legacy_flags)
    if not ok or corners is None:
        return False, None, "failed"

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        80,
        1e-4,
    )
    refined = cv2.cornerSubPix(
        gray8,
        corners.astype(np.float32),
        (11, 11),
        (-1, -1),
        criteria,
    )
    return (
        True,
        refined.reshape(-1, 2).astype(np.float64),
        "legacy+cornerSubPix",
    )


def solve_pose_iterative(
    obj: np.ndarray,
    img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    refine_lm: bool,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, str]:
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return False, None, None, "ITERATIVE"

    if refine_lm and hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
        except cv2.error:
            pass
    return True, rvec, tvec, "ITERATIVE"


def solve_pose_ippe(
    obj: np.ndarray,
    img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    refine_lm: bool,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, str]:
    if not hasattr(cv2, "SOLVEPNP_IPPE"):
        return solve_pose_iterative(obj, img, K, D, refine_lm)

    try:
        result = cv2.solvePnPGeneric(
            obj,
            img,
            K,
            D,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return solve_pose_iterative(obj, img, K, D, refine_lm)

    if not result or not bool(result[0]):
        return False, None, None, "IPPE"

    rvecs = result[1]
    tvecs = result[2]
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for rvec, tvec in zip(rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        residual = img - projected.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

        R, _ = cv2.Rodrigues(rvec)
        camera_points = (R @ obj.T + tvec.reshape(3, 1)).T
        positive_fraction = float(np.mean(camera_points[:, 2] > 0))
        if positive_fraction < 1.0:
            continue

        if best is None or rmse < best[0]:
            best = (rmse, rvec, tvec)

    if best is None:
        return False, None, None, "IPPE"

    _, rvec, tvec = best
    if refine_lm and hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
        except cv2.error:
            pass

    return True, rvec, tvec, "IPPE"


def solve_pose(
    method: str,
    obj: np.ndarray,
    img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    refine_lm: bool,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, str]:
    if method == "ippe":
        return solve_pose_ippe(obj, img, K, D, refine_lm)
    return solve_pose_iterative(obj, img, K, D, refine_lm)


def calculate_image_metric(
    model_label: str,
    image_name: str,
    detection_method: str,
    pnp_method: str,
    detected: np.ndarray,
    projected: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[ImageMetric, np.ndarray]:
    residual = detected - projected
    distances = np.linalg.norm(residual, axis=1)

    metric = ImageMetric(
        model=model_label,
        image=image_name,
        detection_method=detection_method,
        pnp_method=pnp_method,
        point_count=int(len(distances)),
        mean_euclidean_error_px=float(np.mean(distances)),
        median_euclidean_error_px=float(np.median(distances)),
        p95_euclidean_error_px=float(np.percentile(distances, 95)),
        max_euclidean_error_px=float(np.max(distances)),
        rmse_px=float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        x_bias_px=float(np.mean(residual[:, 0])),
        y_bias_px=float(np.mean(residual[:, 1])),
        x_rmse_px=float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        y_rmse_px=float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        rvec_x=float(rvec.reshape(-1)[0]),
        rvec_y=float(rvec.reshape(-1)[1]),
        rvec_z=float(rvec.reshape(-1)[2]),
        tvec_x_mm=float(tvec.reshape(-1)[0]),
        tvec_y_mm=float(tvec.reshape(-1)[1]),
        tvec_z_mm=float(tvec.reshape(-1)[2]),
    )
    return metric, residual


def calculate_model_summary(
    model: IntrinsicsModel,
    metrics: list[ImageMetric],
    residuals: list[np.ndarray],
) -> ModelSummary:
    if not metrics or not residuals:
        raise ValueError(f"{model.label}没有可汇总的结果")

    all_residual = np.concatenate(residuals, axis=0)
    all_distances = np.linalg.norm(all_residual, axis=1)
    per_image_rmse = np.asarray([m.rmse_px for m in metrics], dtype=np.float64)

    return ModelSummary(
        label=model.label,
        source_path=model.source_path,
        image_count=len(metrics),
        point_count=int(len(all_distances)),
        mean_euclidean_error_px=float(np.mean(all_distances)),
        overall_rmse_px=float(
            np.sqrt(np.mean(np.sum(all_residual * all_residual, axis=1)))
        ),
        mean_per_image_rmse_px=float(np.mean(per_image_rmse)),
        median_per_image_rmse_px=float(np.median(per_image_rmse)),
        p95_per_image_rmse_px=float(np.percentile(per_image_rmse, 95)),
        max_per_image_rmse_px=float(np.max(per_image_rmse)),
        x_bias_px=float(np.mean(all_residual[:, 0])),
        y_bias_px=float(np.mean(all_residual[:, 1])),
        x_rmse_px=float(np.sqrt(np.mean(all_residual[:, 0] ** 2))),
        y_rmse_px=float(np.sqrt(np.mean(all_residual[:, 1] ** 2))),
        camera_matrix=model.camera_matrix.tolist(),
        dist_coeffs=model.dist_coeffs.reshape(-1).tolist(),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_residual_overlay(
    image_path: Path,
    detected: np.ndarray,
    projected: np.ndarray,
    output_path: Path,
    residual_scale: float,
) -> bool:
    image = read_image_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return False

    if image.ndim == 2:
        display = cv2.cvtColor(to_gray8(image), cv2.COLOR_GRAY2BGR)
    elif image.dtype != np.uint8:
        display = cv2.cvtColor(to_gray8(image), cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        display = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        display = image.copy()

    for observed, predicted in zip(detected, projected):
        ox, oy = observed
        px, py = predicted
        end_x = px + residual_scale * (ox - px)
        end_y = py + residual_scale * (oy - py)

        cv2.circle(display, (int(round(px)), int(round(py))), 5, (0, 0, 255), 1)
        cv2.drawMarker(
            display,
            (int(round(ox)), int(round(oy))),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )
        cv2.line(
            display,
            (int(round(px)), int(round(py))),
            (int(round(end_x)), int(round(end_y))),
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return write_image_unicode(output_path, display)


def make_plots(
    output_dir: Path,
    summaries: list[ModelSummary],
    per_image_table: dict[str, dict[str, float]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    generated: list[str] = []

    labels = [s.label for s in summaries]
    overall = [s.overall_rmse_px for s in summaries]

    plt.figure(figsize=(max(7, len(labels) * 1.5), 4.8))
    bars = plt.bar(labels, overall)
    plt.ylabel("Overall reprojection RMSE (px)")
    plt.title("Frozen-intrinsics comparison")
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, overall):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    path = output_dir / "summary_overall_rmse.png"
    plt.savefig(path, dpi=180)
    plt.close()
    generated.append(str(path))

    images = sorted(per_image_table)
    plt.figure(figsize=(max(9, len(images) * 0.65), 5.2))
    for summary in summaries:
        values = [
            per_image_table[name].get(summary.label, np.nan)
            for name in images
        ]
        plt.plot(range(len(images)), values, marker="o", label=summary.label)
    plt.ylabel("Per-image RMSE (px)")
    plt.xlabel("Test image")
    plt.title("Per-image frozen-intrinsics RMSE")
    plt.xticks(range(len(images)), images, rotation=70, ha="right", fontsize=8)
    plt.legend()
    plt.tight_layout()
    path = output_dir / "per_image_rmse_comparison.png"
    plt.savefig(path, dpi=180)
    plt.close()
    generated.append(str(path))

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="冻结多组相机内参，在同一棋盘格测试集上公平计算重投影误差。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-dir", type=Path, required=True, help="测试图像目录")
    parser.add_argument("--pattern", default="*.tif", help="测试图像通配符")
    parser.add_argument("--recursive", action="store_true", help="递归搜索子目录")
    parser.add_argument(
        "--intrinsics",
        action="append",
        required=True,
        metavar="LABEL=YAML",
        help="待评价内参，可重复指定；也可只写YAML路径并自动使用文件名作为标签",
    )
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--pattern-cols", type=int, default=6, help="棋盘内角点列数")
    parser.add_argument("--pattern-rows", type=int, default=5, help="棋盘内角点行数")
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=30.0,
        help="棋盘单格物理尺寸，单位mm",
    )
    parser.add_argument(
        "--pnp-method",
        choices=("iterative", "ippe"),
        default="iterative",
        help="每张图的外参求解方法",
    )
    parser.add_argument(
        "--no-refine-lm",
        action="store_true",
        help="不再用solvePnPRefineLM细化外参",
    )
    parser.add_argument(
        "--sb-exhaustive",
        action="store_true",
        help="启用findChessboardCornersSB穷举模式，坏图上可能较慢",
    )
    parser.add_argument(
        "--sb-accuracy",
        action="store_true",
        help="启用findChessboardCornersSB高精度模式",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许部分图检测或PnP失败；公平比较时通常不建议开启",
    )
    parser.add_argument(
        "--save-worst",
        type=int,
        default=3,
        help="每组内参保存逐图RMSE最差的N张残差图；0表示不保存",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=50.0,
        help="残差可视化向量放大倍数",
    )
    parser.add_argument("--no-plots", action="store_true", help="不生成matplotlib对比图")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.pattern_cols <= 1 or args.pattern_rows <= 1:
        print("错误：棋盘内角点行列数必须大于1。", file=sys.stderr)
        return 2
    if args.square_size_mm <= 0:
        print("错误：square-size-mm必须大于0。", file=sys.stderr)
        return 2
    if not args.test_dir.is_dir():
        print(f"错误：测试目录不存在：{args.test_dir}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    # 加载并检查候选内参标签。
    models: list[IntrinsicsModel] = []
    labels_seen: set[str] = set()
    try:
        for spec in args.intrinsics:
            label, yaml_path = parse_intrinsics_spec(spec)
            if label in labels_seen:
                raise ValueError(f"内参标签重复：{label}")
            labels_seen.add(label)
            models.append(load_intrinsics_yaml(label, yaml_path))
    except Exception as exc:
        print(f"错误：读取内参失败：{exc}", file=sys.stderr)
        return 2

    images = find_images(args.test_dir, args.pattern, args.recursive)
    if not images:
        print(
            f"错误：在 {args.test_dir} 中未找到匹配 {args.pattern} 的图像。",
            file=sys.stderr,
        )
        return 2

    print(f"脚本版本：{SCRIPT_VERSION}")
    print(f"测试目录：{args.test_dir.resolve()}")
    print(f"图像数量：{len(images)}")
    print("候选内参：")
    for model in models:
        print(f"  - {model.label}: {model.source_path}")
    print()

    pattern_size = (args.pattern_cols, args.pattern_rows)
    object_points = build_object_points(
        args.pattern_cols,
        args.pattern_rows,
        args.square_size_mm,
    )

    detection_records: list[DetectionRecord] = []
    detected_corners: dict[str, np.ndarray] = {}
    image_paths: dict[str, Path] = {}
    image_size: tuple[int, int] | None = None
    detection_methods: dict[str, str] = {}

    print("【1】统一检测测试图角点")
    for path in images:
        image = read_image_unicode(path, cv2.IMREAD_UNCHANGED)
        relative_name = path.relative_to(args.test_dir).as_posix()
        image_paths[relative_name] = path

        if image is None:
            detection_records.append(
                DetectionRecord(relative_name, False, "读取失败", "", 0, 0)
            )
            print(f"  × {relative_name}: 读取失败")
            continue

        height, width = image.shape[:2]
        current_size = (width, height)
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            reason = f"分辨率不一致：{current_size}，期望{image_size}"
            detection_records.append(
                DetectionRecord(relative_name, False, reason, "", width, height)
            )
            print(f"  × {relative_name}: {reason}")
            continue

        try:
            gray8 = to_gray8(image)
            ok, corners, method = detect_chessboard(
                gray8,
                pattern_size,
                args.sb_exhaustive,
                args.sb_accuracy,
            )
        except Exception as exc:
            ok, corners, method = False, None, "failed"
            reason = f"处理异常：{exc}"
            detection_records.append(
                DetectionRecord(relative_name, False, reason, method, width, height)
            )
            print(f"  × {relative_name}: {reason}")
            continue

        if not ok or corners is None or len(corners) != len(object_points):
            reason = "未检测到完整棋盘"
            detection_records.append(
                DetectionRecord(relative_name, False, reason, method, width, height)
            )
            print(f"  × {relative_name}: {reason}")
            continue

        detected_corners[relative_name] = corners
        detection_methods[relative_name] = method
        detection_records.append(
            DetectionRecord(relative_name, True, "", method, width, height)
        )
        print(f"  ✓ {relative_name}: {method}")

    detection_csv = args.output / "corner_detection.csv"
    write_csv(
        detection_csv,
        [asdict(item) for item in detection_records],
        [
            "image", "ok", "reason", "detection_method",
            "image_width", "image_height",
        ],
    )

    valid_names = sorted(detected_corners, key=lambda name: natural_key(Path(name)))
    failed_detection_count = len(images) - len(valid_names)
    if failed_detection_count and not args.allow_partial:
        print(
            f"\n错误：有 {failed_detection_count} 张图角点检测失败。"
            "公平比较默认要求全部成功；可检查corner_detection.csv，"
            "或明确使用 --allow-partial。",
            file=sys.stderr,
        )
        return 3
    if not valid_names or image_size is None:
        print("错误：没有有效测试图。", file=sys.stderr)
        return 3

    # 保存同一批角点，便于审计。
    corners_stack = np.stack([detected_corners[name] for name in valid_names], axis=0)
    np.savez_compressed(
        args.output / "detected_corners_shared.npz",
        image_names=np.asarray(valid_names),
        corners=corners_stack,
    )

    # 检查内参声明分辨率。
    for model in models:
        if (
            model.declared_width is not None
            and model.declared_height is not None
            and (model.declared_width, model.declared_height) != image_size
        ):
            print(
                f"错误：{model.label}声明分辨率"
                f"{(model.declared_width, model.declared_height)}，"
                f"测试图分辨率为{image_size}。脚本不会自动缩放内参。",
                file=sys.stderr,
            )
            return 3

    print("\n【2】冻结K、D，仅求每张图外参并重投影")
    all_model_metrics: dict[str, list[ImageMetric]] = {}
    all_model_residuals: dict[str, list[np.ndarray]] = {}
    all_projected: dict[str, dict[str, np.ndarray]] = {}
    all_summaries: list[ModelSummary] = []
    pose_failures: list[dict[str, str]] = []

    for model in models:
        print(f"\n模型：{model.label}")
        model_metrics: list[ImageMetric] = []
        model_residuals: list[np.ndarray] = []
        projected_by_image: dict[str, np.ndarray] = {}

        for name in valid_names:
            corners = detected_corners[name]
            ok, rvec, tvec, used_method = solve_pose(
                args.pnp_method,
                object_points,
                corners,
                model.camera_matrix,
                model.dist_coeffs,
                refine_lm=not args.no_refine_lm,
            )
            if not ok or rvec is None or tvec is None:
                pose_failures.append({"model": model.label, "image": name})
                print(f"  × {name}: solvePnP失败")
                continue

            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                model.camera_matrix,
                model.dist_coeffs,
            )
            projected = projected.reshape(-1, 2).astype(np.float64)
            metric, residual = calculate_image_metric(
                model.label,
                name,
                detection_methods[name],
                used_method,
                corners,
                projected,
                rvec,
                tvec,
            )
            model_metrics.append(metric)
            model_residuals.append(residual)
            projected_by_image[name] = projected
            print(f"  ✓ {name}: RMSE={metric.rmse_px:.6f} px")

        if len(model_metrics) != len(valid_names) and not args.allow_partial:
            print(
                f"\n错误：{model.label}有外参求解失败图，"
                "无法与其他模型进行完全公平比较。",
                file=sys.stderr,
            )
            return 4
        if not model_metrics:
            print(f"错误：{model.label}无有效评价结果。", file=sys.stderr)
            return 4

        summary = calculate_model_summary(model, model_metrics, model_residuals)
        all_model_metrics[model.label] = model_metrics
        all_model_residuals[model.label] = model_residuals
        all_projected[model.label] = projected_by_image
        all_summaries.append(summary)

        model_dir = args.output / safe_label(model.label)
        model_dir.mkdir(parents=True, exist_ok=True)

        metric_fields = list(asdict(model_metrics[0]).keys())
        write_csv(
            model_dir / "per_image_metrics.csv",
            [asdict(item) for item in model_metrics],
            metric_fields,
        )
        (model_dir / "summary.json").write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.save_worst > 0:
            worst = sorted(
                model_metrics,
                key=lambda item: item.rmse_px,
                reverse=True,
            )[: args.save_worst]
            residual_dir = model_dir / "worst_residual_overlays"
            for rank, metric in enumerate(worst, start=1):
                name = metric.image
                detected = detected_corners[name]
                projected = projected_by_image[name]
                output_name = (
                    f"{rank:02d}_rmse_{metric.rmse_px:.4f}_"
                    f"{safe_label(Path(name).stem)}.png"
                )
                draw_residual_overlay(
                    image_paths[name],
                    detected,
                    projected,
                    residual_dir / output_name,
                    args.residual_scale,
                )

        print(
            f"  总体RMSE={summary.overall_rmse_px:.6f} px，"
            f"平均欧氏误差={summary.mean_euclidean_error_px:.6f} px"
        )

    if pose_failures:
        write_csv(
            args.output / "pose_failures.csv",
            pose_failures,
            ["model", "image"],
        )

    print("\n【3】生成跨模型比较结果")
    summary_rows = [asdict(summary) for summary in all_summaries]
    summary_fields = [
        "label", "source_path", "image_count", "point_count",
        "mean_euclidean_error_px", "overall_rmse_px",
        "mean_per_image_rmse_px", "median_per_image_rmse_px",
        "p95_per_image_rmse_px", "max_per_image_rmse_px",
        "x_bias_px", "y_bias_px", "x_rmse_px", "y_rmse_px",
        "camera_matrix", "dist_coeffs",
    ]
    write_csv(
        args.output / "comparison_summary.csv",
        summary_rows,
        summary_fields,
    )
    (args.output / "comparison_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 宽表：每行一张图，每个模型两列。
    per_image_map: dict[str, dict[str, ImageMetric]] = {}
    for label, metrics in all_model_metrics.items():
        for metric in metrics:
            per_image_map.setdefault(metric.image, {})[label] = metric

    comparison_rows: list[dict[str, Any]] = []
    for name in sorted(per_image_map, key=lambda item: natural_key(Path(item))):
        row: dict[str, Any] = {"image": name}
        for model in models:
            metric = per_image_map[name].get(model.label)
            row[f"{model.label}__rmse_px"] = (
                metric.rmse_px if metric is not None else ""
            )
            row[f"{model.label}__mean_euclidean_px"] = (
                metric.mean_euclidean_error_px if metric is not None else ""
            )
            row[f"{model.label}__p95_euclidean_px"] = (
                metric.p95_euclidean_error_px if metric is not None else ""
            )
            row[f"{model.label}__max_euclidean_px"] = (
                metric.max_euclidean_error_px if metric is not None else ""
            )
        comparison_rows.append(row)

    comparison_fields = ["image"]
    for model in models:
        comparison_fields.extend(
            [
                f"{model.label}__rmse_px",
                f"{model.label}__mean_euclidean_px",
                f"{model.label}__p95_euclidean_px",
                f"{model.label}__max_euclidean_px",
            ]
        )
    write_csv(
        args.output / "per_image_comparison.csv",
        comparison_rows,
        comparison_fields,
    )

    plot_input = {
        name: {
            label: metric.rmse_px
            for label, metric in model_map.items()
        }
        for name, model_map in per_image_map.items()
    }
    generated_plots: list[str] = []
    if not args.no_plots:
        generated_plots = make_plots(
            args.output,
            all_summaries,
            plot_input,
        )

    run_metadata = {
        "script_version": SCRIPT_VERSION,
        "test_dir": str(args.test_dir.resolve()),
        "pattern": args.pattern,
        "recursive": bool(args.recursive),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "board": {
            "pattern_cols": args.pattern_cols,
            "pattern_rows": args.pattern_rows,
            "square_size_mm": args.square_size_mm,
            "points_per_image": len(object_points),
        },
        "pnp_method": args.pnp_method,
        "refine_lm": not args.no_refine_lm,
        "detected_image_count": len(valid_names),
        "detection_failure_count": failed_detection_count,
        "models": [
            {
                "label": model.label,
                "source_path": model.source_path,
            }
            for model in models
        ],
        "outputs": {
            "corner_detection_csv": str(detection_csv),
            "shared_corners_npz": str(
                args.output / "detected_corners_shared.npz"
            ),
            "comparison_summary_csv": str(
                args.output / "comparison_summary.csv"
            ),
            "comparison_summary_json": str(
                args.output / "comparison_summary.json"
            ),
            "per_image_comparison_csv": str(
                args.output / "per_image_comparison.csv"
            ),
            "plots": generated_plots,
        },
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ranked = sorted(all_summaries, key=lambda item: item.overall_rmse_px)
    print("\n【4】总体RMSE排名")
    for rank, summary in enumerate(ranked, start=1):
        print(
            f"  {rank}. {summary.label}: "
            f"{summary.overall_rmse_px:.6f} px "
            f"(P95逐图={summary.p95_per_image_rmse_px:.6f} px)"
        )

    print("\n输出目录：", args.output.resolve())
    print("关键文件：")
    print("  comparison_summary.csv")
    print("  per_image_comparison.csv")
    print("  detected_corners_shared.npz")
    print("  各模型目录/per_image_metrics.csv")
    print("  各模型目录/summary.json")
    if generated_plots:
        print("  summary_overall_rmse.png")
        print("  per_image_rmse_comparison.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
