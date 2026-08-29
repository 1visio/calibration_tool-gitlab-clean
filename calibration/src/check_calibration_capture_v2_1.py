#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""棋盘格标定采集预检 V2。

用途：
1. 在正式相机标定前检查拟合集的姿态、距离、画面覆盖、清晰度和数量；
2. 检查独立验证集逐图是否清晰、曝光合理、可稳定检测；
3. 输出控制台报告、逐图 CSV、汇总 JSON 和 PNG 看板；
4. 使用安全缓存，补拍后仅处理新增或发生变化的图像。

示例：
    python check_calibration_capture_v2.py --dir ./fit --pattern "chess *.tif" \
        --pattern-cols 6 --pattern-rows 5 --square-size-mm 30 \
        --reference-yaml camera_intrinsics.yaml --mode fit

    python check_calibration_capture_v2.py --dir ./test --pattern "chess *.tif" \
        --pattern-cols 6 --pattern-rows 5 --square-size-mm 30 \
        --reference-yaml camera_intrinsics.yaml --mode validation
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
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

SCRIPT_VERSION = "2.1.0"
CACHE_VERSION = 2
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SECTORS = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]
ZONE_NAMES = [["左上", "上中", "右上"], ["左中", "正中", "右中"], ["左下", "下中", "右下"]]
GAUSS_EDGE = math.sqrt(2.0 * math.pi)
BLUR_FLOOR_PX = 0.77


@dataclass
class Shot:
    name: str
    ok: bool
    reason: str = ""
    image_width: int = 0
    image_height: int = 0
    detection_method: str = ""
    tilt_deg: float = math.nan
    tilt_azimuth_deg: float = math.nan
    inplane_rotation_deg: float = math.nan
    depth_mm: float = math.nan
    tvec_x_mm: float = math.nan
    tvec_y_mm: float = math.nan
    blur_sigma_median: float = math.nan
    blur_sigma_p95: float = math.nan
    clipped_fraction: float = math.nan
    board_area_fraction: float = math.nan
    centre_u: float = math.nan
    centre_v: float = math.nan
    radius_max: float = math.nan
    reprojection_rmse_px: float = math.nan
    reprojection_max_px: float = math.nan
    duplicate_of: str = ""


@dataclass
class CheckItem:
    name: str
    value: float
    limit: float
    mode: str
    unit: str
    status: str
    decisive: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# 通用工具

def natural_key(path: Path, root: Optional[Path] = None) -> list[tuple[int, Any]]:
    try:
        text = path.relative_to(root).as_posix() if root is not None else path.as_posix()
    except ValueError:
        text = path.as_posix()
    return [
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", text)
    ]


def relative_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def safe_nanmedian(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(np.median(arr)) if arr.size else math.nan


def safe_nanmax(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(np.max(arr)) if arr.size else math.nan


def safe_nanmin(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(np.min(arr)) if arr.size else math.nan


def circular_distance_deg(a: float, b: float, period: float = 360.0) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    diff = abs(a - b) % period
    return min(diff, period - diff)


def format_value(value: float, unit: str = "", digits: int = 3) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}{unit}"


def verdict_mark(status: str) -> str:
    return {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "INFO": "·"}.get(status, "?")


def grade(value: float, limit: float, mode: str) -> str:
    if not math.isfinite(value):
        return "WARN"
    if mode == "min":
        return "PASS" if value >= limit else ("WARN" if value >= 0.7 * limit else "FAIL")
    if mode == "max":
        return "PASS" if value <= limit else ("WARN" if value <= 1.5 * limit else "FAIL")
    raise ValueError(f"未知门槛模式：{mode}")


# ---------------------------------------------------------------------------
# 图像、角点和清晰度

def read_image_unchanged(path: Path) -> Optional[np.ndarray]:
    """兼容中文路径，保留原始通道和位深。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error):
        return None


def to_gray8(image: np.ndarray) -> np.ndarray:
    """把灰度/彩色、8/16 位图统一转换为角点检测使用的 uint8 灰度图。"""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"不支持的图像形状：{image.shape}")

    if gray.dtype == np.uint8:
        return gray

    grayf = gray.astype(np.float32)
    finite = grayf[np.isfinite(grayf)]
    if finite.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)

    # 采用稳健百分位拉伸，避免 12/16 位图仅占低灰度范围时直接右移导致过暗。
    lo, hi = np.percentile(finite, [0.1, 99.9])
    if hi <= lo + 1e-6:
        return np.zeros(gray.shape, dtype=np.uint8)
    scaled = np.clip((grayf - lo) * (255.0 / (hi - lo)), 0, 255)
    return scaled.astype(np.uint8)


def detect(
    gray8: np.ndarray,
    pattern_size: tuple[int, int],
    exhaustive: bool = False,
    high_accuracy: bool = False,
) -> tuple[Optional[np.ndarray], str]:
    """检测完整棋盘内角点，优先使用 SB，失败后回退传统检测器。"""
    # 低纹理图像无需进入耗时的棋盘搜索。
    if float(np.std(gray8)) < 4.0:
        return None, ""
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if high_accuracy:
            flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
        if exhaustive:
            flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        try:
            found, corners = cv2.findChessboardCornersSB(gray8, pattern_size, flags=flags)
            if found and corners is not None:
                return corners.reshape(-1, 2).astype(np.float64), "findChessboardCornersSB"
        except cv2.error:
            pass

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    try:
        found, corners = cv2.findChessboardCorners(gray8, pattern_size, flags)
    except cv2.error:
        return None, ""
    if not found or corners is None:
        return None, ""

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    try:
        refined = cv2.cornerSubPix(
            gray8,
            corners.astype(np.float32),
            (11, 11),
            (-1, -1),
            criteria,
        )
    except cv2.error:
        return None, ""
    return refined.reshape(-1, 2).astype(np.float64), "findChessboardCorners+cornerSubPix"


def blur_sigma_at(
    gray32: np.ndarray,
    corners: np.ndarray,
    half: int = 15,
    floor: float = BLUR_FLOOR_PX,
) -> tuple[float, float, float]:
    """在每个角点邻域用最大梯度法估计等效高斯模糊 σ（像素）。"""
    h, w = gray32.shape
    sigmas: list[float] = []
    clipped_sum = 0.0
    valid_patches = 0

    for u, v in corners:
        x, y = int(round(float(u))), int(round(float(v)))
        if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
            continue
        patch = gray32[y - half : y + half + 1, x - half : x + half + 1]
        lo, hi = np.percentile(patch, [2, 98])
        contrast = float(hi - lo)
        if contrast < 12.0:
            continue

        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        grad = np.hypot(gx, gy)
        peak = float(np.percentile(grad, 99.0))
        if peak <= 1e-6:
            continue

        raw = contrast / (peak * GAUSS_EDGE)
        sigmas.append(math.sqrt(max(raw * raw - floor * floor, 0.0)))
        clipped_sum += float(np.mean((patch <= 1) | (patch >= 254)))
        valid_patches += 1

    if not sigmas:
        return math.nan, math.nan, math.nan
    return (
        float(np.median(sigmas)),
        float(np.percentile(sigmas, 95)),
        clipped_sum / valid_patches,
    )


def inplane_rotation(corners: np.ndarray, pattern_cols: int) -> float:
    """棋盘第一行在图像中的方向，按 180° 周期归一化到 [0, 180)。"""
    if len(corners) < pattern_cols:
        return math.nan
    dx, dy = corners[pattern_cols - 1] - corners[0]
    return math.degrees(math.atan2(float(dy), float(dx))) % 180.0


# ---------------------------------------------------------------------------
# YAML / 内参读取

def _matrix_from_mapping(value: Any, shape: Optional[tuple[int, ...]] = None) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("data", value.get("values"))
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    if shape is not None:
        try:
            arr = arr.reshape(shape)
        except ValueError:
            return None
    return arr


def _read_opencv_filestorage(path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """尝试按 OpenCV FileStorage 读取。

    普通 YAML 也可能被 FileStorage 打开，但其节点未必是 ``!!opencv-matrix``。
    对这种节点直接调用 ``node.mat()`` 会触发 isMap() 断言。这里逐节点捕获
    OpenCV 异常，并返回 None，让上层继续使用 PyYAML 解析，而不是提前中断。
    """
    try:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    except (cv2.error, SystemError):
        return None, None
    if not fs.isOpened():
        return None, None

    def node_matrix(keys: Sequence[str], min_size: int) -> Optional[np.ndarray]:
        for key in keys:
            try:
                node = fs.getNode(key)
                if node.empty():
                    continue
                matrix = node.mat()
            except (cv2.error, SystemError):
                # 该节点很可能是普通 YAML 列表/字典，不是 OpenCV matrix。
                continue
            if matrix is not None and matrix.size >= min_size:
                return np.asarray(matrix, dtype=np.float64)
        return None

    try:
        matrix_k = node_matrix(("camera_matrix", "K", "cameraMatrix", "intrinsic_matrix"), 9)
        matrix_d = node_matrix(
            ("dist_coeffs", "distortion_coefficients", "D", "distCoeffs", "distortion"),
            4,
        )
        K = None if matrix_k is None or matrix_k.size != 9 else matrix_k.reshape(3, 3)
        D = None if matrix_d is None else matrix_d.reshape(-1)
        return K, D
    finally:
        fs.release()


def load_reference_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """兼容 OpenCV FileStorage YAML 和普通 PyYAML 列表/字典格式。"""
    K, D = _read_opencv_filestorage(path)
    if K is not None and D is not None:
        return K, D

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "无法按普通 YAML 解析内参；请安装 PyYAML：pip install pyyaml，"
            "或把内参保存为 OpenCV FileStorage YAML。"
        ) from exc

    text = path.read_text(encoding="utf-8-sig")
    # PyYAML 不认识 OpenCV 的 %YAML:1.0 指令；FileStorage 已失败时再做兼容清理。
    text = re.sub(r"^%YAML:\s*1\.0\s*$", "%YAML 1.1", text, flags=re.MULTILINE)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("内参 YAML 顶层必须是字典。")

    K = None
    for key in ("camera_matrix", "K", "cameraMatrix", "intrinsic_matrix"):
        K = _matrix_from_mapping(data.get(key), (3, 3))
        if K is not None:
            break

    D = None
    for key in ("dist_coeffs", "distortion_coefficients", "D", "distCoeffs", "distortion"):
        D = _matrix_from_mapping(data.get(key))
        if D is not None:
            D = D.reshape(-1)
            break

    if K is None:
        # 兼容把 fx、fy、cx、cy 分开保存的普通 YAML。
        scalar_sources = [data]
        for parent_key in ("intrinsics", "camera_intrinsics", "camera"):
            candidate = data.get(parent_key)
            if isinstance(candidate, dict):
                scalar_sources.append(candidate)
        for source_data in scalar_sources:
            try:
                fx = float(source_data["fx"])
                fy = float(source_data["fy"])
                cx = float(source_data["cx"])
                cy = float(source_data["cy"])
            except (KeyError, TypeError, ValueError):
                continue
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            break

    if D is None:
        # 常见的 k1/k2/p1/p2/k3 分字段格式。
        scalar_sources = [data]
        for parent_key in ("intrinsics", "camera_intrinsics", "camera"):
            candidate = data.get(parent_key)
            if isinstance(candidate, dict):
                scalar_sources.append(candidate)
        for source_data in scalar_sources:
            try:
                D = np.array(
                    [
                        float(source_data["k1"]),
                        float(source_data["k2"]),
                        float(source_data.get("p1", 0.0)),
                        float(source_data.get("p2", 0.0)),
                        float(source_data.get("k3", 0.0)),
                    ],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError):
                continue
            break

    if K is None or D is None:
        raise ValueError(
            "内参 YAML 中未找到可识别的相机矩阵和畸变参数。支持："
            "camera_matrix/K 的二维列表、含 data 字段的 OpenCV 格式，"
            "或 fx/fy/cx/cy 与 k1/k2/p1/p2/k3 分字段格式。"
        )
    if K.shape != (3, 3) or D.size < 4:
        raise ValueError(f"内参尺寸异常：K={K.shape}, D={D.shape}")
    return K, D


# ---------------------------------------------------------------------------
# 位姿、重投影和重复姿态

def solve_board_pose(
    object_points: np.ndarray,
    corners: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray, float, float]]:
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            corners,
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    normal = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if normal[2] > 0:
        normal = -normal
    nz = float(np.clip(abs(normal[2]), 0.0, 1.0))
    tilt = math.degrees(math.acos(nz))
    azimuth = math.degrees(math.atan2(float(normal[1]), float(normal[0]))) % 360.0
    return rvec.reshape(3, 1), tvec.reshape(3, 1), tilt, azimuth


def reprojection_error(
    object_points: np.ndarray,
    corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[float, float]:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    residual = projected.reshape(-1, 2) - corners.reshape(-1, 2)
    distances = np.linalg.norm(residual, axis=1)
    rmse = math.sqrt(float(np.mean(np.square(distances))))
    return rmse, float(np.max(distances))


def find_duplicates(
    records: Sequence[Shot],
    width: int,
    height: int,
    tilt_tol: float,
    azimuth_tol: float,
    depth_rel_tol: float,
    centre_norm_tol: float,
    rotation_tol: float,
) -> int:
    """按倾角、方位、距离、板心和板内旋转的联合相似度标记近重复姿态。"""
    duplicate_count = 0
    accepted: list[Shot] = []
    diagonal = max(math.hypot(width, height), 1.0)

    for record in records:
        record.duplicate_of = ""
        for base in accepted:
            centre_dist = math.hypot(record.centre_u - base.centre_u, record.centre_v - base.centre_v) / diagonal
            depth_scale = max(abs(record.depth_mm), abs(base.depth_mm), 1.0)
            depth_rel = abs(record.depth_mm - base.depth_mm) / depth_scale
            azimuth_dist = circular_distance_deg(record.tilt_azimuth_deg, base.tilt_azimuth_deg)
            # 小倾角下方位角不稳定，不把方位角作为重复判断的硬条件。
            azimuth_close = (
                min(record.tilt_deg, base.tilt_deg) < 10.0
                or azimuth_dist <= azimuth_tol
            )
            if (
                abs(record.tilt_deg - base.tilt_deg) <= tilt_tol
                and azimuth_close
                and depth_rel <= depth_rel_tol
                and centre_dist <= centre_norm_tol
                and circular_distance_deg(
                    record.inplane_rotation_deg,
                    base.inplane_rotation_deg,
                    period=180.0,
                ) <= rotation_tol
            ):
                record.duplicate_of = base.name
                duplicate_count += 1
                break
        if not record.duplicate_of:
            accepted.append(record)
    return duplicate_count


# ---------------------------------------------------------------------------
# 缓存

def cache_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "pattern_cols": args.pattern_cols,
        "pattern_rows": args.pattern_rows,
        "blur_floor_px": round(float(args.blur_floor_px), 6),
        "opencv_version": cv2.__version__,
        "exhaustive_detection": bool(args.exhaustive_detection),
        "high_accuracy_detection": bool(args.high_accuracy_detection),
    }


def load_cache(path: Path, signature: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """返回缓存条目和是否命中当前配置；旧版/损坏缓存自动忽略。"""
    if not path.exists():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}, False
    if not isinstance(payload, dict):
        return {}, False
    if payload.get("cache_version") != CACHE_VERSION:
        return {}, False
    cached_signature = payload.get("signature")
    if not isinstance(cached_signature, dict):
        return {}, False
    # 小版本修复若未改变角点检测算法，可继续复用旧缓存；忽略脚本展示版本。
    cached_compare = dict(cached_signature)
    current_compare = dict(signature)
    cached_compare.pop("script_version", None)
    current_compare.pop("script_version", None)
    if cached_compare != current_compare:
        return {}, False
    entries = payload.get("entries")
    return (entries if isinstance(entries, dict) else {}), True


def save_cache(path: Path, signature: dict[str, Any], entries: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "signature": signature,
        "entries": entries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def cache_key(path: Path, root: Path) -> str:
    stat = path.stat()
    return f"{relative_name(path, root)}|{stat.st_mtime_ns}|{stat.st_size}"


# ---------------------------------------------------------------------------
# 结果输出

def write_csv(path: Path, records: Sequence[Shot]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(Shot("", False)).keys())
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_partial_error_outputs(
    report_path: Path,
    summary_path: Path,
    records: Sequence[Shot],
    *,
    root: Path,
    args: argparse.Namespace,
    stage: str,
    error: str,
    cache_path: Path,
    fresh: int,
    cached_hits: int,
) -> None:
    """在正式统计尚未完成时，也保存角点检测阶段的结果。"""
    write_csv(report_path, records)
    write_summary_json(
        summary_path,
        {
            "script_version": SCRIPT_VERSION,
            "status": "ERROR",
            "stage": stage,
            "error": error,
            "mode": args.mode,
            "input_dir": str(root),
            "pattern": args.pattern,
            "board": {
                "pattern_cols": args.pattern_cols,
                "pattern_rows": args.pattern_rows,
                "square_size_mm": args.square_size_mm,
            },
            "counts": {
                "input_images": len(records),
                "corner_detected": sum(1 for record in records if record.ok),
                "failed_before_pose": sum(1 for record in records if not record.ok),
                "freshly_processed": fresh,
                "cache_hits": cached_hits,
            },
            "note": "当前仅完成图像读取、角点检测和清晰度预计算；位姿、覆盖与重投影尚未计算。",
            "outputs": {
                "shot_report_csv": str(report_path),
                "summary_json": str(summary_path),
                "cache": None if args.no_cache else str(cache_path),
            },
        },
    )


def draw_dashboard(
    path: Path,
    corners: np.ndarray,
    occupancy: np.ndarray,
    tilts: np.ndarray,
    sector_hit: np.ndarray,
    depths: np.ndarray,
    blur: np.ndarray,
    reproj: np.ndarray,
    image_size: tuple[int, int],
    overall: str,
    max_blur_sigma: float,
    max_reprojection_rmse: float,
) -> None:
    width, height = image_size
    scale = 520.0 / max(width, 1)
    ch, cw = max(1, int(height * scale)), 520
    right_w, pad = 330, 20
    body_h = max(ch + 40, 760)
    canvas = np.full((body_h + 60, cw + right_w + pad, 3), 255, np.uint8)

    top = 34
    for i in range(1, 8):
        cv2.line(canvas, (int(cw * i / 8), top), (int(cw * i / 8), top + ch), (228, 228, 228), 1)
        cv2.line(canvas, (0, top + int(ch * i / 8)), (cw, top + int(ch * i / 8)), (228, 228, 228), 1)
    for x, y in corners:
        cv2.circle(canvas, (int(x * scale), top + int(y * scale)), 2, (190, 70, 40), -1)
    for row in range(8):
        for col in range(8):
            if occupancy[row, col] == 0:
                cv2.rectangle(
                    canvas,
                    (int(cw * col / 8) + 1, top + int(ch * row / 8) + 1),
                    (int(cw * (col + 1) / 8) - 1, top + int(ch * (row + 1) / 8) - 1),
                    (0, 0, 230),
                    1,
                )
    cv2.putText(
        canvas,
        f"corner coverage   {(occupancy == 0).sum()}/64 cells empty (red)",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    def hist_panel(
        x0: int,
        y0: int,
        w: int,
        h: int,
        values: np.ndarray,
        bins: Any,
        title: str,
        limit: Optional[float] = None,
        limit_label: str = "",
    ) -> None:
        values = values[np.isfinite(values)]
        cv2.putText(canvas, title, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
        if values.size == 0:
            cv2.putText(canvas, "N/A", (x0 + 8, y0 + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
            return
        counts, edges = np.histogram(values, bins=bins)
        top_n = max(1, int(counts.max()))
        for i, count in enumerate(counts):
            bx = x0 + int(i * w / len(counts))
            bw = max(4, int(w / len(counts)) - 4)
            bh = int(int(count) / top_n * h)
            if bh:
                cv2.rectangle(canvas, (bx, y0 + h - bh), (bx + bw, y0 + h), (200, 120, 60), -1)
            cv2.putText(canvas, str(int(count)), (bx + 1, y0 + h - bh - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(canvas, (x0, y0 + h), (x0 + w, y0 + h), (120, 120, 120), 1)
        cv2.putText(canvas, f"{edges[0]:.1f}", (x0, y0 + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{edges[-1]:.1f}", (x0 + w - 38, y0 + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (110, 110, 110), 1, cv2.LINE_AA)
        if limit is not None and edges[-1] > edges[0]:
            lx = x0 + int((limit - edges[0]) / (edges[-1] - edges[0]) * w)
            if x0 <= lx <= x0 + w:
                cv2.line(canvas, (lx, y0 - 2), (lx, y0 + h), (0, 0, 220), 1)
                cv2.putText(canvas, limit_label, (max(x0, lx - 24), y0 + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 200), 1, cv2.LINE_AA)

    rx = cw + pad
    hist_panel(rx, 48, right_w - pad, 82, tilts, [0, 10, 20, 30, 40, 60, 90], "board tilt (deg)", 30, "min")
    hist_panel(rx, 180, right_w - pad, 82, depths, 6, "working distance (mm)")
    hist_panel(rx, 312, right_w - pad, 82, blur, 6, "corner blur sigma (px)", max_blur_sigma, "max")
    hist_panel(rx, 444, right_w - pad, 82, reproj, 6, "reprojection RMSE (px)", max_reprojection_rmse, "max")

    cv2.putText(canvas, "tilt azimuth coverage (>=20 deg)", (rx, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
    cx0, cy0, rad = rx + (right_w - pad) // 2, 665, 58
    cv2.circle(canvas, (cx0, cy0), rad, (235, 235, 235), 1)
    for i in range(8):
        angle = math.radians(i * 45)
        hit = sector_hit[i] > 0
        px = int(cx0 + rad * math.cos(angle))
        py = int(cy0 + rad * math.sin(angle))
        colour = (60, 165, 60) if hit else (205, 205, 205)
        cv2.line(canvas, (cx0, cy0), (px, py), colour, 2)
        cv2.circle(canvas, (px, py), 11, colour, -1)
        cv2.putText(canvas, str(int(sector_hit[i])), (px - 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{int((sector_hit > 0).sum())}/8 sectors", (cx0 - 34, cy0 + rad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)

    label_map = {
        "合格": ("PASS - ready to calibrate", (60, 160, 60)),
        "基本可用，建议补拍": ("USABLE - shoot more", (0, 150, 220)),
        "不合格": ("FAIL - do not calibrate yet", (40, 40, 220)),
    }
    label = label_map[overall]
    cv2.rectangle(canvas, (0, body_h + 6), (canvas.shape[1], body_h + 54), label[1], -1)
    cv2.putText(canvas, label[0], (16, body_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise OSError(f"PNG 编码失败：{path}")
    buf.tofile(path)


# ---------------------------------------------------------------------------
# 参数与主流程

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="棋盘格标定采集预检 V2：检查拟合集或验证集是否满足使用要求",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dir", type=Path, required=True, help="图像目录")
    parser.add_argument("--pattern", default="chess *.tif", help="文件 glob，例如 'chess *.tif'")
    parser.add_argument("--recursive", action="store_true", help="递归搜索子目录")
    parser.add_argument("--mode", choices=("fit", "validation"), default="fit", help="fit 检查数据多样性；validation 侧重逐图质量")

    board = parser.add_argument_group("棋盘参数")
    board.add_argument("--pattern-cols", type=int, default=6, help="横向内角点数")
    board.add_argument("--pattern-rows", type=int, default=5, help="纵向内角点数")
    board.add_argument("--square-size-mm", type=float, default=30.0, help="单格边长，单位 mm")
    board.add_argument(
        "--high-accuracy-detection",
        action="store_true",
        help="启用 SB 的高精度模式，检测更慢；普通预检通常无需启用",
    )
    board.add_argument(
        "--exhaustive-detection",
        action="store_true",
        help="启用 SB 穷举搜索，检测更慢；仅在大倾角难检图上尝试",
    )

    intr = parser.add_argument_group("内参来源（优先级：reference-yaml > 名义内参 > 内部快速标定）")
    intr.add_argument("--reference-yaml", type=Path, default=None, help="已有内参 YAML")
    intr.add_argument("--focal-length-mm", type=float, default=None, help="镜头标称焦距")
    intr.add_argument("--pixel-um", type=float, default=3.45, help="像元尺寸，单位 µm")

    out = parser.add_argument_group("输出")
    out.add_argument("--output-dir", type=Path, default=None, help="输出目录；默认 <dir>/capture_check_output")
    out.add_argument("--report", type=Path, default=None, help="逐图 CSV；默认输出目录/shot_report.csv")
    out.add_argument("--dashboard", type=Path, default=None, help="PNG 看板；默认输出目录/dashboard.png")
    out.add_argument("--summary-json", type=Path, default=None, help="汇总 JSON；默认输出目录/summary.json")
    out.add_argument("--no-dashboard", action="store_true", help="不生成 PNG 看板")

    cache = parser.add_argument_group("缓存")
    cache.add_argument("--cache", type=Path, default=None, help="缓存文件；默认 <dir>/.capture_check_cache_v2.json")
    cache.add_argument("--no-cache", action="store_true", help="禁用缓存")
    cache.add_argument("--clear-cache", action="store_true", help="运行前删除当前缓存")

    quality = parser.add_argument_group("图像质量门槛")
    quality.add_argument("--blur-floor-px", type=float, default=BLUR_FLOOR_PX, help="模糊估计器基底")
    quality.add_argument("--max-blur-sigma", type=float, default=1.20, help="逐图模糊 σ 中位数上限，px")
    quality.add_argument("--max-blur-p95", type=float, default=1.92, help="逐图角点模糊 σ 第95百分位的全局上限，px")
    quality.add_argument(
        "--max-clipped",
        type=float,
        default=0.20,
        help="角点邻域接近0/255的极值像素比例参考上限；默认仅作辅助提醒",
    )
    quality.add_argument(
        "--clipping-decisive",
        action="store_true",
        help="把极值像素占比纳入最终判定；普通黑白棋盘通常不建议启用",
    )
    quality.add_argument("--max-reprojection-rmse", type=float, default=0.80, help="逐图重投影 RMSE 中位数上限，px")
    quality.add_argument("--max-reprojection-worst", type=float, default=1.50, help="最差图逐图重投影 RMSE 上限，px")

    diversity = parser.add_argument_group("拟合集多样性门槛（mode=fit 时参与最终判定）")
    diversity.add_argument("--min-tilt-median", type=float, default=20.0)
    diversity.add_argument("--min-tilt-max", type=float, default=30.0)
    diversity.add_argument("--min-strong-tilt-images", type=int, default=6, help="倾角 ≥30° 的图像数")
    diversity.add_argument("--min-tilt-sectors", type=int, default=6, help="倾角 ≥20° 时需覆盖的方位扇区数，共8个")
    diversity.add_argument("--min-inplane-sectors", type=int, default=3, help="棋盘平面内旋转需覆盖的30°扇区数，共6个")
    diversity.add_argument("--min-depth-span", type=float, default=0.20, help="(最大Z-最小Z)/平均Z")
    diversity.add_argument("--min-radius-coverage", type=float, default=0.85, help="角点最大归一化半径/图像角半径")
    diversity.add_argument("--max-empty-cells", type=int, default=12, help="8×8覆盖网格允许的空格数")
    diversity.add_argument("--min-board-area", type=float, default=0.03, help="棋盘角点凸包面积占全图比例的中位数下限")
    diversity.add_argument("--min-images", type=int, default=None, help="有效图像下限；fit 默认20，validation 默认1")
    diversity.add_argument(
        "--min-valid-rate",
        type=float,
        default=None,
        help="输入图像中最终有效图像的比例下限；fit 默认0.85，validation 默认1.0",
    )
    diversity.add_argument("--min-total-points", type=int, default=None, help="总角点下限；默认自动等于 min-images×每张角点数")
    diversity.add_argument("--max-duplicate-fraction", type=float, default=0.25, help="近重复姿态占有效图像比例上限")

    duplicate = parser.add_argument_group("近重复姿态判定阈值")
    duplicate.add_argument("--duplicate-tilt-tol", type=float, default=2.0, help="倾角差阈值，度")
    duplicate.add_argument("--duplicate-azimuth-tol", type=float, default=10.0, help="倾斜方位差阈值，度")
    duplicate.add_argument("--duplicate-depth-rel-tol", type=float, default=0.03, help="相对深度差阈值")
    duplicate.add_argument("--duplicate-centre-norm-tol", type=float, default=0.04, help="板心距离/图像对角线阈值")
    duplicate.add_argument("--duplicate-rotation-tol", type=float, default=5.0, help="平面内旋转差阈值，度")

    args = parser.parse_args()
    if args.pattern_cols < 2 or args.pattern_rows < 2:
        parser.error("pattern-cols 和 pattern-rows 必须至少为 2")
    if args.square_size_mm <= 0:
        parser.error("square-size-mm 必须大于 0")
    if args.pixel_um <= 0:
        parser.error("pixel-um 必须大于 0")
    if args.min_images is None:
        args.min_images = 20 if args.mode == "fit" else 1
    if args.min_valid_rate is None:
        args.min_valid_rate = 0.85 if args.mode == "fit" else 1.0
    if not 0.0 <= args.min_valid_rate <= 1.0:
        parser.error("min-valid-rate 必须在 0～1 之间")
    if args.min_total_points is None:
        args.min_total_points = args.min_images * args.pattern_cols * args.pattern_rows
    return args


def main() -> int:
    args = parse_args()
    root = args.dir.resolve()
    if not root.exists() or not root.is_dir():
        print(f"错误：目录不存在或不是文件夹：{root}", file=sys.stderr)
        return 2

    output_dir = (args.output_dir or (root / "capture_check_output")).resolve()
    report_path = (args.report or (output_dir / "shot_report.csv")).resolve()
    dashboard_path = (args.dashboard or (output_dir / "dashboard.png")).resolve()
    summary_path = (args.summary_json or (output_dir / "summary.json")).resolve()
    cache_path = (args.cache or (root / ".capture_check_cache_v2.json")).resolve()

    pattern_size = (args.pattern_cols, args.pattern_rows)
    points_per_image = args.pattern_cols * args.pattern_rows
    object_points = np.zeros((points_per_image, 3), dtype=np.float64)
    object_points[:, :2] = (
        np.mgrid[0 : args.pattern_cols, 0 : args.pattern_rows].T.reshape(-1, 2)
        * args.square_size_mm
    )

    globber = root.rglob if args.recursive else root.glob
    paths = sorted(
        (
            p
            for p in globber(args.pattern)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: natural_key(p, root),
    )
    if not paths:
        print(f"错误：{root} 下没有匹配 {args.pattern!r} 的图像", file=sys.stderr)
        return 2

    signature = cache_signature(args)
    if args.clear_cache and cache_path.exists():
        try:
            cache_path.unlink()
        except OSError as exc:
            print(f"错误：无法删除缓存 {cache_path}：{exc}", file=sys.stderr)
            return 2

    cache_entries: dict[str, Any] = {}
    cache_valid = False
    if not args.no_cache:
        cache_entries, cache_valid = load_cache(cache_path, signature)

    records: list[Shot] = []
    corner_sets: dict[str, np.ndarray] = {}
    image_size: Optional[tuple[int, int]] = None
    fresh = 0
    cached_hits = 0

    print(f"扫描到 {len(paths)} 张图像，开始检测……")
    print(f"结果目录：{output_dir}")
    for path in paths:
        name = relative_name(path, root)
        image = read_image_unchanged(path)
        if image is None:
            records.append(Shot(name=name, ok=False, reason="读取失败"))
            print(f"  ✗ {name:32s} 读取失败")
            continue

        try:
            gray8 = to_gray8(image)
        except ValueError as exc:
            records.append(Shot(name=name, ok=False, reason=str(exc)))
            print(f"  ✗ {name:32s} {exc}")
            continue

        current_size = (gray8.shape[1], gray8.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            records.append(
                Shot(
                    name=name,
                    ok=False,
                    reason=f"分辨率不一致：{current_size[0]}×{current_size[1]}，期望 {image_size[0]}×{image_size[1]}",
                    image_width=current_size[0],
                    image_height=current_size[1],
                )
            )
            print(f"  ✗ {name:32s} 分辨率不一致")
            continue

        key = cache_key(path, root)
        entry = cache_entries.get(key) if not args.no_cache else None
        corners: Optional[np.ndarray] = None
        method = ""
        blur = (math.nan, math.nan, math.nan)

        if isinstance(entry, dict):
            status = entry.get("status")
            if status == "ok" and entry.get("corners") is not None:
                try:
                    corners = np.asarray(entry["corners"], dtype=np.float64).reshape(-1, 2)
                    method = str(entry.get("detection_method", "cache"))
                    cached_blur = entry.get("blur", [math.nan, math.nan, math.nan])
                    values = list(cached_blur[:3]) if isinstance(cached_blur, (list, tuple)) else []
                    values += [math.nan] * (3 - len(values))
                    blur = tuple(math.nan if v is None else float(v) for v in values[:3])  # type: ignore[assignment]
                    cached_hits += 1
                except (TypeError, ValueError):
                    corners = None
            elif status == "not_found":
                records.append(
                    Shot(
                        name=name,
                        ok=False,
                        reason="未检出完整棋盘",
                        image_width=current_size[0],
                        image_height=current_size[1],
                    )
                )
                cached_hits += 1
                print(f"  ✗ {name:32s} 未检出完整棋盘（缓存）")
                continue

        if corners is None:
            corners, method = detect(
                gray8,
                pattern_size,
                exhaustive=args.exhaustive_detection,
                high_accuracy=args.high_accuracy_detection,
            )
            fresh += 1
            if corners is None:
                records.append(
                    Shot(
                        name=name,
                        ok=False,
                        reason="未检出完整棋盘",
                        image_width=current_size[0],
                        image_height=current_size[1],
                    )
                )
                cache_entries[key] = {"status": "not_found"}
                print(f"  ✗ {name:32s} 未检出完整棋盘")
                continue
            blur = blur_sigma_at(gray8.astype(np.float32), corners, floor=args.blur_floor_px)
            cache_entries[key] = {
                "status": "ok",
                "corners": corners.tolist(),
                "blur": [None if not math.isfinite(v) else float(v) for v in blur],
                "detection_method": method,
            }
        else:
            blur = tuple(math.nan if v is None else float(v) for v in blur)  # type: ignore[assignment]

        if corners.shape != (points_per_image, 2):
            records.append(
                Shot(
                    name=name,
                    ok=False,
                    reason=f"角点数量异常：{len(corners)}，期望 {points_per_image}",
                    image_width=current_size[0],
                    image_height=current_size[1],
                )
            )
            continue

        corner_sets[name] = corners
        records.append(
            Shot(
                name=name,
                ok=True,
                image_width=current_size[0],
                image_height=current_size[1],
                detection_method=method,
                inplane_rotation_deg=inplane_rotation(corners, args.pattern_cols),
                blur_sigma_median=float(blur[0]),
                blur_sigma_p95=float(blur[1]),
                clipped_fraction=float(blur[2]),
                centre_u=float(np.mean(corners[:, 0])),
                centre_v=float(np.mean(corners[:, 1])),
            )
        )
        print(f"  ✓ {name:32s} {method}")

    if not args.no_cache:
        try:
            save_cache(cache_path, signature, cache_entries)
        except OSError as exc:
            print(f"警告：缓存写入失败：{exc}", file=sys.stderr)

    detected = [record for record in records if record.ok]
    if image_size is None or not detected:
        write_csv(report_path, records)
        print(f"\n没有可用的完整棋盘图像（0/{len(records)}）。")
        print(f"逐图报告已输出：{report_path}")
        return 2
    if (
        args.reference_yaml is None
        and args.focal_length_mm is None
        and len(detected) < 3
    ):
        write_csv(report_path, records)
        print(
            f"\n仅检出 {len(detected)} 张图，且未提供参考/名义内参，"
            "不足以执行内部快速标定。"
        )
        print("请提供 --reference-yaml，或同时提供 --focal-length-mm 与 --pixel-um。")
        print(f"逐图报告已输出：{report_path}")
        return 2

    width, height = image_size

    # ---- 内参来源 ----
    K: np.ndarray
    D: np.ndarray
    source: str
    calibration_rms = math.nan
    internal_rvecs: Optional[Sequence[np.ndarray]] = None
    internal_tvecs: Optional[Sequence[np.ndarray]] = None

    if args.reference_yaml is not None:
        reference_path = args.reference_yaml.resolve()
        if not reference_path.exists():
            message = f"内参文件不存在：{reference_path}"
            write_partial_error_outputs(
                report_path, summary_path, records,
                root=root, args=args, stage="load_intrinsics", error=message,
                cache_path=cache_path, fresh=fresh, cached_hits=cached_hits,
            )
            print(f"错误：{message}", file=sys.stderr)
            print(f"角点检测阶段结果已保存：{report_path}")
            print(f"错误摘要已保存：{summary_path}")
            return 2
        try:
            K, D = load_reference_intrinsics(reference_path)
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            message = f"读取内参失败：{exc}"
            write_partial_error_outputs(
                report_path, summary_path, records,
                root=root, args=args, stage="load_intrinsics", error=message,
                cache_path=cache_path, fresh=fresh, cached_hits=cached_hits,
            )
            print(f"错误：{message}", file=sys.stderr)
            print(f"角点检测阶段结果已保存：{report_path}")
            print(f"错误摘要已保存：{summary_path}")
            return 2
        source = f"参考内参 {reference_path.name}"
        reprojection_decisive = True
    elif args.focal_length_mm is not None:
        focal_px = args.focal_length_mm / (args.pixel_um * 1e-3)
        K = np.array(
            [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        D = np.zeros(5, dtype=np.float64)
        source = f"名义内参（{args.focal_length_mm:g} mm / {args.pixel_um:g} µm → f={focal_px:.1f} px）"
        reprojection_decisive = False
    else:
        object_sets = [object_points.astype(np.float32).reshape(-1, 1, 3) for _ in detected]
        image_sets = [corner_sets[record.name].astype(np.float32).reshape(-1, 1, 2) for record in detected]
        try:
            calibration_rms, K, D, internal_rvecs, internal_tvecs = cv2.calibrateCamera(
                object_sets,
                image_sets,
                (width, height),
                None,
                None,
                flags=cv2.CALIB_FIX_K3,
            )
        except cv2.error as exc:
            message = f"内部快速标定失败：{exc}"
            write_partial_error_outputs(
                report_path, summary_path, records,
                root=root, args=args, stage="internal_calibration", error=message,
                cache_path=cache_path, fresh=fresh, cached_hits=cached_hits,
            )
            print(f"错误：{message}", file=sys.stderr)
            print(f"角点检测阶段结果已保存：{report_path}")
            print(f"错误摘要已保存：{summary_path}")
            return 2
        source = "内部快速标定（仅用于预检；数据退化时内参本身不可信）"
        reprojection_decisive = True

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if fx <= 0 or fy <= 0:
        print("错误：内参焦距必须为正。", file=sys.stderr)
        return 2
    r_corner = max(
        math.hypot((x - cx) / fx, (y - cy) / fy)
        for x in (0.0, float(width))
        for y in (0.0, float(height))
    )

    # ---- 位姿和重投影 ----
    for index, record in enumerate(detected):
        corners = corner_sets[record.name]
        if internal_rvecs is not None and internal_tvecs is not None:
            rvec = np.asarray(internal_rvecs[index], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(internal_tvecs[index], dtype=np.float64).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            normal = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if normal[2] > 0:
                normal = -normal
            tilt = math.degrees(math.acos(float(np.clip(abs(normal[2]), 0.0, 1.0))))
            azimuth = math.degrees(math.atan2(float(normal[1]), float(normal[0]))) % 360.0
        else:
            pose = solve_board_pose(object_points, corners, K, D)
            if pose is None:
                record.ok = False
                record.reason = "位姿求解失败"
                continue
            rvec, tvec, tilt, azimuth = pose

        record.tilt_deg = tilt
        record.tilt_azimuth_deg = azimuth
        record.tvec_x_mm = float(tvec[0, 0])
        record.tvec_y_mm = float(tvec[1, 0])
        record.depth_mm = float(tvec[2, 0])
        if record.depth_mm <= 0:
            record.ok = False
            record.reason = "位姿深度非正"
            continue

        hull = cv2.convexHull(corners.astype(np.float32))
        record.board_area_fraction = float(cv2.contourArea(hull) / (width * height))
        record.radius_max = float(
            np.max(
                np.hypot(
                    (corners[:, 0] - cx) / fx,
                    (corners[:, 1] - cy) / fy,
                )
            )
        )
        record.reprojection_rmse_px, record.reprojection_max_px = reprojection_error(
            object_points, corners, rvec, tvec, K, D
        )

    good = [record for record in detected if record.ok]
    if not good:
        write_csv(report_path, records)
        print("所有已检出棋盘图像均未能成功求解位姿。")
        print(f"逐图报告已输出：{report_path}")
        return 2

    duplicate_count = find_duplicates(
        good,
        width,
        height,
        args.duplicate_tilt_tol,
        args.duplicate_azimuth_tol,
        args.duplicate_depth_rel_tol,
        args.duplicate_centre_norm_tol,
        args.duplicate_rotation_tol,
    )

    # ---- 汇总统计 ----
    tilts = finite_values(record.tilt_deg for record in good)
    depths = finite_values(record.depth_mm for record in good)
    blur_medians = finite_values(record.blur_sigma_median for record in good)
    blur_p95 = finite_values(record.blur_sigma_p95 for record in good)
    clipped = finite_values(record.clipped_fraction for record in good)
    reprojection = finite_values(record.reprojection_rmse_px for record in good)
    board_areas = finite_values(record.board_area_fraction for record in good)
    rotations = finite_values(record.inplane_rotation_deg for record in good)
    all_corners = np.vstack([corner_sets[record.name] for record in good])

    radius_max = float(
        np.max(
            np.hypot(
                (all_corners[:, 0] - cx) / fx,
                (all_corners[:, 1] - cy) / fy,
            )
        )
    )
    radius_coverage = radius_max / max(r_corner, 1e-12)

    occupancy = np.zeros((8, 8), dtype=int)
    for x, y in all_corners:
        if 0 <= x < width and 0 <= y < height:
            row = min(7, int(y / height * 8))
            col = min(7, int(x / width * 8))
            occupancy[row, col] += 1
    empty_cells = int(np.sum(occupancy == 0))

    zones = np.zeros((3, 3), dtype=int)
    for record in good:
        row = min(2, max(0, int(record.centre_v / height * 3)))
        col = min(2, max(0, int(record.centre_u / width * 3)))
        zones[row, col] += 1

    sector_hit = np.zeros(8, dtype=int)
    for record in good:
        if record.tilt_deg >= 20.0:
            sector = int(((record.tilt_azimuth_deg + 22.5) % 360.0) // 45.0)
            sector_hit[sector] += 1

    rotation_hit = np.zeros(6, dtype=int)
    for rotation in rotations:
        rotation_hit[min(5, int(rotation // 30.0))] += 1

    mean_depth = float(np.mean(depths)) if depths.size else math.nan
    depth_span = (
        float((np.max(depths) - np.min(depths)) / mean_depth)
        if depths.size and mean_depth > 0
        else math.nan
    )
    strong_count = int(np.sum(tilts >= 30.0))
    total_points = int(len(all_corners))
    duplicate_fraction = duplicate_count / max(len(good), 1)
    valid_rate = len(good) / max(len(records), 1)

    # ---- 检查项 ----
    checks: list[CheckItem] = []

    def add_check(
        name: str,
        value: float,
        limit: float,
        mode: str,
        unit: str = "",
        decisive: bool = True,
        note: str = "",
    ) -> None:
        status = grade(value, limit, mode)
        if not decisive and status == "FAIL":
            status = "WARN"
        checks.append(CheckItem(name, value, limit, mode, unit, status, decisive, note))

    fit_decisive = args.mode == "fit"
    add_check("板面倾角中位", safe_nanmedian(tilts), args.min_tilt_median, "min", "°", fit_decisive)
    add_check("板面倾角最大", safe_nanmax(tilts), args.min_tilt_max, "min", "°", fit_decisive)
    add_check("≥30° 的图像数", float(strong_count), float(args.min_strong_tilt_images), "min", "张", fit_decisive)
    add_check("倾斜方位扇区数", float(np.sum(sector_hit > 0)), float(args.min_tilt_sectors), "min", "/8", fit_decisive)
    add_check("平面内旋转扇区数", float(np.sum(rotation_hit > 0)), float(args.min_inplane_sectors), "min", "/6", fit_decisive)
    add_check("工作距离相对跨度", depth_span, args.min_depth_span, "min", "", fit_decisive)
    add_check("半径覆盖率", radius_coverage, args.min_radius_coverage, "min", "", fit_decisive)
    add_check("8×8 空格子", float(empty_cells), float(args.max_empty_cells), "max", "格", fit_decisive)
    add_check("棋盘面积占比中位", safe_nanmedian(board_areas), args.min_board_area, "min", "", fit_decisive)
    add_check("近重复姿态占比", duplicate_fraction, args.max_duplicate_fraction, "max", "", fit_decisive)

    add_check("角点模糊 σ 中位", safe_nanmedian(blur_medians), args.max_blur_sigma, "max", "px", True)
    add_check("角点模糊 σ 最差P95", safe_nanmax(blur_p95), args.max_blur_p95, "max", "px", True)
    add_check(
        "极值像素占比",
        safe_nanmax(clipped),
        args.max_clipped,
        "max",
        "",
        args.clipping_decisive,
        "黑白棋盘中仅作曝光风险参考" if not args.clipping_decisive else "",
    )
    add_check(
        "重投影 RMSE 中位",
        safe_nanmedian(reprojection),
        args.max_reprojection_rmse,
        "max",
        "px",
        reprojection_decisive,
        "名义内参下仅供参考" if not reprojection_decisive else "",
    )
    add_check(
        "重投影 RMSE 最差图",
        safe_nanmax(reprojection),
        args.max_reprojection_worst,
        "max",
        "px",
        reprojection_decisive,
        "名义内参下仅供参考" if not reprojection_decisive else "",
    )
    add_check("有效图像比例", valid_rate, float(args.min_valid_rate), "min", "", True)
    add_check("有效图像数", float(len(good)), float(args.min_images), "min", "张", True)
    add_check("总角点数", float(total_points), float(args.min_total_points), "min", "点", True)

    decisive_fails = [item for item in checks if item.decisive and item.status == "FAIL"]
    decisive_warns = [item for item in checks if item.decisive and item.status == "WARN"]
    advisory_warns = [item for item in checks if not item.decisive and item.status != "PASS"]
    overall = "不合格" if decisive_fails else ("基本可用，建议补拍" if decisive_warns else "合格")

    # ---- 控制台输出 ----
    print("\n" + "=" * 78)
    print(f"棋盘格标定采集预检 V{SCRIPT_VERSION}　模式：{args.mode}")
    print(f"图像目录：{root}")
    print("=" * 78)
    print(
        f"图像 {len(records)} 张｜检测成功 {len(detected)} 张｜位姿有效 {len(good)} 张｜"
        f"失败 {len(records) - len(good)} 张｜新处理 {fresh} 张｜缓存命中 {cached_hits} 张"
    )
    print(f"图像尺寸：{width}×{height}")
    print(f"内参来源：{source}")
    if math.isfinite(calibration_rms):
        print(f"内部快速标定 RMS：{calibration_rms:.4f} px")
    print(f"输出目录：{output_dir}")

    failures = [record for record in records if not record.ok]
    if failures:
        print("失败图：")
        for record in failures[:12]:
            print(f"    - {record.name}: {record.reason}")
        if len(failures) > 12:
            print(f"    ……另有 {len(failures) - 12} 张，详见 CSV")

    print("\n【1】姿态与距离")
    print(
        f"    倾角：中位 {safe_nanmedian(tilts):.1f}°，范围 "
        f"{safe_nanmin(tilts):.1f}°～{safe_nanmax(tilts):.1f}°，≥30° 共 {strong_count} 张"
    )
    print(
        "    倾斜方位（≥20°）："
        + " ".join(
            f"{SECTORS[i]}{int(sector_hit[i])}" if sector_hit[i] else f"{SECTORS[i]}·"
            for i in range(8)
        )
    )
    print(
        "    平面内旋转（每30°）："
        + " ".join(f"{i * 30:3d}°:{int(rotation_hit[i])}" for i in range(6))
    )
    print(
        f"    工作距离 Z：{safe_nanmin(depths):.1f}～{safe_nanmax(depths):.1f} mm，"
        f"中位 {safe_nanmedian(depths):.1f} mm，相对跨度 {depth_span * 100:.1f}%"
    )
    print(f"    近重复姿态：{duplicate_count}/{len(good)}（{duplicate_fraction * 100:.1f}%）")
    duplicates = [record for record in good if record.duplicate_of]
    for record in duplicates[:8]:
        print(f"        {record.name} ≈ {record.duplicate_of}")
    if len(duplicates) > 8:
        print(f"        ……另有 {len(duplicates) - 8} 组，详见 CSV")

    print("\n【2】画面覆盖")
    print(f"    半径覆盖率：{radius_coverage * 100:.1f}%")
    print(f"    8×8 网格空格：{empty_cells}/64")
    for row in occupancy:
        print("      " + " ".join(f"{int(value):4d}" if value else "   ." for value in row))
    print("    板心 3×3 分区：")
    for row in zones:
        print("      " + " ".join(f"{int(value):3d}" for value in row))
    print(
        f"    棋盘面积占比：中位 {safe_nanmedian(board_areas) * 100:.2f}%｜"
        f"最小 {safe_nanmin(board_areas) * 100:.2f}%｜最大 {safe_nanmax(board_areas) * 100:.2f}%"
    )

    print("\n【3】图像质量与几何一致性")
    print(
        f"    模糊 σ：逐图中位的总体中位 {safe_nanmedian(blur_medians):.3f} px，"
        f"最差图 P95 {safe_nanmax(blur_p95):.3f} px"
    )
    print(f"    接近0/255的极值像素占比最高：{safe_nanmax(clipped) * 100:.2f}%"
        + ("（默认仅作辅助曝光风险参考）" if not args.clipping_decisive else ""))
    print(
        f"    重投影 RMSE：中位 {safe_nanmedian(reprojection):.3f} px，"
        f"最差 {safe_nanmax(reprojection):.3f} px"
        + ("（名义内参下仅作相对参考）" if not reprojection_decisive else "")
    )
    worst_reproj = max(good, key=lambda record: record.reprojection_rmse_px if math.isfinite(record.reprojection_rmse_px) else -1.0)
    print(f"    最差重投影图：{worst_reproj.name} = {worst_reproj.reprojection_rmse_px:.3f} px")

    print("\n【4】验收结果")
    print(f"    {'项目':<22}{'实测':>13}  {'判据':<17}{'状态':<8}说明")
    for item in checks:
        op = "≥" if item.mode == "min" else "≤"
        decisive_text = "" if item.decisive else "辅助项"
        note = "；".join(part for part in (decisive_text, item.note) if part)
        value_text = format_value(item.value, item.unit)
        limit_text = f"{op} {item.limit:g}{item.unit}"
        print(
            f"    {item.name:<22}{value_text:>13}  {limit_text:<17}"
            f"{verdict_mark(item.status)} {item.status:<5}{note}"
        )

    final_status = "FAIL" if decisive_fails else ("WARN" if decisive_warns else "PASS")
    print(
        f"\n    最终判定：{verdict_mark(final_status)} {overall} "
        f"（决定性 FAIL {len(decisive_fails)}｜决定性 WARN {len(decisive_warns)}｜辅助提醒 {len(advisory_warns)}）"
    )

    # ---- 补拍/处理建议 ----
    todo: list[str] = []
    if args.mode == "fit":
        if safe_nanmedian(tilts) < args.min_tilt_median or strong_count < args.min_strong_tilt_images:
            todo.append(f"增加倾角 ≥30° 的图像，当前还差约 {max(0, args.min_strong_tilt_images - strong_count)} 张。")
        missing = [SECTORS[i] for i in range(8) if sector_hit[i] == 0]
        if len(missing) > 8 - args.min_tilt_sectors:
            todo.append(f"补充这些大倾角方向：{' '.join(missing)}；→ 为图像右，↓ 为图像下。")
        if np.sum(rotation_hit > 0) < args.min_inplane_sectors:
            todo.append("增加棋盘绕自身法向的旋转，例如横放、斜放和接近竖放，避免所有棋盘边方向一致。")
        if depth_span < args.min_depth_span:
            median_depth = safe_nanmedian(depths)
            todo.append(f"工作距离过于集中，可在约 {median_depth * 0.85:.0f} mm 和 {median_depth * 1.15:.0f} mm 各补拍 2～3 张。")
        empty_zones = [ZONE_NAMES[i][j] for i in range(3) for j in range(3) if zones[i, j] == 0]
        if empty_zones:
            todo.append(f"棋盘中心未覆盖：{'、'.join(empty_zones)}；优先各补拍 1～2 张。")
        if radius_coverage < args.min_radius_coverage or empty_cells > args.max_empty_cells:
            todo.append("边角覆盖不足：把棋盘整体移动到四边和四角附近，但普通棋盘必须保持完整可见。")
        if safe_nanmedian(board_areas) < args.min_board_area:
            todo.append("棋盘在图像中偏小：适当靠近相机，或使用更大尺寸/更多角点的棋盘。")
        if duplicate_fraction > args.max_duplicate_fraction:
            todo.append("近重复姿态较多：删除重复图或补拍在位置、倾角、距离、板内旋转上明显不同的图。")

    if safe_nanmedian(blur_medians) > args.max_blur_sigma or safe_nanmax(blur_p95) > args.max_blur_p95:
        todo.append("图像偏糊：重新精确对焦；必要时收小光圈增加景深，并提高稳定照明，避免单纯延长曝光造成运动模糊。")
    if safe_nanmax(clipped) > args.max_clipped:
        todo.append(
            "角点邻域接近0/255的像素较多：结合原图确认是否真的发生高光饱和、暗部压死或边缘膨胀；"
            "黑白棋盘本身含大量明暗区域，因此该项默认不作为硬性淘汰依据。"
        )
    if reprojection_decisive and safe_nanmax(reprojection) > args.max_reprojection_worst:
        todo.append("检查重投影误差最大的图：可能存在角点误检、棋盘翘曲、运动模糊或内参不匹配。")
    if valid_rate < args.min_valid_rate:
        todo.append(
            f"有效图像比例仅 {valid_rate * 100:.1f}%，低于要求 {args.min_valid_rate * 100:.1f}%；"
            "请替换未检出、分辨率不一致或位姿失败的图像。"
        )
    if len(good) < args.min_images:
        todo.append(f"有效图像还差 {args.min_images - len(good)} 张。")
    if total_points < args.min_total_points:
        todo.append(
            f"总角点数还差 {args.min_total_points - total_points} 点；当前每张完整棋盘贡献 {points_per_image} 点。"
        )
    if failures:
        todo.append(f"替换 {len(failures)} 张读取、分辨率、角点检测或位姿求解失败的图像。")

    print("\n【5】建议")
    if todo:
        for item in todo:
            print(f"    • {item}")
    else:
        print("    无明显问题，可以进入正式标定或独立验证。")

    # ---- 写文件 ----
    write_csv(report_path, records)
    summary = {
        "script_version": SCRIPT_VERSION,
        "mode": args.mode,
        "input_dir": str(root),
        "pattern": args.pattern,
        "image_size": {"width": width, "height": height},
        "board": {
            "pattern_cols": args.pattern_cols,
            "pattern_rows": args.pattern_rows,
            "square_size_mm": args.square_size_mm,
            "points_per_image": points_per_image,
        },
        "intrinsics_source": source,
        "camera_matrix": K,
        "dist_coeffs": D,
        "internal_calibration_rms_px": calibration_rms,
        "counts": {
            "input_images": len(records),
            "corner_detected": len(detected),
            "pose_valid": len(good),
            "failed": len(records) - len(good),
            "freshly_processed": fresh,
            "cache_hits": cached_hits,
            "duplicate_count": duplicate_count,
        },
        "metrics": {
            "tilt_median_deg": safe_nanmedian(tilts),
            "tilt_min_deg": safe_nanmin(tilts),
            "tilt_max_deg": safe_nanmax(tilts),
            "strong_tilt_images": strong_count,
            "tilt_sector_hits": sector_hit,
            "inplane_rotation_sector_hits": rotation_hit,
            "depth_min_mm": safe_nanmin(depths),
            "depth_median_mm": safe_nanmedian(depths),
            "depth_max_mm": safe_nanmax(depths),
            "depth_relative_span": depth_span,
            "radius_coverage": radius_coverage,
            "empty_8x8_cells": empty_cells,
            "board_centre_3x3": zones,
            "board_area_fraction_median": safe_nanmedian(board_areas),
            "duplicate_fraction": duplicate_fraction,
            "valid_rate": valid_rate,
            "blur_sigma_median_px": safe_nanmedian(blur_medians),
            "blur_sigma_worst_p95_px": safe_nanmax(blur_p95),
            "clipped_fraction_max": safe_nanmax(clipped),
            "reprojection_rmse_median_px": safe_nanmedian(reprojection),
            "reprojection_rmse_worst_px": safe_nanmax(reprojection),
            "total_points": total_points,
        },
        "checks": [asdict(item) for item in checks],
        "overall": overall,
        "suggestions": todo,
        "outputs": {
            "shot_report_csv": str(report_path),
            "summary_json": str(summary_path),
            "dashboard_png": None if args.no_dashboard else str(dashboard_path),
            "cache": None if args.no_cache else str(cache_path),
        },
    }
    write_summary_json(summary_path, summary)

    if not args.no_dashboard:
        draw_dashboard(
            dashboard_path,
            all_corners,
            occupancy,
            tilts,
            sector_hit,
            depths,
            blur_medians,
            reprojection,
            (width, height),
            overall,
            args.max_blur_sigma,
            args.max_reprojection_rmse,
        )

    print("\n【6】输出文件")
    print(f"    逐图 CSV：{report_path}")
    print(f"    汇总 JSON：{summary_path}")
    if not args.no_dashboard:
        print(f"    PNG 看板：{dashboard_path}")
    if not args.no_cache:
        print(f"    缓存文件：{cache_path}")

    return 1 if decisive_fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
