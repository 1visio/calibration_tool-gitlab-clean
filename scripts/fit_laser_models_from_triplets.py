#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从“棋盘格高曝光 / 低曝光无激光 / 低曝光有激光”三联图中：
1) 估计每一姿态的棋盘格平面；
2) 通过背景差分 + Steger 亚像素中心提取获得激光像素；
3) 用相机射线与棋盘格平面求交，得到不依赖旧激光面的三维标定点；
4) 比较三种激光表面模型：
   - global_plane：传统全局平面；
   - quadratic_graph：稳定的二次图曲面（自动选择因变量轴）；
   - circular_cone：论文启发的空间标准圆锥二次曲面，使用一般相机坐标参数化，
     不强制论文中的“仅绕 Y 轴旋转”安装假设；
5) 在按图像划分的独立验证集上，比较重建点到真实棋盘格平面的误差。

运行：
    python fit_laser_models_from_triplets.py --config laser_model_fit_config.yaml

依赖：numpy opencv-python scipy pandas matplotlib pyyaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# The script is intentionally runnable directly from ``scripts/`` as well as
# imported by the calibration-tool workflow.  Add the package root only for
# the lightweight model-name/legacy-config compatibility layer.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from calibration_tool.laser_models import (  # noqa: E402
    SUPPORTED_LASER_MODEL_TYPES,
    LaserModelConfigError,
    normalize_model_type,
    select_default_model,
)
from calibration_tool.laser import (  # noqa: E402
    normalize_laser_orientation,
    parse_laser_config,
)
from calibration_tool.board_mask import (  # noqa: E402
    FULL_BOARD_PHYSICAL,
    full_board_physical_mask,
    projected_board_boundary,
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import least_squares

EPS = 1.0e-12


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------

def imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """兼容 Windows 中文路径的图像读取。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint16:
        scale = 255.0 / max(float(np.percentile(image, 99.9)), 1.0)
        return np.clip(image.astype(np.float32) * scale, 0.0, 255.0)
    return image.astype(np.float32)


def safe_yaml_load(path: Path) -> Dict[str, Any]:
    """读取普通 YAML，也兼容部分 OpenCV YAML 头。"""
    text = path.read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if not line.startswith("%YAML")]
    text = "\n".join(lines).replace("!!opencv-matrix", "")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 根节点不是字典：{path}")
    return data


def matrix_from_yaml_value(value: Any, name: str) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        rows = int(value.get("rows", 1))
        cols = int(value.get("cols", len(value["data"])))
        return np.asarray(value["data"], dtype=np.float64).reshape(rows, cols)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} 为空")
    return arr


def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    data = safe_yaml_load(path)
    k_value = data.get("camera_matrix", data.get("K", data.get("intrinsic_matrix")))
    d_value = data.get("dist_coeffs", data.get("distortion_coefficients", data.get("D", [])))
    if k_value is None:
        raise KeyError(f"内参文件中未找到 camera_matrix/K：{path}")
    k = matrix_from_yaml_value(k_value, "camera_matrix").reshape(3, 3)
    d = matrix_from_yaml_value(d_value, "dist_coeffs").reshape(-1) if np.size(d_value) else np.zeros(5)
    size = None
    if "image_width" in data and "image_height" in data:
        size = (int(data["image_width"]), int(data["image_height"]))
    elif "image_size" in data:
        vals = list(data["image_size"])
        if len(vals) >= 2:
            size = (int(vals[0]), int(vals[1]))
    return k, d, size


def format_pattern(pattern: str, image_id: Any) -> str:
    try:
        return pattern.format(id=image_id)
    except (ValueError, TypeError):
        return pattern.replace("{id}", str(image_id))


def robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(1.4826 * mad, 1.0e-9)


def metric_dict(values: np.ndarray, prefix: str = "") -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean_signed_mm": float("nan"),
            f"{prefix}mae_mm": float("nan"),
            f"{prefix}rmse_mm": float("nan"),
            f"{prefix}median_abs_mm": float("nan"),
            f"{prefix}p95_abs_mm": float("nan"),
            f"{prefix}max_abs_mm": float("nan"),
        }
    abs_v = np.abs(values)
    return {
        f"{prefix}count": int(values.size),
        f"{prefix}mean_signed_mm": float(np.mean(values)),
        f"{prefix}mae_mm": float(np.mean(abs_v)),
        f"{prefix}rmse_mm": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}median_abs_mm": float(np.median(abs_v)),
        f"{prefix}p95_abs_mm": float(np.percentile(abs_v, 95)),
        f"{prefix}max_abs_mm": float(np.max(abs_v)),
    }


def uniform_subsample(indices: np.ndarray, max_count: int) -> np.ndarray:
    if max_count <= 0 or indices.size <= max_count:
        return indices
    pos = np.linspace(0, indices.size - 1, max_count).round().astype(int)
    return indices[pos]


# -----------------------------------------------------------------------------
# 棋盘格与激光中心提取
# -----------------------------------------------------------------------------

@dataclass
class BoardPose:
    rvec: np.ndarray
    tvec: np.ndarray
    normal: np.ndarray
    d: float
    reprojection_rmse_px: float
    corners: np.ndarray


def make_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    obj = np.zeros((cols * rows, 3), dtype=np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, :2] *= float(square_size_mm)
    return obj


def detect_board_pose(
    image: np.ndarray,
    k: np.ndarray,
    d: np.ndarray,
    cols: int,
    rows: int,
    square_size_mm: float,
    max_rmse_px: float,
) -> BoardPose:
    gray = to_gray_float(image).astype(np.uint8)
    pattern = (cols, rows)
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags_sb)
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if ok:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0e-4),
            )
    if not ok or corners is None:
        raise RuntimeError("棋盘格角点检测失败")

    obj = make_object_points(cols, rows, square_size_mm)
    success, rvec, tvec = cv2.solvePnP(obj, corners, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("solvePnP 失败")
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(obj, corners, k, d, rvec, tvec)

    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, d)
    err = projected.reshape(-1, 2) - corners.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    if rmse > max_rmse_px:
        raise RuntimeError(f"PnP 重投影 RMSE={rmse:.4f}px，超过阈值 {max_rmse_px:.4f}px")

    rot, _ = cv2.Rodrigues(rvec)
    normal = rot[:, 2].astype(np.float64)
    normal /= np.linalg.norm(normal)
    d_plane = -float(normal @ tvec.reshape(3))
    # 统一让法向大致朝向相机，便于不同图像的符号一致。
    if normal[2] > 0:
        normal = -normal
        d_plane = -d_plane

    return BoardPose(
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        normal=normal,
        d=d_plane,
        reprojection_rmse_px=rmse,
        corners=corners.reshape(-1, 2),
    )


def board_inner_mask(shape: Tuple[int, int], corners: np.ndarray, margin_px: int) -> np.ndarray:
    h, w = shape
    hull = cv2.convexHull(np.round(corners).astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if margin_px != 0:
        size = 2 * abs(int(margin_px)) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        if margin_px > 0:
            mask = cv2.dilate(mask, kernel)
        else:
            mask = cv2.erode(mask, kernel)
    return mask > 0


def board_mask_for_pose(
    shape: Tuple[int, int],
    pose: BoardPose,
    k: np.ndarray,
    d: np.ndarray,
    board_cfg: Mapping[str, Any],
    extraction_cfg: Mapping[str, Any],
) -> np.ndarray:
    """按配置生成正式激光点提取 mask。

    ``board_inner_mask`` 的语义保持不变，仅在显式选择
    ``inner_corner_hull`` 时使用。正式 laser-plane 三联图默认使用完整
    棋盘物理边界，且不再使用像素腐蚀/膨胀。
    """
    mode = str(extraction_cfg.get("board_mask_mode", FULL_BOARD_PHYSICAL)).strip().lower()
    if mode == FULL_BOARD_PHYSICAL:
        return full_board_physical_mask(
            shape,
            pose.rvec,
            pose.tvec,
            k,
            d,
            pattern_cols=int(board_cfg["pattern_cols"]),
            pattern_rows=int(board_cfg["pattern_rows"]),
            square_size_mm=float(board_cfg["square_size_mm"]),
            inset_mm=float(extraction_cfg.get("board_mask_inset_mm", 0.0)),
        )
    if mode == "inner_corner_hull":
        return board_inner_mask(
            shape,
            pose.corners,
            margin_px=int(extraction_cfg.get("board_mask_margin_px", -2)),
        )
    raise ValueError(
        f"不支持的 extraction.board_mask_mode={mode!r}；"
        "可选 full_board_physical 或 inner_corner_hull"
    )


def positive_difference(laser: np.ndarray, background: np.ndarray) -> np.ndarray:
    a = to_gray_float(laser)
    b = to_gray_float(background)
    if a.shape != b.shape:
        raise ValueError(f"有激光/无激光图像尺寸不一致：{a.shape} vs {b.shape}")
    # 去除全局亮度漂移，避免曝光微小变化在棋盘格边缘产生大面积响应。
    delta = float(np.median(a - b))
    diff = a - b - delta
    return np.clip(diff, 0.0, None).astype(np.float32)


def steger_candidates(
    diff: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    min_intensity: float,
    min_response: float,
    max_subpixel_offset: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """二维 Hessian/Steger 亮脊线亚像素候选点。返回 x, y, response。"""
    image = diff.astype(np.float64)
    gx = gaussian_filter(image, sigma=sigma, order=(0, 1), mode="nearest")
    gy = gaussian_filter(image, sigma=sigma, order=(1, 0), mode="nearest")
    gxx = gaussian_filter(image, sigma=sigma, order=(0, 2), mode="nearest")
    gxy = gaussian_filter(image, sigma=sigma, order=(1, 1), mode="nearest")
    gyy = gaussian_filter(image, sigma=sigma, order=(2, 0), mode="nearest")

    trace = gxx + gyy
    disc = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4.0 * gxy ** 2, 0.0))
    lam = 0.5 * (trace - disc)  # 亮脊线对应更负的特征值

    nx = gxy.copy()
    ny = lam - gxx
    norm = np.hypot(nx, ny)
    fallback = norm < 1.0e-10
    nx[fallback] = 1.0
    ny[fallback] = 0.0
    norm[fallback] = 1.0
    nx /= norm
    ny /= norm

    denom = nx * (gxx * nx + gxy * ny) + ny * (gxy * nx + gyy * ny)
    numer = gx * nx + gy * ny
    t = np.zeros_like(image)
    good_denom = np.abs(denom) > 1.0e-12
    t[good_denom] = -numer[good_denom] / denom[good_denom]
    dx = t * nx
    dy = t * ny
    response = -lam

    valid = (
        mask
        & good_denom
        & (image >= min_intensity)
        & (response >= min_response)
        & (np.abs(dx) <= max_subpixel_offset)
        & (np.abs(dy) <= max_subpixel_offset)
    )
    yy, xx = np.nonzero(valid)
    return xx.astype(np.float64) + dx[valid], yy.astype(np.float64) + dy[valid], response[valid]


def select_one_per_scanline(
    x: np.ndarray,
    y: np.ndarray,
    response: np.ndarray,
    poly_degree: int,
    outlier_threshold_px: float,
    max_points: int,
    orientation: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按激光延伸方向每条扫描线保留最强候选，并沿该方向做连续性过滤。"""
    orientation = normalize_laser_orientation(orientation)
    if x.size == 0:
        return x, y, response
    scan_coordinate = x if orientation == "horizontal" else y
    scanline = np.round(scan_coordinate).astype(int)
    order = np.lexsort((-response, scanline))
    scanline_o = scanline[order]
    first = np.r_[True, scanline_o[1:] != scanline_o[:-1]]
    keep = order[first]
    x, y, response = x[keep], y[keep], response[keep]

    scan_coordinate = x if orientation == "horizontal" else y
    dependent_coordinate = y if orientation == "horizontal" else x
    order = np.argsort(scan_coordinate)
    x, y, response = x[order], y[order], response[order]
    scan_coordinate = scan_coordinate[order]
    dependent_coordinate = dependent_coordinate[order]
    if x.size >= max(poly_degree + 2, 12):
        # 两轮稳健多项式连续性过滤，只用于剔除二次反射和棋盘格边缘假响应。
        active = np.ones(x.size, dtype=bool)
        for _ in range(2):
            coef = np.polyfit(
                scan_coordinate[active],
                dependent_coordinate[active],
                deg=poly_degree,
            )
            residual = dependent_coordinate - np.polyval(coef, scan_coordinate)
            scale = robust_scale(residual[active])
            threshold = max(float(outlier_threshold_px), 3.5 * scale)
            active = np.abs(residual) <= threshold
            if np.count_nonzero(active) < poly_degree + 2:
                break
        x, y, response = x[active], y[active], response[active]

    idx = uniform_subsample(np.arange(x.size), max_points)
    return x[idx], y[idx], response[idx]


def select_one_per_column(
    x: np.ndarray,
    y: np.ndarray,
    response: np.ndarray,
    poly_degree: int,
    outlier_threshold_px: float,
    max_points: int,
    orientation: str = "horizontal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """旧函数名保留兼容；orientation 可切换为按 row 处理 vertical 激光线。"""
    return select_one_per_scanline(
        x,
        y,
        response,
        poly_degree,
        outlier_threshold_px,
        max_points,
        orientation=orientation,
    )


def extract_laser_centers(
    laser: np.ndarray,
    background: np.ndarray,
    board_mask: np.ndarray,
    cfg: Dict[str, Any],
    orientation: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diff = positive_difference(laser, background)
    method = str(cfg.get("method", "steger")).lower()
    if method != "steger":
        raise ValueError("当前脚本仅实现 method: steger")
    x, y, response = steger_candidates(
        diff=diff,
        mask=board_mask,
        sigma=float(cfg.get("sigma", 1.5)),
        min_intensity=float(cfg.get("min_intensity", 8.0)),
        min_response=float(cfg.get("min_response", 0.8)),
        max_subpixel_offset=float(cfg.get("max_subpixel_offset", 0.6)),
    )
    x, y, response = select_one_per_scanline(
        x,
        y,
        response,
        poly_degree=int(cfg.get("continuity_poly_degree", 2)),
        outlier_threshold_px=float(cfg.get("continuity_threshold_px", 2.0)),
        max_points=int(cfg.get("max_points_per_image", 900)),
        orientation=orientation,
    )
    return x, y, response, diff


def pixels_to_rays(u: np.ndarray, v: np.ndarray, k: np.ndarray, d: np.ndarray) -> np.ndarray:
    pts = np.stack([u, v], axis=1).reshape(-1, 1, 2).astype(np.float64)
    normalized = cv2.undistortPoints(pts, k, d).reshape(-1, 2)
    return np.column_stack([normalized, np.ones(normalized.shape[0])])


def intersect_rays_with_plane(rays: np.ndarray, normal: np.ndarray, d: float) -> Tuple[np.ndarray, np.ndarray]:
    denom = rays @ normal
    lam = np.full(rays.shape[0], np.nan, dtype=np.float64)
    valid = np.abs(denom) > 1.0e-12
    lam[valid] = -float(d) / denom[valid]
    valid &= lam > 0
    points = rays * lam[:, None]
    points[~valid] = np.nan
    return points, valid


def draw_preview(
    chess: np.ndarray,
    pose: BoardPose,
    u: np.ndarray,
    v: np.ndarray,
    output: Path,
    title: str,
) -> None:
    if chess.ndim == 2:
        vis = cv2.cvtColor(chess, cv2.COLOR_GRAY2BGR)
    else:
        vis = chess.copy()
    for px, py in pose.corners:
        cv2.circle(vis, (int(round(px)), int(round(py))), 2, (0, 255, 0), -1)
    for px, py in zip(u, v):
        cv2.circle(vis, (int(round(px)), int(round(py))), 1, (0, 0, 255), -1)
    cv2.putText(vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    imwrite_unicode(output, vis)


# -----------------------------------------------------------------------------
# 三种表面模型
# -----------------------------------------------------------------------------

class LaserModel:
    name: str

    def fit(self, points: np.ndarray, frame_ids: np.ndarray) -> None:
        raise NotImplementedError

    def surface_distance(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def intersect_rays(self, rays: np.ndarray, lambda_hint: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


def frame_equal_weights(frame_ids: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(frame_ids, return_counts=True)
    lookup = {key: count for key, count in zip(ids, counts)}
    w = np.asarray([1.0 / lookup[key] for key in frame_ids], dtype=np.float64)
    return w / np.mean(w)


class PlaneModel(LaserModel):
    name = "global_plane"

    def __init__(self) -> None:
        self.normal = np.array([0.0, 1.0, 0.0])
        self.d = 0.0
        self.z_range = (0.0, np.inf)

    def fit(self, points: np.ndarray, frame_ids: np.ndarray) -> None:
        base = frame_equal_weights(frame_ids)
        robust = np.ones(points.shape[0])
        for _ in range(8):
            w = base * robust
            centroid = np.sum(points * w[:, None], axis=0) / np.sum(w)
            centered = points - centroid
            _, _, vh = np.linalg.svd(centered * np.sqrt(w[:, None]), full_matrices=False)
            normal = vh[-1]
            normal /= np.linalg.norm(normal)
            d = -float(normal @ centroid)
            residual = points @ normal + d
            scale = robust_scale(residual)
            c = 1.5 * scale
            robust = np.minimum(1.0, c / np.maximum(np.abs(residual), EPS))
        if normal[1] < 0:
            normal, d = -normal, -d
        self.normal, self.d = normal, d
        self.z_range = (float(np.min(points[:, 2])), float(np.max(points[:, 2])))

    def surface_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.d

    def intersect_rays(self, rays: np.ndarray, lambda_hint: Optional[np.ndarray] = None) -> np.ndarray:
        denom = rays @ self.normal
        lam = np.full(rays.shape[0], np.nan)
        valid = np.abs(denom) > 1.0e-12
        lam[valid] = -self.d / denom[valid]
        lam[(lam <= 0) | ~np.isfinite(lam)] = np.nan
        return lam

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.name,
            "equation": "a*X + b*Y + c*Z + d = 0",
            "normal": self.normal.tolist(),
            "d_mm": float(self.d),
            # Legacy consumers can still read a four-coefficient plane.
            "plane_abcd": [*self.normal.tolist(), float(self.d)],
            "z_valid_range_mm": list(self.z_range),
        }


class QuadraticGraphModel(LaserModel):
    name = "quadratic_graph"

    def __init__(self, ridge: float = 1.0e-10) -> None:
        self.dep_axis = 1
        self.ind_axes = (0, 2)
        self.center = np.zeros(2)
        self.scale = np.ones(2)
        self.beta = np.zeros(6)
        self.ridge = ridge
        self.z_range = (0.0, np.inf)
        self.plane_hint: Optional[PlaneModel] = None

    @staticmethod
    def design(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(p), p, q, p * p, p * q, q * q])

    def fit(self, points: np.ndarray, frame_ids: np.ndarray, plane: Optional[PlaneModel] = None) -> None:
        if plane is None:
            plane = PlaneModel()
            plane.fit(points, frame_ids)
        self.plane_hint = plane
        self.dep_axis = int(np.argmax(np.abs(plane.normal)))
        self.ind_axes = tuple(axis for axis in range(3) if axis != self.dep_axis)  # type: ignore[assignment]
        p_raw = points[:, self.ind_axes[0]]
        q_raw = points[:, self.ind_axes[1]]
        dep = points[:, self.dep_axis]
        self.center = np.array([np.median(p_raw), np.median(q_raw)])
        self.scale = np.array([
            max(float(np.std(p_raw)), 1.0),
            max(float(np.std(q_raw)), 1.0),
        ])
        p = (p_raw - self.center[0]) / self.scale[0]
        q = (q_raw - self.center[1]) / self.scale[1]
        a = self.design(p, q)
        base = frame_equal_weights(frame_ids)
        robust = np.ones(points.shape[0])
        beta = np.zeros(6)
        for _ in range(10):
            w = base * robust
            aw = a * np.sqrt(w[:, None])
            yw = dep * np.sqrt(w)
            lhs = aw.T @ aw + self.ridge * np.eye(a.shape[1])
            rhs = aw.T @ yw
            beta = np.linalg.solve(lhs, rhs)
            residual = dep - a @ beta
            scale = robust_scale(residual)
            c = 1.5 * scale
            robust = np.minimum(1.0, c / np.maximum(np.abs(residual), EPS))
        self.beta = beta
        self.z_range = (float(np.min(points[:, 2])), float(np.max(points[:, 2])))

    def _f_and_grad(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_raw = points[:, self.ind_axes[0]]
        q_raw = points[:, self.ind_axes[1]]
        dep = points[:, self.dep_axis]
        p = (p_raw - self.center[0]) / self.scale[0]
        q = (q_raw - self.center[1]) / self.scale[1]
        b = self.beta
        pred = self.design(p, q) @ b
        f = dep - pred
        d_pred_dp_raw = (b[1] + 2.0 * b[3] * p + b[4] * q) / self.scale[0]
        d_pred_dq_raw = (b[2] + b[4] * p + 2.0 * b[5] * q) / self.scale[1]
        grad = np.zeros_like(points)
        grad[:, self.dep_axis] = 1.0
        grad[:, self.ind_axes[0]] = -d_pred_dp_raw
        grad[:, self.ind_axes[1]] = -d_pred_dq_raw
        return f, grad

    def surface_distance(self, points: np.ndarray) -> np.ndarray:
        f, grad = self._f_and_grad(points)
        return f / np.maximum(np.linalg.norm(grad, axis=1), EPS)

    def intersect_rays(self, rays: np.ndarray, lambda_hint: Optional[np.ndarray] = None) -> np.ndarray:
        rp = rays[:, self.ind_axes[0]]
        rq = rays[:, self.ind_axes[1]]
        rd = rays[:, self.dep_axis]
        ap = rp / self.scale[0]
        aq = rq / self.scale[1]
        bp = -self.center[0] / self.scale[0]
        bq = -self.center[1] / self.scale[1]
        b = self.beta

        quad = b[3] * ap * ap + b[4] * ap * aq + b[5] * aq * aq
        linear_rhs = (
            b[1] * ap
            + b[2] * aq
            + 2.0 * b[3] * ap * bp
            + b[4] * (ap * bq + aq * bp)
            + 2.0 * b[5] * aq * bq
        )
        const_rhs = b[0] + b[1] * bp + b[2] * bq + b[3] * bp * bp + b[4] * bp * bq + b[5] * bq * bq
        # lambda*rd - rhs(lambda) = 0
        aa = -quad
        bb = rd - linear_rhs
        cc = -const_rhs
        lam = solve_quadratic_roots(aa, bb, cc, lambda_hint, self.z_range)
        return lam

    def to_dict(self) -> Dict[str, Any]:
        axis_name = ["X", "Y", "Z"]
        return {
            "model_type": self.name,
            "dependent_axis": axis_name[self.dep_axis],
            "independent_axes": [axis_name[i] for i in self.ind_axes],
            "equation": "D = beta0 + beta1*p + beta2*q + beta3*p^2 + beta4*p*q + beta5*q^2",
            "normalization": {
                "independent_center_mm": self.center.tolist(),
                "independent_scale_mm": self.scale.tolist(),
            },
            "coefficients": self.beta.tolist(),
            "z_valid_range_mm": list(self.z_range),
        }


def vector_to_angles(axis: np.ndarray) -> Tuple[float, float]:
    a = axis / np.linalg.norm(axis)
    theta = math.acos(float(np.clip(a[2], -1.0, 1.0)))
    phi = math.atan2(float(a[1]), float(a[0]))
    return theta, phi


def angles_to_vector(theta: float, phi: float) -> np.ndarray:
    return np.array([
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ], dtype=np.float64)


class CircularConeModel(LaserModel):
    """论文标准圆锥二次曲面的相机坐标一般化参数形式。"""

    name = "circular_cone"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.axis = np.array([0.0, 1.0, 0.0])
        self.apex = np.zeros(3)
        self.alpha_deg = 89.0
        self.sheet_sign = 1.0
        self.z_range = (0.0, np.inf)
        self.cfg = cfg
        self.fit_success = False
        self.cost = float("nan")

    def _residual(self, params: np.ndarray, points: np.ndarray, sqrt_w: np.ndarray) -> np.ndarray:
        theta, phi, cx, cy, cz, alpha = params
        axis = angles_to_vector(theta, phi)
        apex = np.array([cx, cy, cz])
        q = points - apex
        axial = q @ axis
        radial2 = np.sum(q * q, axis=1) - axial * axial
        radial = np.sqrt(np.maximum(radial2, 0.0))
        tan_alpha = math.tan(alpha)
        residual = radial / max(tan_alpha, 1.0e-9) - axial
        # 论文使用单叶物理光面；负轴向点增加软惩罚，避免拟合到另一叶。
        penalty_weight = float(self.cfg.get("negative_axial_penalty", 2.0))
        penalty = penalty_weight * np.minimum(axial, 0.0)
        return np.concatenate([sqrt_w * residual, sqrt_w * penalty])

    def fit(self, points: np.ndarray, frame_ids: np.ndarray, plane: Optional[PlaneModel] = None) -> None:
        if plane is None:
            plane = PlaneModel()
            plane.fit(points, frame_ids)
        self.z_range = (float(np.min(points[:, 2])), float(np.max(points[:, 2])))

        max_points = int(self.cfg.get("fit_max_points", 3000))
        chosen: List[int] = []
        for fid in np.unique(frame_ids):
            idx = np.flatnonzero(frame_ids == fid)
            chosen.extend(uniform_subsample(idx, max(20, max_points // max(len(np.unique(frame_ids)), 1))).tolist())
        chosen_arr = np.asarray(chosen, dtype=int)
        if chosen_arr.size > max_points:
            chosen_arr = uniform_subsample(chosen_arr, max_points)
        pts = points[chosen_arr]
        fids = frame_ids[chosen_arr]
        w = frame_equal_weights(fids)
        sqrt_w = np.sqrt(w)

        axis0 = plane.normal.copy()
        apex_plane = -plane.d * plane.normal  # 原点到平面的垂足，通常接近激光器所在光面
        centroid = np.mean(pts, axis=0)

        alpha_min = math.radians(float(self.cfg.get("alpha_min_deg", 60.0)))
        alpha_max = math.radians(float(self.cfg.get("alpha_max_deg", 89.95)))
        bounds_cfg = self.cfg.get("apex_bounds_mm", [[-1000, -1000, -500], [1000, 1000, 500]])
        lower_apex = np.asarray(bounds_cfg[0], dtype=float)
        upper_apex = np.asarray(bounds_cfg[1], dtype=float)
        lower = np.array([0.0, -math.pi, *lower_apex, alpha_min], dtype=float)
        upper = np.array([math.pi, math.pi, *upper_apex, alpha_max], dtype=float)

        explicit_guess = self.cfg.get("apex_initial_mm")
        apex_guesses = [apex_plane, np.zeros(3)]
        if explicit_guess is not None:
            apex_guesses.insert(0, np.asarray(explicit_guess, dtype=float))
        # 额外给出少量可解释多起点，防止近 90° 圆锥退化导致局部极小。
        _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        tangent = vh[0]
        apex_guesses += [apex_plane + 100.0 * tangent, apex_plane - 100.0 * tangent]
        alpha_guesses = [
            float(self.cfg.get("alpha_initial_deg", 89.0)),
            85.0,
            89.8,
        ]

        best = None
        for axis_init in (axis0, -axis0):
            theta0, phi0 = vector_to_angles(axis_init)
            for apex0 in apex_guesses:
                apex0 = np.clip(apex0, lower_apex + 1.0e-6, upper_apex - 1.0e-6)
                for alpha_deg in alpha_guesses:
                    x0 = np.array([theta0, phi0, *apex0, math.radians(alpha_deg)], dtype=float)
                    try:
                        result = least_squares(
                            self._residual,
                            x0,
                            args=(pts, sqrt_w),
                            bounds=(lower, upper),
                            loss=str(self.cfg.get("loss", "soft_l1")),
                            f_scale=float(self.cfg.get("f_scale_mm", 0.1)),
                            max_nfev=int(self.cfg.get("max_nfev", 3000)),
                            verbose=0,
                        )
                    except Exception:
                        continue
                    if not result.success and result.cost <= 0:
                        continue
                    if best is None or result.cost < best.cost:
                        best = result

        if best is None:
            raise RuntimeError("圆锥模型优化失败：所有初值均未收敛")
        theta, phi, cx, cy, cz, alpha = best.x
        axis = angles_to_vector(theta, phi)
        apex = np.array([cx, cy, cz])
        axial = (pts - apex) @ axis
        if np.median(axial) < 0:
            axis = -axis
            axial = -axial
        self.axis = axis
        self.apex = apex
        self.alpha_deg = math.degrees(alpha)
        self.sheet_sign = 1.0 if np.median(axial) >= 0 else -1.0
        self.fit_success = bool(best.success)
        self.cost = float(best.cost)

    def surface_distance(self, points: np.ndarray) -> np.ndarray:
        q = points - self.apex
        axial = q @ self.axis
        cos2 = math.cos(math.radians(self.alpha_deg)) ** 2
        f = axial * axial - cos2 * np.sum(q * q, axis=1)
        grad = 2.0 * (axial[:, None] * self.axis[None, :] - cos2 * q)
        dist = f / np.maximum(np.linalg.norm(grad, axis=1), EPS)
        return dist

    def intersect_rays(self, rays: np.ndarray, lambda_hint: Optional[np.ndarray] = None) -> np.ndarray:
        a = self.axis
        c = self.apex
        cos2 = math.cos(math.radians(self.alpha_deg)) ** 2
        ra = rays @ a
        ca = float(c @ a)
        aa = ra * ra - cos2 * np.sum(rays * rays, axis=1)
        bb = -2.0 * ra * ca + 2.0 * cos2 * (rays @ c)
        cc = ca * ca - cos2 * float(c @ c)
        roots = solve_quadratic_all(aa, bb, cc)
        lam = choose_roots(roots, rays, self.apex, self.axis, lambda_hint, self.z_range)
        return lam

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.name,
            "description": "general circular cone equivalent to a standard cone after rigid transform",
            "axis_unit_camera": self.axis.tolist(),
            "apex_camera_mm": self.apex.tolist(),
            "half_apex_angle_deg": float(self.alpha_deg),
            "fit_success": bool(self.fit_success),
            "optimizer_cost": float(self.cost),
            "z_valid_range_mm": list(self.z_range),
        }


def solve_quadratic_all(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """返回每行两个实根；无实根用 NaN。"""
    a, b, c = np.broadcast_arrays(a, b, c)
    roots = np.full((a.size, 2), np.nan, dtype=np.float64)
    af, bf, cf = a.ravel(), b.ravel(), c.ravel()
    linear = np.abs(af) < 1.0e-12
    valid_linear = linear & (np.abs(bf) > 1.0e-12)
    roots[valid_linear, 0] = -cf[valid_linear] / bf[valid_linear]
    quad = ~linear
    disc = bf * bf - 4.0 * af * cf
    valid_quad = quad & (disc >= 0)
    sd = np.sqrt(np.maximum(disc[valid_quad], 0.0))
    # 数值稳定形式
    q = -0.5 * (bf[valid_quad] + np.sign(bf[valid_quad] + EPS) * sd)
    r1 = q / af[valid_quad]
    r2 = np.where(np.abs(q) > EPS, cf[valid_quad] / q, (-bf[valid_quad] - sd) / (2.0 * af[valid_quad]))
    roots[valid_quad, 0] = r1
    roots[valid_quad, 1] = r2
    return roots


def choose_roots(
    roots: np.ndarray,
    rays: np.ndarray,
    apex: Optional[np.ndarray],
    axis: Optional[np.ndarray],
    lambda_hint: Optional[np.ndarray],
    z_range: Tuple[float, float],
) -> np.ndarray:
    z_min, z_max = z_range
    margin = max(50.0, 0.25 * max(z_max - z_min, 1.0))
    lo, hi = max(1.0e-6, z_min - margin), z_max + margin
    out = np.full(roots.shape[0], np.nan)
    if lambda_hint is None:
        lambda_hint = np.full(roots.shape[0], 0.5 * (z_min + z_max))
    for i in range(roots.shape[0]):
        candidates = roots[i]
        valid = np.isfinite(candidates) & (candidates > 0) & (candidates >= lo) & (candidates <= hi)
        candidates = candidates[valid]
        if candidates.size == 0:
            continue
        if apex is not None and axis is not None:
            pts = candidates[:, None] * rays[i][None, :]
            axial = (pts - apex[None, :]) @ axis
            forward = axial >= 0
            if np.any(forward):
                candidates = candidates[forward]
        out[i] = candidates[np.argmin(np.abs(candidates - lambda_hint[i]))]
    return out


def solve_quadratic_roots(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    lambda_hint: Optional[np.ndarray],
    z_range: Tuple[float, float],
) -> np.ndarray:
    roots = solve_quadratic_all(a, b, c)
    dummy_rays = np.column_stack([np.zeros(a.size), np.zeros(a.size), np.ones(a.size)])
    return choose_roots(roots, dummy_rays, None, None, lambda_hint, z_range)


# -----------------------------------------------------------------------------
# 数据集、评估和输出
# -----------------------------------------------------------------------------


def process_dataset(
    split_name: str,
    dataset_cfg: Dict[str, Any],
    patterns: Dict[str, str],
    k: np.ndarray,
    d: np.ndarray,
    image_size: Optional[Tuple[int, int]],
    board_cfg: Dict[str, Any],
    extraction_cfg: Dict[str, Any],
    laser_orientation: str,
    preview_dir: Path,
) -> pd.DataFrame:
    root = Path(dataset_cfg["root"])
    ids = dataset_cfg.get("ids", [])
    if not ids:
        raise ValueError(f"datasets.{split_name}.ids 为空")
    records: List[Dict[str, Any]] = []
    failures: List[str] = []
    for image_id in ids:
        frame_key = f"{split_name}:{image_id}"
        chess_path = root / format_pattern(patterns["chess"], image_id)
        bg_path = root / format_pattern(patterns["background"], image_id)
        laser_path = root / format_pattern(patterns["laser"], image_id)
        try:
            chess = imread_unicode(chess_path)
            background = imread_unicode(bg_path)
            laser = imread_unicode(laser_path)
            h, w = to_gray_float(chess).shape
            if image_size is not None and (w, h) != image_size:
                raise ValueError(f"图像尺寸 {(w, h)} 与内参尺寸 {image_size} 不一致")
            pose = detect_board_pose(
                chess,
                k,
                d,
                cols=int(board_cfg["pattern_cols"]),
                rows=int(board_cfg["pattern_rows"]),
                square_size_mm=float(board_cfg["square_size_mm"]),
                max_rmse_px=float(board_cfg.get("max_pnp_rmse_px", 0.4)),
            )
            mask = board_mask_for_pose(
                (h, w),
                pose,
                k,
                d,
                board_cfg,
                extraction_cfg,
            )
            u, v, response, diff = extract_laser_centers(
                laser,
                background,
                mask,
                extraction_cfg,
                laser_orientation,
            )
            min_points = int(extraction_cfg.get("min_points_per_image", 80))
            if u.size < min_points:
                raise RuntimeError(f"有效激光中心点仅 {u.size}，小于 {min_points}")
            rays = pixels_to_rays(u, v, k, d)
            points, valid = intersect_rays_with_plane(rays, pose.normal, pose.d)
            u, v, response, rays, points = u[valid], v[valid], response[valid], rays[valid], points[valid]
            if points.shape[0] < min_points:
                raise RuntimeError(f"射线-棋盘格平面有效交点仅 {points.shape[0]}")

            for idx in range(points.shape[0]):
                records.append({
                    "split": split_name,
                    "image_id": str(image_id),
                    "frame_key": frame_key,
                    "u_px": float(u[idx]),
                    "v_px": float(v[idx]),
                    "response": float(response[idx]),
                    "ray_x": float(rays[idx, 0]),
                    "ray_y": float(rays[idx, 1]),
                    "ray_z": float(rays[idx, 2]),
                    "Xc_mm": float(points[idx, 0]),
                    "Yc_mm": float(points[idx, 1]),
                    "Zc_mm": float(points[idx, 2]),
                    "board_nx": float(pose.normal[0]),
                    "board_ny": float(pose.normal[1]),
                    "board_nz": float(pose.normal[2]),
                    "board_d_mm": float(pose.d),
                    "pnp_rmse_px": float(pose.reprojection_rmse_px),
                })
            draw_preview(
                chess,
                pose,
                u,
                v,
                preview_dir / f"{split_name}_{image_id}_extraction.png",
                f"{frame_key}: {len(u)} pts, PnP {pose.reprojection_rmse_px:.3f}px",
            )
            print(f"[OK] {frame_key}: {len(u)} points, PnP RMSE={pose.reprojection_rmse_px:.4f}px")
        except Exception as exc:
            message = f"{frame_key}: {exc}"
            failures.append(message)
            print(f"[FAIL] {message}", file=sys.stderr)
    if failures:
        (preview_dir.parent / f"{split_name}_failures.txt").write_text("\n".join(failures), encoding="utf-8")
    if not records:
        raise RuntimeError(f"{split_name} 没有得到任何有效标定点")
    return pd.DataFrame.from_records(records)


def dataframe_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = df[["Xc_mm", "Yc_mm", "Zc_mm"]].to_numpy(dtype=float)
    rays = df[["ray_x", "ray_y", "ray_z"]].to_numpy(dtype=float)
    frame_ids = df["frame_key"].astype(str).to_numpy()
    return points, rays, frame_ids


def evaluate_model(
    model: LaserModel,
    df: pd.DataFrame,
    plane_hint: PlaneModel,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    points, rays, frame_ids = dataframe_arrays(df)
    lambda_hint = plane_hint.intersect_rays(rays)
    lam = model.intersect_rays(rays, lambda_hint=lambda_hint)
    pred = rays * lam[:, None]
    normals = df[["board_nx", "board_ny", "board_nz"]].to_numpy(dtype=float)
    d_board = df["board_d_mm"].to_numpy(dtype=float)
    board_error = np.sum(pred * normals, axis=1) + d_board
    gt_lambda = points[:, 2] / np.maximum(rays[:, 2], EPS)
    ray_error = (lam - gt_lambda) * np.linalg.norm(rays, axis=1)
    surface_error = model.surface_distance(points)
    valid = np.isfinite(lam) & np.all(np.isfinite(pred), axis=1)
    board_error[~valid] = np.nan
    ray_error[~valid] = np.nan

    metrics: Dict[str, Any] = {
        "model": model.name,
        "split": str(df["split"].iloc[0]),
        "total_points": int(len(df)),
        "valid_intersections": int(np.count_nonzero(valid)),
        "valid_rate": float(np.mean(valid)),
    }
    metrics.update(metric_dict(surface_error, "surface_"))
    metrics.update(metric_dict(board_error, "board_"))
    metrics.update(metric_dict(ray_error, "ray_"))

    detail = df[["split", "image_id", "frame_key", "u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm"]].copy()
    detail["model"] = model.name
    detail["lambda_pred_mm"] = lam
    detail["board_error_mm"] = board_error
    detail["ray_euclidean_error_mm"] = ray_error
    detail["surface_distance_mm"] = surface_error
    detail["valid"] = valid
    return metrics, detail


def per_image_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split, image_id), group in detail.groupby(["model", "split", "image_id"]):
        values = group["board_error_mm"].to_numpy(dtype=float)
        row: Dict[str, Any] = {"model": model, "split": split, "image_id": image_id}
        row.update(metric_dict(values))
        row["valid_rate"] = float(np.mean(group["valid"].to_numpy(dtype=bool)))
        rows.append(row)
    return pd.DataFrame(rows)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def plot_error_vs_variable(detail: pd.DataFrame, variable: str, xlabel: str, output: Path) -> None:
    plt.figure(figsize=(10, 6))
    for model, group in detail.groupby("model"):
        clean = group[[variable, "board_error_mm"]].dropna().sort_values(variable)
        if clean.empty:
            continue
        # 散点只抽样，叠加分箱中位数，兼顾可读性。
        sample = clean.iloc[:: max(1, len(clean) // 3000)]
        plt.scatter(sample[variable], sample["board_error_mm"], s=3, alpha=0.18, label=f"{model} points")
        bins = pd.qcut(clean[variable], q=min(40, max(5, len(clean) // 50)), duplicates="drop")
        med = clean.groupby(bins, observed=True).median(numeric_only=True)
        plt.plot(med[variable], med["board_error_mm"], linewidth=2, label=f"{model} median")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("重建点到真实棋盘格平面的有符号距离 / mm")
    plt.title("独立验证集：模型重建误差分布")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def generate_report(
    output: Path,
    comparison: pd.DataFrame,
    per_image: pd.DataFrame,
    models: Sequence[LaserModel],
    train_count: int,
    val_count: int,
    default_model: str,
) -> None:
    val = comparison[comparison["split"] == "validation"].copy()
    if not val.empty:
        val = val.sort_values("board_rmse_mm")
    lines = [
        "# 激光表面三模型拟合比较报告",
        "",
        f"- 训练标定点：{train_count}",
        f"- 验证标定点：{val_count}",
        f"- 默认选用模型：`{default_model}`",
        f"- 支持模型：{', '.join(SUPPORTED_LASER_MODEL_TYPES)}",
        "- 核心评价量：使用模型从激光像素反算三维点，再计算该点到该幅棋盘格真实平面的有符号距离。",
        "- 数据划分：按完整图像姿态划分训练/验证，避免同一图像相邻点泄漏。",
        "",
        "## 独立验证集汇总",
        "",
    ]
    if val.empty:
        lines.append("未配置独立验证集。")
    else:
        cols = [
            "model", "valid_rate", "board_mean_signed_mm", "board_mae_mm",
            "board_rmse_mm", "board_p95_abs_mm", "board_max_abs_mm",
            "surface_rmse_mm",
        ]
        lines.append(val[cols].to_markdown(index=False, floatfmt=".6f"))
    lines += [
        "",
        "## 判读规则",
        "",
        "1. 优先比较 validation 的 `board_rmse_mm`、`board_p95_abs_mm` 与残差随 u、Z 的趋势。",
        "2. 仅训练表面距离很小而验证棋盘格误差不降，属于过拟合或模型求交不稳定。",
        "3. 二次图曲面若明显优于平面，说明全视场确有稳定二次弯曲；若圆锥模型同时更优，论文的物理圆锥假设更适合当前光片。",
        "4. 圆锥半顶角逼近上界、顶点落在边界或有效求交率低，说明圆锥参数退化，不应直接用于测量。",
        "5. 所有模型都存在随深度单调变化的误差时，应优先检查棋盘格尺寸、相机内参深度尺度和三联图之间的位姿同步。",
        "",
        "## 输出模型参数",
        "",
    ]
    for model in models:
        lines += [f"### {model.name}", "", "```yaml", yaml.safe_dump(model.to_dict(), allow_unicode=True, sort_keys=False).strip(), "```", ""]
    if not per_image.empty:
        lines += ["## 每幅验证图像 RMSE", ""]
        piv = per_image[per_image["split"] == "validation"].pivot(index="image_id", columns="model", values="rmse_mm")
        lines.append(piv.to_markdown(floatfmt=".6f"))
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="基于激光三联图拟合并比较三种激光表面模型")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument(
        "--model",
        "--laser-model",
        dest="model",
        default=None,
        help=(
            "输出目录中的默认模型；支持 global_plane、quadratic_graph、"
            "circular_cone，旧别名 plane_abcd 仍可用"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="覆盖配置中的输出目录（供统一 workflow stage 使用）",
    )
    parser.add_argument(
        "--laser-orientation",
        choices=("horizontal", "vertical"),
        default=None,
        help="覆盖配置中的 laser.orientation（workflow 从项目配置显式传入）",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    cfg = safe_yaml_load(config_path)
    laser_cfg = parse_laser_config(cfg.get("laser"))
    laser_orientation = normalize_laser_orientation(
        args.laser_orientation or laser_cfg.orientation
    )
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/laser_model_comparison"))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_path = Path(cfg["intrinsics"])
    if not intrinsics_path.is_absolute():
        intrinsics_path = (config_path.parent / intrinsics_path).resolve()
    k, dist, image_size = load_intrinsics(intrinsics_path)
    print("Camera K:\n", k)
    print("Distortion:", dist)

    patterns = cfg.get("patterns", {})
    for key in ("chess", "background", "laser"):
        if key not in patterns:
            raise KeyError(f"patterns.{key} 未配置")
    board_cfg = cfg["board"]
    extraction_cfg = cfg.get("extraction", {})
    datasets_cfg = cfg["datasets"]
    print(
        "Board mask:",
        str(extraction_cfg.get("board_mask_mode", FULL_BOARD_PHYSICAL)),
        "inset_mm=",
        float(extraction_cfg.get("board_mask_inset_mm", 0.0)),
    )

    def resolve_dataset(dataset: Mapping[str, Any]) -> Dict[str, Any]:
        resolved = dict(dataset)
        root = Path(str(resolved["root"]))
        if not root.is_absolute():
            root = (config_path.parent / root).resolve()
        resolved["root"] = str(root)
        return resolved

    train_df = process_dataset(
        "train", resolve_dataset(datasets_cfg["train"]), patterns, k, dist, image_size,
        board_cfg, extraction_cfg, laser_orientation, preview_dir,
    )
    frames = [train_df]
    if "validation" in datasets_cfg and datasets_cfg["validation"].get("ids"):
        validation_df = process_dataset(
            "validation", resolve_dataset(datasets_cfg["validation"]), patterns, k, dist, image_size,
            board_cfg, extraction_cfg, laser_orientation, preview_dir,
        )
        frames.append(validation_df)
    else:
        validation_df = pd.DataFrame(columns=train_df.columns)
        print("[WARN] 未配置独立验证集；模型比较可信度会明显下降。")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(output_dir / "calibration_points.csv", index=False, encoding="utf-8-sig")

    train_points, _, train_frame_ids = dataframe_arrays(train_df)
    try:
        default_model = normalize_model_type(args.model or select_default_model(cfg))
    except LaserModelConfigError as exc:
        parser.error(str(exc))
    plane = PlaneModel()
    plane.fit(train_points, train_frame_ids)

    quadratic = QuadraticGraphModel(ridge=float(cfg.get("models", {}).get("quadratic", {}).get("ridge", 1.0e-10)))
    quadratic.fit(train_points, train_frame_ids, plane=plane)

    cone = CircularConeModel(cfg.get("models", {}).get("cone", {}))
    cone.fit(train_points, train_frame_ids, plane=plane)

    models: List[LaserModel] = [plane, quadratic, cone]
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for model in models:
        save_yaml(model_dir / f"{model.name}.yaml", model.to_dict())

    comparison_rows: List[Dict[str, Any]] = []
    detail_frames: List[pd.DataFrame] = []
    for split_df in [train_df, validation_df]:
        if split_df.empty:
            continue
        for model in models:
            metrics, detail = evaluate_model(model, split_df, plane)
            comparison_rows.append(metrics)
            detail_frames.append(detail)
            print(
                f"[{metrics['split']}] {model.name}: "
                f"board RMSE={metrics['board_rmse_mm']:.6f} mm, "
                f"P95={metrics['board_p95_abs_mm']:.6f} mm, "
                f"valid={metrics['valid_rate']:.3f}"
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    details = pd.concat(detail_frames, ignore_index=True)
    details.to_csv(output_dir / "pointwise_model_errors.csv", index=False, encoding="utf-8-sig")
    per_img = per_image_metrics(details)
    per_img.to_csv(output_dir / "per_image_metrics.csv", index=False, encoding="utf-8-sig")

    selected = next(model for model in models if model.name == default_model)
    selected_metrics: Dict[str, Any] = {}
    for split in ("train", "validation"):
        rows = comparison[
            (comparison["model"] == default_model) & (comparison["split"] == split)
        ]
        if not rows.empty:
            selected_metrics[split] = rows.iloc[0].to_dict()
    selected_document = selected.to_dict()
    selected_document["laser"] = {"orientation": laser_orientation}
    selected_document["model_selection"] = {
        "default_model": default_model,
        "supported_models": list(SUPPORTED_LASER_MODEL_TYPES),
        "source": str(config_path),
    }
    selected_document["metrics"] = selected_metrics
    save_yaml(output_dir / "laser_model.yaml", selected_document)
    # ``laser_plane.yaml`` is the historical artifact name.  Its contents are
    # now the selected model document, while old plane_abcd files remain
    # readable through the compatibility loader.
    save_yaml(output_dir / "laser_plane.yaml", selected_document)
    save_yaml(
        output_dir / "laser_model_selection.yaml",
        {
            "model_type": default_model,
            "supported_models": list(SUPPORTED_LASER_MODEL_TYPES),
            "model_file": "laser_model.yaml",
            "legacy_model_file": "laser_plane.yaml",
            "laser": {"orientation": laser_orientation},
        },
    )

    validation_details = details[details["split"] == "validation"]
    if not validation_details.empty:
        plot_error_vs_variable(validation_details, "u_px", "图像横坐标 u / px", output_dir / "validation_error_vs_u.png")
        plot_error_vs_variable(validation_details, "v_px", "图像纵坐标 v / px", output_dir / "validation_error_vs_v.png")
        plot_error_vs_variable(validation_details, "Zc_mm", "真实棋盘格点深度 Zc / mm", output_dir / "validation_error_vs_depth.png")

    generate_report(
        output_dir / "comparison_report.md",
        comparison,
        per_img,
        models,
        train_count=len(train_df),
        val_count=len(validation_df),
        default_model=default_model,
    )
    print(f"\n完成。输出目录：{output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
