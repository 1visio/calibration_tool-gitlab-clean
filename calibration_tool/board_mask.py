"""棋盘格物理边界投影与图像 mask 工具。

该模块只负责几何构造，不参与棋盘检测、Steger 提取或激光模型拟合。
对于 ``pattern_cols x pattern_rows`` 个内角点，完整棋盘包含外侧各一格，
因此默认物理边界为：

``X=[-square_size_mm, pattern_cols*square_size_mm]``
``Y=[-square_size_mm, pattern_rows*square_size_mm]``

例如 11x8、20 mm 棋盘对应 ``[-20, 220] x [-20, 160]`` mm。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


FULL_BOARD_PHYSICAL = "full_board_physical"


def full_board_object_corners(
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    inset_mm: float = 0.0,
) -> np.ndarray:
    """返回完整棋盘四角的物理坐标，顺序为左上、右上、右下、左下。

    ``pattern_cols``/``pattern_rows`` 是 OpenCV 内角点数，不是方格数。
    ``inset_mm`` 只允许向棋盘内部收缩；默认 0 表示完整 12x9 方格边界。
    """
    cols = int(pattern_cols)
    rows = int(pattern_rows)
    square = float(square_size_mm)
    inset = float(inset_mm)
    if cols < 2 or rows < 2:
        raise ValueError("pattern_cols 和 pattern_rows 必须至少为 2")
    if not np.isfinite(square) or square <= 0.0:
        raise ValueError("square_size_mm 必须为正数")
    if not np.isfinite(inset) or inset < 0.0:
        raise ValueError("inset_mm 必须为非负数")

    x_min = -square + inset
    x_max = cols * square - inset
    y_min = -square + inset
    y_max = rows * square - inset
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("inset_mm 过大，无法形成有效棋盘边界")
    return np.asarray(
        [
            [x_min, y_min, 0.0],
            [x_max, y_min, 0.0],
            [x_max, y_max, 0.0],
            [x_min, y_max, 0.0],
        ],
        dtype=np.float64,
    )


def projected_board_boundary(
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    inset_mm: float = 0.0,
) -> np.ndarray:
    """将完整棋盘物理边界按当前 PnP pose 投影到图像坐标。"""
    object_points = full_board_object_corners(
        pattern_cols,
        pattern_rows,
        square_size_mm,
        inset_mm=inset_mm,
    )
    projected, _ = cv2.projectPoints(
        object_points,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def polygon_mask(shape: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """把凸四边形投影区域栅格化为布尔图像 mask。"""
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("mask shape 必须为正数")
    points = np.rint(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
    if points.shape != (4, 2):
        raise ValueError("完整棋盘边界必须包含 4 个二维顶点")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, points.reshape(-1, 1, 2), 255)
    return mask > 0


def full_board_physical_mask(
    shape: Tuple[int, int],
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    inset_mm: float = 0.0,
) -> np.ndarray:
    """生成完整棋盘物理边界 mask，不做额外像素腐蚀或膨胀。"""
    boundary = projected_board_boundary(
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
        pattern_cols,
        pattern_rows,
        square_size_mm,
        inset_mm=inset_mm,
    )
    return polygon_mask(shape, boundary)

