#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量评估不同曝光时间下的棋盘格角点检测率与灰度质量。

推荐目录结构：
dataset/
├─ 500us/
│  ├─ pose_01.tif
│  ├─ pose_02.tif
│  └─ ...
├─ 1000us/
│  ├─ pose_01.tif
│  └─ ...
└─ 2000us/
   └─ ...

每个一级子文件夹代表一个曝光时间，文件夹内放置不同姿态的棋盘格图像。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估不同曝光时间下的棋盘格角点检测率、灰度、对比度、饱和率和清晰度。"
    )
    parser.add_argument("--input", required=True, type=Path, help="输入数据集根目录")
    parser.add_argument("--output", type=Path, default=Path("exposure_eval_output"),
                        help="输出目录，默认 exposure_eval_output")
    parser.add_argument("--cols", type=int, default=6,
                        help="棋盘格横向内角点数量，默认 6")
    parser.add_argument("--rows", type=int, default=5,
                        help="棋盘格纵向内角点数量，默认 5")
    parser.add_argument(
        "--max-value",
        type=float,
        default=0,
        help="传感器有效最大灰度。Mono8填255，Mono12填4095；0表示自动判断",
    )
    parser.add_argument(
        "--sat-ratio",
        type=float,
        default=0.98,
        help="达到最大灰度多少比例视为饱和，默认 0.98",
    )
    parser.add_argument(
        "--dark-ratio",
        type=float,
        default=0.02,
        help="低于最大灰度多少比例视为过暗，默认 0.02",
    )
    return parser.parse_args()


def imread_unicode(path: Path) -> np.ndarray | None:
    """兼容 Windows 中文路径的图像读取。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """兼容 Windows 中文路径的图像保存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def to_gray(image: np.ndarray) -> np.ndarray:
    """将输入图像转换为单通道灰度图，保留原始位深。"""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"不支持的图像形状：{image.shape}")


def infer_max_value(gray: np.ndarray, user_max: float) -> float:
    """
    推断有效最大灰度。
    uint8 默认 255；uint16 默认按 Mono12 的 4095 处理。
    若实际是 Mono16，请通过 --max-value 65535 显式指定。
    """
    if user_max > 0:
        return user_max
    if gray.dtype == np.uint8:
        return 255.0
    if gray.dtype == np.uint16:
        return 4095.0
    observed = float(np.nanmax(gray))
    return observed if observed > 0 else 1.0


def normalize_to_u8(gray: np.ndarray, max_value: float) -> np.ndarray:
    """将原始灰度线性映射到 uint8，仅用于角点检测与可视化。"""
    clipped = np.clip(gray.astype(np.float32), 0, max_value)
    return np.round(clipped * (255.0 / max_value)).astype(np.uint8)


def detect_chessboard(gray8: np.ndarray, pattern_size: tuple[int, int]):
    """
    检测完整棋盘格内角点，并执行亚像素优化。
    先使用 FAST_CHECK 加速；失败后再进行完整搜索。
    """
    flags_fast = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray8, pattern_size, flags_fast)

    if not found:
        flags_full = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray8, pattern_size, flags_full)

    if found and corners is not None:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            50,
            1e-3,
        )
        corners = cv2.cornerSubPix(
            gray8,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
    return found, corners


def make_board_mask(shape: tuple[int, int], corners: np.ndarray | None) -> np.ndarray:
    """
    根据检测到的内角点生成棋盘格局部区域掩膜。
    使用角点外接矩形并增加少量边缘，避免全图背景影响灰度统计。
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if corners is None:
        mask[:, :] = 255
        return mask

    pts = corners.reshape(-1, 2)
    x_min, y_min = np.floor(pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(pts.max(axis=0)).astype(int)

    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)
    margin_x = max(int(width * 0.12), 5)
    margin_y = max(int(height * 0.12), 5)

    x_min = max(x_min - margin_x, 0)
    y_min = max(y_min - margin_y, 0)
    x_max = min(x_max + margin_x, shape[1] - 1)
    y_max = min(y_max + margin_y, shape[0] - 1)

    mask[y_min:y_max + 1, x_min:x_max + 1] = 255
    return mask


def gray_metrics(gray: np.ndarray, mask: np.ndarray, max_value: float,
                 sat_ratio: float, dark_ratio: float) -> dict[str, float]:
    """计算掩膜区域内的灰度、对比度、饱和率和过暗比例。"""
    pixels = gray[mask > 0].astype(np.float64)
    if pixels.size == 0:
        return {
            "mean_gray": np.nan,
            "std_gray": np.nan,
            "p05_gray": np.nan,
            "median_gray": np.nan,
            "p95_gray": np.nan,
            "contrast_p95_p05": np.nan,
            "saturation_ratio": np.nan,
            "dark_ratio": np.nan,
        }

    p05, median, p95 = np.percentile(pixels, [5, 50, 95])
    return {
        "mean_gray": float(np.mean(pixels)),
        "std_gray": float(np.std(pixels)),
        "p05_gray": float(p05),
        "median_gray": float(median),
        "p95_gray": float(p95),
        "contrast_p95_p05": float(p95 - p05),
        "saturation_ratio": float(np.mean(pixels >= sat_ratio * max_value)),
        "dark_ratio": float(np.mean(pixels <= dark_ratio * max_value)),
    }


def sharpness_laplacian(gray8: np.ndarray, mask: np.ndarray) -> float:
    """使用拉普拉斯方差估计局部清晰度，数值越大通常越清晰。"""
    lap = cv2.Laplacian(gray8, cv2.CV_64F)
    values = lap[mask > 0]
    return float(np.var(values)) if values.size else np.nan


def extract_exposure_us(group_name: str) -> float | None:
    """从文件夹名中提取曝光时间，例如 500us、1000 μs、exp_2000。"""
    match = re.search(r"(\d+(?:\.\d+)?)", group_name)
    return float(match.group(1)) if match else None


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def group_name_for_image(root: Path, image_path: Path) -> str:
    """
    使用相对于根目录的第一级文件夹作为曝光组名。
    例如 dataset/1000us/pose01.tif -> 1000us。
    """
    rel = image_path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def finite_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    pattern_size = (args.cols, args.rows)
    expected_corners = args.cols * args.rows

    if not input_dir.exists():
        print(f"[错误] 输入目录不存在：{input_dir}", file=sys.stderr)
        return 1

    images = list_images(input_dir)
    if not images:
        print(f"[错误] 未找到图像：{input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_root = output_dir / "annotated"

    image_rows: list[dict[str, Any]] = []

    for index, image_path in enumerate(images, start=1):
        group = group_name_for_image(input_dir, image_path)
        image = imread_unicode(image_path)

        if image is None:
            print(f"[跳过] 无法读取：{image_path}")
            continue

        try:
            gray = to_gray(image)
        except ValueError as exc:
            print(f"[跳过] {image_path}: {exc}")
            continue

        max_value = infer_max_value(gray, args.max_value)
        gray8 = normalize_to_u8(gray, max_value)
        found, corners = detect_chessboard(gray8, pattern_size)

        board_mask = make_board_mask(gray.shape, corners if found else None)
        metrics = gray_metrics(
            gray, board_mask, max_value, args.sat_ratio, args.dark_ratio
        )
        sharpness = sharpness_laplacian(gray8, board_mask)

        row = {
            "group": group,
            "exposure_us": extract_exposure_us(group),
            "filename": str(image_path.relative_to(input_dir)),
            "detected": int(found),
            "corner_count": expected_corners if found else 0,
            "expected_corners": expected_corners,
            "max_value": max_value,
            **metrics,
            "sharpness_laplacian": sharpness,
        }
        image_rows.append(row)

        # 保存角点可视化图
        vis = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        if found and corners is not None:
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            status = f"OK {expected_corners}/{expected_corners}"
        else:
            status = f"FAIL 0/{expected_corners}"

        cv2.putText(
            vis,
            status,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0) if found else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        rel_png = image_path.relative_to(input_dir).with_suffix(".png")
        imwrite_unicode(annotated_root / rel_png, vis)

        print(f"[{index:>3}/{len(images)}] {group:<12} "
              f"{'成功' if found else '失败'}  {image_path.name}")

    if not image_rows:
        print("[错误] 没有可分析的图像。", file=sys.stderr)
        return 1

    image_fields = [
        "group", "exposure_us", "filename", "detected",
        "corner_count", "expected_corners", "max_value",
        "mean_gray", "std_gray", "p05_gray", "median_gray", "p95_gray",
        "contrast_p95_p05", "saturation_ratio", "dark_ratio",
        "sharpness_laplacian",
    ]
    save_csv(output_dir / "image_metrics.csv", image_rows, image_fields)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        grouped[str(row["group"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group, rows in grouped.items():
        total = len(rows)
        detected = sum(int(r["detected"]) for r in rows)
        successful = [r for r in rows if int(r["detected"]) == 1]

        # 棋盘局部灰度统计只汇总检测成功的图像，避免失败图像使用全图统计造成偏差。
        source = successful if successful else rows
        summary_rows.append({
            "group": group,
            "exposure_us": extract_exposure_us(group),
            "image_count": total,
            "detected_count": detected,
            "detection_rate": detected / total if total else 0.0,
            "mean_gray": finite_mean([float(r["mean_gray"]) for r in source]),
            "median_contrast_p95_p05": finite_median(
                [float(r["contrast_p95_p05"]) for r in source]
            ),
            "mean_saturation_ratio": finite_mean(
                [float(r["saturation_ratio"]) for r in source]
            ),
            "mean_dark_ratio": finite_mean(
                [float(r["dark_ratio"]) for r in source]
            ),
            "median_sharpness_laplacian": finite_median(
                [float(r["sharpness_laplacian"]) for r in source]
            ),
        })

    summary_rows.sort(
        key=lambda r: (
            float("inf") if r["exposure_us"] is None else float(r["exposure_us"])
        )
    )

    summary_fields = [
        "group", "exposure_us", "image_count", "detected_count",
        "detection_rate", "mean_gray", "median_contrast_p95_p05",
        "mean_saturation_ratio", "mean_dark_ratio",
        "median_sharpness_laplacian",
    ]
    save_csv(output_dir / "exposure_summary.csv", summary_rows, summary_fields)

    # 输出候选曝光：优先检测率高，其次饱和率低，再次对比度高。
    ranked = sorted(
        summary_rows,
        key=lambda r: (
            -float(r["detection_rate"]),
            float(r["mean_saturation_ratio"])
            if np.isfinite(float(r["mean_saturation_ratio"])) else float("inf"),
            -float(r["median_contrast_p95_p05"])
            if np.isfinite(float(r["median_contrast_p95_p05"])) else float("inf"),
        ),
    )

    print("\n=== 各曝光汇总 ===")
    print("组名\t检测率\t平均灰度\t对比度\t饱和率\t清晰度")
    for r in summary_rows:
        print(
            f"{r['group']}\t"
            f"{100 * float(r['detection_rate']):.1f}%\t"
            f"{float(r['mean_gray']):.1f}\t"
            f"{float(r['median_contrast_p95_p05']):.1f}\t"
            f"{100 * float(r['mean_saturation_ratio']):.3f}%\t"
            f"{float(r['median_sharpness_laplacian']):.1f}"
        )

    print("\n=== 建议优先复核的曝光组 ===")
    for r in ranked[:3]:
        print(
            f"- {r['group']}: 检测率 {100 * float(r['detection_rate']):.1f}%，"
            f"饱和率 {100 * float(r['mean_saturation_ratio']):.3f}%，"
            f"对比度 {float(r['median_contrast_p95_p05']):.1f}"
        )

    print(f"\n结果已保存到：{output_dir}")
    print("选择曝光时，应先保证检测率接近 100%，再选择饱和率较低且对比度、清晰度较高的较短曝光。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
