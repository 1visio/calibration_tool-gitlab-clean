from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import cv2
import numpy as np

from ..laser import normalize_laser_orientation
from .models import FrameQuality, QualityThresholds


def laser_column_metrics(
    image: np.ndarray,
    *,
    sensor_max_value: float,
) -> dict[str, np.ndarray | float]:
    """Return the per-column laser diagnostics used by ``analyze_frame``.

    These values describe contrast, width and saturation only; they are not a
    centre extractor and must not replace the shared Steger implementation.
    """
    return laser_scanline_metrics(
        image,
        sensor_max_value=sensor_max_value,
        scan_axis="column",
    )


def laser_scanline_metrics(
    image: np.ndarray,
    *,
    sensor_max_value: float,
    scan_axis: str,
) -> dict[str, np.ndarray | float]:
    """按原图 row/column 计算完全相同的激光强度质量指标。

    ``scan_axis=column`` 沿原图每列搜索 peak，法向为 row/v；
    ``scan_axis=row`` 沿原图每行搜索 peak，法向为 column/u。row 路径直接
    在原图 axis=1 上归约，避免 ``image.T`` 非连续视图的高额内存访问开销。
    """

    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError("激光逐扫描线质量分析仅支持二维 uint8/uint16 灰度图")
    if sensor_max_value <= 0:
        raise ValueError("sensor_max_value 必须为正数")
    if scan_axis not in {"column", "row"}:
        raise ValueError("scan_axis 必须是 column 或 row")
    values = image.astype(np.float32, copy=False)
    p01, p99 = (float(value) for value in np.percentile(values, (1, 99)))
    reduction_axis = 0 if scan_axis == "column" else 1
    background = np.percentile(values, 50, axis=reduction_axis)
    peak = np.max(values, axis=reduction_axis)
    minimum_contrast = max(sensor_max_value * 0.05, (p99 - p01) * 0.20)
    active = (peak - background) >= minimum_contrast
    saturated = values >= sensor_max_value * 0.995
    near_saturated = peak >= sensor_max_value * 0.98
    saturated_width = np.sum(saturated, axis=reduction_axis)
    half_max = background + (peak - background) * 0.5
    if scan_axis == "column":
        fwhm = np.sum(values >= half_max[None, :], axis=0)
    else:
        fwhm = np.sum(values >= half_max[:, None], axis=1)
    return {
        "background_dn": background,
        "peak_dn": peak,
        "peak_contrast_dn": peak - background,
        "minimum_contrast_dn": float(minimum_contrast),
        "active": active,
        "peak_saturated": peak >= sensor_max_value * 0.995,
        "peak_near_saturated": near_saturated,
        "saturated_width_px": saturated_width,
        # 保留旧 key，避免影响已有调用方；其含义在 row 模式下是逐 row。
        "column_saturation_fraction": np.mean(saturated, axis=reduction_axis),
        "fwhm_px": fwhm,
    }


def analyze_frame(
    image: np.ndarray,
    *,
    sensor_max_value: float,
    mode: str = "generic",
    thresholds: QualityThresholds = QualityThresholds(),
    board_pattern: tuple[int, int] | None = None,
    laser_orientation: str = "horizontal",
) -> FrameQuality:
    """计算可用于实时提示和采集清单的轻量质量指标。"""
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError("质量分析仅支持二维 uint8/uint16 灰度图")
    if sensor_max_value <= 0:
        raise ValueError("sensor_max_value 必须为正数")
    if mode not in {"generic", "chessboard", "laser"}:
        raise ValueError(f"未知质量模式：{mode}")
    orientation = normalize_laser_orientation(laser_orientation)

    values = image.astype(np.float32, copy=False)
    p01, p50, p99 = (float(value) for value in np.percentile(values, (1, 50, 99)))
    scale_to_u8 = 255.0 / float(sensor_max_value)
    saturation_fraction = float(np.mean(values >= sensor_max_value * 0.995))
    dark_fraction = float(np.mean(values <= sensor_max_value * 0.02))
    normalized = np.clip(values * scale_to_u8, 0, 255).astype(np.uint8)
    focus = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    # 激光线通常只占整幅图像的千分之几，P99 会完全落在背景上；激光模式
    # 使用 P99.9，同时保留 P99 字段，避免把正常的窄亮线误判为低动态范围。
    dynamic_high = float(np.percentile(values, 99.9)) if mode == "laser" else p99
    dynamic_range_u8 = (dynamic_high - p01) * scale_to_u8

    laser_coverage: float | None = None
    laser_peak_saturation_fraction: float | None = None
    laser_peak_near_saturation_fraction: float | None = None
    laser_saturated_width_p95_px: float | None = None
    laser_fwhm_p50_px: float | None = None
    laser_fwhm_p95_px: float | None = None
    chessboard_highlight_fraction: float | None = None
    chessboard_p995_u8: float | None = None
    chessboard_detected: bool | None = None
    chessboard_pattern_used: tuple[int, int] | None = None
    chessboard_detection_method: str | None = None
    chessboard_hint: str | None = None
    warnings: list[str] = []
    if saturation_fraction > thresholds.max_saturation_fraction:
        warnings.append("saturation_high")
    # 线激光图像预期具有大面积暗背景；这类图像应由激光覆盖率和线条对比度
    # 判断，而不是沿用棋盘/普通图像的全局暗像素比例。
    if mode != "laser" and dark_fraction > thresholds.max_dark_fraction:
        warnings.append("image_too_dark")
    if dynamic_range_u8 < thresholds.min_dynamic_range_u8:
        warnings.append("dynamic_range_low")

    if mode == "laser":
        # 横向逐 column、纵向逐 row。两者公式相同；纵向直接沿原图 axis=1
        # 归约，避免非连续 transpose，但不改变 coverage/FWHM 的数值定义。
        column_metrics = laser_scanline_metrics(
            image,
            sensor_max_value=sensor_max_value,
            scan_axis="column" if orientation == "horizontal" else "row",
        )
        active_columns = np.asarray(column_metrics["active"], dtype=bool)
        laser_coverage = float(np.mean(active_columns))
        if laser_coverage < thresholds.min_laser_coverage:
            warnings.append("laser_coverage_low")
        if saturation_fraction > thresholds.max_laser_saturation_fraction:
            warnings.append("laser_saturation_high")
        if np.any(active_columns):
            active_peak_saturated = np.asarray(
                column_metrics["peak_saturated"], dtype=bool
            )[active_columns]
            active_peak_near_saturated = np.asarray(
                column_metrics["peak_near_saturated"], dtype=bool
            )[active_columns]
            laser_peak_saturation_fraction = float(
                np.mean(active_peak_saturated)
            )
            laser_peak_near_saturation_fraction = float(
                np.mean(active_peak_near_saturated)
            )
            saturated_widths = np.asarray(column_metrics["saturated_width_px"])[active_columns]
            laser_saturated_width_p95_px = float(np.percentile(saturated_widths, 95))
            fwhm_widths = np.asarray(column_metrics["fwhm_px"])[active_columns]
            laser_fwhm_p50_px = float(np.percentile(fwhm_widths, 50))
            laser_fwhm_p95_px = float(np.percentile(fwhm_widths, 95))
            if laser_peak_saturation_fraction > thresholds.max_laser_peak_saturation_fraction:
                warnings.append("laser_peak_saturated")
            if laser_peak_near_saturation_fraction > thresholds.max_laser_peak_near_saturation_fraction:
                warnings.append("laser_peak_near_saturation")
            if laser_saturated_width_p95_px > thresholds.max_laser_saturated_width_px:
                warnings.append("laser_saturated_line_wide")
        else:
            laser_peak_saturation_fraction = 0.0
            laser_peak_near_saturation_fraction = 0.0
            laser_saturated_width_p95_px = 0.0
            laser_fwhm_p50_px = 0.0
            laser_fwhm_p95_px = 0.0
    elif mode == "chessboard":
        if board_pattern is None:
            raise ValueError("chessboard 质量模式需要 board_pattern=(cols, rows)")
        chessboard_highlight_fraction = float(np.mean(values >= sensor_max_value * 0.96))
        chessboard_p995_u8 = float(np.percentile(values, 99.5) * scale_to_u8)
        if saturation_fraction > thresholds.max_chessboard_saturation_fraction:
            warnings.append("chessboard_saturation_high")
        if (
            chessboard_highlight_fraction > thresholds.max_chessboard_highlight_fraction
            or chessboard_p995_u8 > thresholds.max_chessboard_p995_u8
        ):
            warnings.append("chessboard_near_saturation")
        detection = detect_chessboard_for_quality(normalized, board_pattern)
        chessboard_detected = detection["detected"]
        chessboard_pattern_used = detection["pattern"]
        chessboard_detection_method = detection["method"]
        chessboard_hint = detection["hint"]
        if thresholds.require_chessboard_detection and not chessboard_detected:
            warnings.append(str(detection["warning"]))

    return FrameQuality(
        mean_dn=float(np.mean(values)),
        p01_dn=p01,
        p50_dn=p50,
        p99_dn=p99,
        dynamic_range_u8=float(dynamic_range_u8),
        saturation_fraction=saturation_fraction,
        dark_fraction=dark_fraction,
        focus_laplacian=focus,
        laser_coverage=laser_coverage,
        laser_peak_saturation_fraction=laser_peak_saturation_fraction,
        laser_peak_near_saturation_fraction=laser_peak_near_saturation_fraction,
        laser_saturated_width_p95_px=laser_saturated_width_p95_px,
        laser_fwhm_p50_px=laser_fwhm_p50_px,
        laser_fwhm_p95_px=laser_fwhm_p95_px,
        chessboard_highlight_fraction=chessboard_highlight_fraction,
        chessboard_p995_u8=chessboard_p995_u8,
        chessboard_detected=chessboard_detected,
        chessboard_pattern_used=chessboard_pattern_used,
        chessboard_detection_method=chessboard_detection_method,
        chessboard_hint=chessboard_hint,
        warnings=tuple(warnings),
    )


def detect_chessboard_for_quality(
    image_u8: np.ndarray,
    board_pattern: tuple[int, int],
) -> dict[str, object]:
    """检测完整棋盘，并区分配置方向错误与激光线遮挡。"""
    height, width = image_u8.shape
    scale = min(1.0, 1400.0 / max(width, height))
    work = (
        cv2.resize(image_u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image_u8
    )
    # 棋盘检测允许为算法增强对比度；这不会改变曝光质量指标本身。
    low, high = (float(value) for value in np.percentile(work, (0.5, 99.5)))
    enhanced = (
        np.clip((work.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        if high > low
        else work
    )

    method = _find_chessboard(enhanced, board_pattern)
    if method:
        return {
            "detected": True,
            "pattern": board_pattern,
            "method": method,
            "hint": "完整棋盘检测通过",
            "warning": None,
        }

    swapped = (board_pattern[1], board_pattern[0])
    if swapped != board_pattern:
        swapped_method = _find_chessboard(enhanced, swapped)
        if swapped_method:
            return {
                "detected": False,
                "pattern": swapped,
                "method": swapped_method,
                "hint": f"检测到 {swapped[0]}×{swapped[1]}，当前配置为 {board_pattern[0]}×{board_pattern[1]}",
                "warning": "chessboard_pattern_mismatch",
            }

    suppressed, laser_row = _suppress_horizontal_laser(enhanced)
    if suppressed is not None:
        suppressed_method = _find_chessboard(suppressed, board_pattern)
        if suppressed_method:
            return {
                "detected": False,
                "pattern": board_pattern,
                "method": f"{suppressed_method}+laser_suppressed",
                "hint": f"激光线（约第 {laser_row} 行）遮挡棋盘；采集内参棋盘时请关闭激光",
                "warning": "chessboard_obscured_by_laser",
            }
        if swapped != board_pattern:
            swapped_method = _find_chessboard(suppressed, swapped)
            if swapped_method:
                return {
                    "detected": False,
                    "pattern": swapped,
                    "method": f"{swapped_method}+laser_suppressed",
                    "hint": (
                        f"激光线遮挡棋盘，且检测到的方向为 {swapped[0]}×{swapped[1]}；"
                        f"请关闭激光并检查棋盘配置"
                    ),
                    "warning": "chessboard_obscured_by_laser",
                }

    return {
        "detected": False,
        "pattern": None,
        "method": None,
        "hint": "未找到完整棋盘：请检查内角点数量、完整入镜、对焦和曝光；内参采集时关闭激光",
        "warning": "chessboard_not_found",
    }


def _find_chessboard(image: np.ndarray, pattern: tuple[int, int]) -> str | None:
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, _ = cv2.findChessboardCorners(image, pattern, classic_flags)
    if found:
        return "classic"
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        found, _ = cv2.findChessboardCornersSB(image, pattern, sb_flags)
        if found:
            return "SB"
    return None


def _suppress_horizontal_laser(image: np.ndarray) -> tuple[np.ndarray | None, int | None]:
    threshold = max(220.0, float(np.percentile(image, 99.0)))
    row_coverage = np.mean(image >= threshold, axis=1)
    row = int(np.argmax(row_coverage))
    if float(row_coverage[row]) < 0.30:
        return None, None
    half_band = max(3, int(round(image.shape[0] * 0.005)))
    top = max(0, row - half_band)
    bottom = min(image.shape[0], row + half_band + 1)
    reference_top = image[max(0, top - 2 * half_band) : max(0, top - half_band)]
    reference_bottom = image[min(image.shape[0], bottom + half_band) : min(image.shape[0], bottom + 2 * half_band)]
    references = [part for part in (reference_top, reference_bottom) if part.size]
    if not references:
        return None, None
    replacement = np.median(np.concatenate(references, axis=0), axis=0).astype(np.uint8)
    cleaned = image.copy()
    cleaned[top:bottom] = replacement
    return cleaned, row


def quality_to_dict(quality: FrameQuality) -> dict[str, object]:
    value = asdict(quality)
    value["passed"] = quality.passed
    value["warnings"] = list(quality.warnings)
    return value


def summarize_quality(items: Iterable[FrameQuality]) -> dict[str, object]:
    qualities = list(items)
    if not qualities:
        return {"frames": 0, "passed": 0, "warnings": 0, "warning_counts": {}}
    warning_counts: dict[str, int] = {}
    for quality in qualities:
        for warning in quality.warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    passed = sum(quality.passed for quality in qualities)
    return {
        "frames": len(qualities),
        "passed": passed,
        "warnings": len(qualities) - passed,
        "warning_counts": warning_counts,
        "mean_dynamic_range_u8": float(np.mean([item.dynamic_range_u8 for item in qualities])),
        "mean_focus_laplacian": float(np.mean([item.focus_laplacian for item in qualities])),
    }
