#!/usr/bin/env python3
"""Leakage-safe V2 laser-plane calibration for paired chess/laser images.

V2 keeps board ROI, local background subtraction, peak prominence, subpixel
centroids, chess-boundary rejection and 3-D RANSAC.  It separates the four
quality filters, replaces column-neighbour continuity with a line RANSAC in
undistorted image coordinates, and gives every usable pose equal fitting weight.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks, peak_widths

import calibrate_laser_plane_opencv as base

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("laser_plane_calibration_v2")
DEFAULT_CONFIG = Path(__file__).with_name("laser_plane_config_v2.yaml")

STATUS_COLOURS = {
    "accepted_final": (0, 255, 0),
    "rejected_saturation": (0, 0, 255),
    "rejected_snr": (0, 255, 255),
    "rejected_fwhm": (255, 0, 0),
    "rejected_multi_peak": (255, 255, 0),
    "rejected_chess_boundary": (180, 0, 180),
    "rejected_line_ransac": (0, 165, 255),
    "rejected_prominence": (220, 220, 220),
    "rejected_roi": (128, 128, 128),
    "rejected_ransac": (255, 128, 255),
}


@dataclass(frozen=True)
class LockedThresholds:
    source_ids: str
    min_prominence: float
    max_saturated_pixels: int
    min_snr: float
    min_fwhm_px: float
    max_fwhm_px: float
    max_multi_peak_ratio: float
    line_ransac_threshold_px: float
    plane_ransac_threshold_mm: float
    pair_movement_warning_px: float
    low_coverage: float
    high_outlier_rate: float
    high_pose_rmse_mm: float


@dataclass
class FeatureFrame:
    image_id: int
    split: str
    chess_path: Path
    laser_path: Path
    image_width: int
    board_plane: np.ndarray
    reprojection_rmse_px: float
    diagnostics: pd.DataFrame
    pair_motion: dict[str, float | bool | str]


@dataclass(frozen=True)
class BalancedFit:
    plane: np.ndarray
    inliers: np.ndarray
    pose_ids: np.ndarray
    balanced_score: float


def cfg(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise base.CalibrationError(f"配置缺少字段：{dotted}")
        value = value[key]
    return value


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"V2 配置不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise base.CalibrationError("V2 配置根节点必须是映射")
    base.validate_config(data)
    required = (
        "quality_filters.multi_peak_enabled",
        "threshold_selection.line_seed_threshold_px",
        "line_ransac.iterations",
        "pair_movement.severe_px",
        "anomaly.minimum_coverage",
    )
    for key in required:
        cfg(data, key)
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--chess-pattern", default="chess {id:03d}.tif")
    parser.add_argument("--laser-pattern", default="laser {id:03d}.tif")
    parser.add_argument("--tune-ids", type=base.parse_id_spec, default="1-18")
    parser.add_argument("--train-ids", type=base.parse_id_spec, default="1-25")
    parser.add_argument("--val-ids", type=base.parse_id_spec, default="26-32")
    parser.add_argument("--pattern-cols", type=base.positive_int, default=6)
    parser.add_argument("--pattern-rows", type=base.positive_int, default=5)
    parser.add_argument("--square-size-mm", type=base.positive_float, default=30.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--old-config",
        type=Path,
        default=Path(__file__).with_name("laser_plane_config.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_split_contract(args: argparse.Namespace) -> None:
    tune = set(args.tune_ids)
    train = set(args.train_ids)
    validation = set(args.val_ids)
    if not tune:
        raise base.CalibrationError("--tune-ids 不能为空")
    if not tune <= train:
        raise base.CalibrationError("阈值选择编号必须是训练集的子集")
    if train & validation:
        raise base.CalibrationError("训练集与验证集不得重叠")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{args.output_dir}")


def image_pair(
    args: argparse.Namespace, image_id: int, split: str
) -> tuple[int, str, Path, Path]:
    chess = base.format_image_path(args.image_dir, args.chess_pattern, image_id)
    laser = base.format_image_path(args.image_dir, args.laser_pattern, image_id)
    missing = [str(path) for path in (chess, laser) if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少配对图像：" + ", ".join(missing))
    return image_id, split, chess, laser


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        output_dir / "processing_log.txt", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def _quantile(values: Sequence[float] | np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise base.CalibrationError("阈值选择没有有限样本")
    return float(np.percentile(finite, percentile))


def format_ids(ids: Sequence[int]) -> str:
    ordered = sorted(ids)
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "（无）"
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, float):
                cells.append("—" if not np.isfinite(value) else f"{value:.6g}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def measure_pair_motion(
    prepared: base.PreparedFrame,
    config: Mapping[str, Any],
) -> dict[str, float | bool | str]:
    chess = base.to_gray(base.read_image(prepared.chess_path), prepared.chess_path)
    laser = base.to_gray(base.read_image(prepared.laser_path), prepared.laser_path)
    scale = float(cfg(config, "pair_movement.ecc_scale"))
    chess_small = cv2.resize(
        chess, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    laser_small = cv2.resize(
        laser, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    chess_small /= max(float(np.max(chess_small)), 1.0)
    laser_small /= max(float(np.max(laser_small)), 1.0)
    mask = np.zeros(chess_small.shape, dtype=np.uint8)
    polygon = np.rint(prepared.pose.roi_polygon * scale).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        int(cfg(config, "pair_movement.ecc_iterations")),
        float(cfg(config, "pair_movement.ecc_epsilon")),
    )
    ecc_score = math.nan
    ecc_displacement = np.empty(0, dtype=np.float64)
    try:
        score, warp = cv2.findTransformECC(
            chess_small,
            laser_small,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            mask,
            5,
        )
        ecc_score = float(score)
        corners = prepared.pose.corners.reshape(-1, 2) * scale
        transformed = cv2.transform(corners.reshape(-1, 1, 2), warp).reshape(-1, 2)
        ecc_displacement = np.linalg.norm(transformed - corners, axis=1) / scale
    except cv2.error:
        pass

    patch_radius = int(cfg(config, "pair_movement.template_patch_radius_px"))
    search_radius = int(cfg(config, "pair_movement.template_search_radius_px"))
    minimum_correlation = float(cfg(config, "pair_movement.template_min_correlation"))
    shifts: list[tuple[float, float]] = []
    correlations: list[float] = []
    for corner_x, corner_y in prepared.pose.corners:
        x = int(round(float(corner_x)))
        y = int(round(float(corner_y)))
        margin = patch_radius + search_radius
        if (
            x - margin < 0
            or y - margin < 0
            or x + margin + 1 > chess.shape[1]
            or y + margin + 1 > chess.shape[0]
        ):
            continue
        template = chess[
            y - patch_radius : y + patch_radius + 1,
            x - patch_radius : x + patch_radius + 1,
        ]
        search = laser[
            y - margin : y + margin + 1,
            x - margin : x + margin + 1,
        ]
        correlation = cv2.matchTemplate(
            search, template, cv2.TM_CCOEFF_NORMED
        )
        _, maximum, _, location = cv2.minMaxLoc(correlation)
        if maximum >= minimum_correlation:
            shifts.append(
                (float(location[0] - search_radius), float(location[1] - search_radius))
            )
            correlations.append(float(maximum))
    template_ratio = len(shifts) / len(prepared.pose.corners)
    template_displacement = np.empty(0, dtype=np.float64)
    template_score = math.nan
    if shifts:
        shift_array = np.asarray(shifts, dtype=np.float64)
        median_shift = np.median(shift_array, axis=0)
        consensus = (
            np.linalg.norm(shift_array - median_shift, axis=1)
            <= float(cfg(config, "pair_movement.template_consensus_radius_px"))
        )
        template_ratio = float(np.count_nonzero(consensus) / len(prepared.pose.corners))
        if np.any(consensus):
            template_displacement = np.linalg.norm(shift_array[consensus], axis=1)
            template_score = float(np.median(np.asarray(correlations)[consensus]))

    if template_ratio >= float(
        cfg(config, "pair_movement.template_min_consensus_ratio")
    ):
        displacement = template_displacement
        method = "local_template"
        registration_score = template_score
        tracked_ratio = template_ratio
        tracking_ok = True
    elif ecc_score >= float(cfg(config, "pair_movement.minimum_ecc_score")):
        displacement = ecc_displacement
        method = "ecc_fallback"
        registration_score = ecc_score
        tracked_ratio = 1.0
        tracking_ok = True
    else:
        displacement = np.empty(0, dtype=np.float64)
        method = "unresolved"
        registration_score = max(template_score, ecc_score)
        tracked_ratio = template_ratio
        tracking_ok = False
    return {
        "tracked_ratio": tracked_ratio,
        "registration_score": registration_score,
        "movement_method": method,
        "ecc_score": ecc_score,
        "template_consensus_ratio": template_ratio,
        "template_median_correlation": template_score,
        "median_displacement_px": float(np.median(displacement)) if len(displacement) else math.nan,
        "p95_displacement_px": float(np.percentile(displacement, 95)) if len(displacement) else math.nan,
        "max_displacement_px": float(np.max(displacement)) if len(displacement) else math.nan,
        "median_forward_backward_px": math.nan,
        "tracking_ok": tracking_ok,
    }


def detect_board_pose_robust(
    gray: np.ndarray,
    args: argparse.Namespace,
    intrinsics: base.Intrinsics,
    config: Mapping[str, Any],
) -> base.BoardPose:
    try:
        return base.detect_board_pose(
            gray,
            args.pattern_cols,
            args.pattern_rows,
            args.square_size_mm,
            intrinsics,
            config,
        )
    except base.CalibrationError as exc:
        if "未检测到完整棋盘格内角点" not in str(exc):
            raise
    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    found, corners = cv2.findChessboardCornersSB(
        gray, (args.pattern_cols, args.pattern_rows), flags
    )
    if not found or corners is None:
        raise base.CalibrationError("传统与 SB 检测均未找到完整棋盘格内角点")
    window = int(cfg(config, "chessboard.corner_window_px"))
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        int(cfg(config, "chessboard.corner_max_iterations")),
        float(cfg(config, "chessboard.corner_epsilon")),
    )
    refined = cv2.cornerSubPix(
        gray, corners.astype(np.float32), (window, window), (-1, -1), criteria
    )
    board_points = base.object_points(
        args.pattern_cols, args.pattern_rows, args.square_size_mm
    )
    solved, rvec, tvec = cv2.solvePnP(
        board_points,
        refined,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise base.CalibrationError("SB 角点的 solvePnP 失败")
    projected, _ = cv2.projectPoints(
        board_points,
        rvec,
        tvec,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    residual = refined.reshape(-1, 2) - projected.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    maximum = float(cfg(config, "chessboard.max_reprojection_rmse_px"))
    if rmse > maximum:
        raise base.CalibrationError(
            f"SB 棋盘 PnP 重投影 RMSE {rmse:.3f}px 超过 {maximum:.3f}px"
        )
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2].astype(np.float64)
    normal /= np.linalg.norm(normal)
    point_on_plane = tvec.reshape(3).astype(np.float64)
    plane = np.append(normal, -float(normal @ point_on_plane))
    border = float(cfg(config, "chessboard.outer_border_squares"))
    inset = float(cfg(config, "chessboard.boundary_inset_mm"))
    square = float(args.square_size_mm)
    x0 = -border * square + inset
    y0 = -border * square + inset
    x1 = (args.pattern_cols - 1 + border) * square - inset
    y1 = (args.pattern_rows - 1 + border) * square - inset
    outer = np.asarray(
        [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]],
        dtype=np.float32,
    )
    polygon, _ = cv2.projectPoints(
        outer, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    return base.BoardPose(
        refined.reshape(-1, 2),
        rvec,
        tvec,
        plane,
        polygon.reshape(-1, 2).astype(np.float32),
        rmse,
    )


def prepare_frame_robust(
    pair: tuple[int, str, Path, Path],
    args: argparse.Namespace,
    intrinsics: base.Intrinsics,
    config: Mapping[str, Any],
) -> base.PreparedFrame:
    image_id, split, chess_path, laser_path = pair
    chess = base.to_gray(base.read_image(chess_path), chess_path)
    laser = base.to_gray(base.read_image(laser_path), laser_path)
    expected = (intrinsics.image_size[1], intrinsics.image_size[0])
    if chess.shape != expected or laser.shape != expected:
        raise base.CalibrationError(
            f"ID {image_id:03d} 尺寸不匹配：期望 {expected[::-1]}，"
            f"chess={chess.shape[::-1]}，laser={laser.shape[::-1]}"
        )
    pose = detect_board_pose_robust(
        base.chessboard_detection_image(chess), args, intrinsics, config
    )
    image = laser.astype(np.float64)
    full_scale = base._full_scale(laser, config)
    if float(np.max(image)) > full_scale:
        raise base.CalibrationError(
            f"ID {image_id:03d} 最大值超过传感器满量程 {full_scale:g}"
        )
    window = int(cfg(config, "laser.background_window_px"))
    percentile = float(cfg(config, "laser.background_percentile")) * 100.0
    background = percentile_filter(
        image, percentile=percentile, size=(window, 1), mode="nearest"
    )
    corrected = np.maximum(image - background, 0.0)
    sigma = float(cfg(config, "laser.profile_smoothing_sigma_px"))
    raw_smoothed = (
        gaussian_filter1d(image, sigma=sigma, axis=0, mode="nearest")
        if sigma > 0
        else image.copy()
    )
    corrected_smoothed = (
        gaussian_filter1d(corrected, sigma=sigma, axis=0, mode="nearest")
        if sigma > 0
        else corrected.copy()
    )
    boundary = base.build_chess_boundary_mask(
        laser.shape, pose, args, intrinsics, config
    )
    return base.PreparedFrame(
        image_id,
        split,
        chess_path,
        laser_path,
        laser,
        full_scale,
        pose,
        background,
        corrected,
        raw_smoothed,
        corrected_smoothed,
        boundary,
    )


def extract_features(
    prepared: base.PreparedFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    raw = prepared.laser_gray.astype(np.float64)
    signal = prepared.corrected_smoothed
    height, width = raw.shape
    half_window = int(cfg(config, "laser.centroid_half_window_px"))
    seed_prominence = (
        float(cfg(config, "laser.min_peak_prominence_ratio")) * prepared.full_scale
    )
    saturation_level = (
        float(cfg(config, "laser.saturation_ratio")) * prepared.full_scale
    )
    noise_floor = float(cfg(config, "laser.noise_floor_ratio")) * prepared.full_scale
    separation = int(cfg(config, "quality_filters.multi_peak_min_separation_px"))
    rows = np.arange(height, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for column in range(width):
        profile = signal[:, column]
        peaks, properties = find_peaks(profile, prominence=seed_prominence)
        record: dict[str, Any] = {
            "column_px": column,
            "x_px": float(column),
            "peak_row_px": math.nan,
            "y_px": math.nan,
            "prominence": math.nan,
            "snr": math.nan,
            "fwhm_px": math.nan,
            "saturated_pixels": 0,
            "multi_peak_ratio": 0.0,
            "line_residual_px": math.nan,
            "structural_status": "rejected_prominence",
            "status": "rejected_prominence",
        }
        if peaks.size == 0:
            records.append(record)
            continue
        selected_index = int(np.argmax(profile[peaks]))
        peak = int(peaks[selected_index])
        start = max(0, peak - half_window)
        stop = min(height, peak + half_window + 1)
        weights = np.maximum(profile[start:stop], 0.0)
        weight_sum = float(np.sum(weights))
        centroid = (
            float(np.sum(rows[start:stop] * weights) / weight_sum)
            if weight_sum > 0
            else math.nan
        )
        other = peaks[np.abs(peaks - peak) >= separation]
        main_height = max(float(profile[peak]), np.finfo(float).eps)
        second_ratio = (
            float(np.max(profile[other]) / main_height) if other.size else 0.0
        )
        raw_window = raw[start:stop, column]
        saturated = int(np.count_nonzero(raw_window >= saturation_level))
        median = float(np.median(profile))
        mad = float(np.median(np.abs(profile - median)))
        noise = max(1.4826 * mad, noise_floor)
        snr = float(profile[peak] / noise)
        fwhm = float(
            peak_widths(profile, np.asarray([peak]), rel_height=0.5)[0][0]
        )
        status = "candidate"
        if not np.isfinite(centroid):
            status = "rejected_centroid"
        elif not base._inside_polygon((float(column), centroid), prepared.pose.roi_polygon):
            status = "rejected_roi"
        else:
            rounded_y = int(np.clip(round(centroid), 0, height - 1))
            if prepared.chess_boundary_mask[rounded_y, column] != 0:
                status = "rejected_chess_boundary"
        record.update(
            peak_row_px=float(peak),
            y_px=centroid,
            prominence=float(properties["prominences"][selected_index]),
            snr=snr,
            fwhm_px=fwhm,
            saturated_pixels=saturated,
            multi_peak_ratio=second_ratio,
            structural_status=status,
            status=status,
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _candidate_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["status"] == "candidate"


def apply_quality_thresholds(
    diagnostics: pd.DataFrame,
    thresholds: LockedThresholds,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    result = diagnostics.copy()
    result["status"] = result["structural_status"]
    active = _candidate_mask(result)
    result.loc[active & (result["prominence"] < thresholds.min_prominence), "status"] = (
        "rejected_prominence"
    )
    if bool(cfg(config, "quality_filters.saturation_enabled")):
        active = _candidate_mask(result)
        result.loc[
            active & (result["saturated_pixels"] > thresholds.max_saturated_pixels),
            "status",
        ] = "rejected_saturation"
    if bool(cfg(config, "quality_filters.snr_enabled")):
        active = _candidate_mask(result)
        result.loc[active & (result["snr"] < thresholds.min_snr), "status"] = (
            "rejected_snr"
        )
    if bool(cfg(config, "quality_filters.fwhm_enabled")):
        active = _candidate_mask(result)
        bad_fwhm = (result["fwhm_px"] < thresholds.min_fwhm_px) | (
            result["fwhm_px"] > thresholds.max_fwhm_px
        )
        result.loc[active & bad_fwhm, "status"] = "rejected_fwhm"
    if bool(cfg(config, "quality_filters.multi_peak_enabled")):
        active = _candidate_mask(result)
        result.loc[
            active & (result["multi_peak_ratio"] > thresholds.max_multi_peak_ratio),
            "status",
        ] = "rejected_multi_peak"
    return result


def _line_from_two(points: np.ndarray) -> np.ndarray | None:
    delta = points[1] - points[0]
    norm = float(np.linalg.norm(delta))
    if norm <= np.finfo(float).eps:
        return None
    normal = np.asarray([delta[1], -delta[0]], dtype=np.float64) / norm
    return np.append(normal, -float(normal @ points[0]))


def _fit_line_svd(points: np.ndarray) -> np.ndarray:
    centre = np.mean(points, axis=0)
    _, singular, right = np.linalg.svd(points - centre, full_matrices=False)
    if len(points) < 2 or singular[0] <= np.finfo(float).eps:
        raise base.CalibrationError("二维激光点退化，无法拟合直线")
    direction = right[0]
    normal = np.asarray([direction[1], -direction[0]])
    normal /= np.linalg.norm(normal)
    return np.append(normal, -float(normal @ centre))


def line_ransac(
    pixels: np.ndarray,
    intrinsics: base.Intrinsics,
    threshold_px: float,
    config: Mapping[str, Any],
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(pixels)
    if count < 2:
        return np.zeros(count, dtype=bool), np.full(count, math.nan)
    ideal = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        P=intrinsics.camera_matrix,
    ).reshape(-1, 2)
    rng = np.random.default_rng(
        int(cfg(config, "line_ransac.random_seed")) + seed_offset
    )
    best: np.ndarray | None = None
    best_median = math.inf
    for _ in range(int(cfg(config, "line_ransac.iterations"))):
        line = _line_from_two(ideal[rng.choice(count, size=2, replace=False)])
        if line is None:
            continue
        residual = np.abs(ideal @ line[:2] + line[2])
        inliers = residual <= threshold_px
        inlier_count = int(np.count_nonzero(inliers))
        median = float(np.median(residual[inliers])) if inlier_count else math.inf
        previous = int(np.count_nonzero(best)) if best is not None else -1
        if inlier_count > previous or (inlier_count == previous and median < best_median):
            best = inliers
            best_median = median
    if best is None or np.count_nonzero(best) < 2:
        return np.zeros(count, dtype=bool), np.full(count, math.nan)
    line = _fit_line_svd(ideal[best])
    residual = np.abs(ideal @ line[:2] + line[2])
    inliers = residual <= threshold_px
    minimum = int(cfg(config, "line_ransac.min_inliers"))
    minimum_ratio = float(cfg(config, "line_ransac.min_inlier_ratio"))
    if np.count_nonzero(inliers) < minimum or np.mean(inliers) < minimum_ratio:
        return np.zeros(count, dtype=bool), residual
    return inliers, residual


def apply_line_constraint(
    diagnostics: pd.DataFrame,
    intrinsics: base.Intrinsics,
    threshold_px: float,
    config: Mapping[str, Any],
    image_id: int,
) -> pd.DataFrame:
    result = diagnostics.copy()
    indices = result.index[_candidate_mask(result)].to_numpy()
    if indices.size == 0:
        return result
    pixels = result.loc[indices, ["x_px", "y_px"]].to_numpy(dtype=np.float64)
    inliers, residual = line_ransac(
        pixels, intrinsics, threshold_px, config, seed_offset=image_id
    )
    result.loc[indices, "line_residual_px"] = residual
    result.loc[indices[~inliers], "status"] = "rejected_line_ransac"
    result.loc[indices[inliers], "status"] = "accepted_2d"
    return result


def feature_frame_from_prepared(
    prepared: base.PreparedFrame,
    config: Mapping[str, Any],
) -> FeatureFrame:
    return FeatureFrame(
        prepared.image_id,
        prepared.split,
        prepared.chess_path,
        prepared.laser_path,
        prepared.laser_gray.shape[1],
        prepared.pose.plane,
        prepared.pose.reprojection_rmse_px,
        extract_features(prepared, config),
        measure_pair_motion(prepared, config),
    )


def select_provisional_thresholds(
    tune_frames: Sequence[FeatureFrame], config: Mapping[str, Any], source_ids: str
) -> LockedThresholds:
    candidates = pd.concat(
        [
            frame.diagnostics[
                frame.diagnostics["structural_status"] == "candidate"
            ]
            for frame in tune_frames
        ],
        ignore_index=True,
    )
    if candidates.empty:
        raise base.CalibrationError("001–018 没有可用于阈值选择的结构有效点")
    full_scale = 255.0
    prominence = max(
        full_scale * float(cfg(config, "threshold_selection.prominence_floor_ratio")),
        _quantile(
            candidates["prominence"],
            float(cfg(config, "threshold_selection.prominence_quantile")),
        ),
    )
    saturated = int(
        math.ceil(
            _quantile(
                candidates["saturated_pixels"],
                float(cfg(config, "threshold_selection.saturation_count_quantile")),
            )
        )
    )
    snr = max(
        float(cfg(config, "threshold_selection.snr_floor")),
        _quantile(
            candidates["snr"],
            float(cfg(config, "threshold_selection.snr_quantile")),
        ),
    )
    padding = float(cfg(config, "threshold_selection.fwhm_padding_px"))
    fwhm_min = max(
        0.25,
        _quantile(
            candidates["fwhm_px"],
            float(cfg(config, "threshold_selection.fwhm_low_quantile")),
        )
        - padding,
    )
    fwhm_max = _quantile(
        candidates["fwhm_px"],
        float(cfg(config, "threshold_selection.fwhm_high_quantile")),
    ) + padding
    multi_peak = min(
        0.995,
        _quantile(
            candidates["multi_peak_ratio"],
            float(cfg(config, "threshold_selection.multi_peak_ratio_quantile")),
        )
        + float(cfg(config, "threshold_selection.multi_peak_ratio_padding")),
    )
    motion = np.asarray(
        [
            frame.pair_motion["median_displacement_px"]
            for frame in tune_frames
            if frame.pair_motion["tracking_ok"]
        ],
        dtype=np.float64,
    )
    motion = motion[np.isfinite(motion)]
    if motion.size:
        median = float(np.median(motion))
        mad = float(np.median(np.abs(motion - median)))
        robust = median + float(
            cfg(config, "pair_movement.threshold_mad_scale")
        ) * 1.4826 * mad
        quantile = _quantile(
            motion, float(cfg(config, "pair_movement.threshold_quantile"))
        )
        motion_warning = float(
            np.clip(
                max(robust, quantile),
                float(cfg(config, "pair_movement.minimum_warning_px")),
                float(cfg(config, "pair_movement.maximum_warning_px")),
            )
        )
    else:
        motion_warning = float(cfg(config, "pair_movement.maximum_warning_px"))
    return LockedThresholds(
        source_ids=source_ids,
        min_prominence=prominence,
        max_saturated_pixels=saturated,
        min_snr=snr,
        min_fwhm_px=fwhm_min,
        max_fwhm_px=fwhm_max,
        max_multi_peak_ratio=multi_peak,
        line_ransac_threshold_px=float(
            cfg(config, "threshold_selection.line_seed_threshold_px")
        ),
        plane_ransac_threshold_mm=float(
            cfg(config, "threshold_selection.plane_seed_threshold_mm")
        ),
        pair_movement_warning_px=motion_warning,
        low_coverage=float(cfg(config, "anomaly.minimum_coverage")),
        high_outlier_rate=float(cfg(config, "anomaly.minimum_outlier_rate")),
        high_pose_rmse_mm=float(cfg(config, "anomaly.minimum_rmse_mm")),
    )


def derive_line_threshold(
    tune_frames: Sequence[FeatureFrame],
    provisional: LockedThresholds,
    intrinsics: base.Intrinsics,
    config: Mapping[str, Any],
) -> float:
    residuals: list[np.ndarray] = []
    seed = float(cfg(config, "threshold_selection.line_seed_threshold_px"))
    for frame in tune_frames:
        filtered = apply_quality_thresholds(frame.diagnostics, provisional, config)
        candidates = filtered[filtered["status"] == "candidate"]
        if len(candidates) < 2:
            continue
        pixels = candidates[["x_px", "y_px"]].to_numpy(dtype=np.float64)
        inliers, residual = line_ransac(
            pixels, intrinsics, seed, config, seed_offset=frame.image_id
        )
        if np.any(inliers):
            residuals.append(residual[inliers])
    if not residuals:
        raise base.CalibrationError("001–018 无法建立去畸变二维激光直线")
    selected = _quantile(
        np.concatenate(residuals),
        float(cfg(config, "threshold_selection.line_residual_quantile")),
    ) * float(cfg(config, "threshold_selection.line_residual_scale"))
    return float(
        np.clip(
            selected,
            float(cfg(config, "threshold_selection.line_min_threshold_px")),
            float(cfg(config, "threshold_selection.line_max_threshold_px")),
        )
    )


def process_feature_frame(
    frame: FeatureFrame,
    thresholds: LockedThresholds,
    intrinsics: base.Intrinsics,
    config: Mapping[str, Any],
) -> base.FrameResult:
    diagnostics = apply_quality_thresholds(frame.diagnostics, thresholds, config)
    diagnostics = apply_line_constraint(
        diagnostics,
        intrinsics,
        thresholds.line_ransac_threshold_px,
        config,
        frame.image_id,
    )
    points, diagnostics = base.intersect_board_plane(
        diagnostics, frame.board_plane, intrinsics, config
    )
    diagnostics.insert(0, "split", frame.split)
    diagnostics.insert(0, "image_id", frame.image_id)
    return base.FrameResult(
        frame.image_id,
        frame.split,
        frame.chess_path,
        frame.laser_path,
        frame.image_width,
        len(points) > 0,
        "ok" if len(points) else "没有有效三维点",
        frame.reprojection_rmse_px,
        points,
        diagnostics,
    )


def weighted_plane_svd(points: np.ndarray, pose_ids: np.ndarray) -> np.ndarray:
    if len(points) < 3 or len(points) != len(pose_ids):
        raise base.CalibrationError("姿态等权 SVD 至少需要三个带姿态编号的点")
    unique, counts = np.unique(pose_ids, return_counts=True)
    count_by_pose = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_pose[int(item)] for item in pose_ids])
    weights /= np.sum(weights)
    centre = np.sum(points * weights[:, None], axis=0)
    centred = points - centre
    covariance = (centred * weights[:, None]).T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= np.finfo(float).eps:
        raise base.CalibrationError("姿态等权三维点退化，无法拟合平面")
    normal = eigenvectors[:, 0]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    return np.append(normal, -float(normal @ centre))


def stack_result_points(
    results: Sequence[base.FrameResult], excluded_ids: set[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    excluded = excluded_ids or set()
    usable = [
        result
        for result in results
        if result.image_id not in excluded and len(result.points_3d)
    ]
    if not usable:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    return (
        np.vstack([result.points_3d for result in usable]),
        np.concatenate(
            [
                np.full(len(result.points_3d), result.image_id, dtype=np.int64)
                for result in usable
            ]
        ),
    )


def balanced_plane_ransac(
    points: np.ndarray,
    pose_ids: np.ndarray,
    threshold_mm: float,
    config: Mapping[str, Any],
) -> BalancedFit:
    unique = np.unique(pose_ids)
    if len(points) < 3 or len(unique) < 2:
        raise base.CalibrationError("姿态等权 RANSAC 至少需要两个姿态和三个点")
    indices_by_pose = {
        int(pose): np.flatnonzero(pose_ids == pose) for pose in unique
    }
    rng = np.random.default_rng(int(cfg(config, "ransac.random_seed")))
    best: np.ndarray | None = None
    best_score = -math.inf
    best_median = math.inf
    for _ in range(int(cfg(config, "ransac.iterations"))):
        if len(unique) >= 3:
            sampled_poses = rng.choice(unique, size=3, replace=False)
            sampled = np.asarray(
                [rng.choice(indices_by_pose[int(pose)]) for pose in sampled_poses]
            )
        else:
            sampled = rng.choice(len(points), size=3, replace=False)
        candidate = base.plane_from_three(points[sampled])
        if candidate is None:
            continue
        distances = base.point_plane_distances(points, candidate)
        inliers = distances <= threshold_mm
        ratios = [float(np.mean(inliers[pose_ids == pose])) for pose in unique]
        score = float(np.mean(ratios))
        median = float(np.median(distances[inliers])) if np.any(inliers) else math.inf
        if score > best_score or (math.isclose(score, best_score) and median < best_median):
            best = inliers
            best_score = score
            best_median = median
    if best is None:
        raise base.CalibrationError("姿态等权 RANSAC 未找到有效平面")
    minimum = int(cfg(config, "ransac.min_inliers"))
    minimum_ratio = float(cfg(config, "ransac.min_inlier_ratio"))
    if np.count_nonzero(best) < minimum or np.mean(best) < minimum_ratio:
        raise base.CalibrationError(
            f"姿态等权 RANSAC 内点不足：{np.count_nonzero(best)}/{len(best)}"
        )
    plane = weighted_plane_svd(points[best], pose_ids[best])
    for _ in range(5):
        refined = base.point_plane_distances(points, plane) <= threshold_mm
        if np.count_nonzero(refined) < 3 or np.array_equal(refined, best):
            best = refined
            break
        best = refined
        plane = weighted_plane_svd(points[best], pose_ids[best])
    best = base.point_plane_distances(points, plane) <= threshold_mm
    return BalancedFit(plane, best, pose_ids, best_score)


def derive_model_thresholds(
    tune_results: Sequence[base.FrameResult],
    provisional: LockedThresholds,
    config: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    points, pose_ids = stack_result_points(tune_results)
    seed = balanced_plane_ransac(
        points,
        pose_ids,
        float(cfg(config, "threshold_selection.plane_seed_threshold_mm")),
        config,
    )
    distances = base.point_plane_distances(points, seed.plane)
    inlier_distances = distances[seed.inliers]
    threshold = _quantile(
        inlier_distances,
        float(cfg(config, "threshold_selection.plane_residual_quantile")),
    ) * float(cfg(config, "threshold_selection.plane_residual_scale"))
    plane_threshold = float(
        np.clip(
            threshold,
            float(cfg(config, "threshold_selection.plane_min_threshold_mm")),
            float(cfg(config, "threshold_selection.plane_max_threshold_mm")),
        )
    )
    coverages = np.asarray(
        [len(result.points_3d) / result.image_width for result in tune_results]
    )
    low_coverage = max(
        float(cfg(config, "anomaly.minimum_coverage")),
        _quantile(coverages, float(cfg(config, "anomaly.coverage_quantile")))
        * float(cfg(config, "anomaly.coverage_scale")),
    )
    outlier_rates: list[float] = []
    pose_rmse: list[float] = []
    for pose in np.unique(pose_ids):
        local = distances[pose_ids == pose]
        outlier_rates.append(float(np.mean(local > plane_threshold)))
        pose_rmse.append(float(np.sqrt(np.mean(local * local))))
    high_outlier = max(
        float(cfg(config, "anomaly.minimum_outlier_rate")),
        _quantile(
            outlier_rates, float(cfg(config, "anomaly.outlier_rate_quantile"))
        )
        + float(cfg(config, "anomaly.outlier_rate_margin")),
    )
    high_rmse = max(
        float(cfg(config, "anomaly.minimum_rmse_mm")),
        _quantile(pose_rmse, float(cfg(config, "anomaly.rmse_quantile")))
        + float(cfg(config, "anomaly.rmse_margin_mm")),
    )
    return plane_threshold, low_coverage, high_outlier, high_rmse


def update_thresholds(
    provisional: LockedThresholds, **changes: float
) -> LockedThresholds:
    values = asdict(provisional)
    values.update(changes)
    return LockedThresholds(**values)


def mark_training_inliers(
    results: Sequence[base.FrameResult],
    fit: BalancedFit,
    excluded_ids: set[int],
) -> None:
    offset = 0
    for result in results:
        indices = result.diagnostics.index[
            result.diagnostics["status"] == "accepted_3d"
        ].to_numpy()
        if result.image_id in excluded_ids:
            result.diagnostics.loc[indices, "status"] = "excluded_pair_movement"
            continue
        count = len(indices)
        local = fit.inliers[offset : offset + count]
        if len(local) != count:
            raise base.CalibrationError("三维 RANSAC 掩膜与姿态点数不一致")
        result.diagnostics.loc[indices[~local], "status"] = "rejected_ransac"
        result.diagnostics.loc[indices[local], "status"] = "accepted_final"
        offset += count


def finalise_validation(results: Sequence[base.FrameResult]) -> None:
    for result in results:
        result.diagnostics.loc[
            result.diagnostics["status"] == "accepted_3d", "status"
        ] = "accepted_final"


def _diagnostic_points(
    result: base.FrameResult, statuses: set[str]
) -> np.ndarray:
    frame = result.diagnostics[result.diagnostics["status"].isin(statuses)]
    columns = ["x_mm", "y_mm", "z_mm"]
    if frame.empty or not set(columns) <= set(frame.columns):
        return np.empty((0, 3), dtype=np.float64)
    points = frame[columns].to_numpy(dtype=np.float64)
    return points[np.all(np.isfinite(points), axis=1)]


def distance_summary(points: np.ndarray, plane: np.ndarray) -> dict[str, float]:
    distances = base.point_plane_distances(points, plane)
    if not len(distances):
        return {key: math.nan for key in ("mae_mm", "rmse_mm", "p95_mm", "max_mm")}
    return {
        "mae_mm": float(np.mean(distances)),
        "rmse_mm": float(np.sqrt(np.mean(distances * distances))),
        "p95_mm": float(np.percentile(distances, 95)),
        "max_mm": float(np.max(distances)),
    }


def overall_metrics(
    algorithm: str,
    split: str,
    results: Sequence[base.FrameResult],
    plane: np.ndarray,
) -> dict[str, Any]:
    points_by_pose = [
        _diagnostic_points(result, {"accepted_final"}) for result in results
    ]
    usable = [points for points in points_by_pose if len(points)]
    points = np.vstack(usable) if usable else np.empty((0, 3))
    summary = distance_summary(points, plane)
    pose_rmse = [
        distance_summary(points, plane)["rmse_mm"] for points in usable
    ]
    return {
        "algorithm": algorithm,
        "split": split,
        "poses": len(results),
        "successful_poses": len(usable),
        "valid_points": len(points),
        "coverage": (
            len(points) / sum(result.image_width for result in results)
            if results
            else math.nan
        ),
        "equal_pose_mean_rmse_mm": float(np.mean(pose_rmse)) if pose_rmse else math.nan,
        **summary,
    }


def pose_metrics(
    algorithm: str,
    results: Sequence[base.FrameResult],
    plane: np.ndarray,
    features: Mapping[int, FeatureFrame],
    thresholds: LockedThresholds,
    config: Mapping[str, Any],
    excluded_ids: set[int] | None = None,
) -> pd.DataFrame:
    excluded = excluded_ids or set()
    rows: list[dict[str, Any]] = []
    extracted_statuses = {
        "accepted_final",
        "rejected_ransac",
        "excluded_pair_movement",
    }
    severe = float(cfg(config, "pair_movement.severe_px"))
    for result in results:
        feature = features[result.image_id]
        extracted = _diagnostic_points(result, extracted_statuses)
        valid = _diagnostic_points(result, {"accepted_final"})
        summary = distance_summary(valid, plane)
        all_distances = base.point_plane_distances(extracted, plane)
        outlier_rate = (
            float(np.mean(all_distances > thresholds.plane_ransac_threshold_mm))
            if len(all_distances)
            else math.nan
        )
        motion = feature.pair_motion
        movement = float(motion["median_displacement_px"])
        flags: list[str] = []
        if not bool(motion["tracking_ok"]):
            flags.append("pair_tracking_low")
        if np.isfinite(movement) and movement > thresholds.pair_movement_warning_px:
            flags.append("pair_movement")
        coverage = len(valid) / result.image_width
        extraction_coverage = len(extracted) / result.image_width
        if extraction_coverage < thresholds.low_coverage:
            flags.append("low_coverage")
        if np.isfinite(outlier_rate) and outlier_rate > thresholds.high_outlier_rate:
            flags.append("high_outlier_rate")
        if np.isfinite(summary["rmse_mm"]) and summary["rmse_mm"] > thresholds.high_pose_rmse_mm:
            flags.append("high_rmse")
        if result.image_id in excluded or (np.isfinite(movement) and movement > severe):
            recommendation = "剔除：配对移动严重"
        elif flags:
            recommendation = "复核：" + ",".join(flags)
        else:
            recommendation = "保留"
        counts = result.diagnostics["status"].value_counts()
        rows.append(
            {
                "algorithm": algorithm,
                "image_id": result.image_id,
                "split": result.split,
                "fit_used": result.split == "train" and result.image_id not in excluded,
                "extracted_3d_points": len(extracted),
                "valid_points": len(valid),
                "extraction_coverage": extraction_coverage,
                "coverage": coverage,
                "outlier_rate": outlier_rate,
                **summary,
                "pnp_reprojection_rmse_px": result.reprojection_rmse_px,
                "pair_tracked_ratio": motion["tracked_ratio"],
                "pair_registration_score": motion.get("registration_score", math.nan),
                "pair_median_displacement_px": movement,
                "pair_p95_displacement_px": motion["p95_displacement_px"],
                "pair_max_displacement_px": motion["max_displacement_px"],
                "rejected_saturation": int(counts.get("rejected_saturation", 0)),
                "rejected_snr": int(counts.get("rejected_snr", 0)),
                "rejected_fwhm": int(counts.get("rejected_fwhm", 0)),
                "rejected_multi_peak": int(counts.get("rejected_multi_peak", 0)),
                "rejected_line_ransac": int(counts.get("rejected_line_ransac", 0)),
                "anomaly_flags": ";".join(flags),
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows)


def save_overlays(
    results: Sequence[base.FrameResult],
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    radius = int(cfg(config, "reporting.overlay_point_radius_px"))
    for result in results:
        directory = output_dir / "images" / result.split / f"{result.image_id:03d}"
        laser_gray = base.to_gray(base.read_image(result.laser_path), result.laser_path)
        overlay = base._overlay_base(laser_gray, config)
        for status, colour in STATUS_COLOURS.items():
            selected = result.diagnostics[result.diagnostics["status"] == status]
            base._draw_points(overlay, selected, colour, radius)
        base.write_image(directory / "laser_filter_overlay.png", overlay)
        chess_gray = base.to_gray(base.read_image(result.chess_path), result.chess_path)
        valid = result.diagnostics[result.diagnostics["status"] == "accepted_final"]
        chess_overlay = base.create_chess_laser_overlay(chess_gray, valid, config)
        base.write_image(directory / "chess_laser_overlay.png", chess_overlay)


def save_plots(
    train_results: Sequence[base.FrameResult],
    validation_results: Sequence[base.FrameResult],
    plane: np.ndarray,
    comparison: pd.DataFrame,
    pose_table: pd.DataFrame,
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    train = np.vstack(
        [
            points
            for result in train_results
            if len(points := _diagnostic_points(result, {"accepted_final"}))
        ]
    )
    validation_parts = [
        points
        for result in validation_results
        if len(points := _diagnostic_points(result, {"accepted_final"}))
    ]
    validation = np.vstack(validation_parts) if validation_parts else np.empty((0, 3))
    maximum = int(cfg(config, "reporting.max_plot_points"))
    rng = np.random.default_rng(20260722)

    def sample(points: np.ndarray) -> np.ndarray:
        if len(points) <= maximum:
            return points
        return points[rng.choice(len(points), maximum, replace=False)]

    train_plot = sample(train)
    validation_plot = sample(validation)
    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(*train_plot.T, s=1, alpha=0.35, label="train")
    if len(validation_plot):
        axis.scatter(*validation_plot.T, s=2, alpha=0.45, label="validation")
    all_points = np.vstack([train_plot, validation_plot]) if len(validation_plot) else train_plot
    x_values = np.linspace(np.percentile(all_points[:, 0], 2), np.percentile(all_points[:, 0], 98), 12)
    y_values = np.linspace(np.percentile(all_points[:, 1], 2), np.percentile(all_points[:, 1], 98), 12)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    if abs(plane[2]) > 1e-9:
        grid_z = -(plane[0] * grid_x + plane[1] * grid_y + plane[3]) / plane[2]
        axis.plot_surface(grid_x, grid_y, grid_z, alpha=0.22, color="tab:green")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "point_cloud_plane.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist(
        base.point_plane_distances(train, plane),
        bins=70,
        alpha=0.65,
        label="train",
    )
    if len(validation):
        axis.hist(
            base.point_plane_distances(validation, plane),
            bins=70,
            alpha=0.65,
            label="validation",
        )
    axis.set_xlabel("Absolute point-to-plane distance (mm)")
    axis.set_ylabel("Count")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "residual_distribution.png", dpi=160)
    plt.close(fig)

    validation_comparison = comparison[comparison["split"] == "validation"]
    fig, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(len(validation_comparison))
    width = 0.18
    for offset, metric in enumerate(("mae_mm", "rmse_mm", "p95_mm", "max_mm")):
        axis.bar(
            x + (offset - 1.5) * width,
            validation_comparison[metric],
            width,
            label=metric.replace("_mm", ""),
        )
    axis.set_xticks(x, validation_comparison["algorithm"])
    axis.set_ylabel("Validation error (mm)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "algorithm_comparison.png", dpi=160)
    plt.close(fig)

    selected = pose_table[pose_table["algorithm"] == "v2_equal_pose"]
    fig, axis = plt.subplots(figsize=(12, 5))
    colours = np.where(selected["split"] == "validation", "tab:orange", "tab:blue")
    axis.bar(selected["image_id"].astype(str), selected["rmse_mm"], color=colours)
    axis.axhline(
        selected.attrs.get("high_pose_rmse_mm", math.nan),
        color="tab:red",
        linestyle="--",
        label="locked anomaly threshold",
    )
    axis.set_xlabel("Pose ID")
    axis.set_ylabel("RMSE (mm)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_pose_rmse.png", dpi=160)
    plt.close(fig)


def write_reports(
    args: argparse.Namespace,
    intrinsics: base.Intrinsics,
    thresholds: LockedThresholds,
    new_plane: np.ndarray,
    old_plane: np.ndarray,
    comparison: pd.DataFrame,
    poses: pd.DataFrame,
    excluded_ids: set[int],
) -> None:
    output = args.output_dir
    coefficients = {key: float(value) for key, value in zip("abcd", new_plane)}
    plane_document = {
        "model": "normalized_plane_ax_by_cz_d_equals_0",
        "coordinate_system": "camera",
        "coordinate_unit": "mm",
        "normal_is_unit_length": True,
        "coefficients": coefficients,
        "threshold_tuning_ids": args.tune_ids,
        "training_ids": args.train_ids,
        "validation_ids": args.val_ids,
        "validation_used_for_thresholds_or_fitting": False,
        "excluded_training_pose_ids": sorted(excluded_ids),
        "fitting_weight": "equal_total_weight_per_pose",
        "intrinsics_source": str(args.intrinsics.resolve()),
        "image_size": list(intrinsics.image_size),
        "locked_thresholds": asdict(thresholds),
    }
    (output / "laser_plane.yaml").write_text(
        yaml.safe_dump(plane_document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output / "locked_thresholds.yaml").write_text(
        yaml.safe_dump(asdict(thresholds), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    old_document = {
        "algorithm": "old_A7_point_weighted",
        "coefficients": {key: float(value) for key, value in zip("abcd", old_plane)},
        "same_intrinsics_and_splits_as_v2": True,
    }
    (output / "old_algorithm_baseline.yaml").write_text(
        yaml.safe_dump(old_document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    new_validation = comparison[
        (comparison["algorithm"] == "v2_equal_pose")
        & (comparison["split"] == "validation")
    ].iloc[0]
    old_validation = comparison[
        (comparison["algorithm"] == "old_A7_point_weighted")
        & (comparison["split"] == "validation")
    ].iloc[0]
    delta = float(new_validation["rmse_mm"] - old_validation["rmse_mm"])
    verdict = "改善" if delta < 0 else "变差"
    anomalies = poses[
        (poses["algorithm"] == "v2_equal_pose") & (poses["anomaly_flags"] != "")
    ]
    md = f"""# calib02 线激光平面标定 V2

## 最终平面

相机坐标系、毫米单位、单位法向量：

`{new_plane[0]:.12g} x + {new_plane[1]:.12g} y + {new_plane[2]:.12g} z + {new_plane[3]:.12g} = 0`

阈值仅由 `{thresholds.source_ids}` 选择；019–025 只补充最终拟合，026–032 在平面冻结后处理。验证集未用于阈值、RANSAC或拟合。

## 处理流程

```mermaid
flowchart LR
    A["001–018 提取特征"] --> B["锁定独立质量阈值"]
    B --> C["去畸变二维直线 RANSAC"]
    C --> D["001–025 射线-板面交点"]
    D --> E["姿态等权三维 RANSAC + SVD"]
    E --> F["冻结激光平面"]
    F --> G["026–032 独立验证"]
    G --> H["异常诊断与旧算法对比"]
```

## 新旧算法对比

{markdown_table(comparison)}

验证 RMSE 相对旧 A7 算法{verdict} `{abs(delta):.6f} mm`。

## 异常姿态

{markdown_table(anomalies[["image_id", "split", "valid_points", "coverage", "outlier_rate", "rmse_mm", "anomaly_flags", "recommendation"]]) if len(anomalies) else "未发现达到锁定阈值的异常姿态。"}

## 图例与文件

- 棋盘叠加图：绿色激光线、洋红色最终有效点。
- 激光分类图：红=饱和，黄=低 SNR，蓝=FWHM，青=多峰，紫=棋盘边界，橙=二维直线 RANSAC 外点，绿=最终有效点。
- 主要数据：`laser_plane.yaml`、`overall_metrics.csv`、`pose_metrics.csv`、`anomaly_diagnostics.csv`、`algorithm_comparison.csv`、`locked_thresholds.yaml`。
"""
    (output / "calibration_report.md").write_text(md, encoding="utf-8")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>calib02 线激光平面标定 V2</title><style>
body{{max-width:1180px;margin:32px auto;padding:0 24px;font:15px/1.6 system-ui;color:#172033}}
h1,h2{{color:#0b3a66}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #ccd5df;padding:6px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.flow{{padding:14px;background:#edf7ff;border-left:4px solid #2780b8}} .grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}
img{{max-width:100%}} code{{background:#edf2f7;padding:2px 5px}}</style></head><body>
<h1>calib02 线激光平面标定 V2</h1>
<p><code>{new_plane[0]:.12g}x + {new_plane[1]:.12g}y + {new_plane[2]:.12g}z + {new_plane[3]:.12g} = 0</code></p>
<p class="flow">001–018 锁阈值 → 001–025 姿态等权拟合 → 冻结平面 → 026–032 独立验证 → 异常与旧算法对比</p>
<h2>新旧算法对比</h2>{comparison.to_html(index=False, float_format=lambda value: f"{value:.6g}")}
<p>验证 RMSE 相对旧 A7 算法{verdict} <b>{abs(delta):.6f} mm</b>。</p>
<h2>异常姿态</h2>{anomalies.to_html(index=False, float_format=lambda value: f"{value:.6g}") if len(anomalies) else '<p>未发现达到锁定阈值的异常姿态。</p>'}
<h2>可视化</h2><div class="grid"><img src="algorithm_comparison.png"><img src="per_pose_rmse.png"><img src="point_cloud_plane.png"><img src="residual_distribution.png"></div>
<h2>图例</h2><p>棋盘图：绿色激光线、洋红色有效点。分类图：红饱和、黄低 SNR、蓝 FWHM、青多峰、紫边界、橙二维直线外点。</p>
</body></html>"""
    (output / "calibration_report.html").write_text(html, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_split_contract(args)
    setup_logging(args.output_dir)
    config = load_config(args.config)
    old_config = base.load_config(args.old_config)
    intrinsics = base.load_intrinsics(args.intrinsics)
    LOGGER.info("阈值选择集：%s；训练集：%s；验证集：%s", args.tune_ids, args.train_ids, args.val_ids)
    LOGGER.info("开始处理前尚未读取任何验证图像")

    old_spec = next(spec for spec in base.build_experiment_specs() if spec.name == "A7")
    features: dict[int, FeatureFrame] = {}
    new_train: list[base.FrameResult] = []
    old_train: list[base.FrameResult] = []
    tune_frames: list[FeatureFrame] = []
    old_tune: list[base.FrameResult] = []

    for image_id in args.tune_ids:
        pair = image_pair(args, image_id, "train")
        prepared = prepare_frame_robust(pair, args, intrinsics, config)
        feature = feature_frame_from_prepared(prepared, config)
        tune_frames.append(feature)
        features[image_id] = feature
        old_tune.append(
            base.process_prepared_frame(prepared, old_spec, intrinsics, old_config)
        )
        LOGGER.info("tune ID %03d 特征提取完成", image_id)

    source_ids = format_ids(args.tune_ids)
    provisional = select_provisional_thresholds(tune_frames, config, source_ids)
    line_threshold = derive_line_threshold(
        tune_frames, provisional, intrinsics, config
    )
    provisional = update_thresholds(
        provisional, line_ransac_threshold_px=line_threshold
    )
    tune_results = [
        process_feature_frame(frame, provisional, intrinsics, config)
        for frame in tune_frames
    ]
    plane_threshold, low_coverage, high_outlier, high_rmse = derive_model_thresholds(
        tune_results, provisional, config
    )
    thresholds = update_thresholds(
        provisional,
        plane_ransac_threshold_mm=plane_threshold,
        low_coverage=low_coverage,
        high_outlier_rate=high_outlier,
        high_pose_rmse_mm=high_rmse,
    )
    (args.output_dir / "locked_thresholds.yaml").write_text(
        yaml.safe_dump(asdict(thresholds), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "threshold": key,
                "value": value,
                "source_ids": source_ids,
                "validation_accessed": False,
            }
            for key, value in asdict(thresholds).items()
            if key != "source_ids"
        ]
    ).to_csv(
        args.output_dir / "threshold_selection_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    LOGGER.info("所有阈值已由 %s 锁定：%s", source_ids, asdict(thresholds))

    new_train.extend(tune_results)
    old_train.extend(old_tune)
    remaining_train = [item for item in args.train_ids if item not in set(args.tune_ids)]
    for image_id in remaining_train:
        pair = image_pair(args, image_id, "train")
        prepared = prepare_frame_robust(pair, args, intrinsics, config)
        feature = feature_frame_from_prepared(prepared, config)
        features[image_id] = feature
        new_train.append(process_feature_frame(feature, thresholds, intrinsics, config))
        old_train.append(
            base.process_prepared_frame(prepared, old_spec, intrinsics, old_config)
        )
        LOGGER.info("train-only ID %03d 使用锁定阈值处理完成", image_id)

    severe = float(cfg(config, "pair_movement.severe_px"))
    excluded_ids = {
        image_id
        for image_id in args.train_ids
        if np.isfinite(float(features[image_id].pair_motion["median_displacement_px"]))
        and float(features[image_id].pair_motion["median_displacement_px"]) > severe
    }
    new_points, new_pose_ids = stack_result_points(new_train, excluded_ids)
    new_fit = balanced_plane_ransac(
        new_points,
        new_pose_ids,
        thresholds.plane_ransac_threshold_mm,
        config,
    )
    mark_training_inliers(new_train, new_fit, excluded_ids)
    LOGGER.info("V2 训练平面已冻结：%s", new_fit.plane.tolist())

    old_run = base.ExperimentRun(old_spec, train_results=old_train)
    base.fit_experiment(old_run, old_config)
    if old_run.plane is None:
        raise base.CalibrationError(f"旧 A7 基线拟合失败：{old_run.fit_error}")
    LOGGER.info("旧 A7 训练平面已冻结：%s", old_run.plane.tolist())

    LOGGER.info("新旧训练平面均已冻结，现在才开始读取验证图像")
    new_validation: list[base.FrameResult] = []
    old_validation: list[base.FrameResult] = []
    for image_id in args.val_ids:
        pair = image_pair(args, image_id, "validation")
        prepared = prepare_frame_robust(pair, args, intrinsics, config)
        feature = feature_frame_from_prepared(prepared, config)
        features[image_id] = feature
        new_validation.append(
            process_feature_frame(feature, thresholds, intrinsics, config)
        )
        old_validation.append(
            base.process_prepared_frame(prepared, old_spec, intrinsics, old_config)
        )
        LOGGER.info("validation ID %03d 使用冻结参数处理完成", image_id)
    finalise_validation(new_validation)
    old_run.validation_results = old_validation
    base.finalise_validation(old_run)

    comparison = pd.DataFrame(
        [
            overall_metrics("v2_equal_pose", "train", new_train, new_fit.plane),
            overall_metrics(
                "v2_equal_pose", "validation", new_validation, new_fit.plane
            ),
            overall_metrics(
                "old_A7_point_weighted", "train", old_train, old_run.plane
            ),
            overall_metrics(
                "old_A7_point_weighted",
                "validation",
                old_validation,
                old_run.plane,
            ),
        ]
    )
    new_pose_table = pose_metrics(
        "v2_equal_pose",
        [*new_train, *new_validation],
        new_fit.plane,
        features,
        thresholds,
        config,
        excluded_ids,
    )
    old_pose_table = pose_metrics(
        "old_A7_point_weighted",
        [*old_train, *old_validation],
        old_run.plane,
        features,
        thresholds,
        config,
    )
    poses = pd.concat([new_pose_table, old_pose_table], ignore_index=True)
    poses.attrs["high_pose_rmse_mm"] = thresholds.high_pose_rmse_mm
    comparison.to_csv(
        args.output_dir / "algorithm_comparison.csv", index=False, encoding="utf-8-sig"
    )
    comparison[comparison["algorithm"] == "v2_equal_pose"].to_csv(
        args.output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig"
    )
    poses.to_csv(args.output_dir / "pose_metrics.csv", index=False, encoding="utf-8-sig")
    new_pose_table[new_pose_table["anomaly_flags"] != ""].to_csv(
        args.output_dir / "anomaly_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {"image_id": image_id, **features[image_id].pair_motion}
            for image_id in [*args.train_ids, *args.val_ids]
        ]
    ).to_csv(args.output_dir / "pair_movement.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "image_id": image_id,
                "split": "train" if image_id in set(args.train_ids) else "validation",
                "chess_path": str(features[image_id].chess_path.resolve()),
                "laser_path": str(features[image_id].laser_path.resolve()),
                "pair_exists": True,
                "image_width": intrinsics.image_size[0],
                "image_height": intrinsics.image_size[1],
                "intrinsics_width": intrinsics.image_size[0],
                "intrinsics_height": intrinsics.image_size[1],
                "size_matches_intrinsics": True,
                "pnp_reprojection_rmse_px": features[image_id].reprojection_rmse_px,
            }
            for image_id in [*args.train_ids, *args.val_ids]
        ]
    ).to_csv(args.output_dir / "input_validation.csv", index=False, encoding="utf-8-sig")
    new_pose_table[
        [
            "rejected_saturation",
            "rejected_snr",
            "rejected_fwhm",
            "rejected_multi_peak",
            "rejected_line_ransac",
        ]
    ].sum().rename("rejected_points").to_csv(
        args.output_dir / "filter_rejection_summary.csv", encoding="utf-8-sig"
    )

    save_overlays([*new_train, *new_validation], args.output_dir, config)
    save_plots(
        new_train,
        new_validation,
        new_fit.plane,
        comparison,
        poses,
        args.output_dir,
        config,
    )
    write_reports(
        args,
        intrinsics,
        thresholds,
        new_fit.plane,
        old_run.plane,
        comparison,
        poses,
        excluded_ids,
    )
    LOGGER.info("全部结果已写入：%s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
