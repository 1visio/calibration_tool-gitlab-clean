#!/usr/bin/env python3
"""Core, leakage-safe line-laser plane calibration from chess/nolaser/laser triplets.

The script intentionally keeps only the geometry and signal-processing steps
that directly support a reliable plane estimate.  It depends on the common
OpenCV YAML and geometry helpers in ``calibrate_laser_plane_opencv.py``.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import calibrate_laser_plane_opencv as common

# Workflow runs inside the PySide/Qt GUI process.  TkAgg cannot be loaded
# after Qt has claimed the application event loop; all figures are written to
# disk, so an off-screen backend is the correct default here.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("laser_plane_core")
DEFAULT_CONFIG = Path(__file__).with_name("laser_plane_core_config.yaml")


@dataclass(frozen=True)
class ImagePair:
    image_id: int
    split: str
    chess_path: Path
    nolaser_path: Path
    laser_path: Path


@dataclass
class CoreFrame:
    result: common.FrameResult
    pair_motion: dict[str, Any]


@dataclass(frozen=True)
class PlaneFit:
    plane: np.ndarray
    inliers: np.ndarray
    balanced_inlier_ratio: float


def nested(config: Mapping[str, Any], key: str) -> Any:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise common.CalibrationError(f"配置缺少字段：{key}")
        value = value[part]
    return value


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise common.CalibrationError("配置文件根节点必须是映射")
    required = (
        "chessboard.corner_window_px",
        "laser.background_window_px",
        "laser.min_peak_prominence_ratio",
        "line_ransac.threshold_px",
        "plane_ransac.threshold_mm",
        "intersection.min_depth_mm",
        "pair_motion.severe_px",
        "visualisation.dpi",
    )
    for key in required:
        nested(data, key)
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument(
        "--fit-dir",
        type=Path,
        required=True,
        help="拟合集目录，目录内应包含同编号 chess/nolaser/laser 图像",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="独立验证集目录，目录内应包含同编号 chess/nolaser/laser 图像",
    )
    parser.add_argument("--chess-pattern", default="chess {id:03d}.tif")
    parser.add_argument("--nolaser-pattern", default="nolaser {id:03d}.tif")
    parser.add_argument("--laser-pattern", default="laser {id:03d}.tif")
    parser.add_argument(
        "--fit-ids",
        type=common.parse_id_spec,
        default=None,
        help="可选；如 1-18。省略时自动扫描拟合集目录中的配对编号",
    )
    parser.add_argument(
        "--test-ids",
        type=common.parse_id_spec,
        default=None,
        help="可选；如 1-6 或 19-24。省略时自动扫描验证集目录中的配对编号",
    )
    parser.add_argument("--manual-exclude-ids", type=common.parse_id_spec, default=[])
    parser.add_argument("--pattern-cols", type=common.positive_int, default=6)
    parser.add_argument("--pattern-rows", type=common.positive_int, default=5)
    parser.add_argument("--square-size-mm", type=common.positive_float, default=30.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        output_dir / "processing_log.txt", encoding="utf-8"
    )
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{args.output_dir}")
    for label, directory in (("拟合集", args.fit_dir), ("验证集", args.test_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label}目录不存在：{directory}")


def image_path(directory: Path, pattern: str, image_id: int) -> Path:
    return common.format_image_path(directory, pattern, image_id)


def pattern_regex(pattern: str) -> re.Pattern[str]:
    """Convert a path pattern containing one ``{id...}`` field to a regex."""
    normalised = pattern.replace("\\", "/")
    parts: list[str] = []
    id_fields = 0
    for literal, field_name, _format_spec, _conversion in Formatter().parse(normalised):
        parts.append(re.escape(literal))
        if field_name is None:
            continue
        if field_name != "id":
            raise common.CalibrationError(
                f"图像模式只允许使用 {{id}} 字段：{pattern!r}"
            )
        id_fields += 1
        parts.append(r"(?P<id>\d+)")
    if id_fields != 1:
        raise common.CalibrationError(
            f"图像模式必须且只能包含一个 {{id}} 字段：{pattern!r}"
        )
    return re.compile("^" + "".join(parts) + "$", flags=re.IGNORECASE)


def discover_ids(directory: Path, pattern: str) -> set[int]:
    matcher = pattern_regex(pattern)
    identifiers: set[int] = set()
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        match = matcher.fullmatch(relative)
        if match is not None:
            identifiers.add(int(match.group("id")))
    return identifiers


def resolve_pair_ids(
    directory: Path,
    chess_pattern: str,
    nolaser_pattern: str,
    laser_pattern: str,
    requested: Sequence[int] | None,
    label: str,
) -> list[int]:
    chess_ids = discover_ids(directory, chess_pattern)
    nolaser_ids = discover_ids(directory, nolaser_pattern)
    laser_ids = discover_ids(directory, laser_pattern)
    complete = chess_ids & nolaser_ids & laser_ids

    all_ids = chess_ids | nolaser_ids | laser_ids
    incomplete = sorted(all_ids - complete)
    if incomplete:
        details = []
        for image_id in incomplete:
            missing = []
            if image_id not in chess_ids:
                missing.append("chess")
            if image_id not in nolaser_ids:
                missing.append("nolaser")
            if image_id not in laser_ids:
                missing.append("laser")
            details.append(f"{image_id:03d} 缺少 {'/'.join(missing)}")
        LOGGER.warning("%s存在不完整三联图：%s", label, "; ".join(details))

    if requested is None:
        identifiers = sorted(complete)
    else:
        identifiers = list(requested)
        missing = sorted(set(identifiers) - complete)
        if missing:
            raise FileNotFoundError(
                f"{label}指定编号缺少 chess/nolaser/laser 完整三联图：{missing}"
            )

    if not identifiers:
        raise common.CalibrationError(
            f"{label}未找到完整三联图；目录={directory}，"
            f"chess-pattern={chess_pattern!r}，"
            f"nolaser-pattern={nolaser_pattern!r}，"
            f"laser-pattern={laser_pattern!r}"
        )
    return identifiers


def validate_inputs(
    args: argparse.Namespace, intrinsics: common.Intrinsics
) -> list[ImagePair]:
    args.fit_ids = resolve_pair_ids(
        args.fit_dir,
        args.chess_pattern,
        args.nolaser_pattern,
        args.laser_pattern,
        args.fit_ids,
        "拟合集",
    )
    args.test_ids = resolve_pair_ids(
        args.test_dir,
        args.chess_pattern,
        args.nolaser_pattern,
        args.laser_pattern,
        args.test_ids,
        "验证集",
    )

    pairs: list[ImagePair] = []
    expected = (intrinsics.image_size[1], intrinsics.image_size[0])
    groups = (
        ("train", args.fit_dir, args.fit_ids),
        ("validation", args.test_dir, args.test_ids),
    )
    for split, directory, identifiers in groups:
        for image_id in identifiers:
            chess = image_path(directory, args.chess_pattern, image_id)
            nolaser = image_path(directory, args.nolaser_pattern, image_id)
            laser = image_path(directory, args.laser_pattern, image_id)
            for path in (chess, nolaser, laser):
                if not path.is_file():
                    raise FileNotFoundError(f"缺少三联图图像：{path}")
                image = common.read_image(path)
                shape = image.shape[:2]
                if shape != expected:
                    raise common.CalibrationError(
                        f"{path.name} 尺寸 {shape[::-1]} 与内参尺寸 "
                        f"{intrinsics.image_size} 不一致"
                    )
            pairs.append(ImagePair(image_id, split, chess, nolaser, laser))
    LOGGER.info(
        "输入检查通过：%d 个拟合姿态，%d 个验证姿态，尺寸=%s",
        len(args.fit_ids),
        len(args.test_ids),
        intrinsics.image_size,
    )
    LOGGER.info("拟合集编号：%s", args.fit_ids)
    LOGGER.info("验证集编号：%s", args.test_ids)
    return pairs


def detect_board_pose(
    gray: np.ndarray,
    args: argparse.Namespace,
    intrinsics: common.Intrinsics,
    config: Mapping[str, Any],
) -> common.BoardPose:
    """Try classic chess detection, then the more robust SB detector."""
    try:
        return common.detect_board_pose(
            gray,
            args.pattern_cols,
            args.pattern_rows,
            args.square_size_mm,
            intrinsics,
            config,
        )
    except (common.CalibrationError, cv2.error):
        pass

    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(
        gray, (args.pattern_cols, args.pattern_rows), flags
    )
    if not found or corners is None:
        raise common.CalibrationError("传统与 SB 方法均未找到完整棋盘格内角点")
    window = int(nested(config, "chessboard.corner_window_px"))
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        int(nested(config, "chessboard.corner_max_iterations")),
        float(nested(config, "chessboard.corner_epsilon")),
    )
    refined = cv2.cornerSubPix(
        gray, corners.astype(np.float32), (window, window), (-1, -1), criteria
    )
    board = common.object_points(
        args.pattern_cols, args.pattern_rows, args.square_size_mm
    )
    solved, rvec, tvec = cv2.solvePnP(
        board,
        refined,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise common.CalibrationError("solvePnP 未能求得棋盘姿态")
    projected, _ = cv2.projectPoints(
        board, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    delta = refined.reshape(-1, 2) - projected.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    maximum = float(nested(config, "chessboard.max_reprojection_rmse_px"))
    if rmse > maximum:
        raise common.CalibrationError(
            f"棋盘 PnP 重投影 RMSE={rmse:.3f}px，超过 {maximum:.3f}px"
        )
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2].astype(np.float64)
    normal /= np.linalg.norm(normal)
    point = tvec.reshape(3).astype(np.float64)
    plane = np.append(normal, -float(normal @ point))
    border = float(nested(config, "chessboard.outer_border_squares"))
    inset = float(nested(config, "chessboard.boundary_inset_mm"))
    square = float(args.square_size_mm)
    outer = np.asarray(
        [
            [-border * square + inset, -border * square + inset, 0.0],
            [(args.pattern_cols - 1 + border) * square - inset, -border * square + inset, 0.0],
            [(args.pattern_cols - 1 + border) * square - inset, (args.pattern_rows - 1 + border) * square - inset, 0.0],
            [-border * square + inset, (args.pattern_rows - 1 + border) * square - inset, 0.0],
        ],
        dtype=np.float32,
    )
    polygon, _ = cv2.projectPoints(
        outer, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    return common.BoardPose(
        refined.reshape(-1, 2),
        rvec,
        tvec,
        plane,
        polygon.reshape(-1, 2).astype(np.float32),
        rmse,
    )


def prepare_frame(
    pair: ImagePair,
    args: argparse.Namespace,
    intrinsics: common.Intrinsics,
    config: Mapping[str, Any],
) -> common.PreparedFrame:
    chess = common.to_gray(common.read_image(pair.chess_path), pair.chess_path)
    nolaser = common.to_gray(common.read_image(pair.nolaser_path), pair.nolaser_path)
    laser = common.to_gray(common.read_image(pair.laser_path), pair.laser_path)
    pose = detect_board_pose(
        common.chessboard_detection_image(chess), args, intrinsics, config
    )

    # nolaser 与 laser 使用相同短曝光，直接相减可更干净地去除棋盘纹理和环境光。
    image = laser.astype(np.float32)
    background = nolaser.astype(np.float32)
    corrected = np.maximum(image - background, 0.0)
    full_scale = common._full_scale(laser, config)
    sigma = float(nested(config, "laser.profile_smoothing_sigma_px"))
    smoothed = (
        gaussian_filter1d(corrected, sigma=sigma, axis=0, mode="nearest")
        if sigma > 0
        else corrected.copy()
    )
    raw_smoothed = (
        gaussian_filter1d(image, sigma=sigma, axis=0, mode="nearest")
        if sigma > 0
        else image.copy()
    )
    boundary = common.build_chess_boundary_mask(
        laser.shape, pose, args, intrinsics, config
    )
    return common.PreparedFrame(
        pair.image_id,
        pair.split,
        pair.chess_path,
        pair.laser_path,
        laser,
        full_scale,
        pose,
        background,
        corrected,
        raw_smoothed,
        smoothed,
        boundary,
    )


def _tracking_image(gray: np.ndarray) -> np.ndarray | None:
    """Make an exposure-invariant edge image for chess-to-laser matching.

    The chess frame is intentionally captured at a much longer exposure than
    the laser frame. Matching raw DN values would therefore turn a valid pair
    into a correlation failure. A local contrast stretch followed by a Sobel
    magnitude keeps board edges while suppressing the absolute exposure
    difference. A completely flat frame is reported as unavailable instead of
    allowing ``matchTemplate`` to return an arbitrary location.
    """
    image = np.asarray(gray)
    if image.ndim != 2 or image.size == 0:
        return None
    values = image.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1.0e-6:
        return None
    scaled = np.clip((values - low) * 255.0 / (high - low), 0.0, 255.0).astype(
        np.uint8
    )
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(scaled)
    gradient_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    if float(np.max(magnitude)) <= 1.0e-6:
        return None
    return cv2.normalize(magnitude, None, 0.0, 255.0, cv2.NORM_MINMAX).astype(
        np.uint8
    )


def measure_pair_motion(
    prepared: common.PreparedFrame, nolaser_path: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Check chess-to-laser registration without treating ``nolaser`` as texture.

    ``nolaser`` is deliberately kept at the short laser exposure so it can be
    subtracted from ``laser``. At that exposure the board is often invisible,
    so using it as a template source produces false boundary matches. The
    long-exposure chess image is the only board-textured reference. If the
    laser image itself contains too little board texture for a reliable match,
    the result is explicitly ``unresolved_low_texture`` and is handled as a
    manual-review warning by the workflow quality gate.
    """
    del nolaser_path  # retained in the signature for callers and old plugins
    motion_config = nested(config, "pair_motion")
    if not bool(motion_config.get("enabled", False)):
        return {
            "tracking_ok": False,
            "movement_method": "disabled",
            "tracking_status": "disabled",
        }

    chess = common.to_gray(common.read_image(prepared.chess_path), prepared.chess_path)
    laser = common.to_gray(common.read_image(prepared.laser_path), prepared.laser_path)
    reference = _tracking_image(chess)
    target = _tracking_image(laser)
    tracking_reference = "chess"
    tracking_target = "laser"
    patch = int(motion_config.get("template_patch_radius_px", 15))
    search = int(motion_config.get("template_search_radius_px", 14))
    minimum_correlation = float(motion_config.get("template_min_correlation", 0.45))
    minimum_patch_std = float(motion_config.get("tracking_min_patch_std_dn", 3.0))
    minimum_valid_ratio = float(
        motion_config.get("tracking_min_valid_patch_ratio", 0.50)
    )
    reject_boundary = bool(motion_config.get("tracking_reject_boundary", True))
    total_corners = len(prepared.pose.corners)

    def unresolved(
        method: str,
        reason: str,
        *,
        valid_patch_count: int = 0,
        matched_patch_count: int = 0,
        consensus_ratio: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "tracking_ok": False,
            "movement_method": method,
            "tracking_status": "not_assessed" if method == "unresolved_low_texture" else "unresolved",
            "tracking_reason": reason,
            "tracking_reference": tracking_reference,
            "tracking_target": tracking_target,
            "valid_patch_count": valid_patch_count,
            "matched_patch_count": matched_patch_count,
            "valid_patch_ratio": float(valid_patch_count / total_corners) if total_corners else 0.0,
            "template_consensus_ratio": consensus_ratio,
            "median_displacement_px": math.nan,
        }

    if reference is None or target is None:
        return unresolved("unresolved_low_texture", "tracking_image_flat")
    if reference.shape != target.shape or total_corners == 0:
        return unresolved("unresolved_low_texture", "tracking_shape_or_corners_invalid")

    shifts: list[tuple[float, float]] = []
    correlations: list[float] = []
    valid_patch_count = 0
    boundary_match_count = 0
    margin = patch + search
    for corner_x, corner_y in prepared.pose.corners:
        x, y = int(round(float(corner_x))), int(round(float(corner_y)))
        if (
            x - margin < 0
            or y - margin < 0
            or x + margin + 1 > reference.shape[1]
            or y + margin + 1 > reference.shape[0]
        ):
            continue
        template = reference[y - patch : y + patch + 1, x - patch : x + patch + 1]
        region = target[y - margin : y + margin + 1, x - margin : x + margin + 1]
        if (
            float(np.std(template)) < minimum_patch_std
            or float(np.std(region)) < minimum_patch_std
        ):
            continue
        valid_patch_count += 1
        score = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, maximum, _, location = cv2.minMaxLoc(score)
        at_boundary = (
            location[0] <= 0
            or location[1] <= 0
            or location[0] >= score.shape[1] - 1
            or location[1] >= score.shape[0] - 1
        )
        if at_boundary and reject_boundary:
            boundary_match_count += 1
            continue
        if maximum >= minimum_correlation:
            shifts.append((location[0] - search, location[1] - search))
            correlations.append(float(maximum))

    minimum_valid_count = max(1, int(math.ceil(minimum_valid_ratio * total_corners)))
    if valid_patch_count < minimum_valid_count:
        return unresolved(
            "unresolved_low_texture",
            "target_patch_texture_insufficient",
            valid_patch_count=valid_patch_count,
            matched_patch_count=len(shifts),
        )
    if not shifts:
        return unresolved(
            "unresolved_low_texture",
            "target_texture_not_matchable"
            if boundary_match_count
            else "no_match_above_correlation",
            valid_patch_count=valid_patch_count,
        )

    values = np.asarray(shifts, dtype=np.float64)
    centre = np.median(values, axis=0)
    consensus = np.linalg.norm(values - centre, axis=1) <= float(
        motion_config.get("template_consensus_radius_px", 1.5)
    )
    ratio = float(np.count_nonzero(consensus) / max(valid_patch_count, 1))
    if len(shifts) < max(1, int(math.ceil(minimum_valid_ratio * valid_patch_count))):
        return unresolved(
            "unresolved_low_texture",
            "target_texture_not_matchable",
            valid_patch_count=valid_patch_count,
            matched_patch_count=len(shifts),
            consensus_ratio=ratio,
        )
    if ratio < float(motion_config.get("template_min_consensus_ratio", 0.50)):
        return unresolved(
            "unresolved",
            "inconsistent_matches",
            valid_patch_count=valid_patch_count,
            matched_patch_count=len(shifts),
            consensus_ratio=ratio,
        )
    consensus_shifts = values[consensus]
    displacement = np.linalg.norm(consensus_shifts, axis=1)
    return {
        "tracking_ok": True,
        "movement_method": "chess_to_laser_gradient_template",
        "tracking_status": "tracked",
        "tracking_reason": "ok",
        "tracking_reference": tracking_reference,
        "tracking_target": tracking_target,
        "valid_patch_count": valid_patch_count,
        "matched_patch_count": len(shifts),
        "valid_patch_ratio": float(valid_patch_count / total_corners),
        "template_consensus_ratio": ratio,
        "median_correlation": float(np.median(np.asarray(correlations)[consensus])),
        "median_dx_px": float(np.median(consensus_shifts[:, 0])),
        "median_dy_px": float(np.median(consensus_shifts[:, 1])),
        "median_displacement_px": float(np.median(displacement)),
        "p95_displacement_px": float(np.percentile(displacement, 95)),
        "max_displacement_px": float(np.max(displacement)),
    }


def extract_centres(
    prepared: common.PreparedFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    signal = prepared.corrected_smoothed
    height, width = signal.shape
    half = int(nested(config, "laser.centroid_half_window_px"))
    minimum = (
        float(nested(config, "laser.min_peak_prominence_ratio"))
        * prepared.full_scale
    )
    rows = np.arange(height, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for column in range(width):
        profile = signal[:, column]
        peaks, properties = find_peaks(profile, prominence=minimum)
        record: dict[str, Any] = {
            "column_px": column,
            "x_px": float(column),
            "peak_row_px": math.nan,
            "y_px": math.nan,
            "prominence": math.nan,
            "line_residual_px": math.nan,
            "status": "rejected_prominence",
        }
        if peaks.size == 0:
            records.append(record)
            continue
        selected = int(np.argmax(profile[peaks]))
        peak = int(peaks[selected])
        start, stop = max(0, peak - half), min(height, peak + half + 1)
        weights = np.maximum(profile[start:stop], 0.0)
        total = float(np.sum(weights))
        if total <= np.finfo(float).eps:
            records.append(record)
            continue
        y = float(np.sum(rows[start:stop] * weights) / total)
        status = "candidate"
        if not common._inside_polygon((float(column), y), prepared.pose.roi_polygon):
            status = "rejected_roi"
        elif prepared.chess_boundary_mask[
            int(np.clip(round(y), 0, height - 1)), column
        ]:
            status = "rejected_chess_boundary"
        record.update(
            peak_row_px=float(peak),
            y_px=y,
            prominence=float(properties["prominences"][selected]),
            status=status,
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def line_from_two(points: np.ndarray) -> np.ndarray | None:
    delta = points[1] - points[0]
    length = float(np.linalg.norm(delta))
    if length <= np.finfo(float).eps:
        return None
    normal = np.asarray([delta[1], -delta[0]], dtype=np.float64) / length
    return np.append(normal, -float(normal @ points[0]))


def fit_line_svd(points: np.ndarray) -> np.ndarray:
    centre = np.mean(points, axis=0)
    _, singular, right = np.linalg.svd(points - centre, full_matrices=False)
    if len(points) < 2 or singular[0] <= np.finfo(float).eps:
        raise common.CalibrationError("二维激光点退化，无法拟合直线")
    direction = right[0]
    normal = np.asarray([direction[1], -direction[0]], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return np.append(normal, -float(normal @ centre))


def apply_line_ransac(
    diagnostics: pd.DataFrame,
    intrinsics: common.Intrinsics,
    config: Mapping[str, Any],
    seed_offset: int,
) -> pd.DataFrame:
    result = diagnostics.copy()
    indices = result.index[result["status"] == "candidate"].to_numpy()
    if indices.size < 2:
        return result
    pixels = result.loc[indices, ["x_px", "y_px"]].to_numpy(dtype=np.float64)
    ideal = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        P=intrinsics.camera_matrix,
    ).reshape(-1, 2)
    threshold = float(nested(config, "line_ransac.threshold_px"))
    rng = np.random.default_rng(
        int(nested(config, "line_ransac.random_seed")) + seed_offset
    )
    best: np.ndarray | None = None
    best_median = math.inf
    for _ in range(int(nested(config, "line_ransac.iterations"))):
        line = line_from_two(ideal[rng.choice(len(ideal), 2, replace=False)])
        if line is None:
            continue
        residual = np.abs(ideal @ line[:2] + line[2])
        inliers = residual <= threshold
        count = int(np.count_nonzero(inliers))
        median = float(np.median(residual[inliers])) if count else math.inf
        previous = int(np.count_nonzero(best)) if best is not None else -1
        if count > previous or (count == previous and median < best_median):
            best, best_median = inliers, median
    if best is None or np.count_nonzero(best) < 2:
        result.loc[indices, "status"] = "rejected_line_ransac"
        return result
    line = fit_line_svd(ideal[best])
    residual = np.abs(ideal @ line[:2] + line[2])
    inliers = residual <= threshold
    sufficient = (
        np.count_nonzero(inliers) >= int(nested(config, "line_ransac.min_inliers"))
        and np.mean(inliers) >= float(nested(config, "line_ransac.min_inlier_ratio"))
    )
    result.loc[indices, "line_residual_px"] = residual
    result.loc[indices, "status"] = "rejected_line_ransac"
    if sufficient:
        result.loc[indices[inliers], "status"] = "accepted_2d"
    return result


def process_pair(
    pair: ImagePair,
    args: argparse.Namespace,
    intrinsics: common.Intrinsics,
    config: Mapping[str, Any],
) -> CoreFrame:
    prepared = prepare_frame(pair, args, intrinsics, config)
    diagnostics = extract_centres(prepared, config)
    diagnostics = apply_line_ransac(
        diagnostics, intrinsics, config, prepared.image_id
    )
    points, diagnostics = common.intersect_board_plane(
        diagnostics, prepared.pose.plane, intrinsics, config
    )
    diagnostics.insert(0, "split", prepared.split)
    diagnostics.insert(0, "image_id", prepared.image_id)
    result = common.FrameResult(
        prepared.image_id,
        prepared.split,
        prepared.chess_path,
        prepared.laser_path,
        prepared.laser_gray.shape[1],
        bool(len(points)),
        "ok" if len(points) else "没有有效三维点",
        prepared.pose.reprojection_rmse_px,
        points,
        diagnostics,
    )
    motion = measure_pair_motion(prepared, pair.nolaser_path, config)
    LOGGER.info(
        "ID %03d %s：3D点=%d，覆盖率=%.3f，PnP RMSE=%.3fpx，配对移动=%s",
        result.image_id,
        result.split,
        len(result.points_3d),
        len(result.points_3d) / result.image_width,
        result.reprojection_rmse_px,
        motion.get("median_displacement_px", math.nan),
    )
    return CoreFrame(result, motion)


def weighted_plane_svd(points: np.ndarray, pose_ids: np.ndarray) -> np.ndarray:
    if len(points) < 3 or len(points) != len(pose_ids):
        raise common.CalibrationError("姿态等权 SVD 至少需要三个带姿态编号的点")
    unique, counts = np.unique(pose_ids, return_counts=True)
    counts_by_pose = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / counts_by_pose[int(item)] for item in pose_ids])
    weights /= np.sum(weights)
    centre = np.sum(points * weights[:, None], axis=0)
    centred = points - centre
    covariance = (centred * weights[:, None]).T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= np.finfo(float).eps:
        raise common.CalibrationError("三维点退化，无法拟合平面")
    normal = eigenvectors[:, 0]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    return np.append(normal, -float(normal @ centre))


def balanced_plane_ransac(
    points: np.ndarray, pose_ids: np.ndarray, config: Mapping[str, Any]
) -> PlaneFit:
    poses = np.unique(pose_ids)
    if len(points) < 3 or len(poses) < 2:
        raise common.CalibrationError("平面拟合至少需要两个姿态和三个三维点")
    threshold = float(nested(config, "plane_ransac.threshold_mm"))
    indices = {int(pose): np.flatnonzero(pose_ids == pose) for pose in poses}
    rng = np.random.default_rng(int(nested(config, "plane_ransac.random_seed")))
    best: np.ndarray | None = None
    best_score, best_median = -math.inf, math.inf
    for _ in range(int(nested(config, "plane_ransac.iterations"))):
        if len(poses) >= 3:
            sampled_poses = rng.choice(poses, 3, replace=False)
            sampled = np.asarray([rng.choice(indices[int(p)]) for p in sampled_poses])
        else:
            sampled = rng.choice(len(points), 3, replace=False)
        plane = common.plane_from_three(points[sampled])
        if plane is None:
            continue
        distances = common.point_plane_distances(points, plane)
        inliers = distances <= threshold
        score = float(np.mean([np.mean(inliers[pose_ids == p]) for p in poses]))
        median = float(np.median(distances[inliers])) if np.any(inliers) else math.inf
        if score > best_score or (math.isclose(score, best_score) and median < best_median):
            best, best_score, best_median = inliers, score, median
    if best is None:
        raise common.CalibrationError("三维 RANSAC 未找到有效平面")
    minimum = int(nested(config, "plane_ransac.min_inliers"))
    minimum_ratio = float(nested(config, "plane_ransac.min_inlier_ratio"))
    if np.count_nonzero(best) < minimum or np.mean(best) < minimum_ratio:
        raise common.CalibrationError(
            f"三维 RANSAC 内点不足：{np.count_nonzero(best)}/{len(best)}"
        )
    plane = weighted_plane_svd(points[best], pose_ids[best])
    for _ in range(5):
        refined = common.point_plane_distances(points, plane) <= threshold
        if np.array_equal(refined, best):
            break
        best = refined
        plane = weighted_plane_svd(points[best], pose_ids[best])
    return PlaneFit(plane, best, best_score)


def stack_training(
    frames: Sequence[CoreFrame], excluded: set[int]
) -> tuple[np.ndarray, np.ndarray, list[tuple[CoreFrame, slice]]]:
    points: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    mapping: list[tuple[CoreFrame, slice]] = []
    offset = 0
    for frame in frames:
        result = frame.result
        if result.image_id in excluded or not len(result.points_3d):
            continue
        count = len(result.points_3d)
        points.append(result.points_3d)
        poses.append(np.full(count, result.image_id, dtype=np.int64))
        mapping.append((frame, slice(offset, offset + count)))
        offset += count
    if not points:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64), mapping
    return np.vstack(points), np.concatenate(poses), mapping


def mark_fit_status(
    mapping: Sequence[tuple[CoreFrame, slice]], inliers: np.ndarray
) -> None:
    for frame, segment in mapping:
        indices = frame.result.diagnostics.index[
            frame.result.diagnostics["status"] == "accepted_3d"
        ].to_numpy()
        local = inliers[segment]
        frame.result.diagnostics.loc[indices, "status"] = "rejected_plane_ransac"
        frame.result.diagnostics.loc[indices[local], "status"] = "accepted_final"


def auto_excluded_ids(
    train_frames: Sequence[CoreFrame], args: argparse.Namespace, config: Mapping[str, Any]
) -> set[int]:
    excluded = set(args.manual_exclude_ids)
    if (
        not bool(nested(config, "pair_motion.enabled"))
        or not bool(nested(config, "pair_motion.auto_exclude_training"))
    ):
        return excluded
    threshold = float(nested(config, "pair_motion.severe_px"))
    for frame in train_frames:
        tracking_ok = bool(frame.pair_motion.get("tracking_ok", False))
        movement = frame.pair_motion.get("median_displacement_px", math.nan)
        if not tracking_ok:
            LOGGER.warning(
                "ID %03d 配对移动无法确认，请人工检查 centerline_on_chess.png",
                frame.result.image_id,
            )
            continue
        if float(movement) >= threshold:
            excluded.add(frame.result.image_id)
            LOGGER.warning(
                "ID %03d 配对位移 %.3fpx >= %.3fpx，自动排除训练拟合",
                frame.result.image_id,
                float(movement),
                threshold,
            )
    return excluded


def point_table(
    frames: Sequence[CoreFrame], plane: np.ndarray, excluded: set[int]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for frame in frames:
        diagnostics = frame.result.diagnostics
        accepted = diagnostics[diagnostics["status"].isin(
            ["accepted_3d", "accepted_final", "rejected_plane_ransac"]
        )]
        for local_index, row in enumerate(accepted.itertuples(index=False)):
            point = np.asarray([row.x_mm, row.y_mm, row.z_mm], dtype=np.float64)
            signed = float(point @ plane[:3] + plane[3])
            status = str(row.status)
            used = status == "accepted_final" and frame.result.image_id not in excluded
            records.append(
                {
                    "image_id": frame.result.image_id,
                    "split": frame.result.split,
                    "point_index": local_index,
                    "column_px": row.column_px,
                    "image_x_px": row.x_px,
                    "image_y_px": row.y_px,
                    "x_mm": point[0],
                    "y_mm": point[1],
                    "z_mm": point[2],
                    "signed_distance_mm": signed,
                    "abs_distance_mm": abs(signed),
                    "plane_ransac_inlier": status == "accepted_final",
                    "used_for_fit": used,
                    "status": status,
                }
            )
    return pd.DataFrame.from_records(records)


def distance_statistics(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    distances = np.asarray(values, dtype=np.float64)
    distances = distances[np.isfinite(distances)]
    if not len(distances):
        return {
            "point_count": 0,
            "mean_distance_mm": math.nan,
            "variance_distance_mm2": math.nan,
            "rmse_mm": math.nan,
            "p95_mm": math.nan,
            "max_mm": math.nan,
        }
    return {
        "point_count": int(len(distances)),
        "mean_distance_mm": float(np.mean(distances)),
        "variance_distance_mm2": float(np.var(distances, ddof=0)),
        "rmse_mm": float(np.sqrt(np.mean(distances**2))),
        "p95_mm": float(np.percentile(distances, 95)),
        "max_mm": float(np.max(distances)),
    }


def build_summaries(
    points: pd.DataFrame,
    frames: Sequence[CoreFrame],
    excluded: set[int],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scopes = {
        "train_fit_inliers": points[(points["split"] == "train") & points["used_for_fit"]],
        "train_all_extracted": points[points["split"] == "train"],
        "validation": points[points["split"] == "validation"],
    }
    summary_rows = []
    for name, frame in scopes.items():
        row = {"scope": name, **distance_statistics(frame["abs_distance_mm"])}
        signed = frame["signed_distance_mm"].to_numpy(dtype=np.float64)
        row["mean_signed_distance_mm"] = float(np.mean(signed)) if len(signed) else math.nan
        row["variance_signed_distance_mm2"] = float(np.var(signed)) if len(signed) else math.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    frame_by_id = {(frame.result.split, frame.result.image_id): frame for frame in frames}
    pose_rows: list[dict[str, Any]] = []
    for (split, image_id), group in points.groupby(["split", "image_id"], sort=True):
        core = frame_by_id[(str(split), int(image_id))]
        stats = distance_statistics(group["abs_distance_mm"])
        plane_threshold = float(nested(config, "plane_ransac.threshold_mm"))
        inlier_count = int(np.count_nonzero(group["abs_distance_mm"] <= plane_threshold))
        outlier_rate = 1.0 - inlier_count / len(group)
        flags: list[str] = []
        coverage = len(group) / core.result.image_width
        if coverage < float(nested(config, "anomaly.minimum_coverage")):
            flags.append("low_coverage")
        if split == "train" and outlier_rate > float(nested(config, "anomaly.high_outlier_rate")):
            flags.append("high_outlier_rate")
        if stats["rmse_mm"] > float(nested(config, "anomaly.high_rmse_mm")):
            flags.append("high_rmse")
        if stats["p95_mm"] > plane_threshold:
            flags.append("high_p95")
        if image_id in excluded:
            flags.append("excluded_from_fit")
        motion_method = str(core.pair_motion.get("movement_method", "unresolved"))
        # Low-texture short-exposure laser frames are expected in this
        # acquisition design. Keep that state in pair_motion_diagnostics.csv
        # for manual review without marking every otherwise good pose anomalous.
        if motion_method not in {"disabled", "unresolved_low_texture"} and not core.pair_motion.get(
            "tracking_ok", False
        ):
            flags.append("pair_motion_unresolved")
        movement = core.pair_motion.get("median_displacement_px", math.nan)
        if core.pair_motion.get("tracking_ok") and movement >= float(
            nested(config, "pair_motion.severe_px")
        ):
            flags.append("pair_movement")
        pose_rows.append(
            {
                "image_id": int(image_id),
                "split": split,
                "valid_3d_points": int(len(group)),
                "points_within_plane_threshold": inlier_count,
                "coverage": coverage,
                "outlier_rate": outlier_rate,
                "pnp_reprojection_rmse_px": core.result.reprojection_rmse_px,
                "pair_motion_method": motion_method,
                "pair_motion_status": core.pair_motion.get("tracking_status", "unknown"),
                "pair_median_displacement_px": movement,
                **{key: value for key, value in stats.items() if key != "point_count"},
                "anomaly_flags": ";".join(flags) if flags else "ok",
            }
        )
    pose_metrics = pd.DataFrame(pose_rows)
    pair_rows = []
    for core in frames:
        pair_rows.append(
            {
                "image_id": core.result.image_id,
                "split": core.result.split,
                **core.pair_motion,
                "excluded_from_fit": core.result.image_id in excluded,
            }
        )
    return summary, pose_metrics, pd.DataFrame(pair_rows)


def display_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    maximum = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else float(np.max(image))
    return np.clip(image.astype(np.float64) * 255.0 / max(maximum, 1.0), 0, 255).astype(np.uint8)


def draw_centres(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    gray = display_u8(image)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    if not len(xy):
        return canvas
    ordered = xy[np.argsort(xy[:, 0])]
    polyline = np.rint(ordered).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [polyline], False, (0, 255, 0), 2, cv2.LINE_AA)
    marker_step = max(1, math.ceil(len(ordered) / 120))
    for x, y in ordered[::marker_step]:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 1, (255, 0, 255), -1, cv2.LINE_AA)
    return canvas


def save_camera_pointcloud_ply(rows: pd.DataFrame, path: Path) -> None:
    """Save one frame of laser feature points as an ASCII PLY in camera coordinates."""
    columns = ["x_mm", "y_mm", "z_mm", "x_px", "y_px"]
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        if not rows.empty:
            raise common.CalibrationError(
                f"PLY export is missing columns: {', '.join(missing)}"
            )
        values = np.empty((0, len(columns)), dtype=np.float64)
    else:
        values = rows[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise common.CalibrationError(f"PLY export contains non-finite values: {path}")

    header = [
        "ply",
        "format ascii 1.0",
        "comment coordinate_system camera",
        "comment units millimeter",
        "comment u_v_are_subpixel_image_coordinates",
        "comment point_selection all_valid_board_intersections_before_plane_filter",
        f"element vertex {len(values)}",
        "property double x",
        "property double y",
        "property double z",
        "property double u",
        "property double v",
        "end_header",
    ]
    try:
        with path.open("w", encoding="ascii", newline="\n") as stream:
            stream.write("\n".join(header) + "\n")
            for x, y, z, u, v in values:
                stream.write(
                    f"{x:.9f} {y:.9f} {z:.9f} {u:.9f} {v:.9f}\n"
                )
    except OSError as exc:
        raise common.CalibrationError(f"Unable to write PLY point cloud: {path}: {exc}") from exc


def save_overlays(frames: Sequence[CoreFrame], output_dir: Path) -> None:
    for core in frames:
        result = core.result
        diagnostics = result.diagnostics
        if result.split == "train":
            rows = diagnostics[diagnostics["status"] == "accepted_final"]
        else:
            rows = diagnostics[diagnostics["status"] == "accepted_3d"]
        point_rows = diagnostics[diagnostics["status"].isin(
            ["accepted_3d", "accepted_final", "rejected_plane_ransac"]
        )]
        xy = rows[["x_px", "y_px"]].to_numpy(dtype=np.float64)
        directory = output_dir / "images" / result.split / f"{result.image_id:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        chess = common.read_image(result.chess_path)
        laser = common.read_image(result.laser_path)
        common.write_image(directory / "centerline_on_chess.png", draw_centres(chess, xy))
        common.write_image(directory / "centerline_on_laser.png", draw_centres(laser, xy))
        rows.to_csv(directory / "feature_points_2d.csv", index=False)
        save_camera_pointcloud_ply(point_rows, directory / "laser_points_camera.ply")


def draw_plane(axis: Any, points: np.ndarray, plane: np.ndarray) -> None:
    if not len(points):
        return
    x = np.linspace(np.percentile(points[:, 0], 2), np.percentile(points[:, 0], 98), 18)
    y = np.linspace(np.percentile(points[:, 1], 2), np.percentile(points[:, 1], 98), 18)
    xx, yy = np.meshgrid(x, y)
    if abs(plane[2]) > 1e-8:
        zz = -(plane[0] * xx + plane[1] * yy + plane[3]) / plane[2]
        axis.plot_surface(xx, yy, zz, alpha=0.22, color="#4c9be8", linewidth=0)


def save_feature_plane_figure(
    points: pd.DataFrame,
    plane: np.ndarray,
    split: str,
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    subset = points[points["split"] == split].copy()
    if split == "train":
        subset = subset[subset["used_for_fit"]]
    if subset.empty:
        return
    figure = plt.figure(figsize=(14, 6.2), constrained_layout=True)
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_distance = figure.add_subplot(1, 2, 2)
    colours = plt.cm.turbo(np.linspace(0, 1, subset["image_id"].nunique()))
    maximum = int(nested(config, "visualisation.max_3d_points_per_pose"))
    for colour, (image_id, group) in zip(colours, subset.groupby("image_id", sort=True)):
        ordered = group.sort_values("column_px")
        sampled = ordered.iloc[:: max(1, math.ceil(len(ordered) / maximum))]
        axis_3d.plot(
            sampled["x_mm"], sampled["y_mm"], sampled["z_mm"],
            color=colour, linewidth=float(nested(config, "visualisation.line_width")),
            marker=".", markersize=1.5, alpha=0.85,
        )
        axis_distance.plot(
            np.arange(len(ordered)), ordered["abs_distance_mm"],
            color=colour, linewidth=0.9, label=f"{int(image_id):03d}",
        )
    xyz = subset[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
    draw_plane(axis_3d, xyz, plane)
    axis_3d.set_title(f"{split}: feature points and fitted laser plane")
    axis_3d.set_xlabel("X (mm)")
    axis_3d.set_ylabel("Y (mm)")
    axis_3d.set_zlabel("Z (mm)")
    axis_distance.set_title(f"{split}: point-to-plane absolute distance")
    axis_distance.set_xlabel("Feature point index (sorted by image column)")
    axis_distance.set_ylabel("Distance (mm)")
    axis_distance.grid(alpha=0.25)
    axis_distance.legend(title="Pose", ncol=2, fontsize=7, loc="upper right")
    figure.savefig(output_path, dpi=int(nested(config, "visualisation.dpi")))
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(figure)


def save_combined_feature_plane_figure(
    points: pd.DataFrame,
    plane: np.ndarray,
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    subset = points[points["split"].isin(["train", "validation"])].copy()
    if subset.empty:
        return
    subset = subset[subset["used_for_fit"] | (subset["split"] == "validation")].copy()
    if subset.empty:
        return
    figure = plt.figure(figsize=(14, 6.2), constrained_layout=True)
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_distance = figure.add_subplot(1, 2, 2)
    colours = plt.cm.turbo(np.linspace(0, 1, subset["image_id"].nunique()))
    maximum = int(nested(config, "visualisation.max_3d_points_per_pose"))
    for colour, (image_id, group) in zip(colours, subset.groupby("image_id", sort=True)):
        ordered = group.sort_values("column_px")
        sampled = ordered.iloc[:: max(1, math.ceil(len(ordered) / maximum))]
        axis_3d.plot(
            sampled["x_mm"], sampled["y_mm"], sampled["z_mm"],
            color=colour, linewidth=float(nested(config, "visualisation.line_width")),
            marker=".", markersize=1.5, alpha=0.85,
        )
        axis_distance.plot(
            np.arange(len(ordered)), ordered["abs_distance_mm"],
            color=colour, linewidth=0.9, label=f"{int(image_id):03d}",
        )
    xyz = subset[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
    draw_plane(axis_3d, xyz, plane)
    axis_3d.set_title("combined: feature points and fitted laser plane")
    axis_3d.set_xlabel("X (mm)")
    axis_3d.set_ylabel("Y (mm)")
    axis_3d.set_zlabel("Z (mm)")
    axis_distance.set_title("combined: point-to-plane absolute distance")
    axis_distance.set_xlabel("Feature point index (sorted by image column)")
    axis_distance.set_ylabel("Distance (mm)")
    axis_distance.grid(alpha=0.25)
    axis_distance.legend(title="Pose", ncol=2, fontsize=7, loc="upper right")
    figure.savefig(output_path, dpi=int(nested(config, "visualisation.dpi")))
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(figure)


def plane_surface_grid(
    points: np.ndarray, plane: np.ndarray, steps: int = 28
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an XYZ mesh for any non-degenerate plane orientation."""
    normal = np.asarray(plane[:3], dtype=np.float64)
    dependent = int(np.argmax(np.abs(normal)))
    bounds: list[tuple[float, float]] = []
    for axis in range(3):
        low, high = np.percentile(points[:, axis], [2, 98])
        if not np.isfinite(low) or not np.isfinite(high):
            raise common.CalibrationError("三维点范围包含非有限值")
        if math.isclose(float(low), float(high)):
            margin = max(abs(float(low)) * 0.02, 1.0)
            low, high = low - margin, high + margin
        bounds.append((float(low), float(high)))

    axes = [axis for axis in range(3) if axis != dependent]
    first = np.linspace(*bounds[axes[0]], steps)
    second = np.linspace(*bounds[axes[1]], steps)
    grid_first, grid_second = np.meshgrid(first, second)
    coordinates: list[np.ndarray | None] = [None, None, None]
    coordinates[axes[0]] = grid_first
    coordinates[axes[1]] = grid_second
    numerator = -(
        plane[3]
        + normal[axes[0]] * grid_first
        + normal[axes[1]] * grid_second
    )
    coordinates[dependent] = numerator / normal[dependent]
    return (
        np.asarray(coordinates[0]),
        np.asarray(coordinates[1]),
        np.asarray(coordinates[2]),
    )


def save_interactive_feature_plane(
    points: pd.DataFrame,
    plane: np.ndarray,
    split: str,
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Write a self-contained Plotly HTML that can be rotated and zoomed."""
    subset = points[points["split"] == split].copy()
    if split == "train":
        subset = subset[subset["used_for_fit"]]
    if subset.empty:
        return

    figure = go.Figure()
    maximum = int(nested(config, "visualisation.max_3d_points_per_pose"))
    for image_id, group in subset.groupby("image_id", sort=True):
        ordered = group.sort_values("column_px")
        sampled = ordered.iloc[:: max(1, math.ceil(len(ordered) / maximum))]
        figure.add_trace(
            go.Scatter3d(
                x=sampled["x_mm"],
                y=sampled["y_mm"],
                z=sampled["z_mm"],
                mode="lines+markers",
                name=f"Pose {int(image_id):03d}",
                marker={"size": 2},
                line={"width": 3},
                customdata=np.column_stack(
                    [sampled["column_px"], sampled["abs_distance_mm"]]
                ),
                hovertemplate=(
                    "X=%{x:.3f} mm<br>Y=%{y:.3f} mm<br>Z=%{z:.3f} mm"
                    "<br>column=%{customdata[0]:.0f}px"
                    "<br>|distance|=%{customdata[1]:.4f} mm<extra></extra>"
                ),
            )
        )

    xyz = subset[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
    xx, yy, zz = plane_surface_grid(xyz, plane)
    figure.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            opacity=0.28,
            showscale=False,
            name="Fitted laser plane",
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title=f"{split}: feature points and fitted laser plane",
        scene={
            "xaxis_title": "X (mm)",
            "yaxis_title": "Y (mm)",
            "zaxis_title": "Z (mm)",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )

def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "（无）"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            cells.append(f"{value:.6g}" if isinstance(value, float) and np.isfinite(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    intrinsics: common.Intrinsics,
    frames: Sequence[CoreFrame],
    fit: PlaneFit,
    excluded: set[int],
    points: pd.DataFrame,
    summary: pd.DataFrame,
    pose_metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
) -> None:
    output = args.output_dir
    points.to_csv(output / "feature_point_distances.csv", index=False)
    summary.to_csv(output / "distance_summary.csv", index=False)
    pose_metrics.to_csv(output / "pose_metrics.csv", index=False)
    pair_metrics.to_csv(output / "pair_motion_diagnostics.csv", index=False)
    pd.concat([frame.result.diagnostics for frame in frames], ignore_index=True).to_csv(
        output / "laser_centres_diagnostics.csv", index=False
    )
    save_overlays(frames, output)
    save_feature_plane_figure(
        points, fit.plane, "train", output / "feature_points_plane_train.png", config
    )
    save_feature_plane_figure(
        points, fit.plane, "validation", output / "feature_points_plane_validation.png", config
    )
    save_combined_feature_plane_figure(
        points, fit.plane, output / "feature_points_plane_all.png", config
    )
    save_interactive_feature_plane(
        points, fit.plane, "train", output / "feature_points_plane_train_interactive.html", config
    )
    save_interactive_feature_plane(
        points, fit.plane, "validation", output / "feature_points_plane_validation_interactive.html", config
    )
    plane_data = {
        "coordinate_system": "camera",
        "units": "mm",
        "equation": "a*x + b*y + c*z + d = 0",
        "plane": {
            "a": float(fit.plane[0]),
            "b": float(fit.plane[1]),
            "c": float(fit.plane[2]),
            "d": float(fit.plane[3]),
            "normal_norm": float(np.linalg.norm(fit.plane[:3])),
        },
        "intrinsics": str(args.intrinsics.resolve()),
        "image_size": list(intrinsics.image_size),
        "fit_dir": str(args.fit_dir.resolve()),
        "test_dir": str(args.test_dir.resolve()),
        "chess_pattern": args.chess_pattern,
        "nolaser_pattern": args.nolaser_pattern,
        "laser_pattern": args.laser_pattern,
        "train_ids": args.fit_ids,
        "validation_ids": args.test_ids,
        "excluded_training_ids": sorted(excluded),
        "plane_ransac_threshold_mm": float(nested(config, "plane_ransac.threshold_mm")),
        "distance_definition": "absolute orthogonal distance to normalized plane",
        "distance_variance_ddof": 0,
        "metrics": summary.set_index("scope").to_dict(orient="index"),
    }
    (output / "laser_plane.yaml").write_text(
        yaml.safe_dump(plane_data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (output / "laser_plane_core_config_used.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    key_metrics = summary[summary["scope"].isin(["train_fit_inliers", "validation"])]
    anomalies = pose_metrics[pose_metrics["anomaly_flags"] != "ok"]
    report = f"""# 核心版线激光平面标定报告

## 平面方程

相机坐标系、单位 mm，归一化平面：

`{fit.plane[0]:+.12f} x {fit.plane[1]:+.12f} y {fit.plane[2]:+.12f} z {fit.plane[3]:+.12f} = 0`

## 点面距离

“距离”采用非负正交距离；方差采用总体方差 `ddof=0`，单位为 mm²。

{markdown_table(key_metrics)}

## 异常姿态

{markdown_table(anomalies)}

## 说明

- 训练误差使用三维 RANSAC 内点；`train_all_extracted` 可用于观察被 RANSAC 剔除前的整体质量。
- 验证集仅在平面锁定后计算，不参与阈值设定、异常剔除或平面拟合。
- PNG 文件用于静态报告；`feature_points_plane_*_interactive.html` 可在浏览器中旋转、缩放和查看点坐标。
- 激光信号优先采用同一短曝光下的 `laser - nolaser` 差分结果。
- 配对运动检查以高曝光 `chess` 为参考；短曝光目标没有可匹配棋盘纹理时记录 `unresolved_low_texture`，不把模板边界位置当成位移。
"""
    (output / "calibration_report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> np.ndarray:
    validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)
    config = load_config(args.config)
    intrinsics = common.load_intrinsics(args.intrinsics)
    pairs = validate_inputs(args, intrinsics)
    train_pairs = [pair for pair in pairs if pair.split == "train"]
    validation_pairs = [pair for pair in pairs if pair.split == "validation"]

    LOGGER.info("阶段 1/4：只处理训练集并建立三维点")
    train_frames = [process_pair(pair, args, intrinsics, config) for pair in train_pairs]
    excluded = auto_excluded_ids(train_frames, args, config)
    LOGGER.info("训练拟合排除编号：%s", sorted(excluded) if excluded else "无")
    train_points, pose_ids, mapping = stack_training(train_frames, excluded)
    fit = balanced_plane_ransac(train_points, pose_ids, config)
    mark_fit_status(mapping, fit.inliers)
    LOGGER.info(
        "阶段 2/4：训练平面锁定，方程=%+.12f x %+.12f y %+.12f z %+.12f = 0",
        *fit.plane,
    )

    LOGGER.info("阶段 3/4：平面锁定后处理独立验证集")
    validation_frames = [
        process_pair(pair, args, intrinsics, config) for pair in validation_pairs
    ]
    frames = [*train_frames, *validation_frames]
    point_data = point_table(frames, fit.plane, excluded)
    summary, pose_metrics, pair_metrics = build_summaries(
        point_data, frames, excluded, config
    )
    LOGGER.info("阶段 4/4：保存距离、姿态指标、叠加图、静态图和可旋转3D图")
    write_outputs(
        args,
        config,
        intrinsics,
        frames,
        fit,
        excluded,
        point_data,
        summary,
        pose_metrics,
        pair_metrics,
    )
    LOGGER.info("完成，结果目录：%s", args.output_dir.resolve())
    return fit.plane


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run(args)
        return 0
    except (common.CalibrationError, FileNotFoundError, FileExistsError, ValueError, cv2.error) as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        LOGGER.exception("标定失败：%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
