#!/usr/bin/env python3
"""Adaptive monochrome M1 front end for checkerboard laser-plane calibration.

The default signal is raw ``laser_gray - chess_gray`` followed by per-cell
median removal.  Coarse candidates use per-cell robust thresholds; final
centres use a local baseline, peak SNR and normalized profile energy.  The
rectification and equal-pose 3-D backend are reused unchanged from the previous
monochrome/V2 implementation.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import yaml

import calibrate_laser_plane_opencv as base
import calibrate_laser_plane_opencv_v2 as v2
import calibrate_laser_plane_rectified_mono as previous


LOGGER = logging.getLogger("laser_plane_rectified_mono_adaptive")
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "calibration"
    / "calib02"
    / "laser_plane_rectified_mono_adaptive_raw_20260722"
)
DEFAULT_COMPARISON_BASELINE = (
    ROOT
    / "calibration"
    / "calib02"
    / "laser_plane_rectified_mono_m1_m2_20260722_reviewed"
)
METHOD = "M1A"
METHOD_LABEL = "M1_adaptive_raw_difference"
GUIDE_METHOD = "GUIDE"


@dataclass(frozen=True)
class AdaptiveConfig(previous.FrontendConfig):
    secondary_background_weight: float = 0.0
    cell_candidate_percentile: float = 97.5
    cell_seed_sigma: float = 3.0
    profile_baseline_percentile: float = 60.0
    profile_noise_floor: float = 0.10
    profile_min_snr: float = 2.0
    profile_min_energy_snr: float = 3.0
    boundary_window_px: float = 5.0
    min_profile_step_px: float = 0.25


@dataclass(frozen=True)
class AdaptiveFrame:
    base_frame: previous.RectifiedFrame
    cell_colours: np.ndarray
    cell_medians: np.ndarray


@dataclass(frozen=True)
class ProfileExtraction:
    points: np.ndarray
    samples: pd.DataFrame
    step_px: float


def build_parser() -> argparse.ArgumentParser:
    parser = previous.build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--comparison-baseline-dir",
        type=Path,
        default=DEFAULT_COMPARISON_BASELINE,
        help="Previous M1/M2/V2 reviewed output used for the four-way comparison.",
    )
    parser.add_argument(
        "--m1-signal-mode",
        choices=("raw", "affine"),
        default="raw",
        help="Raw laser-chess is the default; affine is retained only as a control.",
    )
    parser.add_argument(
        "--secondary-background-weight",
        type=float,
        default=0.0,
        help="Large-scale post-difference Gaussian subtraction weight; default 0 disables it.",
    )
    parser.add_argument(
        "--visual-boundary-review",
        choices=("reduced", "not-reduced", "unclear"),
        default="unclear",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    previous.validate_args(args)
    if args.secondary_background_weight < 0:
        raise base.CalibrationError("--secondary-background-weight 不能为负数")
    required = (
        "algorithm_comparison.csv",
        "pose_metrics.csv",
        "frontend_metrics.csv",
    )
    missing = [
        name
        for name in required
        if not (args.comparison_baseline_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"上一轮比较目录缺少文件：{', '.join(missing)} "
            f"({args.comparison_baseline_dir})"
        )


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


def robust_background(values: np.ndarray, percentile: float = 60.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 0.0
    cutoff = float(np.percentile(finite, percentile))
    background = finite[finite <= cutoff]
    if not len(background):
        background = finite
    baseline = float(np.median(background))
    noise = 1.4826 * float(np.median(np.abs(background - baseline)))
    return baseline, noise


def build_adaptive_signal(
    chess_rectified: np.ndarray,
    laser_rectified: np.ndarray,
    pattern_size: tuple[int, int],
    config: AdaptiveConfig,
    signal_mode: str = "raw",
) -> previous.SignalData:
    chess = chess_rectified.astype(np.float32)
    laser = laser_rectified.astype(np.float32)
    if signal_mode == "affine":
        scale, offset = previous.robust_affine_intensity_match(chess, laser)
        reference = scale * chess + offset
    elif signal_mode == "raw":
        reference = chess
    else:
        raise ValueError(f"Unsupported signal mode: {signal_mode}")
    difference = laser - reference
    difference = previous.subtract_per_square_baseline(
        difference,
        pattern_size,
        config.pixels_per_square,
        config.cell_border_ignore_px,
    )
    difference = np.clip(difference, 0.0, 255.0)
    if config.secondary_background_weight > 0:
        kernel = previous._odd_kernel(config.local_background_kernel)
        background = cv2.GaussianBlur(difference, (kernel, kernel), 0)
        difference = np.clip(
            difference - config.secondary_background_weight * background,
            0.0,
            255.0,
        )
    if config.gaussian_blur_sigma > 0:
        difference = cv2.GaussianBlur(
            difference.astype(np.float32), (0, 0), config.gaussian_blur_sigma
        )
    return previous.SignalData(
        difference.astype(np.float32),
        previous._normalize_debug(difference),
        np.clip(reference, 0, 255).astype(np.uint8),
    )


def build_laser_guide_signal(
    laser_rectified: np.ndarray,
    pattern_size: tuple[int, int],
    config: AdaptiveConfig,
) -> previous.SignalData:
    laser = laser_rectified.astype(np.float32)
    kernel = previous._odd_kernel(config.local_background_kernel)
    background = cv2.GaussianBlur(laser, (kernel, kernel), 0)
    guide = np.clip(laser - background, 0.0, 255.0)
    guide = previous.subtract_per_square_baseline(
        guide,
        pattern_size,
        config.pixels_per_square,
        config.cell_border_ignore_px,
    )
    guide = np.clip(guide, 0.0, 255.0)
    if config.gaussian_blur_sigma > 0:
        guide = cv2.GaussianBlur(
            guide.astype(np.float32), (0, 0), config.gaussian_blur_sigma
        )
    return previous.SignalData(
        guide.astype(np.float32),
        previous._normalize_debug(guide),
        np.clip(background, 0, 255).astype(np.uint8),
    )


def classify_checker_cells(
    chess_rectified: np.ndarray, config: AdaptiveConfig
) -> tuple[np.ndarray, np.ndarray]:
    rows = chess_rectified.shape[0] // config.pixels_per_square
    columns = chess_rectified.shape[1] // config.pixels_per_square
    medians = np.empty((rows, columns), dtype=np.float64)
    border = max(1, config.cell_border_ignore_px)
    for row in range(rows):
        for column in range(columns):
            cell = chess_rectified[
                row * config.pixels_per_square + border : (row + 1) * config.pixels_per_square - border,
                column * config.pixels_per_square + border : (column + 1) * config.pixels_per_square - border,
            ]
            medians[row, column] = float(np.median(cell))
    parity = np.indices(medians.shape).sum(axis=0) % 2
    parity_zero = float(np.median(medians[parity == 0]))
    parity_one = float(np.median(medians[parity == 1]))
    black_parity = 0 if parity_zero <= parity_one else 1
    colours = np.where(parity == black_parity, "black", "white")
    return colours, medians


def prepare_adaptive_frame(
    pair: tuple[int, str, Path, Path],
    args: argparse.Namespace,
    intrinsics: base.Intrinsics,
    v2_config: Mapping[str, Any],
    config: AdaptiveConfig,
) -> AdaptiveFrame:
    frame = previous.prepare_rectified_frame(
        pair, args, intrinsics, v2_config, config
    )
    signal = build_adaptive_signal(
        frame.chess_rectified,
        frame.laser_rectified,
        (args.pattern_cols, args.pattern_rows),
        config,
        args.m1_signal_mode,
    )
    guide = build_laser_guide_signal(
        frame.laser_rectified,
        (args.pattern_cols, args.pattern_rows),
        config,
    )
    colours, medians = classify_checker_cells(frame.chess_rectified, config)
    return AdaptiveFrame(
        replace(frame, signals={METHOD: signal, GUIDE_METHOD: guide}), colours, medians
    )


def adaptive_candidate_mask(
    signal: np.ndarray,
    cell_colours: np.ndarray,
    config: AdaptiveConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    binary = np.zeros(signal.shape, dtype=np.uint8)
    records: list[dict[str, Any]] = []
    border = max(3, config.cell_border_ignore_px)
    pps = config.pixels_per_square
    for row in range(cell_colours.shape[0]):
        for column in range(cell_colours.shape[1]):
            x0, x1 = column * pps + border, (column + 1) * pps - border
            y0, y1 = row * pps + border, (row + 1) * pps - border
            cell = signal[y0:y1, x0:x1]
            baseline, noise = robust_background(
                cell, config.profile_baseline_percentile
            )
            noise = max(noise, config.profile_noise_floor)
            noise_threshold = baseline + config.cell_seed_sigma * noise
            quantile_threshold = float(
                np.percentile(cell, config.cell_candidate_percentile)
            )
            threshold = max(noise_threshold, quantile_threshold)
            selected = cell >= threshold
            binary_cell = binary[y0:y1, x0:x1]
            binary_cell[selected] = 255
            records.append(
                {
                    "cell_row": row,
                    "cell_column": column,
                    "cell_colour": str(cell_colours[row, column]),
                    "baseline": baseline,
                    "noise": noise,
                    "threshold": threshold,
                    "candidate_pixels": int(np.count_nonzero(selected)),
                }
            )
    return binary, pd.DataFrame(records)


def point_cell_colours(
    points: np.ndarray, cell_colours: np.ndarray, pixels_per_square: int
) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=object)
    columns = np.clip(
        np.floor(points[:, 0] / pixels_per_square).astype(int),
        0,
        cell_colours.shape[1] - 1,
    )
    rows = np.clip(
        np.floor(points[:, 1] / pixels_per_square).astype(int),
        0,
        cell_colours.shape[0] - 1,
    )
    return cell_colours[rows, columns]


def cell_balanced_weight_signal(
    signal: np.ndarray, binary: np.ndarray, config: AdaptiveConfig
) -> np.ndarray:
    balanced = np.zeros(signal.shape, dtype=np.float32)
    pps = config.pixels_per_square
    rows = signal.shape[0] // pps
    columns = signal.shape[1] // pps
    for row in range(rows):
        for column in range(columns):
            y0, y1 = row * pps, (row + 1) * pps
            x0, x1 = column * pps, (column + 1) * pps
            selected = binary[y0:y1, x0:x1] > 0
            if not np.any(selected):
                continue
            values = np.maximum(signal[y0:y1, x0:x1], 0.0)
            local = balanced[y0:y1, x0:x1]
            total = float(np.sum(values[selected]))
            local[selected] = (
                values[selected] / total if total > 0 else 1.0 / np.count_nonzero(selected)
            )
    return balanced * 1000.0


def weighted_ransac_points(
    points: np.ndarray,
    weights: np.ndarray,
    threshold_px: float,
    config: AdaptiveConfig,
    seed_offset: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    if len(points) < 30:
        return None, np.zeros(len(points), dtype=bool)
    rng = np.random.default_rng(20260722 + seed_offset)
    best: np.ndarray | None = None
    best_score = -math.inf
    best_median = math.inf
    for _ in range(config.ransac_iterations):
        sampled = points[rng.choice(len(points), size=2, replace=False)]
        line = previous.line_from_two(sampled)
        if line is None:
            continue
        residual = previous.orthogonal_line_distances(points, line)
        inliers = residual <= threshold_px
        if np.count_nonzero(inliers) < 30:
            continue
        score = float(np.sum(weights[inliers]))
        median = float(np.median(residual[inliers]))
        if score > best_score or (math.isclose(score, best_score) and median < best_median):
            best, best_score, best_median = inliers, score, median
    if best is None:
        return None, np.zeros(len(points), dtype=bool)
    line = previous.weighted_line_svd(points[best], weights[best])
    residual = previous.orthogonal_line_distances(points, line)
    return line, residual <= threshold_px


def refine_profile_extraction(
    extraction: ProfileExtraction,
    threshold_px: float,
    config: AdaptiveConfig,
    seed_offset: int,
) -> tuple[np.ndarray | None, ProfileExtraction]:
    samples = extraction.samples.copy()
    samples["line_inlier"] = False
    accepted_indices = samples.index[samples["accepted"]].to_numpy()
    if len(accepted_indices) != len(extraction.points):
        raise base.CalibrationError("亚像素点与剖面记录数量不一致")
    weights = np.clip(
        samples.loc[accepted_indices, "snr"].to_numpy(dtype=np.float64), 1.0, 100.0
    )
    line, inliers = weighted_ransac_points(
        extraction.points,
        weights,
        threshold_px,
        config,
        seed_offset,
    )
    if line is None or np.count_nonzero(inliers) < config.min_points_per_frame:
        samples.loc[accepted_indices, "line_inlier"] = True
        return None, ProfileExtraction(extraction.points, samples, extraction.step_px)
    samples.loc[accepted_indices[inliers], "line_inlier"] = True
    samples.loc[accepted_indices[~inliers], "accepted"] = False
    return line, ProfileExtraction(extraction.points[inliers], samples, extraction.step_px)


def profile_step_for_original_columns(
    frame: previous.RectifiedFrame,
    line: np.ndarray,
    config: AdaptiveConfig,
) -> float:
    intersections = previous.line_intersections_with_rect(
        line,
        frame.chess_rectified.shape[1],
        frame.chess_rectified.shape[0],
    )
    if intersections is None:
        return config.sample_step_px
    endpoints = np.vstack(intersections)
    undistorted = previous.apply_homography(endpoints, frame.inverse_homography)
    horizontal_span = abs(float(undistorted[1, 0] - undistorted[0, 0]))
    rectified_length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    if horizontal_span < 1.0:
        return config.sample_step_px
    return float(
        np.clip(
            rectified_length / horizontal_span,
            config.min_profile_step_px,
            config.sample_step_px,
        )
    )


def extract_adaptive_profiles(
    signal: np.ndarray,
    line: np.ndarray,
    cell_colours: np.ndarray,
    config: AdaptiveConfig,
    step_px: float,
) -> ProfileExtraction:
    intersections = previous.line_intersections_with_rect(
        line, signal.shape[1], signal.shape[0]
    )
    columns = [
        "distance_px",
        "nominal_x_px",
        "nominal_y_px",
        "cell_colour",
        "baseline",
        "noise",
        "peak_height",
        "snr",
        "energy_snr",
        "accepted",
        "subpixel_x_px",
        "subpixel_y_px",
    ]
    if intersections is None:
        return ProfileExtraction(
            np.empty((0, 2), dtype=np.float32), pd.DataFrame(columns=columns), step_px
        )
    start, end = intersections
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 40.0:
        return ProfileExtraction(
            np.empty((0, 2), dtype=np.float32), pd.DataFrame(columns=columns), step_px
        )
    direction /= length
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    offsets = np.arange(
        -config.profile_half_width_px,
        config.profile_half_width_px + 1,
        dtype=np.float32,
    )
    records: list[dict[str, Any]] = []
    points: list[np.ndarray] = []
    for distance in np.arange(0.0, length, step_px):
        nominal = start + direction * distance
        profile = np.asarray(
            [
                previous.bilinear_sample(
                    signal,
                    float(nominal[0] + offset * normal[0]),
                    float(nominal[1] + offset * normal[1]),
                )
                for offset in offsets
            ],
            dtype=np.float32,
        )
        baseline, noise = robust_background(
            profile, config.profile_baseline_percentile
        )
        noise = max(noise, config.profile_noise_floor)
        peak_index = int(np.argmax(profile))
        peak_height = max(0.0, float(profile[peak_index]) - baseline)
        snr = peak_height / noise
        left, right = max(0, peak_index - 2), min(len(profile), peak_index + 3)
        window = profile[left:right]
        weights = np.maximum(window - baseline, 0.0)
        energy = float(np.sum(weights))
        energy_snr = energy / (noise * math.sqrt(max(len(window), 1)))
        subpixel_offset = (
            float(np.sum(offsets[left:right] * weights) / energy)
            if energy > 0
            else math.nan
        )
        accepted = bool(
            snr >= config.profile_min_snr
            and energy_snr >= config.profile_min_energy_snr
            and np.isfinite(subpixel_offset)
            and abs(subpixel_offset) <= config.max_center_offset_px
        )
        point = (
            nominal + normal * subpixel_offset
            if accepted
            else np.asarray([math.nan, math.nan], dtype=np.float32)
        )
        if accepted:
            points.append(point.astype(np.float32))
        colour = point_cell_colours(
            nominal.reshape(1, 2), cell_colours, config.pixels_per_square
        )[0]
        records.append(
            {
                "distance_px": float(distance),
                "nominal_x_px": float(nominal[0]),
                "nominal_y_px": float(nominal[1]),
                "cell_colour": str(colour),
                "baseline": baseline,
                "noise": noise,
                "peak_height": peak_height,
                "snr": snr,
                "energy_snr": energy_snr,
                "accepted": accepted,
                "subpixel_x_px": float(point[0]),
                "subpixel_y_px": float(point[1]),
            }
        )
    return ProfileExtraction(
        np.vstack(points) if points else np.empty((0, 2), dtype=np.float32),
        pd.DataFrame.from_records(records, columns=columns),
        step_px,
    )


def boundary_continuity(
    line: np.ndarray,
    accepted_distances: np.ndarray,
    shape: tuple[int, int],
    config: AdaptiveConfig,
) -> tuple[float, float, int]:
    intersections = previous.line_intersections_with_rect(
        line, shape[1], shape[0]
    )
    if intersections is None or not len(accepted_distances):
        return 0.0, math.inf, 0
    start, end = intersections
    direction = end - start
    length = float(np.linalg.norm(direction))
    direction /= max(length, np.finfo(float).eps)
    crossings: list[float] = []
    a, b, c = (float(value) for value in line)
    pps = config.pixels_per_square
    for x in range(pps, shape[1], pps):
        if abs(b) > 1.0e-9:
            y = -(a * x + c) / b
            if 0 <= y < shape[0]:
                crossings.append(float((np.asarray([x, y]) - start) @ direction))
    for y in range(pps, shape[0], pps):
        if abs(a) > 1.0e-9:
            x = -(b * y + c) / a
            if 0 <= x < shape[1]:
                crossings.append(float((np.asarray([x, y]) - start) @ direction))
    recovered = [
        np.any(np.abs(accepted_distances - crossing) <= config.boundary_window_px)
        for crossing in crossings
    ]
    ordered = np.sort(accepted_distances)
    gaps = np.diff(ordered)
    longest_gap = float(np.max(gaps)) if len(gaps) else math.inf
    rate = float(np.mean(recovered)) if recovered else math.nan
    return rate, longest_gap, len(crossings)


def _safe_percentile(values: pd.Series, percentile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if len(finite) else math.nan


def colour_metrics(
    candidates: np.ndarray,
    inliers: np.ndarray,
    extraction: ProfileExtraction,
    cell_colours: np.ndarray,
    config: AdaptiveConfig,
) -> dict[str, Any]:
    candidate_colours = point_cell_colours(
        candidates, cell_colours, config.pixels_per_square
    )
    metrics: dict[str, Any] = {}
    for colour in ("black", "white"):
        candidate_local = candidate_colours == colour
        profiles = extraction.samples[
            extraction.samples["cell_colour"] == colour
        ]
        accepted = profiles[profiles["accepted"]]
        attempts = len(profiles)
        metrics.update(
            {
                f"candidate_points_{colour}": int(np.count_nonzero(candidate_local)),
                f"ransac_inliers_{colour}": int(
                    np.count_nonzero(candidate_local & inliers)
                ),
                f"profile_attempts_{colour}": attempts,
                f"subpixel_points_{colour}": len(accepted),
                f"profile_coverage_{colour}": len(accepted) / attempts if attempts else math.nan,
                f"peak_height_p10_{colour}": _safe_percentile(accepted["peak_height"], 10),
                f"peak_height_median_{colour}": _safe_percentile(accepted["peak_height"], 50),
                f"peak_height_p90_{colour}": _safe_percentile(accepted["peak_height"], 90),
                f"snr_p10_{colour}": _safe_percentile(accepted["snr"], 10),
                f"snr_median_{colour}": _safe_percentile(accepted["snr"], 50),
                f"snr_p90_{colour}": _safe_percentile(accepted["snr"], 90),
            }
        )
    return metrics


def save_colour_debug(
    frame: previous.RectifiedFrame,
    extraction: ProfileExtraction,
    output_dir: Path,
    config: AdaptiveConfig,
) -> None:
    directory = output_dir / "images" / frame.split / f"{frame.image_id:03d}" / METHOD
    overlay = cv2.cvtColor(frame.signals[METHOD].normalized, cv2.COLOR_GRAY2BGR)
    black = extraction.samples[
        extraction.samples["accepted"]
        & (extraction.samples["cell_colour"] == "black")
    ][["subpixel_x_px", "subpixel_y_px"]].to_numpy(dtype=np.float32)
    white = extraction.samples[
        extraction.samples["accepted"]
        & (extraction.samples["cell_colour"] == "white")
    ][["subpixel_x_px", "subpixel_y_px"]].to_numpy(dtype=np.float32)
    previous._draw_points(overlay, black, (255, 255, 0), 1)
    previous._draw_points(overlay, white, (0, 255, 0), 1)
    pps = config.pixels_per_square
    for x in range(pps, overlay.shape[1], pps):
        cv2.line(overlay, (x, 0), (x, overlay.shape[0] - 1), (80, 80, 80), 1)
    for y in range(pps, overlay.shape[0], pps):
        cv2.line(overlay, (0, y), (overlay.shape[1] - 1, y), (80, 80, 80), 1)
    base.write_image(directory / "06_black_white_recovery.png", overlay)
    extraction.samples.to_csv(
        directory / "profile_samples.csv", index=False, encoding="utf-8-sig"
    )


def process_adaptive_method(
    adaptive: AdaptiveFrame,
    line_threshold_px: float,
    intrinsics: base.Intrinsics,
    v2_config: Mapping[str, Any],
    config: AdaptiveConfig,
    output_dir: Path,
    signal_mode: str = "raw",
) -> previous.FrontendResult:
    frame = adaptive.base_frame
    signal = frame.signals[METHOD].signal
    guide_signal = frame.signals[GUIDE_METHOD].signal
    binary, cell_stats = adaptive_candidate_mask(
        guide_signal, adaptive.cell_colours, config
    )
    balanced_weights = cell_balanced_weight_signal(guide_signal, binary, config)
    line, candidates, inliers, candidate_residuals = previous.weighted_ransac_line(
        binary,
        balanced_weights,
        line_threshold_px,
        config,
        seed_offset=frame.image_id + 2000,
    )
    metrics: dict[str, Any] = {
        "method": METHOD,
        "signal_mode": signal_mode,
        "image_id": frame.image_id,
        "split": frame.split,
        "secondary_background_weight": config.secondary_background_weight,
        "coarse_guide": "laser_local_background_only",
        "candidate_count": len(candidates),
        "ransac_inliers": int(np.count_nonzero(inliers)),
        "subpixel_count": 0,
        "line_residual_px": math.nan,
        "rectified_coverage": 0.0,
        "boundary_recovery_rate": 0.0,
        "longest_gap_px": math.inf,
        "success": False,
        "message": "line model failed",
    }
    if line is None:
        previous.save_frame_debug(
            frame,
            METHOD,
            output_dir,
            binary,
            candidates,
            inliers,
            None,
            np.empty((0, 2)),
            np.empty((0, 2)),
            metrics,
        )
        cell_stats.to_csv(
            output_dir / "images" / frame.split / f"{frame.image_id:03d}" / METHOD / "cell_thresholds.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return previous.failed_frontend_result(
            frame, METHOD, "line model failed", metrics
        )

    step_px = profile_step_for_original_columns(frame, line, config)
    extraction = extract_adaptive_profiles(
        signal, line, adaptive.cell_colours, config, step_px
    )
    refined_line, extraction = refine_profile_extraction(
        extraction,
        line_threshold_px,
        config,
        frame.image_id + 3000,
    )
    if refined_line is not None:
        line = refined_line
    subpixel = extraction.points
    line_residual = (
        float(np.mean(previous.orthogonal_line_distances(subpixel, line)))
        if len(subpixel)
        else math.nan
    )
    spatial_coverage = previous.rectified_coverage(subpixel, signal.shape)
    accepted_distances = extraction.samples.loc[
        extraction.samples["accepted"], "distance_px"
    ].to_numpy(dtype=np.float64)
    boundary_rate, longest_gap, boundary_count = boundary_continuity(
        line, accepted_distances, signal.shape, config
    )
    metrics.update(
        colour_metrics(
            candidates, inliers, extraction, adaptive.cell_colours, config
        )
    )
    metrics.update(
        {
            "subpixel_count": len(subpixel),
            "profile_step_px": step_px,
            "line_residual_px": line_residual,
            "rectified_coverage": spatial_coverage,
            "boundary_recovery_rate": boundary_rate,
            "boundary_crossings": boundary_count,
            "longest_gap_px": longest_gap,
            "candidate_residual_p95_px": (
                float(np.percentile(candidate_residuals[inliers], 95))
                if np.any(inliers)
                else math.nan
            ),
        }
    )
    undistorted = previous.apply_homography(
        subpixel, frame.inverse_homography
    )
    accepted_profiles = extraction.samples[extraction.samples["accepted"]].reset_index(drop=True)
    diagnostics = pd.DataFrame(
        {
            "x_px": undistorted[:, 0] if len(undistorted) else np.empty(0),
            "y_px": undistorted[:, 1] if len(undistorted) else np.empty(0),
            "rect_x_px": subpixel[:, 0] if len(subpixel) else np.empty(0),
            "rect_y_px": subpixel[:, 1] if len(subpixel) else np.empty(0),
            "cell_colour": accepted_profiles["cell_colour"],
            "peak_height": accepted_profiles["peak_height"],
            "profile_snr": accepted_profiles["snr"],
            "energy_snr": accepted_profiles["energy_snr"],
            "line_residual_px": previous.orthogonal_line_distances(subpixel, line),
            "structural_status": "candidate",
            "status": "accepted_2d",
        }
    )
    frame_quality_ok = bool(
        len(subpixel) >= config.min_points_per_frame
        and spatial_coverage >= config.min_rectified_coverage_ratio
        and np.isfinite(line_residual)
        and line_residual <= config.max_line_residual_px
    )
    if frame_quality_ok:
        points, diagnostics = base.intersect_board_plane(
            diagnostics, frame.pose.plane, frame.rectified_intrinsics, v2_config
        )
    else:
        diagnostics["status"] = "rejected_frame_quality"
        points = np.empty((0, 3), dtype=np.float64)

    projected_original = np.empty((0, 2), dtype=np.float32)
    accepted_indices = diagnostics.index[
        diagnostics["status"] == "accepted_3d"
    ].to_numpy()
    if len(points):
        projected, _ = cv2.projectPoints(
            points.astype(np.float32),
            np.zeros(3),
            np.zeros(3),
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
        )
        projected_original = projected.reshape(-1, 2).astype(np.float32)
        diagnostics.loc[accepted_indices, "original_x_px"] = projected_original[:, 0]
        diagnostics.loc[accepted_indices, "original_y_px"] = projected_original[:, 1]
    metrics.update(
        success=bool(len(points)),
        message="ok" if len(points) else "frame quality or board intersection failed",
    )
    diagnostics.insert(0, "method", METHOD)
    diagnostics.insert(0, "split", frame.split)
    diagnostics.insert(0, "image_id", frame.image_id)
    previous.save_frame_debug(
        frame,
        METHOD,
        output_dir,
        binary,
        candidates,
        inliers,
        line,
        subpixel,
        projected_original,
        metrics,
    )
    save_colour_debug(frame, extraction, output_dir, config)
    cell_stats.to_csv(
        output_dir / "images" / frame.split / f"{frame.image_id:03d}" / METHOD / "cell_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result = base.FrameResult(
        frame.image_id,
        frame.split,
        frame.chess_path,
        frame.laser_path,
        frame.laser_gray.shape[1],
        bool(len(points)),
        str(metrics["message"]),
        frame.pose.reprojection_rmse_px,
        points,
        diagnostics,
    )
    feature = v2.FeatureFrame(
        frame.image_id,
        frame.split,
        frame.chess_path,
        frame.laser_path,
        frame.laser_gray.shape[1],
        frame.pose.plane,
        frame.pose.reprojection_rmse_px,
        diagnostics,
        frame.pair_motion,
    )
    return previous.FrontendResult(result, feature, metrics)


def derive_line_threshold(
    frames: Sequence[AdaptiveFrame], config: AdaptiveConfig
) -> float:
    residuals: list[np.ndarray] = []
    for adaptive in frames:
        frame = adaptive.base_frame
        signal = frame.signals[GUIDE_METHOD].signal
        binary, _ = adaptive_candidate_mask(
            signal, adaptive.cell_colours, config
        )
        balanced_weights = cell_balanced_weight_signal(signal, binary, config)
        _, _, inliers, residual = previous.weighted_ransac_line(
            binary,
            balanced_weights,
            config.ransac_seed_threshold_px,
            config,
            seed_offset=frame.image_id + 2000,
        )
        if np.any(inliers):
            residuals.append(residual[inliers])
    if not residuals:
        raise base.CalibrationError("M1A 在 001-018 上无法建立粗激光直线")
    selected = float(np.percentile(np.concatenate(residuals), 99.0)) * 1.5
    return float(np.clip(selected, 0.5, config.ransac_seed_threshold_px))


def save_plane_document(
    args: argparse.Namespace,
    intrinsics: base.Intrinsics,
    thresholds: v2.LockedThresholds,
    fit: v2.BalancedFit,
    excluded_ids: set[int],
    config: AdaptiveConfig,
) -> None:
    directory = args.output_dir / METHOD
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "algorithm": METHOD_LABEL,
        "model": "normalized_plane_ax_by_cz_d_equals_0",
        "coordinate_system": "camera",
        "coordinate_unit": "mm",
        "normal_is_unit_length": True,
        "coefficients": {
            key: float(value) for key, value in zip("abcd", fit.plane)
        },
        "signal_mode": args.m1_signal_mode,
        "secondary_background_weight": config.secondary_background_weight,
        "threshold_tuning_ids": list(args.tune_ids),
        "training_ids": list(args.train_ids),
        "validation_ids": list(args.val_ids),
        "validation_used_for_thresholds_or_fitting": False,
        "excluded_training_pose_ids": sorted(excluded_ids),
        "fitting_weight": "equal_total_weight_per_pose",
        "backend": "calibrate_laser_plane_opencv_v2.balanced_plane_ransac",
        "intrinsics_source": str(args.intrinsics.resolve()),
        "image_size": list(intrinsics.image_size),
        "adaptive_frontend": asdict(config),
        "locked_thresholds": asdict(thresholds),
    }
    (directory / "laser_plane.yaml").write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (directory / "locked_thresholds.yaml").write_text(
        yaml.safe_dump(asdict(thresholds), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def make_contact_sheets(args: argparse.Namespace) -> None:
    output = args.output_dir / "visual_comparison"
    output.mkdir(parents=True, exist_ok=True)
    for image_id in args.val_ids:
        paths = [
            args.v2_baseline_dir / "images" / "validation" / f"{image_id:03d}" / "laser_filter_overlay.png",
            args.comparison_baseline_dir / "images" / "validation" / f"{image_id:03d}" / "M1" / "05_original_overlay.png",
            args.comparison_baseline_dir / "images" / "validation" / f"{image_id:03d}" / "M2" / "05_original_overlay.png",
            args.output_dir / "images" / "validation" / f"{image_id:03d}" / METHOD / "05_original_overlay.png",
        ]
        labels = ["V2 current", "M1 previous", "M2 previous", "M1 adaptive raw"]
        panels: list[np.ndarray] = []
        for path, label in zip(paths, labels):
            image = base.read_image(path)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            scale = 570.0 / image.shape[1]
            panel = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (24, 32, 48), -1)
            cv2.putText(
                panel,
                label,
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
        base.write_image(
            output / f"validation_{image_id:03d}.png", np.hstack(panels)
        )


def _markdown_table(frame: pd.DataFrame) -> str:
    return v2.markdown_table(frame)


def write_reports(
    args: argparse.Namespace,
    comparison: pd.DataFrame,
    poses: pd.DataFrame,
    frontend: pd.DataFrame,
    fit: v2.BalancedFit,
) -> None:
    comparison.to_csv(
        args.output_dir / "algorithm_comparison.csv", index=False, encoding="utf-8-sig"
    )
    poses.to_csv(args.output_dir / "pose_metrics.csv", index=False, encoding="utf-8-sig")
    frontend.to_csv(
        args.output_dir / "black_white_metrics.csv", index=False, encoding="utf-8-sig"
    )
    validation = comparison[comparison["split"] == "validation"].copy()
    current = validation[validation["algorithm"] == METHOD_LABEL].iloc[0]
    v2_row = validation[validation["algorithm"] == "V2_current"].iloc[0]
    errors_not_worse = bool(
        current["mae_mm"] <= v2_row["mae_mm"]
        and current["rmse_mm"] <= v2_row["rmse_mm"]
        and current["p95_mm"] <= v2_row["p95_mm"]
    )
    coverage_recovered = bool(current["coverage"] >= 0.8 * v2_row["coverage"])
    boundary_reduced = args.visual_boundary_review == "reduced"
    recommend = errors_not_worse and coverage_recovered and boundary_reduced
    decision = pd.DataFrame(
        [
            {
                "validation_error_not_worse": errors_not_worse,
                "coverage_at_least_80pct_of_v2": coverage_recovered,
                "visual_boundary_breaks_reduced": boundary_reduced,
                "recommend_replace_frontend": recommend,
            }
        ]
    )
    decision.to_csv(
        args.output_dir / "replacement_decision.csv", index=False, encoding="utf-8-sig"
    )
    validation_frontend = frontend[frontend["split"] == "validation"].copy()
    colour_summary = []
    for colour in ("black", "white"):
        attempts = int(validation_frontend[f"profile_attempts_{colour}"].sum())
        points = int(validation_frontend[f"subpixel_points_{colour}"].sum())
        colour_summary.append(
            {
                "cell_colour": colour,
                "candidate_points": int(
                    validation_frontend[f"candidate_points_{colour}"].sum()
                ),
                "profile_attempts": attempts,
                "subpixel_points": points,
                "acceptance_coverage": points / attempts if attempts else math.nan,
                "median_peak_height_mean": float(
                    validation_frontend[f"peak_height_median_{colour}"].mean()
                ),
                "median_snr_mean": float(
                    validation_frontend[f"snr_median_{colour}"].mean()
                ),
            }
        )
    colour_summary_frame = pd.DataFrame(colour_summary)
    colour_summary_frame.to_csv(
        args.output_dir / "black_white_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_poses = poses[poses["split"] == "validation"]
    md = f"""# calib02 自适应单色 M1 激光面标定

## 数据隔离

- 001-018 选择二维/三维阈值；001-025 拟合；026-032 在阈值和平面冻结后独立验证。
- 默认信号：`laser_gray - chess_gray`；全局仿射匹配与大尺度二次背景扣除均默认关闭。
- 平面：`{fit.plane[0]:.12g} x + {fit.plane[1]:.12g} y + {fit.plane[2]:.12g} z + {fit.plane[3]:.12g} = 0`。

## 流程

```mermaid
flowchart LR
    A["灰度配对 + 正视化"] --> B["laser-chess"]
    B --> C["逐方格中值基线"]
    C --> D["逐格稳健阈值粗候选"]
    D --> E["正交距离 RANSAC 粗直线"]
    E --> F["沿线法向局部基线/峰高/SNR/能量"]
    F --> G["亚像素灰度重心"]
    G --> H["V2 姿态等权三维后端"]
    H --> I["冻结后独立验证"]
```

## 四方法总体比较

{_markdown_table(comparison)}

## 黑白格恢复统计（验证集）

{_markdown_table(colour_summary_frame)}

## 新方法逐姿态前端统计

{_markdown_table(validation_frontend)}

## 验证集逐姿态误差

{_markdown_table(validation_poses[["algorithm", "image_id", "valid_points", "coverage", "mae_mm", "rmse_mm", "p95_mm", "anomaly_flags"]])}

## 替换判据

{_markdown_table(decision)}

图例：四列对比依次为 V2、上一版 M1、上一版 M2、新 M1A；新方法黑格亚像素点用青色、白格用绿色显示。完整图位于 `visual_comparison/`。
"""
    (args.output_dir / "calibration_report.md").write_text(md, encoding="utf-8")
    html_report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>calib02 自适应单色 M1</title><style>
body{{max-width:1280px;margin:32px auto;padding:0 24px;font:15px/1.6 system-ui;color:#172033}}
h1,h2{{color:#0b3a66}}table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border:1px solid #ccd5df;padding:5px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.flow{{padding:14px;background:#edf7ff;border-left:4px solid #2780b8}}.legend{{padding:12px;background:#fff7df;border-left:4px solid #e0a020}}img{{max-width:100%}}</style></head><body>
<h1>calib02 自适应单色 M1 激光面标定</h1>
<p class="flow">逐格灰度差分与中值基线 → 逐格自适应粗候选 → 正交 RANSAC → 法向局部基线/SNR/能量 → 亚像素重心 → V2 三维后端 → 冻结后验证</p>
<h2>四方法总体比较</h2>{comparison.to_html(index=False, float_format=lambda value: f"{value:.6g}")}
<h2>黑白格验证汇总</h2>{colour_summary_frame.to_html(index=False, float_format=lambda value: f"{value:.6g}")}
<h2>新方法逐姿态前端统计</h2>{validation_frontend.to_html(index=False, float_format=lambda value: f"{value:.6g}")}
<h2>验证集逐姿态误差</h2>{validation_poses.to_html(index=False, float_format=lambda value: f"{value:.6g}")}
<h2>替换判据</h2>{decision.to_html(index=False)}
<p class="legend">图例：四列依次为 V2、上一版 M1、上一版 M2、新 M1A；新方法青=黑格中心，绿=白格中心。</p>
{''.join(f'<h3>{image_id:03d}</h3><img src="visual_comparison/validation_{image_id:03d}.png">' for image_id in args.val_ids)}
</body></html>"""
    (args.output_dir / "calibration_report.html").write_text(
        html_report, encoding="utf-8"
    )


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    setup_logging(args.output_dir)
    v2_config = v2.load_config(args.v2_config)
    intrinsics = base.load_intrinsics(args.intrinsics)
    config = AdaptiveConfig(
        morph_open_enabled=bool(args.enable_morph_open),
        secondary_background_weight=float(args.secondary_background_weight),
    )
    LOGGER.info(
        "固定拆分：tune=%s train=%s validation=%s",
        args.tune_ids,
        args.train_ids,
        args.val_ids,
    )
    LOGGER.info(
        "M1 信号=%s；二次背景权重=%g；3x3 开运算=%s",
        args.m1_signal_mode,
        config.secondary_background_weight,
        config.morph_open_enabled,
    )
    LOGGER.info("冻结阈值和平面前不读取验证图像")

    train_frames: dict[int, AdaptiveFrame] = {}
    for pair in previous.collect_pairs(args, args.train_ids, "train"):
        adaptive = prepare_adaptive_frame(
            pair, args, intrinsics, v2_config, config
        )
        train_frames[adaptive.base_frame.image_id] = adaptive
        LOGGER.info("train ID %03d 自适应信号构建完成", adaptive.base_frame.image_id)
    tune_frames = [train_frames[image_id] for image_id in args.tune_ids]
    line_threshold = derive_line_threshold(tune_frames, config)

    train_results: list[base.FrameResult] = []
    features: dict[int, v2.FeatureFrame] = {}
    frontend_metrics: list[dict[str, Any]] = []
    tune_set = set(args.tune_ids)
    for image_id in args.train_ids:
        item = process_adaptive_method(
            train_frames[image_id],
            line_threshold,
            intrinsics,
            v2_config,
            config,
            args.output_dir,
            args.m1_signal_mode,
        )
        train_results.append(item.frame_result)
        features[image_id] = item.feature_frame
        frontend_metrics.append(item.metrics)
    tune_results = [result for result in train_results if result.image_id in tune_set]
    thresholds = previous.lock_model_thresholds(
        tune_results,
        previous.provisional_thresholds(METHOD, line_threshold),
        v2_config,
    )
    base_frames = {
        image_id: adaptive.base_frame
        for image_id, adaptive in train_frames.items()
    }
    excluded_ids = previous.severe_pair_ids(
        base_frames, args.train_ids, v2_config
    )
    points, pose_ids = v2.stack_result_points(train_results, excluded_ids)
    fit = v2.balanced_plane_ransac(
        points, pose_ids, thresholds.plane_ransac_threshold_mm, v2_config
    )
    v2.mark_training_inliers(train_results, fit, excluded_ids)
    LOGGER.info("M1A 训练平面冻结：%s", fit.plane.tolist())
    LOGGER.info("阈值和平面已冻结，现在开始读取验证图像")

    validation_results: list[base.FrameResult] = []
    for pair in previous.collect_pairs(args, args.val_ids, "validation"):
        adaptive = prepare_adaptive_frame(
            pair, args, intrinsics, v2_config, config
        )
        item = process_adaptive_method(
            adaptive,
            line_threshold,
            intrinsics,
            v2_config,
            config,
            args.output_dir,
            args.m1_signal_mode,
        )
        validation_results.append(item.frame_result)
        features[item.frame_result.image_id] = item.feature_frame
        frontend_metrics.append(item.metrics)
        LOGGER.info(
            "validation ID %03d 使用冻结参数处理完成",
            item.frame_result.image_id,
        )
    v2.finalise_validation(validation_results)
    save_plane_document(
        args, intrinsics, thresholds, fit, excluded_ids, config
    )

    new_rows = [
        v2.overall_metrics(METHOD_LABEL, "train", train_results, fit.plane),
        v2.overall_metrics(
            METHOD_LABEL, "validation", validation_results, fit.plane
        ),
    ]
    baseline_comparison = pd.read_csv(
        args.comparison_baseline_dir / "algorithm_comparison.csv"
    )
    comparison = pd.concat(
        [baseline_comparison, pd.DataFrame(new_rows)], ignore_index=True, sort=False
    )
    new_poses = v2.pose_metrics(
        METHOD_LABEL,
        [*train_results, *validation_results],
        fit.plane,
        features,
        thresholds,
        v2_config,
        excluded_ids,
    )
    baseline_poses = pd.read_csv(
        args.comparison_baseline_dir / "pose_metrics.csv"
    )
    poses = pd.concat([baseline_poses, new_poses], ignore_index=True, sort=False)
    frontend = pd.DataFrame(frontend_metrics)
    frontend.to_csv(
        args.output_dir / "frontend_metrics.csv", index=False, encoding="utf-8-sig"
    )
    make_contact_sheets(args)
    write_reports(args, comparison, poses, frontend, fit)

    audit = [
        {
            "threshold": key,
            "value": value,
            "source_ids": "1-18",
            "validation_accessed": False,
        }
        for key, value in asdict(thresholds).items()
        if key != "source_ids"
    ]
    pd.DataFrame(audit).to_csv(
        args.output_dir / "threshold_selection_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    LOGGER.info("全部结果已写入：%s", args.output_dir.resolve())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (
        base.CalibrationError,
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        cv2.error,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
