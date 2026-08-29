#!/usr/bin/env python3
"""Ground extrinsic calibration with Steger laser-centre extraction.

This script intentionally does not modify ``calibrate_ground_extrinsics.py``.
It reuses that module for chessboard normal estimation, coordinate-frame
construction, diagnostics, and file output, and replaces only the flat-ground
laser stripe centre extractor used for the ground laser images.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter, gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_ground_extrinsics as core  # noqa: E402


LOGGER = logging.getLogger("ground_extrinsics_steger")


@dataclass(frozen=True)
class StegerSettings:
    sigma_px: float = 1.2
    max_offset_px: float = 0.75
    min_normal_y: float = 0.5
    min_response_ratio: float = 0.0005
    background_window_px: int = 31
    min_prominence_ratio: float = 0.010
    profile_smoothing_sigma_px: float = 0.8


def unit_interval(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return result


def positive_odd_int(value: str) -> int:
    result = int(value)
    if result <= 0 or result % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--steger-sigma-px",
        type=positive_float,
        default=StegerSettings.sigma_px,
        help="2-D Gaussian derivative scale in pixels; default 1.2.",
    )
    parser.add_argument(
        "--steger-max-offset-px",
        type=positive_float,
        default=StegerSettings.max_offset_px,
        help="Maximum subpixel shift along the ridge normal; default 0.75.",
    )
    parser.add_argument(
        "--steger-min-normal-y",
        type=unit_interval,
        default=StegerSettings.min_normal_y,
        help="Minimum absolute vertical component of the ridge normal; default 0.5.",
    )
    parser.add_argument(
        "--steger-min-response-ratio",
        type=nonnegative_float,
        default=StegerSettings.min_response_ratio,
        help="Minimum negative-curvature response as a fraction of full scale; default 0.0005.",
    )
    parser.add_argument(
        "--steger-background-window-px",
        type=positive_odd_int,
        default=StegerSettings.background_window_px,
        help="Vertical percentile-filter window for local background removal; default 31.",
    )
    parser.add_argument(
        "--steger-min-prominence-ratio",
        type=nonnegative_float,
        default=StegerSettings.min_prominence_ratio,
        help="Minimum column-peak prominence as a fraction of full scale; default 0.010.",
    )
    parser.add_argument(
        "--steger-profile-smoothing-sigma-px",
        type=nonnegative_float,
        default=StegerSettings.profile_smoothing_sigma_px,
        help="Vertical smoothing sigma before peak search; default 0.8.",
    )
    return parser


def _sensor_full_scale(gray: np.ndarray, sensor_max_value: float | None) -> float:
    if sensor_max_value is not None:
        return float(sensor_max_value)
    if gray.dtype == np.uint8:
        return 255.0
    if np.issubdtype(gray.dtype, np.integer):
        maximum = int(np.max(gray))
        return 4095.0 if maximum <= 4095 else float(np.iinfo(gray.dtype).max)
    return float(np.nanmax(gray))


def derivative_images(
    signal: np.ndarray, sigma_px: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(signal, dtype=np.float32)
    kwargs = {"sigma": sigma_px, "mode": "nearest"}
    gx = gaussian_filter(source, order=(0, 1), **kwargs)
    gy = gaussian_filter(source, order=(1, 0), **kwargs)
    gxx = gaussian_filter(source, order=(0, 2), **kwargs)
    gxy = gaussian_filter(source, order=(1, 1), **kwargs)
    gyy = gaussian_filter(source, order=(2, 0), **kwargs)
    return gx, gy, gxx, gxy, gyy


def steger_point(
    column: int,
    row: int,
    derivatives: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float, float, float, float] | None:
    gx, gy, gxx, gxy, gyy = derivatives
    hessian = np.asarray(
        [[gxx[row, column], gxy[row, column]], [gxy[row, column], gyy[row, column]]],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalue = float(eigenvalues[0])
    if not np.isfinite(eigenvalue) or eigenvalue >= -np.finfo(float).eps:
        return None
    normal = eigenvectors[:, 0]
    gradient = np.asarray([gx[row, column], gy[row, column]], dtype=np.float64)
    offset = -float(gradient @ normal) / eigenvalue
    if not np.isfinite(offset):
        return None
    x = float(column + offset * normal[0])
    y = float(row + offset * normal[1])
    return x, y, -eigenvalue, offset, float(normal[0]), float(normal[1])


def extract_laser_centres_steger(
    gray: np.ndarray,
    args: argparse.Namespace,
    settings: StegerSettings,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    image = gray.astype(np.float32, copy=False)
    full_scale = _sensor_full_scale(gray, args.sensor_max_value)
    background = percentile_filter(
        image,
        percentile=float(args.background_percentile),
        size=(settings.background_window_px, 1),
        mode="nearest",
    )
    corrected = np.maximum(image - background, 0.0)
    signal = (
        gaussian_filter1d(
            corrected,
            sigma=settings.profile_smoothing_sigma_px,
            axis=0,
            mode="nearest",
        )
        if settings.profile_smoothing_sigma_px > 0.0
        else corrected
    )
    derivatives = derivative_images(corrected, settings.sigma_px)
    height, width = gray.shape
    min_prominence = settings.min_prominence_ratio * full_scale
    min_response = settings.min_response_ratio * full_scale

    u_px = np.arange(width, dtype=np.float64)
    v_px = np.full(width, np.nan, dtype=np.float64)
    valid = np.zeros(width, dtype=bool)
    reject_counts = {
        "rejected_prominence": 0,
        "rejected_hessian": 0,
        "rejected_steger_response": 0,
        "rejected_steger_offset": 0,
        "rejected_steger_orientation": 0,
        "rejected_image_bounds": 0,
    }
    for column in range(width):
        profile = signal[:, column]
        peaks, properties = find_peaks(profile, prominence=min_prominence)
        if peaks.size == 0:
            reject_counts["rejected_prominence"] += 1
            continue

        selected = int(np.argmax(profile[peaks]))
        peak = int(peaks[selected])
        point = steger_point(column, peak, derivatives)
        if point is None:
            reject_counts["rejected_hessian"] += 1
            continue

        x, y, response, offset, _normal_x, normal_y = point
        if response < min_response:
            reject_counts["rejected_steger_response"] += 1
        elif abs(offset) > settings.max_offset_px:
            reject_counts["rejected_steger_offset"] += 1
        elif abs(normal_y) < settings.min_normal_y:
            reject_counts["rejected_steger_orientation"] += 1
        elif not (0.0 <= x < width and 0.0 <= y < height):
            reject_counts["rejected_image_bounds"] += 1
        else:
            u_px[column] = x
            v_px[column] = y
            valid[column] = True
    return u_px, v_px, valid, reject_counts


def process_laser_image_steger(
    path: Path,
    intrinsics: core.Intrinsics,
    laser_plane: np.ndarray,
    args: argparse.Namespace,
    overlay_path: Path,
) -> core.LaserObservation:
    gray = core.to_gray(core.read_image(path), path)
    expected_shape = (intrinsics.image_size[1], intrinsics.image_size[0])
    if gray.shape != expected_shape:
        raise core.CalibrationError(
            f"Image size {gray.shape[::-1]} does not match intrinsics {intrinsics.image_size}: {path}"
        )

    settings = _settings_from_args(args)
    u_px, v_px, valid, reject_counts = extract_laser_centres_steger(
        gray, args, settings
    )
    quality_valid = core.continuity_filter(
        u_px,
        v_px,
        valid,
        args.continuity_window,
        args.continuity_max_deviation_px,
    )
    quality_indices = np.flatnonzero(quality_valid)
    candidate_pixels = np.column_stack(
        [u_px[quality_indices], v_px[quality_indices]]
    )
    line_inliers, _ = core.line_ransac_filter(
        candidate_pixels,
        intrinsics,
        args.line_ransac_threshold_px,
    )
    quality_valid[quality_indices[~line_inliers]] = False
    candidate_pixels = candidate_pixels[line_inliers]
    points, intersection_valid = core.intersect_laser_plane(
        candidate_pixels,
        intrinsics,
        laser_plane,
        args.min_depth_mm,
        args.max_depth_mm,
    )
    accepted_pixels = candidate_pixels[intersection_valid]
    if len(accepted_pixels) < 30:
        raise core.CalibrationError(f"Too few valid ground laser points: {len(accepted_pixels)}")

    overlay = core.display_bgr(gray)
    rejected = np.flatnonzero(~quality_valid & np.isfinite(v_px))
    for index in rejected[::4]:
        point = (int(round(float(u_px[index]))), int(round(float(v_px[index]))))
        cv2.circle(overlay, point, 1, (0, 0, 255), -1)
    for point in accepted_pixels:
        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 2, (0, 255, 0), -1)
    label = f"Steger accepted={len(accepted_pixels)}/{len(u_px)}"
    cv2.putText(overlay, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    core.write_image(overlay_path, overlay)
    LOGGER.debug("%s Steger rejection counts: %s", path.name, reject_counts)
    return core.LaserObservation(
        path,
        accepted_pixels,
        points,
        len(u_px),
        int(np.count_nonzero(quality_valid)),
    )


def _settings_from_args(args: argparse.Namespace) -> StegerSettings:
    return StegerSettings(
        sigma_px=float(args.steger_sigma_px),
        max_offset_px=float(args.steger_max_offset_px),
        min_normal_y=float(args.steger_min_normal_y),
        min_response_ratio=float(args.steger_min_response_ratio),
        background_window_px=int(args.steger_background_window_px),
        min_prominence_ratio=float(args.steger_min_prominence_ratio),
        profile_smoothing_sigma_px=float(args.steger_profile_smoothing_sigma_px),
    )


def _metadata(settings: StegerSettings, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "steger_2d",
        **asdict(settings),
        "background_percentile": float(args.background_percentile),
        "post_filters": [
            "continuity_filter",
            "undistorted_image_line_ransac",
            "laser_plane_intersection_depth_range",
        ],
    }


def _rewrite_outputs_with_metadata(
    output_dir: Path,
    result: dict[str, Any],
    settings: StegerSettings,
    args: argparse.Namespace,
) -> None:
    result.setdefault("ground_laser", {})
    metadata = _metadata(settings, args)
    result["ground_laser"]["centre_extraction"] = metadata
    result["algorithm_variant"] = {
        "base_module": "calibrate_ground_extrinsics.py",
        "changed_component": "ground laser centre extraction only",
        "centre_extraction": metadata,
    }
    serialised = core._serialisable(result)
    (output_dir / "camera_ground_extrinsics.yaml").write_text(
        yaml.safe_dump(serialised, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "camera_ground_extrinsics.json").write_text(
        json.dumps(serialised, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = _settings_from_args(args)
    original = core.process_laser_image
    core.process_laser_image = process_laser_image_steger
    try:
        result = core.run(args)
    finally:
        core.process_laser_image = original
    _rewrite_outputs_with_metadata(args.output_dir, result, settings, args)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    try:
        run(args)
    except (
        core.CalibrationError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        cv2.error,
        ValueError,
    ) as exc:
        LOGGER.error("Calibration failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
