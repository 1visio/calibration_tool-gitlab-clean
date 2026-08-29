#!/usr/bin/env python3
"""CloudCompare V3 export using Steger laser-centre extraction.

This script reuses ``reconstruct_ground_pointcloud_cloudcompare_v2.py`` for
ground-bias compensation, PLY export, and obstacle segmentation. Only the
laser-centre extractor in ``reconstruct_ground_pointcloud_interactive.py`` is
replaced with a 2-D Steger subpixel ridge extractor.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import reconstruct_ground_pointcloud_cloudcompare_v2 as cloudcompare_v2
import reconstruct_ground_pointcloud_interactive as core


@dataclass(frozen=True)
class StegerSettings:
    sigma_px: float = 1.2
    max_offset_px: float = 0.75
    min_normal_y: float = 0.5
    min_response_ratio: float = 0.0005


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0.0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return result


def unit_interval(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return result


def parse_v3_args(argv: list[str]) -> tuple[StegerSettings, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--steger-sigma-px",
        type=positive_float,
        default=StegerSettings.sigma_px,
        help="2-D Gaussian derivative scale; default 1.2 px.",
    )
    parser.add_argument(
        "--steger-max-offset-px",
        type=positive_float,
        default=StegerSettings.max_offset_px,
        help="Maximum Steger subpixel shift along the normal; default 0.75 px.",
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
    args, remaining = parser.parse_known_args(argv)
    return (
        StegerSettings(
            sigma_px=float(args.steger_sigma_px),
            max_offset_px=float(args.steger_max_offset_px),
            min_normal_y=float(args.steger_min_normal_y),
            min_response_ratio=float(args.steger_min_response_ratio),
        ),
        remaining,
    )


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
    gray: np.ndarray, params: core.ExtractionParams, settings: StegerSettings
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    background = cv2.GaussianBlur(
        gray, (params.background_kernel, params.background_kernel), 0
    )
    signal = cv2.subtract(gray, background).astype(np.float32)
    height, width = gray.shape
    derivatives = derivative_images(signal, settings.sigma_px)
    full_scale = 255.0
    minimum_response = settings.min_response_ratio * full_scale

    candidate_u: list[float] = []
    candidate_v: list[float] = []
    rejected_prominence = 0
    rejected_hessian = 0
    rejected_response = 0
    rejected_offset = 0
    rejected_orientation = 0
    rejected_bounds = 0

    for column in range(width):
        profile = signal[:, column]
        peaks, properties = find_peaks(
            profile, prominence=float(params.min_local_contrast_dn)
        )
        if peaks.size == 0:
            rejected_prominence += 1
            continue

        selected = int(np.argmax(profile[peaks]))
        peak = int(peaks[selected])
        point = steger_point(column, peak, derivatives)
        if point is None:
            rejected_hessian += 1
            continue

        x, y, response, offset, _normal_x, normal_y = point
        if response < minimum_response:
            rejected_response += 1
        elif abs(offset) > settings.max_offset_px:
            rejected_offset += 1
        elif abs(normal_y) < settings.min_normal_y:
            rejected_orientation += 1
        elif not (0.0 <= x < width and 0.0 <= y < height):
            rejected_bounds += 1
        else:
            candidate_u.append(x)
            candidate_v.append(y)

    if candidate_u:
        candidates = np.column_stack([candidate_u, candidate_v]).astype(np.float64)
        order = np.argsort(candidates[:, 0], kind="mergesort")
        candidates = candidates[order]
        breaks = np.where(
            (np.diff(candidates[:, 0]) > params.continuity_max_column_gap)
            | (np.abs(np.diff(candidates[:, 1])) > params.continuity_max_vertical_jump)
        )[0] + 1
        raw_segments = np.split(np.arange(len(candidates)), breaks)
    else:
        candidates = np.empty((0, 2), dtype=np.float64)
        raw_segments = []

    accepted = [
        indexes for indexes in raw_segments if len(indexes) >= params.segment_min_columns
    ]
    points = (
        np.concatenate([candidates[indexes] for indexes in accepted], axis=0)
        if accepted
        else np.empty((0, 2), dtype=np.float64)
    )

    metadata = {
        "method": "steger_2d",
        "min_local_contrast_dn": float(params.min_local_contrast_dn),
        "steger_sigma_px": float(settings.sigma_px),
        "steger_max_offset_px": float(settings.max_offset_px),
        "steger_min_normal_y": float(settings.min_normal_y),
        "steger_min_response_ratio": float(settings.min_response_ratio),
        "candidate_point_count": float(len(candidates)),
        "raw_segment_count": float(len(raw_segments)),
        "accepted_segment_count": float(len(accepted)),
        "extracted_point_count": float(len(points)),
        "rejected_prominence": float(rejected_prominence),
        "rejected_hessian": float(rejected_hessian),
        "rejected_steger_response": float(rejected_response),
        "rejected_steger_offset": float(rejected_offset),
        "rejected_steger_orientation": float(rejected_orientation),
        "rejected_image_bounds": float(rejected_bounds),
    }
    return points, metadata, signal.astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    settings, remaining = parse_v3_args(raw_argv)
    original_extractor = core.extract_laser_centres

    def runtime_extractor(
        gray: np.ndarray, params: core.ExtractionParams
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
        return extract_laser_centres_steger(gray, params, settings)

    core.extract_laser_centres = runtime_extractor
    try:
        if not any(item in ("-h", "--help") for item in raw_argv):
            print(
                "Laser centre extraction: Steger "
                f"(sigma={settings.sigma_px:g}px, max_offset={settings.max_offset_px:g}px, "
                f"min_normal_y={settings.min_normal_y:g}, "
                f"min_response_ratio={settings.min_response_ratio:g})"
            )
        return cloudcompare_v2.main(remaining)
    finally:
        core.extract_laser_centres = original_extractor


if __name__ == "__main__":
    raise SystemExit(main())
