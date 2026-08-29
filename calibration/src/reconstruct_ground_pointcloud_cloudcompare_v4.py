#!/usr/bin/env python3
"""CloudCompare V4 export with selectable Steger centreline extractors.

V4 keeps the PLY export, ground-bias compensation, and multi-obstacle reporting
from ``reconstruct_ground_pointcloud_cloudcompare_v2.py``. Both extractor names
use the same ``calibration/src/realtime_steger.py`` implementation; ``shared``
is retained only as a CLI compatibility alias.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import reconstruct_ground_pointcloud_cloudcompare_v2 as cloudcompare_v2  # noqa: E402
import reconstruct_ground_pointcloud_interactive as core  # noqa: E402
import realtime_steger  # noqa: E402
import steger_laser_center as steger  # noqa: E402


@dataclass(frozen=True)
class V4StegerOptions:
    extractor: str
    settings: steger.StegerSettings
    post_filter: str
    ransac_threshold_px: float
    ransac_min_inlier_ratio: float
    help_steger: bool = False


def build_v4_steger_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    steger.add_steger_arguments(parser)
    parser.add_argument(
        "--steger-extractor",
        choices=("measurement-tool", "shared"),
        default="measurement-tool",
        help=(
            "Steger extractor implementation. Both 'measurement-tool' and "
            "'shared' use the unified realtime extractor; 'shared' is a legacy alias."
        ),
    )
    parser.add_argument(
        "--steger-post-filter",
        choices=("reconstruction", "none", "flat-ground-ransac"),
        default="reconstruction",
        help=(
            "Post-filter mode. 'reconstruction' keeps segmented continuous Steger points "
            "without line RANSAC; 'none' keeps every Steger-valid column; "
            "'flat-ground-ransac' additionally applies undistorted line RANSAC."
        ),
    )
    parser.add_argument(
        "--steger-ransac-threshold-px",
        type=steger.positive_float,
        default=2.0,
        help="Undistorted image-space RANSAC threshold for --steger-post-filter flat-ground-ransac.",
    )
    parser.add_argument(
        "--steger-ransac-min-inlier-ratio",
        type=steger.unit_interval,
        default=0.5,
        help="Minimum RANSAC inlier ratio for --steger-post-filter flat-ground-ransac.",
    )
    parser.add_argument(
        "--help-steger",
        action="store_true",
        help="Show V4/Steger options and exit.",
    )
    return parser


def parse_v4_args(argv: list[str]) -> tuple[V4StegerOptions, list[str]]:
    parser = build_v4_steger_parser()
    args, remaining = parser.parse_known_args(argv)
    return (
        V4StegerOptions(
            extractor=str(args.steger_extractor),
            settings=steger.settings_from_args(args),
            post_filter=str(args.steger_post_filter),
            ransac_threshold_px=float(args.steger_ransac_threshold_px),
            ransac_min_inlier_ratio=float(args.steger_ransac_min_inlier_ratio),
            help_steger=bool(args.help_steger),
        ),
        remaining,
    )


def _tool_steger_options() -> dict[str, Any]:
    return realtime_steger.load_steger_options()


def _post_filters(options: V4StegerOptions) -> list[str]:
    if options.extractor == "measurement-tool":
        return ["measurement_tool_steger_backend"]
    if options.post_filter == "none":
        return ["steger_quality_gate_only"]
    if options.post_filter == "flat-ground-ransac":
        return [
            "segment_continuity_filter",
            "undistorted_image_line_ransac",
            "laser_plane_intersection_depth_range",
        ]
    return ["segment_continuity_filter", "laser_plane_intersection_depth_range"]


def _extract_points_for_reconstruction(
    gray: np.ndarray,
    calibration: core.Calibration,
    extraction_params: core.ExtractionParams,
    options: V4StegerOptions,
) -> tuple[np.ndarray, dict[str, float | str | None], np.ndarray]:
    if options.extractor == "measurement-tool":
        tool_options = _tool_steger_options()
        points = realtime_steger.steger_backend(gray, tool_options)
        metadata: dict[str, float | str | None] = {
            "method": "steger_realtime",
            **{
                key: (float(value) if isinstance(value, (int, float)) else str(value))
                for key, value in tool_options.items()
            },
            "post_filter_mode": "measurement-tool",
            "candidate_point_count": float(len(points)),
            "raw_segment_count": 0.0,
            "accepted_segment_count": 0.0,
            "extracted_point_count": float(len(points)),
            "post_filters": ", ".join(_post_filters(options)),
        }
        return points, metadata, gray

    extracted = steger.extract_steger_columns(gray, options.settings)
    metadata = dict(extracted.metadata)
    metadata["post_filter_mode"] = options.post_filter
    metadata["configured_segment_min_columns"] = float(extraction_params.segment_min_columns)
    metadata["configured_continuity_max_column_gap"] = float(
        extraction_params.continuity_max_column_gap
    )
    metadata["configured_continuity_max_vertical_jump"] = float(
        extraction_params.continuity_max_vertical_jump
    )

    if options.post_filter == "none":
        points = extracted.pixels
        metadata.update(
            {
                "candidate_point_count": float(len(points)),
                "raw_segment_count": 0.0,
                "accepted_segment_count": 0.0,
                "extracted_point_count": float(len(points)),
            }
        )
    else:
        points, segment_meta = steger.points_from_valid_columns(
            extracted.u_px,
            extracted.v_px,
            extracted.valid,
            extraction_params.continuity_max_column_gap,
            extraction_params.continuity_max_vertical_jump,
            extraction_params.segment_min_columns,
        )
        metadata.update(segment_meta)

    if options.post_filter == "flat-ground-ransac" and len(points):
        before = len(points)
        inliers, residuals = steger.line_ransac_filter_undistorted(
            points,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            options.ransac_threshold_px,
            min_inlier_ratio=options.ransac_min_inlier_ratio,
        )
        inlier_count = int(np.count_nonzero(inliers))
        metadata["line_ransac_threshold_px"] = float(options.ransac_threshold_px)
        metadata["line_ransac_min_inlier_ratio"] = float(options.ransac_min_inlier_ratio)
        metadata["line_ransac_input_count"] = float(before)
        metadata["line_ransac_inlier_count"] = float(inlier_count)
        finite_residuals = residuals[np.isfinite(residuals)]
        metadata["line_ransac_residual_median_px"] = (
            float(np.median(finite_residuals)) if finite_residuals.size else None
        )
        if inlier_count < max(2, int(np.ceil(options.ransac_min_inlier_ratio * before))):
            raise core.ReconstructionError(
                "Steger flat-ground RANSAC has too few inliers: "
                f"{inlier_count}/{before}"
            )
        points = points[inliers]
        metadata["extracted_point_count"] = float(len(points))

    display = steger.signal_to_u8(
        extracted.corrected_signal,
        steger.sensor_full_scale(gray, options.settings.sensor_max_value),
    )
    metadata["post_filters"] = ", ".join(_post_filters(options))
    return points, metadata, display


def _make_v4_process_frame(
    options: V4StegerOptions,
) -> Any:
    def process_frame_v4(
        image_path: Path,
        output_dir: Path,
        calibration: core.Calibration,
        extraction: core.ExtractionParams,
        reconstruction: core.ReconstructionParams,
        obstacle_fit_params: core.ObstacleFitParams,
        show_3d: bool,
    ) -> dict[str, Any]:
        gray = core.read_mono8(image_path)
        if calibration.image_size is not None:
            actual_size = (gray.shape[1], gray.shape[0])
            if actual_size != calibration.image_size:
                raise core.ReconstructionError(
                    f"Image size {actual_size} does not match intrinsics {calibration.image_size}"
                )

        frame_dir = output_dir / image_path.stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        pixels, extraction_meta, _display = _extract_points_for_reconstruction(
            gray, calibration, extraction, options
        )
        frame, filtered = core.reconstruct_points(pixels, calibration, reconstruction)
        frame.to_csv(frame_dir / "points_ground.csv", index=False, float_format="%.9f")
        core.save_overlay(gray, pixels, frame_dir / "laser_center_overlay.png")
        core.save_pointcloud_plot(frame, frame_dir / "pointcloud_3d.png", image_path.name)
        core.save_top_view(
            frame,
            frame_dir / "top_view_xy.png",
            image_path.name,
            image_aspect_ratio=gray.shape[1] / gray.shape[0],
        )
        obstacle_fit = core.fit_obstacle_line(frame, obstacle_fit_params)
        core.save_obstacle_line_fit(
            frame,
            obstacle_fit,
            frame_dir / "obstacle_line_fit.png",
            image_path.name,
            image_aspect_ratio=gray.shape[1] / gray.shape[0],
        )
        statistics = core.compute_statistics(
            image_path.name, len(pixels), frame, filtered, extraction_meta
        )
        statistics["obstacle_mean_height_mm"] = obstacle_fit.get("mean_height_mm")
        statistics["obstacle_line_fit_rmse_mm"] = obstacle_fit.get("line_fit_rmse_mm")
        statistics["obstacle_point_count"] = obstacle_fit.get("obstacle_point_count", 0)
        statistics["obstacle_line_fit"] = {
            key: value
            for key, value in obstacle_fit.items()
            if key not in {"obstacle_mask", "line_endpoints_xy"}
        }
        with (frame_dir / "statistics.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(statistics, stream, allow_unicode=True, sort_keys=False)
        if show_3d:
            core.show_interactive_pointcloud(frame, image_path.name)
        return statistics

    return process_frame_v4


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    options, remaining = parse_v4_args(raw_argv)
    if options.help_steger:
        build_v4_steger_parser().print_help()
        return 0
    original_process_frame = core.process_frame
    core.process_frame = _make_v4_process_frame(options)
    try:
        if not any(item in ("-h", "--help") for item in raw_argv):
            print(
                "Laser centre extraction: measurement-tool Steger "
                f"{_tool_steger_options()}"
                if options.extractor == "measurement-tool"
                else (
                    "Laser centre extraction: shared Steger "
                    f"(sigma={options.settings.sigma_px:g}px, "
                    f"max_offset={options.settings.max_offset_px:g}px, "
                    f"background={options.settings.background_percentile:g}th percentile/"
                    f"{options.settings.background_window_px}px, "
                    f"prominence_ratio={options.settings.min_prominence_ratio:g}, "
                    f"post_filter={options.post_filter})"
                )
            )
        return cloudcompare_v2.main(remaining)
    finally:
        core.process_frame = original_process_frame


if __name__ == "__main__":
    raise SystemExit(main())
